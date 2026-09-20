import 'dart:math';
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:onnxruntime/onnxruntime.dart';

import '../models/seal_result.dart';
import '../models/seal_status.dart';
import '../rules/seal_rules.dart';
import '../rules/shipping_line_resolver.dart';
import 'ctc_decoder.dart';

/// 100% On-Device Offline AI Engine for Container Seal Detection and Recognition.
/// Operates entirely on the mobile device without any internet connectivity.
class SealOnDeviceEngine {
  static SealOnDeviceEngine? _instance;
  static bool _isInitialized = false;

  late final OrtSession _detectorSession;
  late final OrtSession _recognizerSession;
  late final CtcDecoder _ctcDecoder;

  SealOnDeviceEngine._();

  /// Gets or initializes the singleton offline engine.
  static Future<SealOnDeviceEngine> getInstance() async {
    if (_instance == null) {
      _instance = SealOnDeviceEngine._();
      await _instance!._init();
    }
    return _instance!;
  }

  Future<void> _init() async {
    if (_isInitialized) return;

    // 1. Initialize ONNX Runtime environment
    OrtEnv.instance.init();
    final sessionOptions = OrtSessionOptions()
      ..setInterOpNumThreads(1)
      ..setIntraOpNumThreads(2)
      ..setSessionGraphOptimizationLevel(GraphOptimizationLevel.ortEnableAll);

    // Helper to load assets in both package import and standalone contexts
    Future<ByteData> loadAsset(String relativePath) async {
      try {
        return await rootBundle.load('packages/container_seal_ocr/$relativePath');
      } catch (_) {
        return await rootBundle.load(relativePath);
      }
    }

    Future<String> loadStringAsset(String relativePath) async {
      try {
        return await rootBundle.loadString('packages/container_seal_ocr/$relativePath');
      } catch (_) {
        return await rootBundle.loadString(relativePath);
      }
    }

    // 2. Load Detector model from assets
    final detectorBytes = await loadAsset('assets/seal_detector.onnx');
    _detectorSession = OrtSession.fromBuffer(
      detectorBytes.buffer.asUint8List(),
      sessionOptions,
    );

    // 3. Load Recognizer model (INT8 quantized) from assets
    final recognizerBytes = await loadAsset('assets/seal_ocr_int8.onnx');
    _recognizerSession = OrtSession.fromBuffer(
      recognizerBytes.buffer.asUint8List(),
      sessionOptions,
    );

    // 4. Load character dictionary
    final dictContent = await loadStringAsset('assets/char_dict.txt');
    final charList = dictContent
        .split('\n')
        .map((e) => e.replaceAll('\r', ''))
        .where((e) => e.isNotEmpty)
        .toList();

    _ctcDecoder = CtcDecoder(characterDict: charList);
    _isInitialized = true;
  }

  /// Runs full offline pipeline: YOLOv8 Detector -> Crop -> SVTR OCR -> CTC Decode -> Postprocessing Rules.
  Future<SealResult> recognize(Uint8List imageBytes) async {
    final stopwatch = Stopwatch()..start();

    // 1. Decode image using pure Dart image package
    final originalImage = img.decodeImage(imageBytes);
    if (originalImage == null) {
      return SealResult.error('Failed to decode image bytes.');
    }

    try {
      // 2. Run Seal Detector to find region of interest
      final croppedSeal = await _detectAndCrop(originalImage);

      // 3. Run Recognition on the cropped seal patch
      final ocrResult = await _runRecognizer(croppedSeal);

      stopwatch.stop();

      // 4. Rule-based postprocessing & homoglyph disambiguation
      final rawText = ocrResult.text;
      final cleanedText = SealRules.cleanSealNumber(rawText);

      // 5. Automatic shipping line resolution
      final shippingLine = ShippingLineResolver.resolve(cleanedText);

      // 6. Confidence & Status classification
      SealStatus status;
      String? reason;

      if (cleanedText.isEmpty) {
        status = SealStatus.recapture;
        reason = 'NO_VALID_SEAL';
      } else if (ocrResult.confidence >= 0.85) {
        status = SealStatus.success;
      } else if (ocrResult.confidence >= 0.60) {
        status = SealStatus.review;
        reason = 'LOW_CONFIDENCE';
      } else {
        status = SealStatus.recapture;
        reason = 'LOW_CONFIDENCE';
      }

      return SealResult(
        sealNumber: cleanedText.isNotEmpty ? cleanedText : null,
        shippingLine: shippingLine,
        confidence: ocrResult.confidence,
        status: status,
        source: SealSource.ocr,
        rawText: rawText,
        reason: reason,
        latencyMs: stopwatch.elapsedMilliseconds,
        modelVersion: 'seal-ocr-det-v1-rec-v1-offline-int8',
      );
    } catch (e) {
      stopwatch.stop();
      return SealResult.error(
        'Offline inference error: $e',
        latencyMs: stopwatch.elapsedMilliseconds,
      );
    }
  }

  /// Detects seal location with YOLOv8 and crops the target bounding box.
  Future<img.Image> _detectAndCrop(img.Image image) async {
    const int targetSize = 960;
    final resized = img.copyResize(image, width: targetSize, height: targetSize);

    // Normalize image to float32 [0.0, 1.0] in CHW format
    final floatList = Float32List(1 * 3 * targetSize * targetSize);
    int rIdx = 0;
    int gIdx = targetSize * targetSize;
    int bIdx = 2 * targetSize * targetSize;

    for (int y = 0; y < targetSize; y++) {
      for (int x = 0; x < targetSize; x++) {
        final pixel = resized.getPixel(x, y);
        floatList[rIdx++] = pixel.r / 255.0;
        floatList[gIdx++] = pixel.g / 255.0;
        floatList[bIdx++] = pixel.b / 255.0;
      }
    }

    final inputTensor = OrtValueTensor.createTensorWithDataList(
      floatList,
      [1, 3, targetSize, targetSize],
    );

    final runOptions = OrtRunOptions();
    final outputs = await _detectorSession.runAsync(
      runOptions,
      {'images': inputTensor},
    );
    inputTensor.release();
    runOptions.release();

    if (outputs == null || outputs.isEmpty || outputs[0] == null) {
      return image;
    }

    final rawOutput = outputs[0]!.value as List<dynamic>;
    outputs[0]!.release();

    // Parse YOLOv8 output of shape [1, 5, 18900] -> (cx, cy, w, h, conf)
    double maxConf = 0.0;
    double bestCx = 0.0, bestCy = 0.0, bestW = 0.0, bestH = 0.0;

    final batch = rawOutput[0] as List<dynamic>;
    final numBoxes = (batch[0] as List<dynamic>).length;

    for (int i = 0; i < numBoxes; i++) {
      final conf = (batch[4][i] as num).toDouble();
      if (conf > maxConf && conf > 0.35) {
        maxConf = conf;
        bestCx = (batch[0][i] as num).toDouble();
        bestCy = (batch[1][i] as num).toDouble();
        bestW = (batch[2][i] as num).toDouble();
        bestH = (batch[3][i] as num).toDouble();
      }
    }

    if (maxConf < 0.35) {
      // Fallback: If no high confidence seal detected, process the entire image
      return image;
    }

    // Convert normalized coordinates back to original image dimensions
    final scaleX = image.width / targetSize;
    final scaleY = image.height / targetSize;

    final x1 = max(0, ((bestCx - bestW / 2) * scaleX).round());
    final y1 = max(0, ((bestCy - bestH / 2) * scaleY).round());
    final w = min(image.width - x1, (bestW * scaleX).round());
    final h = min(image.height - y1, (bestH * scaleY).round());

    if (w <= 10 || h <= 10) return image;

    return img.copyCrop(image, x: x1, y: y1, width: w, height: h);
  }

  /// Runs SVTR text recognition on the cropped seal patch.
  Future<CtcResult> _runRecognizer(img.Image patch) async {
    const int targetH = 48;
    const int targetW = 320;
    final resized = img.copyResize(patch, width: targetW, height: targetH);

    // Normalize: (pixel / 255.0 - 0.5) / 0.5 -> standard PaddleOCR normalization
    final floatList = Float32List(1 * 3 * targetH * targetW);
    int rIdx = 0;
    int gIdx = targetH * targetW;
    int bIdx = 2 * targetH * targetW;

    for (int y = 0; y < targetH; y++) {
      for (int x = 0; x < targetW; x++) {
        final pixel = resized.getPixel(x, y);
        floatList[rIdx++] = (pixel.r / 255.0 - 0.5) / 0.5;
        floatList[gIdx++] = (pixel.g / 255.0 - 0.5) / 0.5;
        floatList[bIdx++] = (pixel.b / 255.0 - 0.5) / 0.5;
      }
    }

    final inputTensor = OrtValueTensor.createTensorWithDataList(
      floatList,
      [1, 3, targetH, targetW],
    );

    final runOptions = OrtRunOptions();
    final outputs = await _recognizerSession.runAsync(
      runOptions,
      {'x': inputTensor},
    );
    inputTensor.release();
    runOptions.release();

    if (outputs == null || outputs.isEmpty || outputs[0] == null) {
      return const CtcResult(text: '', confidence: 0.0);
    }

    final rawOutput = outputs[0]!.value as List<dynamic>;
    outputs[0]!.release();

    // Shape is [1, timeSteps, numClasses]
    final seqData = rawOutput[0] as List<dynamic>;
    final probabilities = <List<double>>[];

    for (final step in seqData) {
      final stepList = (step as List<dynamic>)
          .map((prob) => (prob as num).toDouble())
          .toList();
      probabilities.add(stepList);
    }

    // Decode with pure Dart CTC decoder
    return _ctcDecoder.decode(probabilities);
  }

  /// Releases model resources and sessions.
  void dispose() {
    if (_isInitialized) {
      _detectorSession.release();
      _recognizerSession.release();
      _isInitialized = false;
    }
  }
}
