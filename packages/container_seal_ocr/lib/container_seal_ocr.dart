library container_seal_ocr;

import 'dart:io';
import 'dart:typed_data';

import 'src/client/seal_api_client.dart';
import 'src/models/seal_result.dart';
import 'src/models/seal_status.dart';
import 'src/ondevice/seal_ondevice_engine.dart';
import 'src/rules/seal_rules.dart';
import 'src/rules/shipping_line_resolver.dart';

export 'src/models/seal_result.dart';
export 'src/models/seal_status.dart';
export 'src/rules/seal_rules.dart';
export 'src/rules/shipping_line_resolver.dart';
export 'src/client/seal_api_client.dart';
export 'src/ondevice/seal_ondevice_engine.dart';

/// Container Seal OCR Engine - Single entrypoint for mobile applications.
/// Supports 100% Offline On-Device AI by default (No Internet required).
class ContainerSealOcr {
  static SealApiClient _client = SealApiClient();
  static OcrMode _defaultMode = OcrMode.onDevice;

  /// Configures default engine settings.
  /// Set [defaultMode] to [OcrMode.onDevice] for 100% offline inference (default).
  /// Set [defaultMode] to [OcrMode.cloud] to send requests to an API endpoint.
  static void configure({
    OcrMode defaultMode = OcrMode.onDevice,
    String? baseUrl,
    Duration? timeout,
  }) {
    _defaultMode = defaultMode;
    if (baseUrl != null || timeout != null) {
      _client = SealApiClient(
        baseUrl: baseUrl ?? _client.baseUrl,
        timeout: timeout ?? _client.timeout,
      );
    }
  }

  /// Recognizes container seal number from an image file path.
  ///
  /// [mode] defaults to [OcrMode.onDevice] (100% OFFLINE, no internet required).
  ///
  /// Returns a [SealResult] containing:
  /// - `sealNumber`: The cleaned seal number (e.g. "OOLKCK28059")
  /// - `shippingLine`: The shipping line (e.g. "OOCL", "SITC", "Yang Ming")
  /// - `confidence`: Confidence score (0.0 to 1.0)
  /// - `status`: Status enum (`SealStatus.success`, `SealStatus.review`, `SealStatus.recapture`)
  /// - `source`: Source enum (`SealSource.barcode`, `SealSource.ocr`)
  /// - `latencyMs`: Processing duration in milliseconds
  static Future<SealResult> recognizeFile(
    String filePath, {
    OcrMode? mode,
    bool enableDetector = true,
    bool enableBarcode = true,
    bool enableTta = true,
  }) async {
    final effectiveMode = mode ?? _defaultMode;

    if (effectiveMode == OcrMode.onDevice) {
      final file = File(filePath);
      if (!await file.exists()) {
        return SealResult.error('Image file not found at: $filePath');
      }
      final bytes = await file.readAsBytes();
      return recognizeBytes(bytes, mode: OcrMode.onDevice);
    } else {
      return _client.recognizeFile(
        filePath,
        enableDetector: enableDetector,
        enableBarcode: enableBarcode,
        enableTta: enableTta,
      );
    }
  }

  /// Recognizes container seal number from raw image bytes.
  ///
  /// [mode] defaults to [OcrMode.onDevice] (100% OFFLINE, no internet required).
  static Future<SealResult> recognizeBytes(
    Uint8List bytes, {
    OcrMode? mode,
    String filename = 'seal.jpg',
    bool enableDetector = true,
    bool enableBarcode = true,
    bool enableTta = true,
  }) async {
    final effectiveMode = mode ?? _defaultMode;

    if (effectiveMode == OcrMode.onDevice) {
      final engine = await SealOnDeviceEngine.getInstance();
      return engine.recognize(bytes);
    } else {
      return _client.recognizeBytes(
        bytes,
        filename: filename,
        enableDetector: enableDetector,
        enableBarcode: enableBarcode,
        enableTta: enableTta,
      );
    }
  }

  /// Resolves the shipping line from an existing seal number.
  static String? resolveShippingLine(String sealNumber) {
    return ShippingLineResolver.resolve(sealNumber);
  }

  /// Cleans and corrects a raw OCR string with homoglyph disambiguation and CTC collapse recovery.
  static String cleanSealNumber(String rawText) {
    return SealRules.cleanSealNumber(rawText);
  }
}
