import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:http/http.dart' as http;

import '../models/seal_result.dart';

/// HTTP Client to communicate with the Container Seal OCR backend API.
class SealApiClient {
  /// Base URL of the OCR service.
  final String baseUrl;

  /// Request timeout duration.
  final Duration timeout;

  SealApiClient({
    this.baseUrl = 'http://127.0.0.1:8000',
    this.timeout = const Duration(seconds: 12),
  });

  /// Sends an image file to the OCR API and returns the parsed [SealResult].
  Future<SealResult> recognizeFile(
    String filePath, {
    bool enableDetector = true,
    bool enableBarcode = true,
    bool enableTta = true,
  }) async {
    final file = File(filePath);
    if (!await file.exists()) {
      return SealResult.error('Image file not found at: $filePath');
    }
    final bytes = await file.readAsBytes();
    final filename = filePath.split(Platform.pathSeparator).last;
    return recognizeBytes(
      bytes,
      filename: filename,
      enableDetector: enableDetector,
      enableBarcode: enableBarcode,
      enableTta: enableTta,
    );
  }

  /// Sends raw image bytes to the OCR API and returns the parsed [SealResult].
  Future<SealResult> recognizeBytes(
    Uint8List bytes, {
    String filename = 'seal.jpg',
    bool enableDetector = true,
    bool enableBarcode = true,
    bool enableTta = true,
  }) async {
    final stopwatch = Stopwatch()..start();
    try {
      final uri = Uri.parse('$baseUrl/api/v1/ocr/seal').replace(
        queryParameters: {
          'enable_detector': enableDetector.toString(),
          'enable_barcode': enableBarcode.toString(),
          'enable_tta': enableTta.toString(),
        },
      );

      final request = http.MultipartRequest('POST', uri);
      request.files.add(
        http.MultipartFile.fromBytes(
          'file',
          bytes,
          filename: filename,
        ),
      );

      final streamedResponse = await request.send().timeout(timeout);
      final response = await http.Response.fromStream(streamedResponse);
      stopwatch.stop();

      if (response.statusCode == 200) {
        final data = jsonDecode(utf8.decode(response.bodyBytes))
            as Map<String, dynamic>;
        return SealResult.fromJson(data, latencyMs: stopwatch.elapsedMilliseconds);
      } else if (response.statusCode == 503) {
        return SealResult.error(
          'Server is currently busy processing another seal. Please retry in a moment.',
          latencyMs: stopwatch.elapsedMilliseconds,
        );
      } else {
        return SealResult.error(
          'HTTP ${response.statusCode}: ${response.body}',
          latencyMs: stopwatch.elapsedMilliseconds,
        );
      }
    } on TimeoutException {
      stopwatch.stop();
      return SealResult.error(
        'Request timed out after ${timeout.inSeconds} seconds.',
        latencyMs: stopwatch.elapsedMilliseconds,
      );
    } catch (e) {
      stopwatch.stop();
      return SealResult.error(
        'Connection error: $e',
        latencyMs: stopwatch.elapsedMilliseconds,
      );
    }
  }
}
