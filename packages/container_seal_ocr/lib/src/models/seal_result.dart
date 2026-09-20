import 'seal_status.dart';
import '../rules/shipping_line_resolver.dart';

/// Comprehensive result of a container seal recognition request.
class SealResult {
  /// The final cleaned, rule-corrected seal number (e.g. "OOLKCK28059", "SJJA123456").
  final String? sealNumber;

  /// Identified shipping line or container carrier (e.g. "OOCL", "SITC", "Yang Ming").
  final String? shippingLine;

  /// Model confidence score from 0.0 to 1.0.
  final double confidence;

  /// Classification status: SUCCESS, REVIEW, RECAPTURE, or ERROR.
  final SealStatus status;

  /// Source of the recognition: BARCODE or OCR.
  final SealSource source;

  /// Raw text read by the OCR model before post-processing rules.
  final String? rawText;

  /// Reason code for REVIEW or RECAPTURE (e.g. "LOW_CONFIDENCE", "NO_VALID_SEAL").
  final String? reason;

  /// Total inference and processing latency in milliseconds.
  final int latencyMs;

  /// Version of the model used for recognition.
  final String? modelVersion;

  /// Error message if an exception or network failure occurred.
  final String? errorMessage;

  const SealResult({
    this.sealNumber,
    this.shippingLine,
    this.confidence = 0.0,
    this.status = SealStatus.recapture,
    this.source = SealSource.ocr,
    this.rawText,
    this.reason,
    this.latencyMs = 0,
    this.modelVersion,
    this.errorMessage,
  });

  /// Convenient alias for [sealNumber].
  String? get seal => sealNumber;

  /// True if the seal was recognized with high confidence and valid structure.
  bool get isSuccess => status == SealStatus.success;

  /// True if the result requires human review.
  bool get isReview => status == SealStatus.review;

  /// True if the image could not be read and needs to be recaptured.
  bool get isRecapture => status == SealStatus.recapture;

  /// True if an error occurred during request or inference.
  bool get isError => status == SealStatus.error;

  /// Formatted confidence percentage string (e.g. "98.5%").
  String get confidencePercent => '${(confidence * 100).toStringAsFixed(1)}%';

  /// Factory constructor to parse JSON response from the Seal OCR HTTP API.
  factory SealResult.fromJson(Map<String, dynamic> json, {int latencyMs = 0}) {
    final seal = json['sealNumber'] as String?;
    final resolvedShippingLine = json['shippingLine'] as String? ??
        ShippingLineResolver.resolve(seal);

    return SealResult(
      sealNumber: seal,
      shippingLine: resolvedShippingLine,
      confidence: (json['confidence'] as num?)?.toDouble() ?? 0.0,
      status: SealStatus.fromString(json['status'] as String?),
      source: SealSource.fromString(json['source'] as String?),
      rawText: json['rawText'] as String?,
      reason: json['reason'] as String?,
      latencyMs: json['processingTimeMs'] as int? ?? latencyMs,
      modelVersion: json['modelVersion'] as String?,
      errorMessage: json['detail'] as String?,
    );
  }

  /// Creates a failed or error result.
  factory SealResult.error(String message, {int latencyMs = 0}) {
    return SealResult(
      status: SealStatus.error,
      confidence: 0.0,
      source: SealSource.ocr,
      errorMessage: message,
      latencyMs: latencyMs,
    );
  }

  Map<String, dynamic> toJson() => {
    'sealNumber': sealNumber,
    'shippingLine': shippingLine,
    'confidence': confidence,
    'status': status.toSerializedString(),
    'source': source.name.toUpperCase(),
    'rawText': rawText,
    'reason': reason,
    'latencyMs': latencyMs,
    'modelVersion': modelVersion,
    'errorMessage': errorMessage,
  };

  @override
  String toString() {
    return 'SealResult(seal: $sealNumber, shippingLine: $shippingLine, '
        'status: ${status.name}, confidence: $confidencePercent, '
        'source: ${source.name}, latency: ${latencyMs}ms)';
  }
}
