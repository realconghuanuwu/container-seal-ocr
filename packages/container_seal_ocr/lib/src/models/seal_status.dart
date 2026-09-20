/// Status classifications for Container Seal OCR results.
enum SealStatus {
  /// Seal number read with high confidence and structural validity.
  success,

  /// Seal number detected but confidence is below high threshold, requires human review.
  review,

  /// Seal number could not be reliably recognized or image is blurry. Needs recapture.
  recapture,

  /// Network error, file format error, or unexpected exception.
  error;

  static SealStatus fromString(String? value) {
    switch (value?.toUpperCase()) {
      case 'SUCCESS':
        return SealStatus.success;
      case 'REVIEW':
        return SealStatus.review;
      case 'RECAPTURE':
        return SealStatus.recapture;
      default:
        return SealStatus.error;
    }
  }

  String toSerializedString() {
    switch (this) {
      case SealStatus.success:
        return 'SUCCESS';
      case SealStatus.review:
        return 'REVIEW';
      case SealStatus.recapture:
        return 'RECAPTURE';
      case SealStatus.error:
        return 'ERROR';
    }
  }
}

/// Source of the recognized seal number.
enum SealSource {
  /// Read instantly via barcode (Code 128 / Data Matrix).
  barcode,

  /// Read via Deep Learning OCR model (YOLOv8 + SVTR_LCNet).
  ocr,

  /// Manual entry.
  manual;

  static SealSource fromString(String? value) {
    switch (value?.toUpperCase()) {
      case 'BARCODE':
        return SealSource.barcode;
      case 'OCR':
        return SealSource.ocr;
      default:
        return SealSource.manual;
    }
  }
}

/// Execution mode for recognition.
enum OcrMode {
  /// Send request to backend API (Cloud / Hugging Face Spaces / On-premise).
  cloud,

  /// Run on-device using local ONNX models (100% offline).
  onDevice,

  /// Offline-first: Run on-device; if confidence is low (REVIEW) and online, fallback to Cloud.
  auto;
}
