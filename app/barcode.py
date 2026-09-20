"""Multi-Engine Barcode & QR Code Decoder Module for Shipping Container Seals (ISO 17712).

Supports 1D Barcodes (Code 128, Code 39, ITF) and 2D Codes (QR Code, Micro QR, DataMatrix).
Features multi-perspective rotation scanning and adaptive CLAHE contrast enhancement for laser-etched seals.
Excludes retail grocery barcode formats (UPC-A, EAN-8) to prevent false-positive phantom detections on container door rivets.
"""

import urllib.parse
from typing import List, Optional

import cv2
import numpy as np

try:
    import zxingcpp
    HAS_ZXING = True
except ImportError:
    HAS_ZXING = False

# Fallback OpenCV QR detector
_CV_QR_DETECTOR = None


def _get_cv_qr_detector():
    global _CV_QR_DETECTOR
    if _CV_QR_DETECTOR is None:
        if hasattr(cv2, "QRCodeDetectorAruco"):
            _CV_QR_DETECTOR = cv2.QRCodeDetectorAruco()
        elif hasattr(cv2, "QRCodeDetector"):
            _CV_QR_DETECTOR = cv2.QRCodeDetector()
    return _CV_QR_DETECTOR


def extract_seal_from_payload(payload: str) -> str:
    """Extracts candidate seal number from barcode/QR payload.

    If the payload is a URL (e.g. https://seal.example.com/v?id=SJJA696362 or .../SJJA696362),
    extracts the query parameter or the trailing path component.
    """
    raw = payload.strip()
    if raw.startswith("http://") or raw.startswith("https://"):
        try:
            parsed = urllib.parse.urlparse(raw)
            # Check query params for common seal keys
            query_dict = urllib.parse.parse_qs(parsed.query)
            for key in ("id", "seal", "sn", "code", "no", "number", "s"):
                if key in query_dict and query_dict[key]:
                    return query_dict[key][0].strip()
            # If no query parameter, check the last path component
            path_parts = [p for p in parsed.path.split("/") if p]
            if path_parts:
                return path_parts[-1].strip()
        except Exception:
            pass
    return raw


def _decode_frame(img: np.ndarray) -> List[str]:
    """Decodes barcodes/QRs from a single image frame using industrial seal formats."""
    results: List[str] = []

    # 1. Primary Engine: zxing-cpp (Industrial 1D & 2D formats)
    if HAS_ZXING:
        try:
            zx_codes = zxingcpp.read_barcodes(img)
            # Whitelist genuine industrial container seal formats:
            # Excludes retail UPC/EAN which can trigger on container door grooves.
            valid_formats = (
                zxingcpp.BarcodeFormat.QRCode,
                zxingcpp.BarcodeFormat.MicroQRCode,
                zxingcpp.BarcodeFormat.RMQRCode,
                zxingcpp.BarcodeFormat.DataMatrix,
                zxingcpp.BarcodeFormat.Code128,
                zxingcpp.BarcodeFormat.Code39,
                zxingcpp.BarcodeFormat.Code93,
                zxingcpp.BarcodeFormat.ITF,
            )
            for code in zx_codes:
                if code.text and code.format in valid_formats:
                    extracted = extract_seal_from_payload(code.text)
                    if extracted and extracted not in results:
                        results.append(extracted)
        except Exception:
            pass

    if results:
        return results

    # 2. Secondary Engine: OpenCV QR Detector (QR codes only)
    qr_detector = _get_cv_qr_detector()
    if qr_detector:
        try:
            ok, points = qr_detector.detect(img)
            if ok and points is not None:
                text, _ = qr_detector.decode(img, points)
                if text:
                    extracted = extract_seal_from_payload(text)
                    if extracted and extracted not in results:
                        results.append(extracted)
        except Exception:
            pass

    return results


def decode(image: np.ndarray) -> List[str]:
    """Decodes all 1D barcodes and 2D QR / DataMatrix codes from an image.

    Tries normal orientation, multi-angle rotations (90°, 180°, 270°),
    and adaptive CLAHE contrast enhancement for laser-etched metallic surfaces.
    """
    if image is None or image.size == 0:
        return []

    # Step 1: Normal orientation
    decoded = _decode_frame(image)
    if decoded:
        return decoded

    # Step 2: Try rotations: 90°, 180°, 270° (many container bolt seals are horizontal/upside-down)
    for rot in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE):
        rotated = cv2.rotate(image, rot)
        decoded = _decode_frame(rotated)
        if decoded:
            return decoded

    # Step 3: CLAHE enhancement for laser-etched metal reflections
    try:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        decoded = _decode_frame(enhanced)
        if decoded:
            return decoded

        # Also try rotated on enhanced
        for rot in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE):
            rot_enh = cv2.rotate(enhanced, rot)
            decoded = _decode_frame(rot_enh)
            if decoded:
                return decoded
    except Exception:
        pass

    return []
