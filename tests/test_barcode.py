"""Unit tests for multi-engine barcode and QR code decoding in app/barcode.py."""

import cv2
import numpy as np
import pytest
from app.barcode import decode, extract_seal_from_payload


def test_extract_seal_from_payload():
    # Plain text payload
    assert extract_seal_from_payload("SJJA696362") == "SJJA696362"
    assert extract_seal_from_payload("  28705257  ") == "28705257"

    # URL query parameter with 'id', 'seal', 'sn', etc.
    assert extract_seal_from_payload("https://seal.example.com/v?id=SJJA696362") == "SJJA696362"
    assert extract_seal_from_payload("http://verify.megafortris.com/check?seal=MF123456&t=1") == "MF123456"
    assert extract_seal_from_payload("https://tracker.com/seal?sn=OOLKCK1476") == "OOLKCK1476"

    # URL trailing path component
    assert extract_seal_from_payload("https://verify.seal.com/seals/SJJA696362") == "SJJA696362"
    assert extract_seal_from_payload("https://api.line.com/v1/28705257/") == "28705257"


def test_decode_qr_benchmark_image():
    # Test on a known benchmark image containing a QR code: e61a3904... -> R5847075
    img = cv2.imread("benchmark/images/e61a3904-a666-4d34-8f6d-f1cdd6397ef4.jpg")
    assert img is not None, "Benchmark image must exist"
    codes = decode(img)
    assert "R5847075" in codes


def test_decode_qr_benchmark_image_sjja():
    # Test on a known benchmark image containing a QR code: e6c30f73... -> SJJA700724
    img = cv2.imread("benchmark/images/e6c30f73-fac5-46fd-b294-a3d86b37852f.jpg")
    assert img is not None, "Benchmark image must exist"
    codes = decode(img)
    assert "SJJA700724" in codes


def test_decode_1d_barcode_benchmark_image():
    # Test on a known benchmark image containing 1D barcode Code 128: e67a59e4... -> FX37404335
    img = cv2.imread("benchmark/images/e67a59e4-a451-48c1-ad92-e48603edd7b6.jpg")
    assert img is not None, "Benchmark image must exist"
    codes = decode(img)
    assert "FX37404335" in codes


def test_decode_rotated_barcode():
    # Test that a 90-degree rotated barcode image is successfully decoded
    img = cv2.imread("benchmark/images/e67a59e4-a451-48c1-ad92-e48603edd7b6.jpg")
    assert img is not None
    rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    codes = decode(rotated)
    assert "FX37404335" in codes


def test_decode_blank_or_noise_image():
    blank = np.zeros((200, 200, 3), dtype=np.uint8)
    assert decode(blank) == []

    empty = np.array([], dtype=np.uint8)
    assert decode(empty) == []
