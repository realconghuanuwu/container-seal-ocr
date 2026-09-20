from dataclasses import dataclass
from io import BytesIO

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .config import Settings


MIME_FORMATS = {
    "image/jpeg": "JPEG",
    "image/png": "PNG",
    "image/webp": "WEBP",
}


class ImageError(ValueError):
    def __init__(self, code: str, http_status: int = 400):
        super().__init__(code)
        self.code = code
        self.http_status = http_status


@dataclass
class PreparedImage:
    original: np.ndarray
    normalized: np.ndarray


def prepare(data: bytes, mime: str | None, settings: Settings) -> PreparedImage:
    if mime not in MIME_FORMATS:
        raise ImageError("UNSUPPORTED_MIME", 415)
    if not data:
        raise ImageError("EMPTY_UPLOAD")
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise ImageError("FILE_TOO_LARGE", 413)
    try:
        with Image.open(BytesIO(data)) as image:
            if image.format != MIME_FORMATS[mime]:
                raise ImageError("MIME_CONTENT_MISMATCH", 415)
            width, height = image.size
            if width * height > settings.max_pixels:
                raise ImageError("IMAGE_TOO_LARGE", 413)
            image.load()
            original = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
            normalized = cv2.cvtColor(
                np.asarray(ImageOps.exif_transpose(image).convert("RGB")), cv2.COLOR_RGB2BGR
            )
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        if isinstance(exc, ImageError):
            raise
        raise ImageError("CORRUPTED_IMAGE") from exc
    return PreparedImage(original=original, normalized=normalized)


def quality_reason(image: np.ndarray, settings: Settings) -> str | None:
    height, width = image.shape[:2]
    if min(height, width) < settings.min_side:
        return "IMAGE_TOO_SMALL"
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if max(height, width) > 1024:
        scale = 1024 / max(height, width)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    mean = float(gray.mean())
    contrast = float(gray.std())
    if mean < settings.dark_threshold and contrast < 10:
        return "IMAGE_TOO_DARK"
    if mean > settings.bright_threshold and contrast < 10:
        return "IMAGE_TOO_BRIGHT"
    if cv2.Laplacian(gray, cv2.CV_64F).var() < settings.blur_threshold:
        return "IMAGE_TOO_BLURRY"
    return None


def resize(image: np.ndarray, settings: Settings) -> np.ndarray:
    height, width = image.shape[:2]
    if max(height, width) <= settings.max_side:
        return image
    scale = settings.max_side / max(height, width)
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
