import os
import re
from dataclasses import dataclass
from pathlib import Path


def _load_env_file() -> None:
    if "PYTEST_CURRENT_TEST" in os.environ:
        return
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if env_file.is_file():
        try:
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key, val = key.strip(), val.strip()
                if key and key not in os.environ:
                    os.environ[key] = val
        except Exception:
            pass


_load_env_file()


@dataclass(frozen=True)
class Settings:
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "10"))
    max_pixels: int = int(os.getenv("MAX_PIXELS", "40000000"))
    max_side: int = int(os.getenv("MAX_SIDE", "2048"))
    min_side: int = int(os.getenv("MIN_SIDE", "160"))
    blur_threshold: float = float(os.getenv("BLUR_THRESHOLD", "20"))
    dark_threshold: float = float(os.getenv("DARK_THRESHOLD", "15"))
    bright_threshold: float = float(os.getenv("BRIGHT_THRESHOLD", "240"))
    ocr_timeout_seconds: float = float(os.getenv("OCR_TIMEOUT_SECONDS", "9"))
    success_threshold: float = float(os.getenv("SUCCESS_THRESHOLD", "0.90"))
    review_threshold: float = float(os.getenv("REVIEW_THRESHOLD", "0.70"))
    seal_allowed_pattern: str = os.getenv("SEAL_ALLOWED_PATTERN", r"^[A-Z0-9/.]+$")
    seal_min_length: int = int(os.getenv("SEAL_MIN_LENGTH", "5"))
    seal_max_length: int = int(os.getenv("SEAL_MAX_LENGTH", "20"))
    detection_model_dir: str | None = os.getenv("DETECTION_MODEL_DIR") or None
    recognition_model_dir: str | None = os.getenv("RECOGNITION_MODEL_DIR") or None
    seal_detector_model: str | None = os.getenv("SEAL_DETECTOR_MODEL") or None
    seal_detector_input_size: int = int(os.getenv("SEAL_DETECTOR_INPUT_SIZE", "960"))
    seal_detector_confidence: float = float(os.getenv("SEAL_DETECTOR_CONFIDENCE", "0.50"))
    seal_detector_iou: float = float(os.getenv("SEAL_DETECTOR_IOU", "0.45"))
    seal_detector_padding: float = float(os.getenv("SEAL_DETECTOR_PADDING", "0.18"))
    seal_detector_padding_x: float = float(os.getenv("SEAL_DETECTOR_PADDING_X", "0.35"))
    seal_detector_padding_y: float = float(os.getenv("SEAL_DETECTOR_PADDING_Y", "0.20"))
    seal_detector_max_regions: int = int(os.getenv("SEAL_DETECTOR_MAX_REGIONS", "3"))
    text_det_unclip_ratio: float = float(os.getenv("TEXT_DET_UNCLIP_RATIO", "1.60"))
    model_version: str = os.getenv("MODEL_VERSION", "pp-ocrv6-medium-base")
    app_env: str = os.getenv("APP_ENV", "standard")

    def __post_init__(self) -> None:
        re.compile(self.seal_allowed_pattern)
        if not (0 <= self.review_threshold <= self.success_threshold <= 1):
            raise ValueError("Confidence thresholds must satisfy 0 <= review <= success <= 1")
        if not (1 <= self.seal_min_length <= self.seal_max_length):
            raise ValueError("Invalid seal length limits")
        if self.ocr_timeout_seconds <= 0 or self.max_upload_mb <= 0:
            raise ValueError("Timeout and upload limit must be positive")
        if self.seal_detector_input_size <= 0 or self.seal_detector_max_regions <= 0:
            raise ValueError("Seal detector size and maximum regions must be positive")
        if not (0 <= self.seal_detector_confidence <= 1 and 0 <= self.seal_detector_iou <= 1):
            raise ValueError("Seal detector confidence and IoU must be between 0 and 1")
        if self.seal_detector_padding < 0:
            raise ValueError("Seal detector padding cannot be negative")
