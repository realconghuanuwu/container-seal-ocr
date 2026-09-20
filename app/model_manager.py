from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import Settings

logger = logging.getLogger("seal_ocr")

BASE_MODEL_ID = "pp-ocrv6-medium-base"
DEFAULT_BASE_MODEL = {
    "id": BASE_MODEL_ID,
    "name": "PP-OCRv6 Base (Official)",
    "description": "Mô hình gốc PaddleOCR v6 medium, không fine-tune (Zero-shot)",
    "type": "base",
    "recognition_dir": None,
}


def find_models_dir(settings: Settings | None = None) -> Path | None:
    """Find the root directory containing available models."""
    if settings and settings.recognition_model_dir:
        rec_path = Path(settings.recognition_model_dir)
        if rec_path.parent.is_dir():
            return rec_path.parent

    container_models = Path("/models")
    if container_models.is_dir():
        return container_models

    repo_models = Path(__file__).resolve().parent.parent / "models"
    if repo_models.is_dir():
        return repo_models

    return None


def is_recognition_model_dir(path: Path) -> bool:
    """Check if directory contains a PaddleOCR recognition model."""
    if not path.is_dir():
        return False
    rec_indicators = (
        "inference.pdiparams",
        "model.pdiparams",
        "inference.json",
        "inference.yml",
    )
    return any((path / indicator).is_file() for indicator in rec_indicators)


def list_available_models(settings: Settings, active_version: str | None = None) -> list[dict]:
    """List all available recognition models including the official base model."""
    active_version = active_version or settings.model_version
    models: list[dict] = []

    models_dir = find_models_dir(settings)
    fine_tuned_found = False

    if models_dir and models_dir.is_dir():
        for sub_dir in sorted(models_dir.iterdir()):
            if not sub_dir.is_dir():
                continue
            if is_recognition_model_dir(sub_dir):
                fine_tuned_found = True
                model_id = sub_dir.name
                is_active = (
                    active_version == model_id
                    or (model_id in (active_version or ""))
                    or (
                        settings.recognition_model_dir
                        and Path(settings.recognition_model_dir).resolve() == sub_dir.resolve()
                    )
                )
                if model_id == "seal-ocr-rec-v1":
                    env_label = getattr(settings, "app_env", "prod").upper()
                    if env_label in ("DEV", "DEVELOPMENT"):
                        name = "Seal OCR Dev v1"
                        desc = "Mô hình Dev v1 fine-tuned trên 14,500+ ảnh seal chuẩn (Môi trường Development)"
                    elif env_label in ("UAT", "STAGING"):
                        name = "Seal OCR UAT v1"
                        desc = "Mô hình UAT Release Candidate v1 (Accuracy 82.67%, CER 7.33%)"
                    else:
                        name = "Seal OCR Production v1"
                        desc = "Mô hình Production v1 fine-tuned trên 14,500+ ảnh seal chuẩn (Accuracy 82.67%, CER 7.33%)"
                elif "-uat" in model_id or "-rc" in model_id:
                    name = f"Seal OCR UAT ({model_id})"
                    desc = f"Mô hình ứng viên UAT/Staging ({model_id})"
                elif "-dev" in model_id:
                    name = f"Seal OCR Dev ({model_id})"
                    desc = f"Mô hình thử nghiệm Development ({model_id})"
                else:
                    name = model_id.replace("-", " ").title()
                    desc = f"Mô hình fine-tune nhận dạng ({model_id})"

                models.append({
                    "id": model_id,
                    "name": name,
                    "description": desc,
                    "type": "fine-tuned",
                    "recognition_dir": str(sub_dir.resolve()),
                    "is_active": bool(is_active),
                })

    is_base_active = (
        active_version == BASE_MODEL_ID
        or (not fine_tuned_found and settings.recognition_model_dir is None)
        or not any(m["is_active"] for m in models)
    )

    base_entry = dict(DEFAULT_BASE_MODEL)
    base_entry["is_active"] = bool(is_base_active)
    # If base is active, ensure other models are marked inactive
    if is_base_active:
        for m in models:
            m["is_active"] = False

    return [base_entry] + models


def resolve_model(model_id: str, settings: Settings) -> tuple[str, str | None]:
    """Resolve a model id to (model_version, recognition_model_dir)."""
    if model_id in (BASE_MODEL_ID, "base", "default"):
        return BASE_MODEL_ID, None

    models_dir = find_models_dir(settings)
    if not models_dir or not models_dir.is_dir():
        raise ValueError(f"Models directory not found; cannot switch to {model_id}")

    target_dir = models_dir / model_id
    if target_dir.is_dir() and is_recognition_model_dir(target_dir):
        rec_suffix = model_id.removeprefix("seal-ocr-")
        version = f"seal-ocr-det-v1-{rec_suffix}" if "rec" in model_id and settings.seal_detector_model else model_id
        return version, str(target_dir)

    # Check if model_id is a full version string like seal-ocr-det-v1-rec-v2
    for sub_dir in models_dir.iterdir():
        if sub_dir.is_dir() and is_recognition_model_dir(sub_dir):
            if sub_dir.name in model_id:
                return model_id, str(sub_dir)

    raise ValueError(f"Unknown or invalid model id: {model_id}")
