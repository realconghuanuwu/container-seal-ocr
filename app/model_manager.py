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


import re


def extract_detector_tag(settings: Settings) -> str:
    """Extract detector tag in SemVer format (e.g. 'det-v1.0.0')."""
    if not settings.seal_detector_model:
        return "det-v1.0.0"
    p = Path(settings.seal_detector_model)
    parent_name = p.parent.name
    match = re.search(r"det(?:ector)?-v?([0-9]+(?:\.[0-9]+)*)", parent_name, re.IGNORECASE)
    if match:
        raw_ver = match.group(1)
        semver = raw_ver if "." in raw_ver else f"{raw_ver}.0.0"
        return f"det-v{semver}"
    return "det-v1.0.0"


def extract_rec_tag(model_id: str) -> str:
    """Extract recognition tag in SemVer format (e.g. 'rec-v1.0.0' or 'rec-v1.0.3')."""
    if "rec-v4-manual" in model_id or "rec-v1.0.3" in model_id:
        return "rec-v1.0.3"
    if "rec-v1" in model_id and "v1.0" not in model_id:
        return "rec-v1.0.0"
    match = re.search(r"rec-v?([0-9]+(?:\.[0-9]+)*)", model_id, re.IGNORECASE)
    if match:
        raw_ver = match.group(1)
        semver = raw_ver if "." in raw_ver else f"{raw_ver}.0.0"
        return f"rec-v{semver}"
    return model_id.removeprefix("seal-ocr-").removeprefix("seal-")


def build_pipeline_version(model_id: str, settings: Settings) -> str:
    """Build standardized pipeline version string: seal-det-vX.Y.Z-rec-vA.B.C."""
    det_tag = extract_detector_tag(settings)
    rec_tag = extract_rec_tag(model_id)
    return f"seal-{det_tag}-{rec_tag}"


def format_model_display(model_id: str, settings: Settings) -> tuple[str, str]:
    """Format human-readable name and description for UI model selector."""
    rec_tag = extract_rec_tag(model_id)

    if rec_tag == "rec-v1.0.0":
        name = "Seal OCR (rec-v1.0.0)"
        desc = "Mô hình v1.0.0 fine-tuned trên 14,500+ ảnh seal chuẩn (Accuracy 82.67%, CER 7.33%)"
    elif rec_tag == "rec-v1.0.3":
        name = "Seal OCR (rec-v1.0.3)"
        desc = "Mô hình v1.0.3 fine-tuned trên 14,500 ảnh base + 100 ảnh manual"
    elif rec_tag == "rec-v1.0.4":
        name = "Seal OCR (rec-v1.0.4)"
        desc = "Mô hình v1.0.4 2-stage (45 base + 10 oversample epochs, target >=90%)"
    else:
        name = f"Seal OCR ({rec_tag})"
        desc = f"Mô hình fine-tune nhận dạng ({model_id})"

    return name, desc


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
                rec_tag = extract_rec_tag(model_id)
                pipeline_ver = build_pipeline_version(model_id, settings)

                is_active = (
                    active_version in (model_id, pipeline_ver, rec_tag)
                    or (model_id in (active_version or ""))
                    or (rec_tag in (active_version or ""))
                    or (
                        settings.recognition_model_dir
                        and Path(settings.recognition_model_dir).resolve() == sub_dir.resolve()
                    )
                )

                name, desc = format_model_display(model_id, settings)

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
    if not (target_dir.is_dir() and is_recognition_model_dir(target_dir)):
        # Check alias / substring match
        target_dir = None
        for sub_dir in sorted(models_dir.iterdir()):
            if not sub_dir.is_dir() or not is_recognition_model_dir(sub_dir):
                continue
            rec_tag = extract_rec_tag(sub_dir.name)
            if (
                sub_dir.name == model_id
                or rec_tag == extract_rec_tag(model_id)
                or sub_dir.name in model_id
                or model_id in sub_dir.name
            ):
                target_dir = sub_dir
                break
        if target_dir is None:
            raise ValueError(f"Unknown or invalid model id: {model_id}")

    version = build_pipeline_version(target_dir.name, settings)
    return version, str(target_dir)

