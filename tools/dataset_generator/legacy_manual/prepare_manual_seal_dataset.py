"""Prepare manually labeled full-frame seal photos for recognition training."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings
from app.postprocess import valid
from app.seal_detector import SealDetector
from app.worker import create_engine
from scripts.create_recognition_dataset_v2 import crop_polygon, normalize_text


LABEL_RE = re.compile(r"^[A-Z0-9]{5,20}$")


def split_sources(names: list[str], seed: int) -> tuple[set[str], set[str]]:
    """Make a deterministic, prefix-stratified split at source-image level."""
    groups: dict[str, list[str]] = {}
    for name in names:
        groups.setdefault(name[:4], []).append(name)
    validation: set[str] = set()
    rng = random.Random(seed)
    for group in groups.values():
        shuffled = sorted(group)
        rng.shuffle(shuffled)
        validation.update(shuffled[:max(1, round(len(shuffled) * 0.2))])
    train = set(names) - validation
    return train, validation


def edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for i, char_left in enumerate(left, 1):
        current = [i]
        for j, char_right in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (char_left != char_right)))
        previous = current
    return previous[-1]


def text_line_crop(engine, crop: np.ndarray, label: str) -> tuple[np.ndarray, int, float] | None:
    candidates = []
    variants = (
        (0, crop),
        (90, cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)),
        (180, cv2.rotate(crop, cv2.ROTATE_180)),
        (270, cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)),
    )
    for rotation, rotated in variants:
        try:
            result = next(iter(engine.predict(rotated)))
        except Exception:
            continue
        data = result.json.get("res", {})
        for text, score, polygon in zip(
            data.get("rec_texts", []), data.get("rec_scores", []),
            data.get("rec_polys", data.get("dt_polys", [])),
        ):
            normalized = normalize_text(text)
            if not normalized or not any(char.isdigit() for char in normalized):
                continue
            distance = edit_distance(normalized, label)
            candidates.append((distance, -float(score), rotation, rotated, polygon, float(score)))
    if not candidates:
        return None
    distance, _, rotation, rotated, polygon, score = min(candidates)
    points = np.asarray(polygon, dtype=np.float32)
    if points.shape != (4, 2):
        return None
    return crop_polygon(rotated, polygon), rotation, score


def prepare(input_dir: Path, output_dir: Path, detector_path: Path, seed: int,
            use_text_crop: bool = False) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing directory: {output_dir}")
    files = sorted(input_dir.glob("*.jpg"))
    if not files:
        raise ValueError(f"No JPG files found in {input_dir}")

    settings = Settings(
        seal_detector_model=str(detector_path),
        model_version="seal-ocr-det-v1-rec-v1",
    )
    detector = SealDetector(settings)
    engine = create_engine(settings) if use_text_crop else None
    labels: dict[str, str] = {}
    for path in files:
        label = path.stem.upper()
        if not LABEL_RE.fullmatch(label) or not valid(label, settings):
            raise ValueError(f"Invalid manual label from filename: {path.name}")
        labels[path.stem] = label

    train_sources, val_sources = split_sources(list(labels), seed)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True)
    rows = []
    failed = []
    for path in files:
        image = cv2.imread(str(path))
        if image is None:
            failed.append({"image": path.name, "reason": "UNREADABLE"})
            continue
        detections = detector.detect(image)
        if not detections:
            failed.append({"image": path.name, "reason": "NO_DETECTION"})
            continue
        crop = detector.crop(image, [detections[0]])[0]
        if crop.size == 0 or min(crop.shape[:2]) < 32:
            failed.append({"image": path.name, "reason": "INVALID_CROP"})
            continue

        text_rotation = None
        text_confidence = None
        if engine is not None:
            extracted = text_line_crop(engine, crop, labels[path.stem])
            if extracted is None or extracted[0].size == 0:
                failed.append({"image": path.name, "reason": "NO_TEXT_LINE"})
                continue
            crop, text_rotation, text_confidence = extracted

        split = "val" if path.stem in val_sources else "train"
        variants = (
            ("r0", crop),
            ("r90", cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)),
            ("r270", cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)),
        )
        for suffix, variant in variants:
            image_name = f"{path.stem}__{suffix}.jpg"
            if not cv2.imwrite(str(image_dir / image_name), variant, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise OSError(f"Could not write {image_name}")
            rows.append({
                "image": f"images/{image_name}",
                "ground_truth": labels[path.stem],
                "source_image": path.name,
                "split": split,
                "rotation": suffix,
                "detector_confidence": round(detections[0]["confidence"], 6),
                "text_rotation": text_rotation,
                "text_confidence": text_confidence,
            })

    if failed:
        raise RuntimeError(f"Detector failed on {len(failed)} images: {failed[:5]}")

    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for split in ("train", "val"):
        lines = [f"{row['image']}\t{row['ground_truth']}" for row in rows if row["split"] == split]
        (output_dir / f"{split}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    metadata = {
        "datasetVersion": "seal-recognition-manual-v1",
        "sourceImages": len(files),
        "cropImages": len(rows),
        "trainImages": sum(row["split"] == "train" for row in rows),
        "validationImages": sum(row["split"] == "val" for row in rows),
        "sourceTrainImages": len(train_sources),
        "sourceValidationImages": len(val_sources),
        "rotations": ["r0", "r90", "r270"],
        "seed": seed,
        "detector": str(detector_path),
        "textCrop": use_text_crop,
        "labelSource": "manually entered filename",
        "labelCounts": dict(Counter(labels.values())),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--detector", type=Path, default=Path("models/seal-detector-v1/best.onnx"))
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--text-crop", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(args.input, args.output, args.detector, args.seed, args.text_crop), indent=2))


if __name__ == "__main__":
    main()
