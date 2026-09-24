"""Prepare and normalize container seal OCR datasets in the simplest way possible.

Supports two simple input modes:
  1. Filename as label:
     A directory of image files whose filename (stem) represents the seal label.
     Example: 'FX40295831.jpg', 'SITR722123.png', 'FX40295831_1.jpg'
  2. Images + Labels file:
     A directory containing images and a 'labels.csv' or 'labels.txt' file mapping:
     'image_name,label' or 'image_name\tlabel'

Outputs standard PaddleOCR dataset format:
  output_dir/
    images/
    train.txt      (images/xxx.jpg\tLABEL)
    val.txt        (images/yyy.jpg\tLABEL)
    metadata.json
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2

# Add parent directory to path to allow importing app modules if available
PARENT_DIR = Path(__file__).resolve().parent.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

try:
    from app.config import Settings
    from app.postprocess import clean, valid
    from app.seal_detector import SealDetector
except ImportError:
    Settings = None
    SealDetector = None
    clean = None
    valid = None

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
LABEL_CLEAN_RE = re.compile(r"^[A-Z0-9]{5,20}$")


def extract_label_from_filename(stem: str) -> str:
    """Extract a clean uppercase alphanumeric seal label from a filename stem.

    Handles common patterns like:
      - 'FX40295831.jpg' -> 'FX40295831'
      - 'FX40295831_1.jpg' -> 'FX40295831'
      - 'SITR722123 (1).png' -> 'SITR722123'
      - 'OOLKCK28059__r90.webp' -> 'OOLKCK28059'
      - 'ABC-12345.jpg' -> 'ABC12345'
    """
    stem = Path(stem).stem
    cleaned = re.sub(r"__r(0|90|180|270)$", "", stem, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*-\s*copy(\s*\d+)?$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*\(\d+\)$", "", cleaned).strip()

    # If format is BASE_INDEX (e.g. FX40295831_1 or FX40295831_copy),
    # only strip the suffix if the base part is already a valid seal label.
    match = re.match(r"^(.*?)[_]+(\d{1,3}|copy)$", cleaned, flags=re.IGNORECASE)
    if match:
        base_candidate = re.sub(r"[^A-Za-z0-9]", "", match.group(1)).upper()
        if is_valid_seal_label(base_candidate):
            return base_candidate

    label = re.sub(r"[^A-Za-z0-9]", "", cleaned).upper()
    return label


def is_valid_seal_label(label: str) -> bool:
    """Validate that the seal label conforms to container seal standards."""
    if not label or len(label) < 5 or len(label) > 20:
        return False
    if not LABEL_CLEAN_RE.fullmatch(label):
        return False
    # Must contain at least one digit
    if not any(char.isdigit() for char in label):
        return False
    return True


def load_dataset_samples(input_dir: Path, labels_file: Path | None = None) -> list[tuple[Path, str]]:
    """Discover all valid (image_path, label) pairs from the input directory."""
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    # Check for labels file explicitly provided or present in directory
    target_labels_file = labels_file
    if target_labels_file is None:
        for candidate in ("labels.csv", "labels.txt", "label.csv", "label.txt"):
            p = input_dir / candidate
            if p.is_file():
                target_labels_file = p
                break

    samples: list[tuple[Path, str]] = []

    if target_labels_file and target_labels_file.is_file():
        # Mode 2: Images + labels file
        print(f"[INFO] Using labels file: {target_labels_file}")
        images_sub = input_dir / "images"
        search_dirs = [images_sub, input_dir] if images_sub.is_dir() else [input_dir]

        content = target_labels_file.read_text(encoding="utf-8-sig")
        lines = [line.strip() for line in content.splitlines() if line.strip()]

        delimiter = "\t" if any("\t" in line for line in lines[:10]) else ","

        for line in lines:
            parts = [p.strip() for p in line.split(delimiter, 1)]
            if len(parts) < 2:
                continue
            img_ref, raw_label = parts[0], parts[1]

            # Skip header line if present
            if img_ref.lower() in ("image", "filename", "file", "path") and raw_label.lower() in ("label", "ground_truth"):
                continue

            label = re.sub(r"[^A-Za-z0-9]", "", raw_label).upper()
            if not is_valid_seal_label(label):
                continue

            img_path = None
            for s_dir in search_dirs:
                p = s_dir / Path(img_ref).name
                if p.is_file():
                    img_path = p
                    break
                # Try direct relative path
                p_rel = input_dir / img_ref
                if p_rel.is_file():
                    img_path = p_rel
                    break

            if img_path and img_path.is_file():
                samples.append((img_path, label))
    else:
        # Mode 1: Filename as label
        print(f"[INFO] Scanning directory for image files (using filename as label)...")
        all_files = sorted(input_dir.rglob("*"))
        for file_path in all_files:
            if not file_path.is_file() or file_path.suffix.lower() not in SUPPORTED_IMAGE_EXTS:
                continue
            label = extract_label_from_filename(file_path.stem)
            if is_valid_seal_label(label):
                samples.append((file_path, label))
            else:
                print(f"[WARN] Skipping file with invalid seal label: {file_path.name} -> '{label}'")

    if not samples:
        raise ValueError(f"No valid image-label samples found in {input_dir}")

    return samples


def prepare_dataset(
    input_dir: Path,
    output_dir: Path,
    labels_file: Path | None = None,
    val_ratio: float = 0.2,
    seed: int = 42,
    crop_seal: bool = False,
    detector_path: Path | None = None,
    rotations: tuple[int, ...] = (0,),
) -> dict:
    """Normalize input samples, optionally crop seals, split train/val, and write PaddleOCR format."""
    samples = load_dataset_samples(input_dir, labels_file)
    print(f"[INFO] Found {len(samples)} valid samples.")

    # Initialize detector if cropping requested
    detector = None
    if crop_seal:
        if SealDetector is None:
            raise RuntimeError("SealDetector is not available. Please ensure app dependencies are installed.")
        d_path = detector_path or (PARENT_DIR / "models/seal-detector-v1/best.onnx")
        if not d_path.is_file():
            raise FileNotFoundError(f"Detector model not found: {d_path}")
        settings = Settings(seal_detector_model=str(d_path))
        detector = SealDetector(settings)
        print(f"[INFO] Initialized SealDetector with model: {d_path}")

    # Stratified or group-based split by label to prevent data leakage
    groups: dict[str, list[tuple[Path, str]]] = {}
    for img_path, label in samples:
        groups.setdefault(label, []).append((img_path, label))

    unique_labels = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(unique_labels)

    val_label_count = max(1, round(len(unique_labels) * val_ratio)) if val_ratio > 0 else 0
    val_labels = set(unique_labels[:val_label_count])

    out_images_dir = output_dir / "images"
    out_images_dir.mkdir(parents=True, exist_ok=True)

    train_entries: list[str] = []
    val_entries: list[str] = []
    manifest_rows: list[dict] = []

    for label, group_samples in groups.items():
        split = "val" if label in val_labels else "train"

        for idx, (img_path, sample_label) in enumerate(group_samples):
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"[WARN] Could not read image: {img_path}")
                continue

            target_crops: list[tuple[str, any]] = []

            if detector is not None:
                detections = detector.detect(image)
                if detections:
                    crop = detector.crop(image, [detections[0]])[0]
                    if crop.size > 0 and min(crop.shape[:2]) >= 24:
                        target_crops.append(("cropped", crop))
                if not target_crops:
                    # Fallback to full image if detection fails
                    target_crops.append(("full", image))
            else:
                target_crops.append(("orig", image))

            for crop_type, base_crop in target_crops:
                for rot in rotations:
                    if rot == 90:
                        cur_crop = cv2.rotate(base_crop, cv2.ROTATE_90_CLOCKWISE)
                    elif rot == 180:
                        cur_crop = cv2.rotate(base_crop, cv2.ROTATE_180)
                    elif rot == 270:
                        cur_crop = cv2.rotate(base_crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    else:
                        cur_crop = base_crop

                    rot_suffix = f"__r{rot}" if len(rotations) > 1 else ""
                    out_filename = f"{sample_label}_{idx:03d}_{crop_type}{rot_suffix}.jpg"
                    out_file_path = out_images_dir / out_filename

                    cv2.imwrite(str(out_file_path), cur_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    rel_img_path = f"images/{out_filename}"

                    entry_line = f"{rel_img_path}\t{sample_label}"
                    if split == "train":
                        train_entries.append(entry_line)
                    else:
                        val_entries.append(entry_line)

                    manifest_rows.append({
                        "image": rel_img_path,
                        "label": sample_label,
                        "split": split,
                        "source_file": img_path.name,
                        "rotation": rot,
                        "crop_type": crop_type,
                    })

    # Write train.txt, val.txt, manifest.csv
    (output_dir / "train.txt").write_text("\n".join(train_entries) + "\n", encoding="utf-8")
    (output_dir / "val.txt").write_text("\n".join(val_entries) + "\n", encoding="utf-8")

    if manifest_rows:
        with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(manifest_rows[0]))
            writer.writeheader()
            writer.writerows(manifest_rows)

    metadata = {
        "datasetVersion": "simple-seal-v1",
        "totalSamples": len(manifest_rows),
        "trainSamples": len(train_entries),
        "valSamples": len(val_entries),
        "uniqueLabels": len(unique_labels),
        "valRatio": val_ratio,
        "seed": seed,
        "cropSeal": crop_seal,
        "rotations": list(rotations),
    }

    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"[SUCCESS] Dataset prepared successfully in: {output_dir}")
    print(f"          Total: {metadata['totalSamples']} | Train: {metadata['trainSamples']} | Val: {metadata['valSamples']}")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare normalized container seal recognition dataset.")
    parser.add_argument("--input", "-i", type=Path, required=True,
                        help="Input folder containing images (or images/ and labels.csv)")
    parser.add_argument("--output", "-o", type=Path, required=True,
                        help="Output directory for normalized PaddleOCR dataset")
    parser.add_argument("--labels", "-l", type=Path, default=None,
                        help="Path to labels.csv or labels.txt file (optional if filename is label)")
    parser.add_argument("--val-ratio", type=float, default=0.2,
                        help="Validation ratio (default: 0.2 = 20%%)")
    parser.add_argument("--crop", action="store_true",
                        help="Automatically crop seal area using YOLO detector ONNX")
    parser.add_argument("--detector", type=Path, default=None,
                        help="Custom path to YOLO seal detector ONNX model")
    parser.add_argument("--rotations", type=str, default="0",
                        help="Comma-separated rotations, e.g. '0' or '0,90,270' (default: '0')")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for deterministic split")

    args = parser.parse_args()
    rotations = tuple(int(r.strip()) for r in args.rotations.split(","))

    metadata = prepare_dataset(
        input_dir=args.input,
        output_dir=args.output,
        labels_file=args.labels,
        val_ratio=args.val_ratio,
        seed=args.seed,
        crop_seal=args.crop,
        detector_path=args.detector,
        rotations=rotations,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
