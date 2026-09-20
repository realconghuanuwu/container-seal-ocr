"""Validate source labels and create a PaddleOCR recognition dataset."""

import argparse
import csv
import hashlib
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from app.config import Settings
from app.postprocess import normalize, valid
from scripts.benchmark import distance


FORMATS = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".webp": "WEBP"}
MANIFEST_FIELDS = ["source_image", "crop", "ground_truth", "split", "prediction",
                   "edit_ratio", "rotation", "sha256"]
REJECTED_FIELDS = ["source_image", "ground_truth", "reason", "prediction", "edit_ratio"]


def phash(path: Path) -> list[int]:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Invalid image: {path.name}")
    hashes = []
    for rotation in range(4):
        resized = cv2.resize(np.rot90(image, rotation).copy(), (32, 32), interpolation=cv2.INTER_AREA)
        values = cv2.dct(np.float32(resized))[:8, :8].flatten()
        median = float(np.median(values[1:]))
        hashes.append(sum(int(value > median) << index for index, value in enumerate(values)))
    return hashes


def hamming(left: list[int], right: list[int]) -> int:
    return min((a ^ b).bit_count() for a in left for b in right)


def validate_dataset(source: Path, benchmark: Path, settings: Settings | None = None) -> tuple[list[dict], dict]:
    settings = settings or Settings()
    images_dir = (source / "images").resolve()
    labels_path = source / "labels.csv"
    if not images_dir.is_dir() or not labels_path.is_file():
        raise ValueError("Source must contain images/ and labels.csv")
    with labels_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"image", "ground_truth"}.issubset(reader.fieldnames):
            raise ValueError("labels.csv must contain image,ground_truth columns")
        rows = list(reader)
    if not rows:
        raise ValueError("labels.csv is empty")

    manifest = json.loads((benchmark / "manifest.json").read_text(encoding="utf-8"))
    benchmark_hashes = {item["sha256"]: item["image"] for item in manifest["images"]}
    benchmark_phashes = [(item["image"], phash(benchmark / "images" / item["image"]))
                         for item in manifest["images"]]
    names, by_hash, samples, duplicates, inventory = set(), {}, [], [], []
    for row in rows:
        name = (row.get("image") or "").strip()
        truth = (row.get("ground_truth") or "").strip()
        if not name or not truth or None in row:
            raise ValueError("Empty or malformed labels.csv row")
        if name != Path(name).name or any(char in name for char in ("/", "\\", ":")):
            raise ValueError(f"Image must be a filename under images/: {name}")
        if name in names:
            raise ValueError(f"Duplicate image reference: {name}")
        names.add(name)
        if truth != normalize(truth) or not valid(truth, settings):
            raise ValueError(f"Invalid ground truth for {name}: {truth}")
        path = (images_dir / name).resolve()
        if not path.is_relative_to(images_dir) or not path.is_file():
            raise ValueError(f"Missing or unsafe image: {name}")
        expected = FORMATS.get(path.suffix.lower())
        if expected is None:
            raise ValueError(f"Unsupported image type: {name}")
        try:
            with Image.open(path) as image:
                if image.format != expected:
                    raise ValueError(f"Image extension/content mismatch: {name}")
                image.load()
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise ValueError(f"Invalid image: {name}") from exc
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        inventory.append((name, truth, digest))
        if digest in benchmark_hashes:
            raise ValueError(f"Benchmark leakage: {name} matches {benchmark_hashes[digest]}")
        if digest in by_hash:
            original = by_hash[digest]
            if original["ground_truth"] != truth:
                raise ValueError(f"Conflicting labels for duplicate images: {original['image']} and {name}")
            duplicates.append({"kept": original["image"], "excluded": name, "sha256": digest})
            continue
        sample = {"image": name, "ground_truth": truth, "sha256": digest, "phash": phash(path)}
        by_hash[digest] = sample
        samples.append(sample)
    disk_names = {path.name for path in images_dir.iterdir() if path.is_file()}
    if names != disk_names:
        raise ValueError(f"Image inventory differs: unlisted={sorted(disk_names - names)}, missing={sorted(names - disk_names)}")

    near_pairs = []
    for index, left in enumerate(samples):
        for right in samples[index + 1:]:
            value = hamming(left["phash"], right["phash"])
            if value <= 6:
                near_pairs.append({"left": left["image"], "right": right["image"], "hammingDistance": value})
    near_benchmark = []
    for sample in samples:
        value, image = min((hamming(sample["phash"], other_hash), image)
                           for image, other_hash in benchmark_phashes)
        if value <= 6:
            near_benchmark.append({"trainingImage": sample["image"], "benchmarkImage": image,
                                   "hammingDistance": value})
    fingerprint = hashlib.sha256("\n".join(
        f"{name}\t{truth}\t{digest}" for name, truth, digest in sorted(inventory)
    ).encode()).hexdigest()
    report = {"sourceSamples": len(rows), "usableSamples": len(samples), "datasetSha256": fingerprint,
              "exactDuplicateImages": duplicates, "benchmarkExactOverlaps": 0,
              "nearDuplicatePairs": near_pairs, "nearBenchmarkPairs": near_benchmark}
    return sorted(samples, key=lambda sample: sample["image"]), report


def create_engine():
    from paddleocr import PaddleOCR

    return PaddleOCR(text_detection_model_name="PP-OCRv6_medium_det",
                     text_recognition_model_name="PP-OCRv6_medium_rec",
                     use_doc_orientation_classify=False, use_doc_unwarping=False,
                     use_textline_orientation=False, device="cpu")


def polygon_metrics(polygon) -> tuple[float, float]:
    points = np.asarray(polygon, dtype=np.float32)
    width = max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3]))
    height = max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2]))
    area = abs(cv2.contourArea(points))
    return max(width, height) / max(min(width, height), 1), area


def candidates(result: dict, truth: str, image_area: int) -> list[dict]:
    values = []
    for text, confidence, polygon in zip(result.get("rec_texts", []), result.get("rec_scores", []),
                                         result.get("rec_polys", result.get("dt_polys", []))):
        prediction = normalize(text)
        ratio = distance(truth, prediction) / max(len(truth), len(prediction), 1)
        aspect, area = polygon_metrics(polygon)
        values.append({"prediction": prediction, "confidence": float(confidence), "polygon": polygon,
                       "edit_ratio": ratio, "aspect": aspect, "area_ratio": area / image_area})
    return values


def accepted(candidate: dict) -> bool:
    return candidate["edit_ratio"] <= 0.4 and candidate["aspect"] >= 1.5 and candidate["area_ratio"] <= 0.25


def crop_polygon(image: np.ndarray, polygon) -> np.ndarray:
    points = np.asarray(polygon, dtype=np.float32)
    width = max(1, round(max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3]))))
    height = max(1, round(max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2]))))
    target = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    crop = cv2.warpPerspective(image, cv2.getPerspectiveTransform(points, target), (width, height),
                               borderMode=cv2.BORDER_REPLICATE)
    if crop.shape[0] / crop.shape[1] >= 1.5:
        crop = np.rot90(crop).copy()
    return crop


def best_candidate(engine, image: np.ndarray, truth: str) -> tuple[dict | None, np.ndarray | None, int]:
    image_area = image.shape[0] * image.shape[1]
    best, best_image, best_rotation = None, None, 0
    best_accepted, accepted_image, accepted_rotation = None, None, 0
    for low_threshold in (False, True):
        for rotation in range(4):
            rotated = np.rot90(image, rotation).copy()
            kwargs = {"text_det_thresh": 0.1, "text_det_box_thresh": 0.3} if low_threshold else {}
            results = list(engine.predict(rotated, **kwargs))
            result = results[0].json["res"] if results else {}
            for candidate in candidates(result, truth, image_area):
                rank = (candidate["edit_ratio"], -candidate["confidence"])
                if best is None or rank < (best["edit_ratio"], -best["confidence"]):
                    best, best_image, best_rotation = candidate, rotated, rotation * 90
                if accepted(candidate) and (best_accepted is None or rank < (best_accepted["edit_ratio"], -best_accepted["confidence"])):
                    best_accepted, accepted_image, accepted_rotation = candidate, rotated, rotation * 90
        if best_accepted is not None and best_accepted["edit_ratio"] == 0:
            return best_accepted, accepted_image, accepted_rotation
    if best_accepted is not None:
        return best_accepted, accepted_image, accepted_rotation
    return best, best_image, best_rotation


def split_groups(samples: list[dict], seed: int) -> tuple[set[str], set[str]]:
    parents = list(range(len(samples)))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parents[right] = left

    for left, sample in enumerate(samples):
        for right in range(left + 1, len(samples)):
            other = samples[right]
            if sample["ground_truth"] == other["ground_truth"] or hamming(sample["phash"], other["phash"]) <= 6:
                union(left, right)
    groups = {}
    for index, sample in enumerate(samples):
        groups.setdefault(find(index), []).append(sample["image"])
    groups = list(groups.values())
    random.Random(seed).shuffle(groups)
    target = max(1, round(len(samples) * 0.1))
    validation = set()
    for group in groups:
        if len(validation) >= target:
            break
        validation.update(group)
    training = {sample["image"] for sample in samples} - validation
    if not training or not validation:
        raise ValueError("Cannot create a non-empty train/validation split")
    return training, validation


def prepare(source: Path, benchmark: Path, output: Path, seed: int = 20260917, engine=None) -> dict:
    samples, validation = validate_dataset(source, benchmark)
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    staging = output.with_name(f".{output.name}.staging")
    if staging.exists():
        raise ValueError(f"Staging directory already exists: {staging}")
    crops_dir = staging / "images"
    crops_dir.mkdir(parents=True)
    engine = engine or create_engine()
    accepted_samples, rejected = [], []
    try:
        for sample in samples:
            image = cv2.imread(str(source / "images" / sample["image"]))
            candidate, rotated, rotation = best_candidate(engine, image, sample["ground_truth"])
            if candidate is None:
                rejected.append({"source_image": sample["image"], "ground_truth": sample["ground_truth"],
                                 "reason": "NO_TEXT_DETECTED", "prediction": "", "edit_ratio": ""})
                continue
            if not accepted(candidate):
                rejected.append({"source_image": sample["image"], "ground_truth": sample["ground_truth"],
                                 "reason": "UNTRUSTED_CROP", "prediction": candidate["prediction"],
                                 "edit_ratio": round(candidate["edit_ratio"], 4)})
                continue
            crop = crop_polygon(rotated, candidate["polygon"])
            if crop.shape[0] < 8 or crop.shape[1] < 24:
                rejected.append({"source_image": sample["image"], "ground_truth": sample["ground_truth"],
                                 "reason": "CROP_TOO_SMALL", "prediction": candidate["prediction"],
                                 "edit_ratio": round(candidate["edit_ratio"], 4)})
                continue
            crop_name = f"{sample['sha256'][:16]}.jpg"
            if not cv2.imwrite(str(crops_dir / crop_name), crop, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise OSError(f"Could not write crop: {crop_name}")
            accepted_samples.append({**sample, "crop": f"images/{crop_name}", "prediction": candidate["prediction"],
                                     "edit_ratio": round(candidate["edit_ratio"], 4), "rotation": rotation})
        train, validation_set = split_groups(accepted_samples, seed)
        for sample in accepted_samples:
            sample["split"] = "train" if sample["image"] in train else "val"
        for split in ("train", "val"):
            lines = [f"{sample['crop']}\t{sample['ground_truth']}" for sample in accepted_samples
                     if sample["split"] == split]
            (staging / f"{split}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        with (staging / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(({**sample, "source_image": sample["image"]} for sample in accepted_samples))
        with (staging / "rejected.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=REJECTED_FIELDS)
            writer.writeheader()
            writer.writerows(rejected)
        (staging / "validation_report.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
        metadata = {"datasetVersion": "seal-training-v1", "createdAtUtc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "sourceDatasetSha256": validation["datasetSha256"], "sourceSamples": validation["sourceSamples"],
                    "duplicateExcluded": len(validation["exactDuplicateImages"]), "acceptedSamples": len(accepted_samples),
                    "rejectedSamples": len(rejected), "trainSamples": len(train), "validationSamples": len(validation_set),
                    "seed": seed, "validation": validation}
        (staging / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        os.replace(staging, output)
        return metadata
    except Exception:
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate_command = commands.add_parser("validate")
    prepare_command = commands.add_parser("prepare")
    for command in (validate_command, prepare_command):
        command.add_argument("--source", type=Path, default=Path("training_source"))
        command.add_argument("--benchmark", type=Path, default=Path("benchmark"))
    validate_command.add_argument("--report", type=Path)
    prepare_command.add_argument("--output", type=Path, default=Path("recognition_dataset"))
    prepare_command.add_argument("--seed", type=int, default=20260917)
    args = parser.parse_args()
    if args.command == "validate":
        _, result = validate_dataset(args.source, args.benchmark)
        if args.report:
            args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    else:
        result = prepare(args.source, args.benchmark, args.output, args.seed)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
