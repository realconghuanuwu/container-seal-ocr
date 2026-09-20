"""Validate and package the frozen recognition dataset for Phase 4."""

import argparse
import csv
import hashlib
import json
import re
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from app.config import Settings
from app.postprocess import normalize, valid


EXPECTED = {
    "datasetVersion": "seal-training-v1",
    "sourceDatasetSha256": "c6ed49813f1145adea6edc3e2de6d07c76b2f144af238056bdc97c7b832a07e2",
    "datasetContentSha256": "7fab45cc90fb5f5243c91047a95f791b00e3b1e21bdf35eb61107751e3474f1a",
    "sourceSamples": 300,
    "acceptedSamples": 291,
    "trainSamples": 262,
    "validationSamples": 29,
    "seed": 20260917,
}

V2_DATASET_VERSION = "seal-recognition-v2"
V3_DATASET_VERSION = "seal-recognition-v3"
LABEL_V3_RE = re.compile(r"^[A-Z0-9]{5,20}$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted((path for path in root.rglob("*") if path.is_file()),
                   key=lambda path: path.relative_to(root).as_posix())
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def labels(path: Path, root: Path, settings: Settings,
           allow_noise_terms: bool = False) -> dict[str, str]:
    result = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            name, truth = line.split("\t", 1)
        except ValueError as exc:
            raise ValueError(f"Malformed label at {path.name}:{number}") from exc
        candidate = (root / name).resolve()
        if not name or Path(name).is_absolute() or not candidate.is_relative_to(root) or not candidate.is_file():
            raise ValueError(f"Missing or unsafe crop at {path.name}:{number}: {name}")
        if name in result:
            raise ValueError(f"Duplicate crop reference in {path.name}: {name}")
        structurally_valid = (
            settings.seal_min_length <= len(truth) <= settings.seal_max_length
            and re.fullmatch(settings.seal_allowed_pattern, truth) is not None
        )
        if truth != normalize(truth) or not (
            structurally_valid if allow_noise_terms else valid(truth, settings)
        ):
            raise ValueError(f"Invalid label at {path.name}:{number}: {truth}")
        try:
            with Image.open(candidate) as image:
                image.verify()
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise ValueError(f"Unreadable crop at {path.name}:{number}: {name}") from exc
        result[name] = truth
    return result


def validate_v2_dataset(root: Path, metadata: dict,
                        settings: Settings | None = None) -> dict:
    """Validate the finalized v2 dataset without depending on its work directory."""
    settings = settings or Settings()
    required_files = ("labels.csv", "manifest.csv", "train.txt", "val.txt",
                      "validation_report.json")
    missing = [name for name in required_files if not (root / name).is_file()]
    if missing:
        raise ValueError(f"Dataset is missing required files: {', '.join(missing)}")

    with (root / "manifest.csv").open(newline="", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))
    required_columns = {"image", "ground_truth", "split", "sha256"}
    if not rows or not required_columns.issubset(rows[0] or {}):
        raise ValueError("manifest.csv is missing required v2 columns")
    if len(rows) != metadata.get("totalVerified"):
        raise ValueError("manifest.csv count does not match totalVerified")

    train = labels(root / "train.txt", root, settings, allow_noise_terms=True)
    validation = labels(root / "val.txt", root, settings, allow_noise_terms=True)
    if set(train) & set(validation):
        raise ValueError("train.txt and val.txt overlap")
    if len(train) != metadata.get("trainSamples") or len(validation) != metadata.get("valSamples"):
        raise ValueError("Split count does not match metadata")
    combined = {**train, **validation}
    if len(combined) != len(rows):
        raise ValueError("Label files and manifest.csv contain different sample counts")

    fingerprint_lines = []
    seen = set()
    for row in rows:
        image, truth, split, expected_hash = (
            row.get("image", ""), row.get("ground_truth", ""),
            row.get("split", ""), row.get("sha256", ""),
        )
        if not image or image in seen or split not in {"train", "val"}:
            raise ValueError("Malformed or duplicate manifest.csv row")
        seen.add(image)
        if combined.get(image) != truth or (image in train) != (split == "train"):
            raise ValueError(f"Label file disagrees with manifest.csv: {image}")
        image_path = (root / image).resolve()
        if not image_path.is_relative_to(root) or not image_path.is_file():
            raise ValueError(f"Missing or unsafe crop in manifest.csv: {image}")
        actual_hash = sha256(image_path)
        if actual_hash != expected_hash:
            raise ValueError(f"Crop hash mismatch: {image}")
        fingerprint_lines.append(f"{image}\t{truth}\t{expected_hash}")

    dataset_hash = hashlib.sha256(
        "\n".join(sorted(fingerprint_lines)).encode()
    ).hexdigest()
    if dataset_hash != metadata.get("datasetSha256"):
        raise ValueError("Dataset fingerprint differs from metadata.json")

    report = json.loads((root / "validation_report.json").read_text(encoding="utf-8"))
    expected_report = {
        "totalImages": len(rows),
        "trainCount": len(train),
        "valCount": len(validation),
        "datasetSha256": dataset_hash,
        "benchmarkOverlapCount": 0,
        "status": "PASS",
    }
    for key, value in expected_report.items():
        if report.get(key) != value:
            raise ValueError(f"validation_report.json mismatch for {key}: {report.get(key)!r}")

    return {
        "datasetVersion": V2_DATASET_VERSION,
        "datasetSha256": dataset_hash,
        "totalVerified": len(rows),
        "trainSamples": len(train),
        "validationSamples": len(validation),
    }


def validate_v3_dataset(root: Path, metadata: dict,
                        settings: Settings | None = None) -> dict:
    """Validate the frozen v3 dataset, including provenance and split groups."""
    summary = validate_v2_dataset(root, metadata, settings)
    total_verified = metadata.get("totalVerified")
    base_samples = metadata.get("baseSamples", 5_000)
    new_samples = metadata.get("newSamples")
    train_samples = metadata.get("trainSamples")
    val_samples = metadata.get("valSamples")
    if total_verified != base_samples + new_samples:
        raise ValueError(f"V3 dataset total samples mismatch: {total_verified} != {base_samples} + {new_samples}")
    if total_verified != train_samples + val_samples:
        raise ValueError(f"V3 split samples mismatch: {total_verified} != {train_samples} + {val_samples}")
    with (root / "manifest.csv").open(newline="", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))
    required = {"source_image", "phash", "group_id", "origin", "bucket"}
    if not rows or not required.issubset(rows[0] or {}):
        raise ValueError("manifest.csv is missing required v3 provenance columns")

    origins = Counter(row["origin"] for row in rows)
    if origins != Counter({"v2": base_samples, "v3": new_samples}):
        raise ValueError(f"V3 provenance counts are invalid: {dict(origins)}")
    labels_count = Counter(row["ground_truth"] for row in rows)
    if max(labels_count.values(), default=0) > 3:
        raise ValueError("A normalized seal label has more than three variants")
    for row in rows:
        if LABEL_V3_RE.fullmatch(row["ground_truth"]) is None:
            raise ValueError(f"Invalid v3 label: {row['ground_truth']}")

    for field in ("source_image", "ground_truth", "group_id"):
        split_by_value: dict[str, str] = {}
        for row in rows:
            value = row[field]
            previous = split_by_value.setdefault(value, row["split"])
            if previous != row["split"]:
                raise ValueError(f"Split leakage through {field}: {value}")
    report = json.loads((root / "validation_report.json").read_text(encoding="utf-8"))
    for key, expected in {
        "baseSamples": base_samples,
        "newSamples": new_samples,
        "duplicateCount": 0,
        "groupLeakageCount": 0,
    }.items():
        if report.get(key) != expected:
            raise ValueError(f"validation_report.json mismatch for {key}: {report.get(key)!r}")
    return {
        **summary,
        "datasetVersion": V3_DATASET_VERSION,
        "baseSamples": base_samples,
        "newSamples": new_samples,
    }


def validate_prepared_dataset(root: Path, expected: dict | None = None,
                              settings: Settings | None = None) -> dict:
    root = root.resolve()
    settings = settings or Settings()
    metadata_path, manifest_path = root / "metadata.json", root / "manifest.csv"
    if not root.is_dir() or not metadata_path.is_file() or not manifest_path.is_file():
        raise ValueError("Dataset must contain metadata.json and manifest.csv")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if expected is None and metadata.get("datasetVersion") == V3_DATASET_VERSION:
        return validate_v3_dataset(root, metadata, settings)
    if expected is None and metadata.get("datasetVersion") == V2_DATASET_VERSION:
        return validate_v2_dataset(root, metadata, settings)
    expected = expected or EXPECTED
    for key, value in expected.items():
        if key == "datasetContentSha256":
            continue
        if metadata.get(key) != value:
            raise ValueError(f"Dataset metadata mismatch for {key}: {metadata.get(key)!r}")
    with manifest_path.open(newline="", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))
    required = {"crop", "ground_truth", "split"}
    if not rows or not required.issubset(rows[0] or {}):
        raise ValueError("manifest.csv is missing required columns")
    if len(rows) != metadata["acceptedSamples"]:
        raise ValueError("manifest.csv count does not match acceptedSamples")
    manifest = {}
    for row in rows:
        crop, truth, split = row.get("crop"), row.get("ground_truth"), row.get("split")
        if not crop or not truth or split not in {"train", "val"} or crop in manifest:
            raise ValueError("Malformed manifest.csv row")
        manifest[crop] = (truth, split)

    train = labels(root / "train.txt", root, settings)
    validation = labels(root / "val.txt", root, settings)
    if set(train) & set(validation):
        raise ValueError("train.txt and val.txt overlap")
    if len(train) != metadata["trainSamples"] or len(validation) != metadata["validationSamples"]:
        raise ValueError("Split count does not match metadata")
    combined = {**train, **validation}
    if set(combined) != set(manifest):
        raise ValueError("Label files and manifest.csv reference different crops")
    for crop, truth in combined.items():
        expected_truth, split = manifest[crop]
        if truth != expected_truth or (crop in train) != (split == "train"):
            raise ValueError(f"Label file disagrees with manifest.csv: {crop}")
    content_hash = content_sha256(root)
    if expected.get("datasetContentSha256") and content_hash != expected["datasetContentSha256"]:
        raise ValueError("Dataset content hash differs from the frozen Phase 3 artifact")
    return {"datasetVersion": metadata["datasetVersion"],
            "sourceDatasetSha256": metadata["sourceDatasetSha256"],
            "datasetContentSha256": content_hash,
            "acceptedSamples": len(rows), "trainSamples": len(train),
            "validationSamples": len(validation)}


def archive_dataset(root: Path, output: Path, expected: dict | None = None) -> dict:
    summary = validate_prepared_dataset(root, expected)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite archive: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(root))
    return {**summary, "archive": str(output), "archiveSha256": sha256(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "archive"):
        command = commands.add_parser(name)
        command.add_argument("--dataset", type=Path, default=Path("recognition_dataset"))
    commands.choices["archive"].add_argument("--output", type=Path,
                                                default=Path("recognition_dataset.zip"))
    args = parser.parse_args()
    result = (archive_dataset(args.dataset, args.output) if args.command == "archive"
              else validate_prepared_dataset(args.dataset))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
