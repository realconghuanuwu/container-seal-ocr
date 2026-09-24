"""Merge a prepared manual dataset into the frozen recognition-v3 dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_split(root: Path, name: str) -> list[tuple[str, str]]:
    rows = []
    for line in (root / name).read_text(encoding="utf-8").splitlines():
        image, label = line.split("\t", 1)
        rows.append((image, label))
    return rows


def merge(base: Path, manual: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing directory: {output}")
    output_images = output / "images"
    output_images.mkdir(parents=True)
    rows = []

    def add_dataset(root: Path, source_name: str, prefix: str) -> None:
        for split in ("train", "val"):
            for image, label in read_split(root, f"{split}.txt"):
                source = root / image
                target_name = f"{prefix}_{Path(image).name}"
                target = output_images / target_name
                shutil.copy2(source, target)
                rows.append({
                    "image": f"images/{target_name}",
                    "ground_truth": label,
                    "split": split,
                    "source_dataset": source_name,
                    "source_image": image,
                    "sha256": sha256(target),
                })

    add_dataset(base, "seal-recognition-v3", "v3")
    add_dataset(manual, "manual-user-100", "manual")

    for split in ("train", "val"):
        lines = [f"{row['image']}\t{row['ground_truth']}" for row in rows if row["split"] == split]
        (output / f"{split}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    dataset_sha256 = hashlib.sha256("\n".join(sorted(
        f"{row['image']}\t{row['ground_truth']}\t{row['sha256']}" for row in rows
    )).encode()).hexdigest()
    train_count = sum(row["split"] == "train" for row in rows)
    val_count = sum(row["split"] == "val" for row in rows)
    metadata = {
        "datasetVersion": "seal-recognition-v4-manual",
        "baseDataset": str(base),
        "manualDataset": str(manual),
        "totalSamples": len(rows),
        "totalVerified": len(rows),
        "trainSamples": train_count,
        "validationSamples": val_count,
        "valSamples": val_count,
        "datasetSha256": dataset_sha256,
        "baseSamples": 14500,
        "manualSamples": sum(row["source_dataset"] == "manual-user-100" for row in rows),
        "manualSourceImages": 100,
        "labelSource": "manually entered filename",
        "sourceCounts": dict(Counter(row["source_dataset"] for row in rows)),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--manual", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(merge(args.base, args.manual, args.output), indent=2))


if __name__ == "__main__":
    main()
