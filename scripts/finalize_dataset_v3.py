"""Merge verified v2 data with 15,000 v3 samples and freeze a 20k dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import Counter
from datetime import datetime, timezone
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

from scripts.create_recognition_dataset_v2 import compute_phash
from scripts.create_recognition_dataset_v3 import (
    QUOTAS,
    PHashIndex,
    _load_base_index,
    effective_record,
    label_bucket,
    load_latest_state,
    load_reviews,
    normalize_label,
    valid_label,
)
from scripts.training_model import sha256, validate_prepared_dataset


DATASET_VERSION = "seal-recognition-v3"
TOTAL = 14_500
BASE_TOTAL = 5_000
NEW_TOTAL = 9_500
TRAIN_TOTAL = 13_000
VAL_TOTAL = 1_500
SEED = 20260919


def _read_manifest(root: Path) -> list[dict[str, str]]:
    with (root / "manifest.csv").open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _selected_new(work_dir: Path, base_dataset: Path, benchmark_hashes: set[str], target_count: int = NEW_TOTAL) -> list[dict[str, Any]]:
    latest = load_latest_state(work_dir / ".pipeline_state.jsonl")
    reviews = load_reviews(work_dir)
    base_hashes, base_index, _, _ = _load_base_index(base_dataset, work_dir / ".v2_index.json")

    phash_index = PHashIndex()
    for h, name in zip(base_index.hashes, base_index.payloads):
        phash_index.add(h, name)

    seen_sha = set(base_hashes) | set(benchmark_hashes)
    label_counts: Counter[str] = Counter()

    with (base_dataset / "manifest.csv").open(newline="", encoding="utf-8-sig") as s:
        for r in csv.DictReader(s):
            label_counts[normalize_label(r["ground_truth"])] += 1

    selected = []
    for record in latest.values():
        item = effective_record(record, reviews)
        if item.get("status") not in {"AUTO_ACCEPTED", "VERIFIED"}:
            continue
        label = normalize_label(item.get("proposed_label"))
        if not valid_label(label):
            continue
        crop = (work_dir / item.get("image", "")).resolve()
        if not crop.is_relative_to(work_dir.resolve()) or not crop.is_file():
            continue
        c_sha = item.get("crop_sha256") or sha256(crop)
        if c_sha in seen_sha:
            continue
        if label_counts[label] >= 3:
            continue
        h = compute_phash(crop)
        if phash_index.near(h) is not None:
            continue
        seen_sha.add(c_sha)
        phash_index.add(h, item.get("image", ""))
        label_counts[label] += 1
        selected.append({
            **item,
            "ground_truth": label,
            "crop_path": crop,
            "origin": "v3",
            "computed_phash": h,
            "crop_sha256": c_sha,
        })
        if len(selected) >= target_count:
            break

    if len(selected) != target_count:
        raise ValueError(f"Expected exactly {target_count} clean accepted v3 crops, found {len(selected)}")
    return sorted(selected, key=lambda item: item["crop_sha256"])


def _union_groups(items: list[dict[str, Any]]) -> list[list[int]]:
    parents = list(range(len(items)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    by_source: dict[str, int] = {}
    by_label: dict[str, int] = {}
    phashes = PHashIndex()
    for index, item in enumerate(items):
        source = item["source_image"]
        label = item["ground_truth"]
        if source in by_source:
            union(index, by_source[source])
        else:
            by_source[source] = index
        if label in by_label:
            union(index, by_label[label])
        else:
            by_label[label] = index
        for near, _ in phashes.near_all(item["phash"]):
            union(index, int(near))
        phashes.add(item["phash"], str(index))

    groups: dict[int, list[int]] = {}
    for index in range(len(items)):
        groups.setdefault(find(index), []).append(index)
    return list(groups.values())


def choose_validation_groups(groups: list[list[int]], target: int, seed: int = SEED) -> set[int]:
    ordered = list(groups)
    random.Random(seed).shuffle(ordered)
    reachable: dict[int, tuple[int, int] | None] = {0: None}
    for group_index, group in enumerate(ordered):
        size = len(group)
        for current in sorted(list(reachable), reverse=True):
            total = current + size
            if total <= target and total not in reachable:
                reachable[total] = (current, group_index)
        if target in reachable:
            break
    if target not in reachable:
        raise ValueError(f"Cannot make an exact {target}-sample validation split without group leakage")
    chosen_groups = set()
    current = target
    while current:
        previous, group_index = reachable[current]  # type: ignore[misc]
        chosen_groups.add(group_index)
        current = previous
    return {item for group_index in chosen_groups for item in ordered[group_index]}


def finalize_dataset(base_dataset: Path, work_dir: Path, output_dir: Path,
                     benchmark_dir: Path, seed: int = SEED) -> dict[str, Any]:
    base_summary = validate_prepared_dataset(base_dataset)
    if base_summary.get("datasetVersion") != "seal-recognition-v2" or base_summary.get("totalVerified") != BASE_TOTAL:
        raise ValueError(f"Unexpected v2 base dataset: {base_summary}")
    base_rows = _read_manifest(base_dataset)
    benchmark = json.loads((benchmark_dir / "manifest.json").read_text(encoding="utf-8"))
    benchmark_hashes = {item["sha256"] for item in benchmark["images"]}
    new_rows = _selected_new(work_dir, base_dataset, benchmark_hashes, NEW_TOTAL)

    items: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    label_counts: Counter[str] = Counter()
    phash_index = PHashIndex()

    def add_item(path: Path, label: str, source: str, origin: str, bucket: str = "",
                 precomputed_hash: list[int] | None = None, precomputed_sha: str | None = None) -> None:
        with Image.open(path) as image:
            image.verify()
        digest = precomputed_sha or sha256(path)
        hashes = precomputed_hash or compute_phash(path)
        if digest in benchmark_hashes:
            raise ValueError(f"Benchmark overlap: {path}")
        if digest in seen_hashes:
            raise ValueError(f"Duplicate crop: {path}")
        if label_counts[label] >= 3:
            raise ValueError(f"More than three variants for label {label}")
        seen_hashes.add(digest)
        phash_index.add(hashes, str(path))
        label_counts[label] += 1
        items.append({
            "source_path": path, "ground_truth": label, "source_image": source.replace("\\", "/"),
            "sha256": digest, "phash": hashes, "origin": origin, "bucket": bucket,
        })

    for row in base_rows:
        add_item(base_dataset / row["image"], normalize_label(row["ground_truth"]),
                 row.get("source_image", ""), "v2")
    for row in new_rows:
        add_item(row["crop_path"], row["ground_truth"], row.get("source_image", ""), "v3",
                 row.get("bucket") or label_bucket(row["ground_truth"]),
                 precomputed_hash=row.get("computed_phash"), precomputed_sha=row.get("crop_sha256"))
    if len(items) != TOTAL:
        raise ValueError(f"Expected {TOTAL} total crops, found {len(items)}")

    groups = _union_groups(items)
    validation_indices = choose_validation_groups(groups, VAL_TOTAL, seed)
    group_by_index = {item_index: f"group_{group_index:05d}"
                      for group_index, group in enumerate(groups) for item_index in group}
    staging = output_dir.with_name(f".{output_dir.name}.staging")
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "images").mkdir(parents=True)
    final_rows = []
    for index, item in enumerate(items, 1):
        image_rel = f"images/crop_{index:06d}.jpg"
        shutil.copy2(item["source_path"], staging / image_rel)
        split = "val" if index - 1 in validation_indices else "train"
        final_rows.append({
            "image": image_rel, "ground_truth": item["ground_truth"],
            "source_image": item["source_image"], "split": split, "sha256": item["sha256"],
            "phash": ":".join(f"{value:016x}" for value in item["phash"]),
            "group_id": group_by_index[index - 1], "origin": item["origin"], "bucket": item["bucket"],
        })

    train = [f"{row['image']}\t{row['ground_truth']}" for row in final_rows if row["split"] == "train"]
    validation = [f"{row['image']}\t{row['ground_truth']}" for row in final_rows if row["split"] == "val"]
    if len(train) != TRAIN_TOTAL or len(validation) != VAL_TOTAL:
        raise AssertionError("Unexpected split count")
    (staging / "train.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    (staging / "val.txt").write_text("\n".join(validation) + "\n", encoding="utf-8")
    for name in ("labels.csv", "manifest.csv"):
        with (staging / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(final_rows[0]))
            writer.writeheader(); writer.writerows(final_rows)

    fingerprint = hashlib.sha256("\n".join(
        f"{row['image']}\t{row['ground_truth']}\t{row['sha256']}" for row in final_rows
    ).encode()).hexdigest()
    report = {
        "totalImages": TOTAL, "trainCount": TRAIN_TOTAL, "valCount": VAL_TOTAL,
        "baseSamples": BASE_TOTAL, "newSamples": NEW_TOTAL, "datasetSha256": fingerprint,
        "benchmarkOverlapCount": 0, "duplicateCount": 0, "groupLeakageCount": 0,
        "groupsCount": len(groups), "status": "PASS",
    }
    metadata = {
        "datasetVersion": DATASET_VERSION, "createdAtUtc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "totalVerified": TOTAL, "baseSamples": BASE_TOTAL, "newSamples": NEW_TOTAL,
        "trainSamples": TRAIN_TOTAL, "valSamples": VAL_TOTAL, "datasetSha256": fingerprint,
        "baseDatasetSha256": base_summary["datasetSha256"], "seed": seed,
        "bucketCounts": dict(Counter(row["bucket"] for row in final_rows if row["origin"] == "v3")),
    }
    (staging / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (staging / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    validate_prepared_dataset(staging)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite finalized dataset: {output_dir}")
    staging.rename(output_dir)
    archive = shutil.make_archive(str(output_dir), "zip", output_dir)
    archive_hash = sha256(Path(archive))
    Path(f"{archive}.sha256").write_text(f"{archive_hash}  {Path(archive).name}\n", encoding="ascii")
    return {**metadata, "archive": archive, "archiveSha256": archive_hash}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-dataset", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path,
                        default=Path(os.getenv("V3_WORK_DIR", "data/recognition_dataset_v3_work")))
    parser.add_argument("--output-dir", type=Path,
                        default=Path(os.getenv("V3_DATASET_DIR", "data/recognition_dataset_v3")))
    parser.add_argument("--benchmark-dir", type=Path, default=Path("benchmark"))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    print(json.dumps(finalize_dataset(args.v2_dataset, args.work_dir, args.output_dir,
                                      args.benchmark_dir, args.seed), indent=2))


if __name__ == "__main__":
    main()
