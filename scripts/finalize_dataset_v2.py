"""Finalize recognition dataset v2 with strict acceptance checks and group-aware split."""

import argparse
import csv
import hashlib
import json
import os
import random
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

TARGET_TOTAL = 5000
TARGET_TRAIN = 4500
TARGET_VAL = 500
ALLOWED_LABEL_RE = re.compile(r"^[A-Z0-9]{5,20}$")

BRAND_NAMES = {
    "WANHAI", "WHL", "YANGMING", "YML", "MAERSK", "MSK", "COSCO", "ONE",
    "HAPAG", "LLOYD", "HAPAGLLOYD", "EVERGREEN", "EMC", "CMA", "CGM", "CMACGM",
    "MSC", "PIL", "SITC", "ZIM", "OOCL", "HMM", "KMTC", "SEAL", "SECURITY",
    "CONTAINER", "HIGH", "BOLT", "LOCK", "MEGA", "CUSTOMS", "LINE", "SHIPPING",
    "CHINA", "SHANGHAI", "GENSTAR", "INTERPOOL", "TEX", "BEACON", "TRITON", "CAI", "SEACUBE"
}


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finalize_dataset(
    work_dir: Path,
    output_dir: Path,
    benchmark_dir: Path,
    target_total: int = TARGET_TOTAL,
    target_train: int = TARGET_TRAIN,
    target_val: int = TARGET_VAL,
    seed: int = 20260918,
) -> dict[str, Any]:
    reviewed_file = work_dir / "reviewed_labels.csv"
    if not reviewed_file.is_file():
        raise FileNotFoundError(f"reviewed_labels.csv not found in {work_dir}. Human review must be conducted first!")

    verified_samples = []
    skipped = []
    with reviewed_file.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") == "VERIFIED":
                crop_rel = row.get("image", "").strip()
                if not crop_rel:
                    skipped.append((row, "EMPTY_IMAGE_PATH"))
                    continue
                label = row.get("proposed_label", "").strip().upper()
                if not ALLOWED_LABEL_RE.match(label):
                    skipped.append((row, f"INVALID_PATTERN: {label}"))
                    continue
                if any(b in label for b in BRAND_NAMES):
                    skipped.append((row, f"BRAND_NAME: {label}"))
                    continue
                src_crop = work_dir / crop_rel
                if not src_crop.is_file():
                    src_crop = work_dir / "images" / Path(crop_rel).name
                if not src_crop.is_file():
                    skipped.append((row, "FILE_NOT_FOUND"))
                    continue

                verified_samples.append({
                    "crop_rel": crop_rel,
                    "ground_truth": label,
                    "source_image": row.get("source_image", ""),
                })

    print(f"Loaded {len(verified_samples)} clean VERIFIED samples (skipped {len(skipped)} invalid/brand/missing).")

    if len(verified_samples) < target_total:
        missing = target_total - len(verified_samples)
        error_msg = (
            f"ERROR: Only {len(verified_samples)} VERIFIED samples available. "
            f"Need exactly {target_total}. Still missing {missing} samples! "
            f"Do not lower quality gate. Please review more samples from review_queue.csv."
        )
        print(error_msg)
        return {"status": "INCOMPLETE", "verified_count": len(verified_samples), "missing": missing}

    # If more than target_total, pick exactly target_total
    if len(verified_samples) > target_total:
        print(f"Pruning to exactly {target_total} samples...")
        rng = random.Random(seed)
        rng.shuffle(verified_samples)
        verified_samples = verified_samples[:target_total]

    # Staging directory for atomic write
    staging_dir = output_dir.with_name(f".{output_dir.name}.staging")
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)
    staging_images = staging_dir / "images"
    staging_images.mkdir(parents=True)

    # Benchmark overlap check
    bm_manifest = json.loads((benchmark_dir / "manifest.json").read_text(encoding="utf-8"))
    bm_hashes = {item["sha256"]: item["image"] for item in bm_manifest["images"]}

    # Copy crops and rename cleanly
    final_items = []
    crop_hashes = {}
    for idx, sample in enumerate(verified_samples, 1):
        src_crop = work_dir / sample["crop_rel"]
        if not src_crop.is_file():
            src_crop = work_dir / "images" / Path(sample["crop_rel"]).name
        if not src_crop.is_file():
            raise FileNotFoundError(f"Crop image file missing: {sample['crop_rel']}")

        with Image.open(src_crop) as img:
            img.verify()

        sha = compute_sha256(src_crop)
        if sha in bm_hashes:
            raise ValueError(f"CRITICAL: Benchmark overlap detected for crop {src_crop.name} with benchmark {bm_hashes[sha]}")

        new_crop_name = f"crop_{idx:06d}.jpg"
        dst_crop = staging_images / new_crop_name
        shutil.copy2(src_crop, dst_crop)

        final_items.append({
            "image": f"images/{new_crop_name}",
            "ground_truth": sample["ground_truth"],
            "source_image": sample["source_image"],
            "sha256": sha,
            "original_crop": sample["crop_rel"]
        })

    # Group-aware splitting
    # Group by source_image and identical ground_truth
    parents = list(range(len(final_items)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parents[rj] = ri

    source_to_indices = {}
    for i, item in enumerate(final_items):
        source_to_indices.setdefault(item["source_image"], []).append(i)

    for indices in source_to_indices.values():
        for k in range(len(indices) - 1):
            union(indices[k], indices[k + 1])

    groups = {}
    for i in range(len(final_items)):
        groups.setdefault(find(i), []).append(i)

    group_list = list(groups.values())
    rng = random.Random(seed)
    rng.shuffle(group_list)

    val_indices = set()
    train_indices = set()

    for grp in group_list:
        if len(val_indices) + len(grp) <= target_val:
            val_indices.update(grp)
        else:
            train_indices.update(grp)

    # If small imbalance due to group boundary, adjust if necessary
    for item_idx in range(len(final_items)):
        if item_idx not in val_indices:
            train_indices.add(item_idx)

    for i in range(len(final_items)):
        final_items[i]["split"] = "val" if i in val_indices else "train"

    train_lines = [f"{it['image']}\t{it['ground_truth']}" for it in final_items if it["split"] == "train"]
    val_lines = [f"{it['image']}\t{it['ground_truth']}" for it in final_items if it["split"] == "val"]

    (staging_dir / "train.txt").write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    (staging_dir / "val.txt").write_text("\n".join(val_lines) + "\n", encoding="utf-8")

    with (staging_dir / "labels.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "ground_truth", "source_image", "split", "sha256"])
        writer.writeheader()
        for it in final_items:
            writer.writerow({
                "image": it["image"],
                "ground_truth": it["ground_truth"],
                "source_image": it["source_image"],
                "split": it["split"],
                "sha256": it["sha256"]
            })

    # Copy manifests and logs from work_dir
    for name in ["rejected.csv", "review_queue.csv"]:
        src = work_dir / name
        if src.is_file():
            shutil.copy2(src, staging_dir / name)

    with (staging_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "ground_truth", "source_image", "split", "sha256", "original_crop"])
        writer.writeheader()
        writer.writerows(final_items)

    dataset_fingerprint = hashlib.sha256("\n".join(
        f"{it['image']}\t{it['ground_truth']}\t{it['sha256']}" for it in sorted(final_items, key=lambda x: x["image"])
    ).encode()).hexdigest()

    validation_report = {
        "totalImages": len(final_items),
        "trainCount": len(train_lines),
        "valCount": len(val_lines),
        "datasetSha256": dataset_fingerprint,
        "benchmarkOverlapCount": 0,
        "emptyLabels": 0,
        "invalidCharLabels": 0,
        "groupsCount": len(group_list),
        "status": "PASS"
    }
    (staging_dir / "validation_report.json").write_text(json.dumps(validation_report, indent=2) + "\n", encoding="utf-8")

    metadata = {
        "datasetVersion": "seal-recognition-v2",
        "createdAtUtc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "totalVerified": len(final_items),
        "trainSamples": len(train_lines),
        "valSamples": len(val_lines),
        "datasetSha256": dataset_fingerprint,
        "seed": seed
    }
    (staging_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    staging_dir.rename(output_dir)

    zip_path = output_dir.with_name(f"{output_dir.name}.zip")
    print(f"Creating archive {zip_path}...")
    shutil.make_archive(str(output_dir), "zip", output_dir)
    metadata["archiveSha256"] = compute_sha256(zip_path)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"\nSuccessfully finalized dataset at: {output_dir}")
    print(f"Archive created at: {zip_path}")
    print(json.dumps(metadata, indent=2))
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, default=Path("recognition_dataset_v2_work"))
    parser.add_argument("--output-dir", type=Path, default=Path("recognition_dataset_v2"))
    parser.add_argument("--benchmark-dir", type=Path, default=Path("benchmark"))
    parser.add_argument("--target-total", type=int, default=TARGET_TOTAL)
    parser.add_argument("--target-train", type=int, default=TARGET_TRAIN)
    parser.add_argument("--target-val", type=int, default=TARGET_VAL)
    parser.add_argument("--seed", type=int, default=20260918)
    args = parser.parse_args()

    finalize_dataset(
        work_dir=args.work_dir,
        output_dir=args.output_dir,
        benchmark_dir=args.benchmark_dir,
        target_total=args.target_total,
        target_train=args.target_train,
        target_val=args.target_val,
        seed=args.seed
    )
