"""High-throughput batch export, import, and calibration workflow for Recognition Dataset v3."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
from PIL import Image

from app.seal_registry import GLOBAL_CARRIER_PREFIXES, get_expected_seal_length
from create_recognition_dataset_v2 import (
    SUPPORTED_EXTENSIONS,
    SealDetectorModel,
    compute_phash,
    compute_sha256,
    evaluate_image_quality,
    extract_best_candidate,
    hamming_distance,
    load_benchmark_index,
    normalize_text,
)
from create_recognition_dataset_v3 import (
    HARD_FAMILIES,
    LABEL_RE,
    QUOTAS,
    SEED,
    STATE_FIELDS,
    PHashIndex,
    _base_record,
    _create_ocr_engine,
    _load_base_index,
    append_state,
    effective_record,
    label_bucket,
    load_latest_state,
    load_reviews,
    normalize_label,
    predict_local_label,
    route_decision,
    summarize_records,
    valid_label,
    write_views,
)

LABEL_PROMPT = """You are labeling container seal serials from the attached crop images.

For every image, read only the container seal serial. Ignore logos, shipping-line names, ISO code, size/type, decorative text, and hardware markings. Preserve repeated characters. Do not invent missing characters.

Rules:
- `id` is the exact filename without extension.
- `label` is uppercase A-Z0-9 only, 7–12 characters.
- If truly unreadable, use an empty label and `readable: false`.
- If uncertain but readable, set `readable: true` and provide at most 3 normalized alternatives.
- Return exactly one record for every attached file.
- Return JSON only, no prose or Markdown:

{"results":[{"id":"filename_without_extension","label":"","readable":false,"alternatives":[]}]}"""


def export_calibration(
    v2_dir: Path,
    work_dir: Path,
    sample_count: int = 500,
    seed: int = SEED,
) -> dict[str, Any]:
    """Export calibration batches from verified v2 dataset."""
    calib_dir = work_dir / "calibration"
    batches_dir = calib_dir / "batches"
    responses_dir = calib_dir / "responses"
    batches_dir.mkdir(parents=True, exist_ok=True)
    responses_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = v2_dir / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"V2 manifest not found: {manifest_path}")

    with manifest_path.open(newline="", encoding="utf-8-sig") as stream:
        all_rows = list(csv.DictReader(stream))

    rng = random.Random(seed)
    rng.shuffle(all_rows)
    selected = all_rows[:sample_count]

    # Save full calibration manifest
    full_manifest = calib_dir / "full_manifest.csv"
    with full_manifest.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = ["id", "v2_image", "ground_truth", "source_image", "sha256", "bucket"]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected:
            writer.writerow({
                "id": row["sha256"],
                "v2_image": row["image"],
                "ground_truth": normalize_label(row["ground_truth"]),
                "source_image": row.get("source_image", ""),
                "sha256": row["sha256"],
                "bucket": label_bucket(row["ground_truth"]),
            })

    # Write labeling prompt template
    (calib_dir / "prompt.txt").write_text(LABEL_PROMPT, encoding="utf-8")

    batch_specs = [
        ("smoke_5", selected[:5]),
        ("rampup_25", selected[:25]),
        ("batch_001", selected[0:100]),
        ("batch_002", selected[100:200]),
        ("batch_003", selected[200:300]),
        ("batch_004", selected[300:400]),
        ("batch_005", selected[400:500]),
    ]

    summary_batches = []
    for batch_name, items in batch_specs:
        b_dir = batches_dir / batch_name
        b_img_dir = b_dir / "images"
        b_img_dir.mkdir(parents=True, exist_ok=True)

        b_manifest = b_dir / "manifest.csv"
        with b_manifest.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["id", "filename", "v2_image", "ground_truth", "bucket"])
            writer.writeheader()
            for item in items:
                src = v2_dir / item["image"]
                item_id = item["sha256"]
                dst = b_img_dir / f"{item_id}.jpg"
                if not dst.is_file():
                    shutil.copy2(src, dst)
                writer.writerow({
                    "id": item_id,
                    "filename": f"{item_id}.jpg",
                    "v2_image": item["image"],
                    "ground_truth": normalize_label(item["ground_truth"]),
                    "bucket": label_bucket(item["ground_truth"]),
                })

        (b_dir / "prompt.txt").write_text(LABEL_PROMPT, encoding="utf-8")
        summary_batches.append({
            "name": batch_name,
            "count": len(items),
            "dir": str(b_dir),
        })

    return {
        "status": "CALIBRATION_EXPORTED",
        "sampleCount": len(selected),
        "calibrationDir": str(calib_dir),
        "batches": summary_batches,
    }


def parse_labeling_json(raw_text: str | None) -> list[dict[str, Any]]:
    """Parse and clean JSON response from multimodal labeling with regex fallback."""
    if not raw_text:
        return []
    text = raw_text.strip()
    code_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if code_match:
        text = code_match.group(1).strip()
    elif text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.S).strip()
    else:
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            text = text[first_brace:last_brace + 1]

    try:
        data = json.loads(text)
        if isinstance(data, list):
            data = {"results": data}
        results = data.get("results")
    except Exception:
        # Robust regex fallback to extract individual result objects
        results = []
        item_pattern = re.compile(
            r'\{\s*"id"\s*:\s*"([^"]+)"\s*,\s*"label"\s*:\s*"([^"]*)"\s*,\s*"readable"\s*:\s*(true|false)'
            r'(?:[^\}]*?"alternatives"\s*:\s*\[([^\]]*)\])?\s*\}',
            re.IGNORECASE | re.DOTALL,
        )
        for m in item_pattern.finditer(raw_text):
            item_id = m.group(1)
            lbl = m.group(2)
            readable = m.group(3).lower() == "true"
            alts_raw = m.group(4) or ""
            alts = re.findall(r'"([^"]+)"', alts_raw)
            results.append({
                "id": item_id,
                "label": lbl,
                "readable": readable,
                "alternatives": alts,
            })
        if not results:
            raise

    if not isinstance(results, list):
        raise ValueError("Response does not contain a 'results' list")

    cleaned = []
    for item in results:
        if not isinstance(item, dict) or "id" not in item:
            raise ValueError(f"Malformed result item: {item}")
        cleaned.append({
            "id": str(item["id"]).strip(),
            "label": normalize_label(item.get("label", "")),
            "readable": bool(item.get("readable", True)),
            "alternatives": [normalize_label(a) for a in item.get("alternatives", []) if normalize_label(a)],
        })
    return cleaned


def evaluate_calibration(
    calib_dir: Path,
    v2_dir: Path | None = None,
) -> dict[str, Any]:
    """Evaluate all calibration batch responses against v2 ground truth."""
    full_manifest_path = calib_dir / "full_manifest.csv"
    if not full_manifest_path.is_file():
        raise FileNotFoundError(f"Calibration manifest not found: {full_manifest_path}")

    with full_manifest_path.open(newline="", encoding="utf-8") as stream:
        ground_truth_map = {row["id"]: row for row in csv.DictReader(stream)}

    responses_dir = calib_dir / "responses"
    response_files = sorted(responses_dir.glob("batch_*.json"))
    if not response_files:
        # Check if smoke_5 or rampup_25 responses exist
        response_files = sorted(responses_dir.glob("*.json"))

    if not response_files:
        raise FileNotFoundError(f"No calibration response files found in {responses_dir}")

    total_evaluated = 0
    parse_success_count = 0
    auto_accepted_count = 0
    correct_count = 0
    exact_match_count = 0
    mismatches = []
    batch_reports = []
    seen_ids = set()

    for resp_file in response_files:
        batch_name = resp_file.stem
        raw_text = resp_file.read_text(encoding="utf-8")
        b_manifest_path = calib_dir / "batches" / batch_name / "manifest.csv"
        expected_ids = set()
        if b_manifest_path.is_file():
            with b_manifest_path.open(newline="", encoding="utf-8") as s:
                expected_ids = {r["id"] for r in csv.DictReader(s)}

        try:
            parsed = parse_labeling_json(raw_text)
            parsed_ids = {item["id"] for item in parsed}
            is_valid_parse = bool(parsed and (not expected_ids or parsed_ids == expected_ids))
        except Exception as exc:
            batch_reports.append({
                "batch": batch_name,
                "parsed": False,
                "error": str(exc),
                "count": 0,
            })
            continue

        b_auto = 0
        b_correct = 0
        b_exact = 0

        for item in parsed:
            item_id = item["id"]
            if item_id in seen_ids:
                continue
            seen_ids.add(item_id)

            gt_row = ground_truth_map.get(item_id)
            if not gt_row:
                continue

            total_evaluated += 1
            parse_success_count += 1
            truth = gt_row["ground_truth"]
            pred = item["label"]
            readable = item["readable"]

            is_auto = readable and valid_label(pred)
            if is_auto:
                auto_accepted_count += 1
                b_auto += 1
                if pred == truth:
                    correct_count += 1
                    b_correct += 1
                else:
                    mismatches.append({
                        "id": item_id,
                        "batch": batch_name,
                        "ground_truth": truth,
                        "prediction": pred,
                        "alternatives": item.get("alternatives", []),
                    })

            if pred == truth:
                exact_match_count += 1
                b_exact += 1

        batch_reports.append({
            "batch": batch_name,
            "parsed": True,
            "count": len(parsed),
            "autoAccepted": b_auto,
            "precision": b_correct / b_auto if b_auto else 0.0,
            "exactMatches": b_exact,
        })

    parse_success_rate = parse_success_count / total_evaluated if total_evaluated else 0.0
    auto_accept_rate = auto_accepted_count / total_evaluated if total_evaluated else 0.0
    precision = correct_count / auto_accepted_count if auto_accepted_count else 0.0
    accuracy = exact_match_count / total_evaluated if total_evaluated else 0.0

    passed_gates = (
        parse_success_rate == 1.0
        and precision >= 0.995
        and auto_accept_rate >= 0.85
    )

    report = {
        "status": "PASS" if passed_gates else "FAIL",
        "totalEvaluated": total_evaluated,
        "parseSuccessRate": parse_success_rate,
        "autoAcceptRate": auto_accept_rate,
        "precision": precision,
        "accuracy": accuracy,
        "gates": {
            "parseSuccess": {"required": 1.0, "actual": parse_success_rate, "passed": parse_success_rate == 1.0},
            "precision": {"required": 0.995, "actual": precision, "passed": precision >= 0.995},
            "autoAcceptRate": {"required": 0.85, "actual": auto_accept_rate, "passed": auto_accept_rate >= 0.85},
        },
        "batches": batch_reports,
        "mismatches": mismatches,
    }

    report_json = calib_dir / "v2_calibration_report.json"
    report_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    # Generate Markdown report
    md_lines = [
        "# V2 Calibration Report: Gemini Multimodal Labeling",
        "",
        f"- **Status**: `{'PASS' if passed_gates else 'FAIL'}`",
        f"- **Total Evaluated Samples**: {total_evaluated}",
        f"- **Parse Success Rate**: {parse_success_rate * 100:.2f}% (Target: 100%)",
        f"- **Auto-Accept Rate**: {auto_accept_rate * 100:.2f}% (Target: ≥85.0%)",
        f"- **Auto-Label Precision**: {precision * 100:.3f}% (Target: ≥99.5%)",
        f"- **Exact Match Accuracy**: {accuracy * 100:.2f}%",
        "",
        "## Batch Breakdown",
        "",
        "| Batch | Samples | Auto-Accepted | Precision | Exact Matches |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]
    for b in batch_reports:
        md_lines.append(
            f"| `{b['batch']}` | {b['count']} | {b.get('autoAccepted', 0)} | "
            f"{b.get('precision', 0.0) * 100:.2f}% | {b.get('exactMatches', 0)} |"
        )

    if mismatches:
        md_lines.extend([
            "",
            "## Discrepancies / Mismatches",
            "",
            "| ID | Batch | Ground Truth | Prediction | Alternatives |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ])
        for m in mismatches:
            md_lines.append(
                f"| `{m['id'][:12]}` | `{m['batch']}` | `{m['ground_truth']}` | "
                f"`{m['prediction']}` | {m['alternatives']} |"
            )

    report_md = calib_dir / "v2_calibration_report.md"
    report_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return report


def export_candidates(
    source_dir: Path,
    benchmark_dir: Path,
    v2_dir: Path,
    work_dir: Path,
    detector_model: Path,
    rec_model_dir: Path,
    batch_size: int = 100,
    max_batches: int = 0,
    scan_limit: int = 0,
) -> dict[str, Any]:
    """Extract, filter, and export seal candidates into batches for multimodal labeling."""
    work_dir.mkdir(parents=True, exist_ok=True)
    images_dir = work_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    batches_dir = work_dir / "candidate_batches"
    batches_dir.mkdir(parents=True, exist_ok=True)

    state_path = work_dir / ".pipeline_state.jsonl"
    latest = load_latest_state(state_path)
    reviews = load_reviews(work_dir)
    effective = [effective_record(item, reviews) for item in latest.values()]

    bucket_counts = Counter(
        item.get("bucket") for item in effective
        if item.get("status") in {"AUTO_ACCEPTED", "VERIFIED"}
    )
    base_hashes, crop_index, label_counts, base_sources = _load_base_index(
        v2_dir, work_dir / ".v2_index.json"
    )
    for item in effective:
        if item.get("status") in {"AUTO_ACCEPTED", "VERIFIED"}:
            label_counts[normalize_label(item.get("proposed_label"))] += 1
        if item.get("crop_phash") and item.get("status") in {"AUTO_ACCEPTED", "VERIFIED", "NEEDS_REVIEW", "PENDING"}:
            crop_index.add(item["crop_phash"], item.get("record_key", ""))

    source_hashes = {item.get("source_sha256") for item in latest.values() if item.get("source_sha256")}
    source_paths = {item.get("source_image") for item in latest.values() if item.get("source_image")}
    benchmark_hashes, benchmark_phashes = load_benchmark_index(benchmark_dir)
    benchmark_index = PHashIndex()
    for name, hashes in benchmark_phashes:
        benchmark_index.add(hashes, name)

    # Track already exported crop sha256
    exported_crops = set()
    for b_manifest in batches_dir.glob("batch_*/manifest.csv"):
        with b_manifest.open(newline="", encoding="utf-8") as s:
            for r in csv.DictReader(s):
                exported_crops.add(r["id"])

    detector = SealDetectorModel(detector_model)
    engine = _create_ocr_engine(rec_model_dir)

    file_cache = work_dir / ".source_file_list.txt"
    if file_cache.is_file():
        files = [source_dir / line.strip() for line in file_cache.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        files = sorted(
            (path for path in source_dir.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS),
            key=lambda path: path.relative_to(source_dir).as_posix(),
        )
        try:
            file_cache.write_text("\n".join(p.relative_to(source_dir).as_posix() for p in files), encoding="utf-8")
        except Exception:
            pass

    scanned = 0
    current_batch_items: list[dict[str, Any]] = []
    exported_batches_count = len(list(batches_dir.glob("batch_*")))

    for path in files:
        if all(bucket_counts[name] >= target for name, target in QUOTAS.items()):
            break
        if scan_limit and scanned >= scan_limit:
            break
        if max_batches and (len(current_batch_items) // batch_size + exported_batches_count >= max_batches):
            break

        relative = path.relative_to(source_dir).as_posix()
        if relative in source_paths:
            continue
        source_sha = compute_sha256(path)
        if source_sha in source_hashes:
            continue
        scanned += 1

        if relative in base_sources or path.name in base_sources:
            record = {**_base_record(relative, source_sha, []), "status": "REJECTED", "reason": "V2_SOURCE_OVERLAP"}
            append_state(state_path, record)
            latest[record["record_key"]] = record
            source_hashes.add(source_sha)
            continue

        try:
            with Image.open(path) as source_image:
                source_image.verify()
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError("cv2.imread failed")
            source_phash = compute_phash(image)
        except Exception as exc:
            record = {**_base_record(relative, source_sha, []), "status": "REJECTED",
                      "reason": "CORRUPTED_IMAGE", "details": str(exc)}
            append_state(state_path, record)
            latest[record["record_key"]] = record
            source_hashes.add(source_sha)
            continue

        base = _base_record(relative, source_sha, source_phash)
        if source_sha in benchmark_hashes:
            record = {**base, "status": "REJECTED", "reason": "BENCHMARK_EXACT_MATCH"}
            append_state(state_path, record)
            latest[record["record_key"]] = record
            source_hashes.add(source_sha)
            continue
        if (near := benchmark_index.near(source_phash)) is not None:
            record = {**base, "status": "REJECTED", "reason": "BENCHMARK_NEAR_MATCH",
                      "details": f"{near[0]} distance={near[1]}"}
            append_state(state_path, record)
            latest[record["record_key"]] = record
            source_hashes.add(source_sha)
            continue

        detections = detector.detect(image)
        if not detections:
            record = {**base, "status": "REJECTED", "reason": "NO_SEAL_DETECTED"}
            append_state(state_path, record)
            latest[record["record_key"]] = record
            source_hashes.add(source_sha)
            continue

        seal_crop, _ = detector.crop_detections(image, detections)[0]
        result = extract_best_candidate(engine, seal_crop)
        crop = result.get("crop")
        if crop is None:
            record = {**base, "status": "REJECTED", "reason": result.get("reason", "NO_LINE_CROP")}
            append_state(state_path, record)
            latest[record["record_key"]] = record
            source_hashes.add(source_sha)
            continue

        quality_ok, quality_reason = evaluate_image_quality(crop)
        if not quality_ok or "CLIPPED_CROP" in result.get("reason", ""):
            record = {**base, "status": "REJECTED", "reason": quality_reason or "CLIPPED_CROP"}
            append_state(state_path, record)
            latest[record["record_key"]] = record
            source_hashes.add(source_sha)
            continue

        encoded_ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not encoded_ok:
            continue
        crop_bytes = encoded.tobytes()
        crop_sha = hashlib.sha256(crop_bytes).hexdigest()
        crop_phash = compute_phash(crop)

        if crop_sha in base_hashes or crop_sha in exported_crops or crop_index.near(crop_phash):
            record = {**base, "status": "REJECTED", "reason": "DUPLICATE_CROP"}
            append_state(state_path, record)
            latest[record["record_key"]] = record
            source_hashes.add(source_sha)
            continue

        local_label = normalize_label(result.get("proposed_label"))

        # Smart Pre-Filter: skip candidates if variant limit or bucket quota is already reached
        if valid_label(local_label):
            if label_counts[local_label] >= 3:
                record = {**base, "status": "REJECTED", "reason": "LABEL_VARIANT_LIMIT_REACHED"}
                append_state(state_path, record)
                latest[record["record_key"]] = record
                source_hashes.add(source_sha)
                continue

            pred_bucket = label_bucket(local_label)
            pending_in_bucket = sum(1 for item in current_batch_items if item.get("pred_bucket") == pred_bucket)
            if bucket_counts[pred_bucket] + pending_in_bucket >= QUOTAS[pred_bucket]:
                record = {**base, "status": "REJECTED", "reason": "BUCKET_QUOTA_FILLED"}
                append_state(state_path, record)
                latest[record["record_key"]] = record
                source_hashes.add(source_sha)
                continue
            label_counts[local_label] += 1
        else:
            pred_bucket = ""

        crop_rel = f"images/{crop_sha[:20]}.jpg"
        (work_dir / crop_rel).write_bytes(crop_bytes)

        record = {
            **base,
            "record_key": crop_sha,
            "image": crop_rel,
            "crop_sha256": crop_sha,
            "crop_phash": crop_phash,
            "rotation": result.get("rotation", 0),
            "local_v2": local_label,
            "pred_bucket": pred_bucket,
            "status": "PENDING",
            "reason": "AWAITING_LABEL",
        }
        append_state(state_path, record)
        latest[record["record_key"]] = record
        source_hashes.add(source_sha)
        crop_index.add(crop_phash, crop_sha)
        exported_crops.add(crop_sha)
        current_batch_items.append(record)

        if len(current_batch_items) >= batch_size:
            exported_batches_count += 1
            _write_candidate_batch(batches_dir, exported_batches_count, current_batch_items, work_dir)
            current_batch_items = []
            if max_batches and exported_batches_count >= max_batches:
                break

    if current_batch_items and not (max_batches and exported_batches_count >= max_batches):
        exported_batches_count += 1
        _write_candidate_batch(batches_dir, exported_batches_count, current_batch_items, work_dir)

    write_views(work_dir, latest)
    return {
        "status": "CANDIDATES_EXPORTED",
        "scanned": scanned,
        "totalBatches": exported_batches_count,
        "batchDir": str(batches_dir),
    }


def _write_candidate_batch(
    batches_dir: Path,
    batch_idx: int,
    items: list[dict[str, Any]],
    work_dir: Path,
) -> None:
    b_dir = batches_dir / f"batch_{batch_idx:03d}"
    b_img_dir = b_dir / "images"
    b_img_dir.mkdir(parents=True, exist_ok=True)

    manifest_file = b_dir / "manifest.csv"
    with manifest_file.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["id", "filename", "crop_path", "source_image", "local_v2", "rotation"],
        )
        writer.writeheader()
        for item in items:
            crop_sha = item["crop_sha256"]
            src_crop = work_dir / item["image"]
            dst_crop = b_img_dir / f"{crop_sha}.jpg"
            if not dst_crop.is_file():
                shutil.copy2(src_crop, dst_crop)
            writer.writerow({
                "id": crop_sha,
                "filename": f"{crop_sha}.jpg",
                "crop_path": item["image"],
                "source_image": item["source_image"],
                "local_v2": item.get("local_v2", ""),
                "rotation": item.get("rotation", 0),
            })
    (b_dir / "prompt.txt").write_text(LABEL_PROMPT, encoding="utf-8")


def _parse_reset_seconds(text: str) -> int:
    """Extract wait duration from AGY quota exhaustion message."""
    m = re.search(r"Resets in (?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?", text, re.IGNORECASE)
    if m:
        h = int(m.group(1) or 0)
        mins = int(m.group(2) or 0)
        s = int(m.group(3) or 0)
        total = h * 3600 + mins * 60 + s
        if total > 0:
            return total + 60
    return 3600


def label_batch(
    batch_dir: Path,
    output_json: Path | None = None,
    model: str = "gemini-3.8-flash-low",
    agy_exe: str = "agy",
    chunk_size: int = 25,
    timeout: int = 300,
) -> dict[str, Any]:
    """Run multimodal labeling on a candidate batch using agy with Gemini, chunking for stability."""
    manifest_path = batch_dir / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Batch manifest not found: {manifest_path}")

    with manifest_path.open(newline="", encoding="utf-8") as s:
        rows = list(csv.DictReader(s))

    if not rows:
        return {"batch": batch_dir.name, "count": 0, "results": []}

    all_parsed: list[dict[str, Any]] = []
    num_chunks = (len(rows) + chunk_size - 1) // chunk_size

    for c_idx in range(0, len(rows), chunk_size):
        chunk_rows = rows[c_idx:c_idx + chunk_size]
        chunk_num = c_idx // chunk_size + 1
        if num_chunks > 1:
            print(f"  [Chunk {chunk_num}/{num_chunks}] Labeling {len(chunk_rows)} images with {model}...")

        lines = [
            "You are an expert OCR annotator for container seal serials.",
            f"For each of the {len(chunk_rows)} images below, view the image using view_file and read the container seal serial.",
            "Rules:",
            "- Ignore logos, brand names, ISO markings, and decorative text.",
            "- Read only the seal serial (uppercase letters and digits).",
            "- Preserve repeated characters accurately.",
            "- If unreadable, set readable=false and label=''.",
            "- If uncertain, provide up to 3 alternatives in the alternatives list.",
            'Return ONLY valid JSON: {"results": [{"id": "<id>", "label": "<serial>", "readable": true, "alternatives": []}]}',
            "",
            "Images to inspect:",
        ]
        for r in chunk_rows:
            img_path = (batch_dir / "images" / r["filename"]).resolve()
            lines.append(f"- ID: {r['id']}, Path: {img_path}")

        prompt = "\n".join(lines)
        cmd = [
            agy_exe,
            "--model",
            model,
            "--dangerously-skip-permissions",
            "--output-format",
            "text",
        ]
        parsed = None
        attempt = 0
        while attempt < 3:
            proc = subprocess.run(
                cmd,
                input=prompt,
                text=True,
                capture_output=True,
                cwd=str(batch_dir),
                timeout=timeout,
            )
            out_text = proc.stdout or ""
            err_text = proc.stderr or ""
            err_msg = err_text + " " + out_text
            if "RESOURCE_EXHAUSTED" in err_msg or "code 429" in err_msg or "quota reached" in err_msg.lower():
                wait_sec = _parse_reset_seconds(err_msg)
                print(f"    [Quota Limit] Pausing for {wait_sec}s ({wait_sec // 60}m) until quota resets...")
                time.sleep(wait_sec)
                continue

            if proc.returncode == 0 and out_text.strip():
                try:
                    parsed = parse_labeling_json(out_text)
                    if parsed:
                        break
                    print(f"    [Retry chunk {chunk_num}] Empty parsed results, retrying...")
                    attempt += 1
                except Exception as exc:
                    print(f"    [Retry chunk {chunk_num}] Parse error: {exc}, retrying...")
                    attempt += 1
            else:
                snippet = (err_text[:300] or out_text[:300]).strip()
                print(f"    [Retry chunk {chunk_num}] Exit {proc.returncode} / empty output: {snippet}, retrying...")
                attempt += 1

            if attempt < 3:
                time.sleep(3)

        if not parsed:
            print(f"    [Notice chunk {chunk_num}] Could not obtain valid labels for chunk after retries. Marking {len(chunk_rows)} items for review.")
            parsed = []

        parsed_ids = {p["id"] for p in parsed}
        for r in chunk_rows:
            if r["id"] not in parsed_ids:
                parsed.append({"id": r["id"], "label": "", "readable": False, "alternatives": []})
        all_parsed.extend(parsed)

    if output_json is None:
        output_json = batch_dir / "response.json"

    output_json.write_text(json.dumps({"results": all_parsed}, indent=2), encoding="utf-8")
    return {
        "batch": batch_dir.name,
        "count": len(all_parsed),
        "output": str(output_json),
    }


def import_batch(
    batch_dir: Path,
    response_json: Path,
    work_dir: Path,
) -> dict[str, Any]:
    """Import Gemini multimodal response for a candidate batch and apply routing/quotas."""
    state_path = work_dir / ".pipeline_state.jsonl"
    latest = load_latest_state(state_path)
    reviews = load_reviews(work_dir)
    effective = [effective_record(item, reviews) for item in latest.values()]

    bucket_counts = Counter(
        item.get("bucket") for item in effective
        if item.get("status") in {"AUTO_ACCEPTED", "VERIFIED"}
    )
    label_counts = Counter(
        normalize_label(item.get("proposed_label")) for item in effective
        if item.get("status") in {"AUTO_ACCEPTED", "VERIFIED"}
    )

    b_manifest_path = batch_dir / "manifest.csv"
    if not b_manifest_path.is_file():
        raise FileNotFoundError(f"Batch manifest not found: {b_manifest_path}")

    with b_manifest_path.open(newline="", encoding="utf-8") as s:
        batch_manifest = {r["id"]: r for r in csv.DictReader(s)}

    raw_text = response_json.read_text(encoding="utf-8")
    parsed = parse_labeling_json(raw_text)
    parsed_map = {item["id"]: item for item in parsed}

    missing_ids = set(batch_manifest) - set(parsed_map)
    if missing_ids:
        if len(missing_ids) > len(batch_manifest) * 0.2:
            raise ValueError(f"Response missing {len(missing_ids)} IDs from batch manifest: {list(missing_ids)[:5]}")
        for m_id in missing_ids:
            parsed_map[m_id] = {"id": m_id, "label": "", "readable": False, "alternatives": []}

    accepted = 0
    routed_review = 0
    rejected = 0

    for item_id, row in batch_manifest.items():
        resp_item = parsed_map[item_id]
        record = latest.get(item_id)
        if not record:
            continue

        label = normalize_label(resp_item.get("label", ""))
        readable = bool(resp_item.get("readable", True))
        local_label = normalize_label(row.get("local_v2", ""))

        status, proposed, reason = route_decision(local_label, resp_item, None)

        record["agy_pass1"] = resp_item
        record["proposed_label"] = proposed
        record["status"] = status
        record["reason"] = reason

        if status == "AUTO_ACCEPTED":
            bucket = label_bucket(proposed)
            record["bucket"] = bucket
            if label_counts[proposed] >= 3:
                record["status"] = "REJECTED"
                record["reason"] = "LABEL_VARIANT_LIMIT_REACHED"
                rejected += 1
            elif bucket_counts[bucket] >= QUOTAS[bucket]:
                record["status"] = "REJECTED"
                record["reason"] = "BUCKET_QUOTA_FILLED"
                rejected += 1
            else:
                label_counts[proposed] += 1
                bucket_counts[bucket] += 1
                accepted += 1
        elif status in {"SECOND_PASS", "NEEDS_REVIEW"}:
            record["status"] = "NEEDS_REVIEW"
            record["bucket"] = label_bucket(proposed) if valid_label(proposed) else ""
            routed_review += 1
        else:
            record["status"] = "REJECTED"
            record["reason"] = "UNREADABLE_OR_INVALID"
            rejected += 1

        append_state(state_path, record)
        latest[record["record_key"]] = record

    summary = write_views(work_dir, latest)
    return {
        "batch": batch_dir.name,
        "imported": len(batch_manifest),
        "autoAccepted": accepted,
        "needsReview": routed_review,
        "rejected": rejected,
        "currentQuotas": summary["buckets"],
        "remaining": summary["remaining"],
        "complete": summary["complete"],
    }


def run_pipeline(
    source_dir: Path,
    benchmark_dir: Path,
    v2_dir: Path,
    work_dir: Path,
    detector_model: Path,
    rec_model_dir: Path,
    batch_size: int = 100,
    max_batches: int = 0,
    scan_limit: int = 0,
    model: str = "gemini-3.8-flash-low",
    timeout: int = 600,
) -> dict[str, Any]:
    """End-to-end pipeline: exports batches, labels via Gemini multimodal, and imports state."""
    batches_dir = work_dir / "candidate_batches"
    batches_dir.mkdir(parents=True, exist_ok=True)
    state_path = work_dir / ".pipeline_state.jsonl"

    processed_batches = 0

    while True:
        latest = load_latest_state(state_path)
        reviews = load_reviews(work_dir)
        effective = [effective_record(item, reviews) for item in latest.values()]
        bucket_counts = Counter(
            item.get("bucket") for item in effective
            if item.get("status") in {"AUTO_ACCEPTED", "VERIFIED"}
        )
        if all(bucket_counts[name] >= target for name, target in QUOTAS.items()):
            print("All quotas filled!")
            break

        if max_batches and processed_batches >= max_batches:
            print(f"Reached max batches limit: {max_batches}")
            break

        existing_batches = sorted(batches_dir.glob("batch_*"), key=lambda p: p.name)
        pending_batch = None
        for b in existing_batches:
            manifest_file = b / "manifest.csv"
            if not manifest_file.is_file():
                continue
            with manifest_file.open(newline="", encoding="utf-8") as s:
                b_ids = [r["id"] for r in csv.DictReader(s)]
            if any(latest.get(i, {}).get("status") == "PENDING" for i in b_ids):
                pending_batch = b
                break

        if pending_batch is None:
            print(f"Exporting next batch of up to {batch_size} candidates...")
            export_candidates(
                source_dir=source_dir,
                benchmark_dir=benchmark_dir,
                v2_dir=v2_dir,
                work_dir=work_dir,
                detector_model=detector_model,
                rec_model_dir=rec_model_dir,
                batch_size=batch_size,
                max_batches=len(existing_batches) + 1,
                scan_limit=scan_limit,
            )
            new_batches = sorted(batches_dir.glob("batch_*"), key=lambda p: p.name)
            if len(new_batches) <= len(existing_batches):
                print("No more candidates could be extracted.")
                break
            pending_batch = new_batches[-1]

        resp_file = pending_batch / "response.json"
        if not resp_file.is_file():
            print(f"Labeling {pending_batch.name} ({len(list((pending_batch / 'images').glob('*.jpg')))} images) using {model}...")
            label_batch(pending_batch, resp_file, model=model, timeout=timeout)

        print(f"Importing {pending_batch.name}...")
        import_res = import_batch(pending_batch, resp_file, work_dir)
        processed_batches += 1
        print(f"Batch {pending_batch.name} imported: accepted={import_res['autoAccepted']}, "
              f"review={import_res['needsReview']}, quotas={import_res['currentQuotas']}")

    latest = load_latest_state(state_path)
    views = write_views(work_dir, latest)

    if views.get("complete"):
        print("\nALL 15,000 CROPS COMPLETED! Finalizing v3 dataset...")
        from finalize_dataset_v3 import finalize_dataset
        out_dataset = Path(os.getenv("V3_DATASET_DIR", "data/recognition_dataset_v3"))
        fin_res = finalize_dataset(
            base_dataset=v2_dir,
            work_dir=work_dir,
            output_dir=out_dataset,
            benchmark_dir=benchmark_dir,
            seed=SEED,
        )
        print("Finalization complete:\n", json.dumps(fin_res, indent=2))
        views["finalized"] = fin_res

    return views



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    # export-calibration
    p_exp_cal = sub.add_parser("export-calibration")
    p_exp_cal.add_argument("--v2-dataset", type=Path, default=Path(os.getenv("V2_DATASET_DIR", "data/recognition_dataset_v2")))
    p_exp_cal.add_argument("--work-dir", type=Path, default=Path(os.getenv("V3_WORK_DIR", "data/recognition_dataset_v3_work")))
    p_exp_cal.add_argument("--samples", type=int, default=500)
    p_exp_cal.add_argument("--seed", type=int, default=SEED)

    # evaluate-calibration
    p_eval_cal = sub.add_parser("evaluate-calibration")
    p_eval_cal.add_argument("--calib-dir", type=Path, default=Path(os.getenv("V3_CALIB_DIR", "data/recognition_dataset_v3_work/calibration")))

    # export-candidates
    p_exp_cand = sub.add_parser("export-candidates")
    p_exp_cand.add_argument("--source-dir", type=Path, default=Path(os.getenv("SOURCE_DIR", "data/new_dataset")))
    p_exp_cand.add_argument("--benchmark-dir", type=Path, default=Path("benchmark"))
    p_exp_cand.add_argument("--v2-dataset", type=Path, default=Path(os.getenv("V2_DATASET_DIR", "data/recognition_dataset_v2")))
    p_exp_cand.add_argument("--work-dir", type=Path, default=Path(os.getenv("V3_WORK_DIR", "data/recognition_dataset_v3_work")))
    p_exp_cand.add_argument("--detector-model", type=Path, default=Path("models/seal-detector-v1/best.onnx"))
    p_exp_cand.add_argument("--rec-model-dir", type=Path, default=Path("models/seal-ocr-rec-v1"))
    p_exp_cand.add_argument("--batch-size", type=int, default=100)
    p_exp_cand.add_argument("--max-batches", type=int, default=0)
    p_exp_cand.add_argument("--scan-limit", type=int, default=0)

    # label-batch
    p_lbl = sub.add_parser("label-batch")
    p_lbl.add_argument("--batch-dir", type=Path, required=True)
    p_lbl.add_argument("--output", type=Path, default=None)
    p_lbl.add_argument("--model", type=str, default="gemini-3.8-flash-low")
    p_lbl.add_argument("--timeout", type=int, default=300)

    # process-batch
    p_proc = sub.add_parser("process-batch")
    p_proc.add_argument("--batch-dir", type=Path, required=True)
    p_proc.add_argument("--work-dir", type=Path, default=Path(os.getenv("V3_WORK_DIR", "data/recognition_dataset_v3_work")))
    p_proc.add_argument("--model", type=str, default="gemini-3.8-flash-low")
    p_proc.add_argument("--timeout", type=int, default=300)

    # import-batch
    p_imp = sub.add_parser("import-batch")
    p_imp.add_argument("--batch-dir", type=Path, required=True)
    p_imp.add_argument("--response", type=Path, required=True)
    p_imp.add_argument("--work-dir", type=Path, default=Path(os.getenv("V3_WORK_DIR", "data/recognition_dataset_v3_work")))

    # run-pipeline
    p_pipe = sub.add_parser("run-pipeline")
    p_pipe.add_argument("--source-dir", type=Path, default=Path(os.getenv("SOURCE_DIR", "data/new_dataset")))
    p_pipe.add_argument("--benchmark-dir", type=Path, default=Path("benchmark"))
    p_pipe.add_argument("--v2-dataset", type=Path, default=Path(os.getenv("V2_DATASET_DIR", "data/recognition_dataset_v2")))
    p_pipe.add_argument("--work-dir", type=Path, default=Path(os.getenv("V3_WORK_DIR", "data/recognition_dataset_v3_work")))
    p_pipe.add_argument("--detector-model", type=Path, default=Path("models/seal-detector-v1/best.onnx"))
    p_pipe.add_argument("--rec-model-dir", type=Path, default=Path("models/seal-ocr-rec-v1"))
    p_pipe.add_argument("--batch-size", type=int, default=100)
    p_pipe.add_argument("--max-batches", type=int, default=0)
    p_pipe.add_argument("--scan-limit", type=int, default=0)
    p_pipe.add_argument("--model", type=str, default="gemini-3.8-flash-low")
    p_pipe.add_argument("--timeout", type=int, default=300)

    # status
    p_stat = sub.add_parser("status")
    p_stat.add_argument("--work-dir", type=Path, default=Path(os.getenv("V3_WORK_DIR", "data/recognition_dataset_v3_work")))

    args = parser.parse_args()

    if args.cmd == "export-calibration":
        res = export_calibration(args.v2_dataset, args.work_dir, args.samples, args.seed)
        print(json.dumps(res, indent=2))
    elif args.cmd == "evaluate-calibration":
        res = evaluate_calibration(args.calib_dir)
        print(json.dumps(res, indent=2))
    elif args.cmd == "export-candidates":
        res = export_candidates(
            args.source_dir, args.benchmark_dir, args.v2_dataset, args.work_dir,
            args.detector_model, args.rec_model_dir, args.batch_size, args.max_batches, args.scan_limit,
        )
        print(json.dumps(res, indent=2))
    elif args.cmd == "label-batch":
        res = label_batch(args.batch_dir, args.output, model=args.model, timeout=args.timeout)
        print(json.dumps(res, indent=2))
    elif args.cmd == "process-batch":
        resp_file = args.batch_dir / "response.json"
        if not resp_file.is_file():
            label_batch(args.batch_dir, resp_file, model=args.model, timeout=args.timeout)
        res = import_batch(args.batch_dir, resp_file, args.work_dir)
        print(json.dumps(res, indent=2))
    elif args.cmd == "import-batch":
        res = import_batch(args.batch_dir, args.response, args.work_dir)
        print(json.dumps(res, indent=2))
    elif args.cmd == "run-pipeline":
        res = run_pipeline(
            args.source_dir, args.benchmark_dir, args.v2_dataset, args.work_dir,
            args.detector_model, args.rec_model_dir, args.batch_size, args.max_batches,
            args.scan_limit, model=args.model, timeout=args.timeout,
        )
        print(json.dumps(res, indent=2))
    elif args.cmd == "status":
        latest = load_latest_state(args.work_dir / ".pipeline_state.jsonl")
        res = write_views(args.work_dir, latest)
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
