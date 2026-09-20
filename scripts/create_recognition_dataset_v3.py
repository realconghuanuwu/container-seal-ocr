"""Build Recognition Dataset v3 with resumable AGY-assisted labeling."""

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
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable

import cv2
import numpy as np
from PIL import Image

from app.seal_registry import GLOBAL_CARRIER_PREFIXES, get_expected_seal_length
from scripts.create_recognition_dataset_v2 import (
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


MODEL = "gemini-3.8-flash-high"
PROMPT_VERSION = "seal-v3-ocr-1"
TRANSFORM_VERSION = "clahe-stretch-1.35-v1"
SEED = 20260919
HARD_FAMILIES = ("OOLKCM", "OOLKCK", "VNHPH", "SJJA", "YMAR", "YMAT", "2869", "HLC", "FX", "SF")
QUOTAS = {"hard": 6512, "registered": 660, "numeric_unknown": 2828}
LABEL_RE = re.compile(r"^[A-Z0-9]{7,12}$")
STATE_FIELDS = (
    "record_key", "source_image", "source_sha256", "source_phash", "image",
    "crop_sha256", "crop_phash", "rotation", "local_v2", "agy_pass1",
    "agy_pass2", "proposed_label", "bucket", "status", "reason",
    "model", "prompt_version", "transform_version",
)


def normalize_label(value: str | None) -> str:
    return normalize_text(value or "")


def valid_label(value: str | None) -> bool:
    label = normalize_label(value)
    if LABEL_RE.fullmatch(label) is None:
        return False
    expected = get_expected_seal_length(label)
    return expected is None or len(label) == expected


def label_bucket(value: str) -> str:
    label = normalize_label(value)
    if any(label.startswith(prefix) for prefix in HARD_FAMILIES):
        return "hard"
    if any(label.startswith(prefix) for prefix in GLOBAL_CARRIER_PREFIXES):
        return "registered"
    return "numeric_unknown"


def route_decision(local_v2: str, pass1: dict[str, Any],
                   pass2: dict[str, Any] | None = None) -> tuple[str, str, str]:
    local = normalize_label(local_v2)
    first = normalize_label(pass1.get("label")) if pass1.get("readable", True) else ""
    if valid_label(first) and first == local:
        return "AUTO_ACCEPTED", first, "LOCAL_V2_AGY_AGREEMENT"
    if pass2 is None:
        return "SECOND_PASS", first or local, "LOCAL_V2_AGY_DISAGREEMENT"
    second = normalize_label(pass2.get("label")) if pass2.get("readable", True) else ""
    if valid_label(first) and first == second:
        return "AUTO_ACCEPTED", first, "AGY_TWO_PASS_AGREEMENT"
    proposed = second if valid_label(second) else first if valid_label(first) else local
    return "NEEDS_REVIEW", proposed, f"local-v2={local or '-'}; agy-1={first or '-'}; agy-2={second or '-'}"


def parse_agy_output(stdout: str) -> list[dict[str, Any]]:
    outer = json.loads(stdout)
    if isinstance(outer, dict) and "status" in outer:
        if outer.get("status") != "SUCCESS":
            raise RuntimeError(outer.get("error") or f"AGY status: {outer.get('status')}")
        payload: Any = outer.get("response", "")
    else:
        payload = outer
    if isinstance(payload, str):
        payload = payload.strip()
        if payload.startswith("```"):
            payload = re.sub(r"^```(?:json)?\s*|\s*```$", "", payload, flags=re.I | re.S)
        payload = json.loads(payload)
    if isinstance(payload, list):
        payload = {"results": payload}
    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, list):
        raise ValueError("AGY response does not contain a results array")
    parsed = []
    for item in results:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            raise ValueError("Malformed AGY result")
        parsed.append({
            "id": item["id"],
            "label": normalize_label(item.get("label")),
            "readable": bool(item.get("readable", True)),
            "alternatives": [normalize_label(v) for v in item.get("alternatives", []) if normalize_label(v)],
        })
    return parsed


def make_composite(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    clahe = cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR)
    original = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    stretched = cv2.resize(original, (int(width * 1.35), height), interpolation=cv2.INTER_CUBIC)
    target_width = max(width, stretched.shape[1])

    def pad(item: np.ndarray) -> np.ndarray:
        return cv2.copyMakeBorder(item, 0, 0, 0, target_width - item.shape[1], cv2.BORDER_CONSTANT,
                                  value=(255, 255, 255))

    return cv2.vconcat([pad(original), pad(clahe), pad(stretched)])


class PHashIndex:
    """Small banded index; candidates are always verified with full Hamming distance."""

    def __init__(self) -> None:
        self.hashes: list[list[int]] = []
        self.payloads: list[str] = []
        self.bands: dict[tuple[int, int], set[int]] = {}

    def add(self, hashes: list[int], payload: str) -> None:
        index = len(self.hashes)
        self.hashes.append([int(value) for value in hashes])
        self.payloads.append(payload)
        for value in hashes:
            for band in range(4):
                part = (value >> (band * 16)) & 0xFFFF
                self.bands.setdefault((band, part), set()).add(index)

    def near(self, hashes: list[int], threshold: int = 6) -> tuple[str, int] | None:
        matches = self.near_all(hashes, threshold)
        return min(matches, key=lambda item: item[1]) if matches else None

    def near_all(self, hashes: list[int], threshold: int = 6) -> list[tuple[str, int]]:
        candidates: set[int] = set()
        for value in hashes:
            for band in range(4):
                part = (value >> (band * 16)) & 0xFFFF
                for neighbor in [part, *(part ^ (1 << bit) for bit in range(16))]:
                    candidates.update(self.bands.get((band, neighbor), ()))
        matches = []
        for index in candidates:
            distance = hamming_distance(hashes, self.hashes[index])
            if distance <= threshold:
                matches.append((self.payloads[index], distance))
        return matches


def _agy_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["results"],
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["id", "label", "readable", "alternatives"],
                    "properties": {
                        "id": {"type": "string"},
                        "label": {"type": "string"},
                        "readable": {"type": "boolean"},
                        "alternatives": {"type": "array", "items": {"type": "string"}},
                    },
                },
            }
        },
    }


def _find_agy_project_id(folder: Path) -> str | None:
    marker = folder / ".agy_project.json"
    if marker.is_file():
        try:
            return json.loads(marker.read_text(encoding="utf-8"))["id"]
        except (KeyError, json.JSONDecodeError):
            pass
    projects = Path.home() / ".gemini" / "config" / "projects"
    def normalized_uri(value: str) -> str:
        return value.rstrip("/").lower().replace("file:///", "file://")

    expected = normalized_uri(folder.resolve().as_uri())
    if projects.is_dir():
        for path in sorted(projects.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                resources = data.get("projectResources", {}).get("resources", [])
                if any(normalized_uri(str(item.get("folderUri", ""))) == expected for item in resources):
                    marker.write_text(json.dumps({"id": data["id"]}), encoding="utf-8")
                    return data["id"]
            except (OSError, KeyError, json.JSONDecodeError):
                continue
    return None


def run_agy(images: list[tuple[str, Path]], staging_root: Path, agy_exe: str = "agy",
            runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
            sleeper: Callable[[float], None] = time.sleep) -> dict[str, dict[str, Any]]:
    staging_root.mkdir(parents=True, exist_ok=True)
    delays = (2, 5, 15)
    last_error: Exception | None = None
    for attempt, delay in enumerate(delays, 1):
        try:
            with tempfile.TemporaryDirectory(prefix="batch-", dir=staging_root) as temporary:
                folder = Path(temporary)
                listed = []
                for index, (item_id, source) in enumerate(images):
                    target = folder / f"image_{index:02d}{source.suffix.lower()}"
                    shutil.copy2(source, target)
                    listed.append({"id": item_id, "path": str(target.resolve())})
                schema_path = folder / "schema.json"
                schema_path.write_text(json.dumps(_agy_schema()), encoding="utf-8")
                prompt = (
                    "OCR only. Read the container seal serial in each listed image. Ignore logos, brands, "
                    "ISO text and hardware markings. Preserve every repeated character. Return only the "
                    "required JSON. Use an empty label and readable=false when the serial is genuinely unreadable. "
                    "Use view_file on every exact absolute path below; do not use run_command, list_dir, search, "
                    "or any other tool.\n"
                    + json.dumps(listed, ensure_ascii=False)
                )
                command = [
                    agy_exe, "--print", prompt, "--model", MODEL, "--sandbox",
                    "--disable-slash-commands", "--output-format", "json", "--json-schema",
                    str(schema_path), "--print-timeout", "2m",
                ]
                project_id = _find_agy_project_id(staging_root)
                command[1:1] = ["--project", project_id] if project_id else ["--new-project"]
                completed = runner(command, cwd=staging_root, capture_output=True, text=True, timeout=150)
                if completed.returncode != 0:
                    raise RuntimeError(completed.stderr.strip() or f"AGY exited {completed.returncode}")
                try:
                    parsed = parse_agy_output(completed.stdout)
                except Exception as exc:
                    raise ValueError(
                        f"Invalid AGY output: stdout={completed.stdout[:500]!r}; "
                        f"stderr={completed.stderr[:500]!r}"
                    ) from exc
                by_id = {item["id"]: item for item in parsed}
                expected = {item_id for item_id, _ in images}
                if set(by_id) != expected:
                    raise ValueError(f"AGY ids differ: expected {sorted(expected)}, received {sorted(by_id)}")
                _find_agy_project_id(staging_root)
                return by_id
        except Exception as exc:  # retried at the process boundary
            last_error = exc
            if attempt < len(delays):
                sleeper(delay)
    raise RuntimeError(f"AGY failed after 3 attempts: {last_error}") from last_error


def load_latest_state(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return latest
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue  # tolerate a truncated final line after interruption
        key = record.get("record_key") or record.get("crop_sha256") or f"source:{record.get('source_sha256', '')}"
        latest[key] = record
    return latest


def append_state(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()


def load_reviews(work_dir: Path) -> dict[str, dict[str, str]]:
    path = work_dir / "reviewed_labels.csv"
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return {row.get("image", ""): row for row in csv.DictReader(stream) if row.get("image")}


def effective_record(record: dict[str, Any], reviews: dict[str, dict[str, str]]) -> dict[str, Any]:
    result = dict(record)
    review = reviews.get(record.get("image", ""))
    if review and record.get("status") == "NEEDS_REVIEW":
        if review.get("status") == "VERIFIED" and valid_label(review.get("proposed_label")):
            result["status"] = "VERIFIED"
            result["proposed_label"] = normalize_label(review["proposed_label"])
            result["bucket"] = label_bucket(result["proposed_label"])
        elif review.get("status") == "REJECTED":
            result["status"] = "REJECTED"
            result["reason"] = "MANUAL_REJECTED"
    return result


def summarize_records(records: Iterable[dict[str, Any]], reviews: dict[str, dict[str, str]]) -> dict[str, Any]:
    effective = [effective_record(item, reviews) for item in records]
    statuses = Counter(item.get("status", "UNKNOWN") for item in effective)
    buckets = Counter(
        item.get("bucket") for item in effective
        if item.get("status") in {"AUTO_ACCEPTED", "VERIFIED"} and item.get("bucket")
    )
    return {
        "statuses": dict(sorted(statuses.items())),
        "buckets": {name: buckets.get(name, 0) for name in QUOTAS},
        "remaining": {name: max(0, target - buckets.get(name, 0)) for name, target in QUOTAS.items()},
        "complete": all(buckets.get(name, 0) >= target for name, target in QUOTAS.items()),
    }


def write_views(work_dir: Path, latest: dict[str, dict[str, Any]]) -> dict[str, Any]:
    work_dir.mkdir(parents=True, exist_ok=True)
    reviews = load_reviews(work_dir)
    records = sorted(latest.values(), key=lambda item: (item.get("source_image", ""), item.get("image", "")))
    prelabels = work_dir / "prelabels.csv"
    with prelabels.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=STATE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    with (work_dir / "review_queue.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["image", "proposed_label", "reason", "source_image"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            if effective_record(record, reviews).get("status") == "NEEDS_REVIEW":
                writer.writerow({name: record.get(name, "") for name in fields})
    with (work_dir / "rejected.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["source_image", "reason", "details"])
        writer.writeheader()
        for record in records:
            item = effective_record(record, reviews)
            if item.get("status") == "REJECTED":
                writer.writerow({"source_image": item.get("source_image", ""), "reason": item.get("reason", ""),
                                 "details": item.get("details", "")})
    summary = summarize_records(records, reviews)
    (work_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _load_base_index(base_dir: Path, cache_path: Path) -> tuple[set[str], PHashIndex, Counter[str], set[str]]:
    metadata = json.loads((base_dir / "metadata.json").read_text(encoding="utf-8"))
    fingerprint = metadata.get("datasetSha256")
    cached: dict[str, Any] | None = None
    if cache_path.is_file():
        candidate = json.loads(cache_path.read_text(encoding="utf-8"))
        if candidate.get("datasetSha256") == fingerprint:
            cached = candidate
    if cached is None:
        with (base_dir / "manifest.csv").open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        items = []
        for row in rows:
            path = base_dir / row["image"]
            items.append({"sha256": row["sha256"], "phash": compute_phash(path),
                          "label": normalize_label(row["ground_truth"]), "image": row["image"],
                          "source_image": row.get("source_image", "")})
        cached = {"datasetSha256": fingerprint, "items": items}
        cache_path.write_text(json.dumps(cached), encoding="utf-8")
    hashes: set[str] = set()
    index = PHashIndex()
    counts: Counter[str] = Counter()
    sources: set[str] = set()
    for item in cached["items"]:
        hashes.add(item["sha256"])
        index.add(item["phash"], f"v2:{item['image']}")
        counts[item["label"]] += 1
        if item.get("source_image"):
            sources.add(item["source_image"].replace("\\", "/"))
    return hashes, index, counts, sources


def _create_ocr_engine(rec_model_dir: Path):
    from paddleocr import PaddleOCR
    return PaddleOCR(
        text_detection_model_name="PP-OCRv6_medium_det",
        text_recognition_model_name="PP-OCRv6_medium_rec",
        text_recognition_model_dir=str(rec_model_dir),
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device="cpu",
    )


def predict_local_label(engine, image: np.ndarray) -> str:
    candidates: list[tuple[float, str]] = []
    for result in engine.predict(image):
        data = result.json.get("res", {})
        for text, score in zip(data.get("rec_texts", []), data.get("rec_scores", [])):
            label = normalize_label(text)
            if label:
                candidates.append((float(score), label))
    return max(candidates, default=(0.0, ""))[1]


def _base_record(source: str, source_sha: str, source_phash: list[int]) -> dict[str, Any]:
    return {
        "record_key": f"source:{source_sha}", "source_image": source, "source_sha256": source_sha,
        "source_phash": source_phash, "model": MODEL, "prompt_version": PROMPT_VERSION,
        "transform_version": TRANSFORM_VERSION,
    }


def _finalize_candidates(candidates: list[dict[str, Any]], work_dir: Path, state_path: Path,
                         latest: dict[str, dict[str, Any]], label_counts: Counter[str],
                         bucket_counts: Counter[str], crop_index: PHashIndex, agy_exe: str) -> None:
    try:
        first = run_agy([(item["record_key"], work_dir / item["image"]) for item in candidates],
                        work_dir / "agy_staging", agy_exe)
    except Exception as exc:
        for item in candidates:
            record = {**item, "status": "PENDING", "reason": "AGY_PASS1_ERROR", "details": str(exc)}
            append_state(state_path, record)
            latest[record["record_key"]] = record
        return

    second_candidates = []
    for item in candidates:
        item["agy_pass1"] = first[item["record_key"]]
        status, _, _ = route_decision(item.get("local_v2", ""), item["agy_pass1"])
        if status == "SECOND_PASS":
            source = cv2.imread(str(work_dir / item["image"]))
            composite_path = work_dir / "agy_composites" / f"{item['crop_sha256']}.jpg"
            composite_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(composite_path), make_composite(source), [cv2.IMWRITE_JPEG_QUALITY, 95])
            second_candidates.append((item["record_key"], composite_path))

    second: dict[str, dict[str, Any]] = {}
    if second_candidates:
        try:
            second = run_agy(second_candidates, work_dir / "agy_staging", agy_exe)
        except Exception as exc:
            for item in candidates:
                if item["record_key"] in {key for key, _ in second_candidates}:
                    record = {**item, "status": "PENDING", "reason": "AGY_PASS2_ERROR", "details": str(exc)}
                    append_state(state_path, record)
                    latest[record["record_key"]] = record

    failed_second = {key for key, _ in second_candidates} - set(second)
    for item in candidates:
        if item["record_key"] in failed_second:
            continue
        pass2 = second.get(item["record_key"])
        status, label, reason = route_decision(item.get("local_v2", ""), item["agy_pass1"], pass2)
        record = {**item, "agy_pass2": pass2 or {}, "proposed_label": label, "status": status, "reason": reason}
        if status == "AUTO_ACCEPTED":
            bucket = label_bucket(label)
            record["bucket"] = bucket
            if label_counts[label] >= 3:
                record.update(status="REJECTED", reason="LABEL_VARIANT_LIMIT_REACHED")
            elif bucket_counts[bucket] >= QUOTAS[bucket]:
                record.update(status="REJECTED", reason="BUCKET_QUOTA_FILLED")
            else:
                label_counts[label] += 1
                bucket_counts[bucket] += 1
        elif status == "NEEDS_REVIEW":
            record["bucket"] = label_bucket(label) if valid_label(label) else ""
        append_state(state_path, record)
        latest[record["record_key"]] = record
        if record["status"] in {"AUTO_ACCEPTED", "NEEDS_REVIEW"}:
            crop_index.add(record["crop_phash"], record["record_key"])


def build_dataset(source_dir: Path, benchmark_dir: Path, work_dir: Path, detector_model: Path,
                  rec_model_dir: Path, base_dataset: Path, agy_exe: str = "agy", limit: int = 0) -> dict[str, Any]:
    profile_path = work_dir / "agy_profile.json"
    if not profile_path.is_file() or not json.loads(profile_path.read_text(encoding="utf-8")).get("passed"):
        raise RuntimeError(f"A passing AGY pilot is required before build: {profile_path}")
    batch_size = int(json.loads(profile_path.read_text(encoding="utf-8"))["selectedBatchSize"])
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "images").mkdir(exist_ok=True)
    state_path = work_dir / ".pipeline_state.jsonl"
    latest = load_latest_state(state_path)
    reviews = load_reviews(work_dir)
    effective = [effective_record(item, reviews) for item in latest.values()]
    bucket_counts = Counter(item.get("bucket") for item in effective
                            if item.get("status") in {"AUTO_ACCEPTED", "VERIFIED"})
    base_hashes, crop_index, label_counts, base_sources = _load_base_index(
        base_dataset, work_dir / ".v2_index.json"
    )
    for item in effective:
        if item.get("status") in {"AUTO_ACCEPTED", "VERIFIED"}:
            label_counts[normalize_label(item.get("proposed_label"))] += 1
        if item.get("crop_phash") and item.get("status") in {"AUTO_ACCEPTED", "VERIFIED", "NEEDS_REVIEW", "PENDING"}:
            crop_index.add(item["crop_phash"], item.get("record_key", ""))
    source_hashes = {item.get("source_sha256") for item in latest.values() if item.get("source_sha256")}
    benchmark_hashes, benchmark_phashes = load_benchmark_index(benchmark_dir)
    benchmark_index = PHashIndex()
    for name, hashes in benchmark_phashes:
        benchmark_index.add(hashes, name)

    detector = SealDetectorModel(detector_model)
    engine = _create_ocr_engine(rec_model_dir)
    files = sorted(
        (path for path in source_dir.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS),
        key=lambda path: path.relative_to(source_dir).as_posix(),
    )
    scanned = 0
    batch: list[dict[str, Any]] = []

    pending = [item for item in latest.values() if item.get("status") == "PENDING" and item.get("image")]
    for offset in range(0, len(pending), batch_size):
        _finalize_candidates([dict(item) for item in pending[offset:offset + batch_size]], work_dir, state_path,
                             latest, label_counts, bucket_counts, crop_index, agy_exe)

    for path in files:
        if all(bucket_counts[name] >= target for name, target in QUOTAS.items()):
            break
        if limit and scanned >= limit:
            break
        relative = path.relative_to(source_dir).as_posix()
        source_sha = compute_sha256(path)
        if source_sha in source_hashes:
            continue
        scanned += 1
        if relative in base_sources or path.name in base_sources:
            record = {**_base_record(relative, source_sha, []), "status": "REJECTED",
                      "reason": "V2_SOURCE_OVERLAP"}
            append_state(state_path, record); latest[record["record_key"]] = record
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
            append_state(state_path, record); latest[record["record_key"]] = record
            source_hashes.add(source_sha)
            continue
        base = _base_record(relative, source_sha, source_phash)
        if source_sha in benchmark_hashes:
            record = {**base, "status": "REJECTED", "reason": "BENCHMARK_EXACT_MATCH"}
        elif (near := benchmark_index.near(source_phash)) is not None:
            record = {**base, "status": "REJECTED", "reason": "BENCHMARK_NEAR_MATCH",
                      "details": f"{near[0]} distance={near[1]}"}
        else:
            detections = detector.detect(image)
            if not detections:
                record = {**base, "status": "REJECTED", "reason": "NO_SEAL_DETECTED"}
            else:
                seal_crop, _ = detector.crop_detections(image, detections)[0]
                result = extract_best_candidate(engine, seal_crop)
                crop = result.get("crop")
                if crop is None:
                    record = {**base, "status": "REJECTED", "reason": result.get("reason", "NO_LINE_CROP")}
                else:
                    quality_ok, quality_reason = evaluate_image_quality(crop)
                    if not quality_ok or "CLIPPED_CROP" in result.get("reason", ""):
                        record = {**base, "status": "REJECTED",
                                  "reason": quality_reason or "CLIPPED_CROP"}
                    else:
                        encoded_ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        if not encoded_ok:
                            raise ValueError("cv2.imencode failed")
                        crop_bytes = encoded.tobytes()
                        crop_sha = hashlib.sha256(crop_bytes).hexdigest()
                        crop_phash = compute_phash(crop)
                        duplicate = crop_sha in base_hashes or crop_index.near(crop_phash)
                        if duplicate:
                            record = {**base, "status": "REJECTED", "reason": "DUPLICATE_CROP",
                                      "details": str(duplicate)}
                        else:
                            crop_rel = f"images/{crop_sha[:20]}.jpg"
                            (work_dir / crop_rel).write_bytes(crop_bytes)
                            record = {
                                **base, "record_key": crop_sha, "image": crop_rel, "crop_sha256": crop_sha,
                                "crop_phash": crop_phash, "rotation": result.get("rotation", 0),
                                "local_v2": normalize_label(result.get("proposed_label")),
                            }
                            batch.append(record)
                            source_hashes.add(source_sha)
                            if len(batch) >= batch_size:
                                _finalize_candidates(batch, work_dir, state_path, latest, label_counts,
                                                     bucket_counts, crop_index, agy_exe)
                                batch = []
                                write_views(work_dir, latest)
                            continue
        append_state(state_path, record)
        latest[record["record_key"]] = record
        source_hashes.add(source_sha)
    if batch:
        _finalize_candidates(batch, work_dir, state_path, latest, label_counts, bucket_counts, crop_index, agy_exe)
    return write_views(work_dir, latest)


def run_pilot(base_dataset: Path, work_dir: Path, rec_model_dir: Path, agy_exe: str = "agy",
              smoke_count: int = 20, sample_count: int = 500, seed: int = SEED) -> dict[str, Any]:
    with (base_dataset / "manifest.csv").open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    rng = random.Random(seed)
    rng.shuffle(rows)
    rows = rows[:sample_count]
    engine = _create_ocr_engine(rec_model_dir)
    samples = []
    for row in rows:
        path = base_dataset / row["image"]
        image = cv2.imread(str(path))
        samples.append({"id": row["sha256"], "path": path, "truth": normalize_label(row["ground_truth"]),
                        "local": predict_local_label(engine, image)})

    def evaluate(items: list[dict[str, Any]], batch_size: int) -> dict[str, Any]:
        auto = correct = parsed = 0
        for offset in range(0, len(items), batch_size):
            part = items[offset:offset + batch_size]
            first = run_agy([(item["id"], item["path"]) for item in part], work_dir / "agy_staging", agy_exe)
            parsed += len(first)
            need_second = []
            for item in part:
                status, _, _ = route_decision(item["local"], first[item["id"]])
                if status == "SECOND_PASS":
                    image = cv2.imread(str(item["path"]))
                    composite = work_dir / "pilot_composites" / f"{item['id']}.jpg"
                    composite.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(composite), make_composite(image), [cv2.IMWRITE_JPEG_QUALITY, 95])
                    need_second.append((item["id"], composite))
            second = run_agy(need_second, work_dir / "agy_staging", agy_exe) if need_second else {}
            for item in part:
                status, label, _ = route_decision(item["local"], first[item["id"]], second.get(item["id"]))
                if status == "AUTO_ACCEPTED":
                    auto += 1
                    correct += int(label == item["truth"])
        return {
            "batchSize": batch_size, "samples": len(items), "parsed": parsed,
            "parseSuccess": parsed / len(items) if items else 0,
            "autoAccepted": auto, "autoAcceptRate": auto / len(items) if items else 0,
            "precision": correct / auto if auto else 0,
        }

    smoke = evaluate(samples[:smoke_count], 1)
    if smoke["parseSuccess"] != 1:
        raise RuntimeError(f"AGY image smoke failed: {smoke}")
    runs = []
    selected = None
    for batch_size in (8, 4, 1):
        run = evaluate(samples, batch_size)
        runs.append(run)
        if run["parseSuccess"] == 1 and run["precision"] >= 0.995 and run["autoAcceptRate"] >= 0.85:
            selected = run
            break
    try:
        version_result = subprocess.run([agy_exe, "--version"], capture_output=True, text=True, timeout=10)
        agy_version = (version_result.stdout or version_result.stderr).strip()
    except Exception:
        agy_version = "unknown"
    result = {"model": MODEL, "agyVersion": agy_version, "smoke": smoke, "runs": runs,
              "passed": selected is not None, "selectedBatchSize": selected["batchSize"] if selected else None}
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "pilot_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (work_dir / "agy_profile.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def run_transport_smoke(base_dataset: Path, work_dir: Path, agy_exe: str = "agy",
                        count: int = 20, batch_size: int = 4) -> dict[str, Any]:
    with (base_dataset / "manifest.csv").open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))[:count]
    parsed = []
    for offset in range(0, len(rows), batch_size):
        part = rows[offset:offset + batch_size]
        results = run_agy([(row["sha256"], base_dataset / row["image"]) for row in part],
                          work_dir / "agy_staging", agy_exe)
        parsed.extend(results.values())
    result = {
        "model": MODEL, "requested": len(rows), "parsed": len(parsed),
        "parseSuccess": len(parsed) == len(rows), "batchSize": batch_size,
        "readable": sum(bool(item.get("readable")) for item in parsed),
    }
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "pilot_smoke.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--work-dir", type=Path,
                        default=Path(os.getenv("V3_WORK_DIR", "data/recognition_dataset_v3_work")))
    common.add_argument("--agy-exe", default="agy")
    pilot = commands.add_parser("pilot", parents=[common])
    pilot.add_argument("--v2-dataset", type=Path, required=True)
    pilot.add_argument("--rec-model-dir", type=Path, default=Path("models/seal-ocr-rec-v1"))
    pilot.add_argument("--smoke-count", type=int, default=20)
    pilot.add_argument("--sample-count", type=int, default=500)
    pilot.add_argument("--transport-only", action="store_true")
    pilot.add_argument("--transport-batch-size", type=int, default=4)
    build = commands.add_parser("build", parents=[common])
    build.add_argument("--source-dir", type=Path, default=Path(os.getenv("SOURCE_DIR", "data/new_dataset")))
    build.add_argument("--benchmark-dir", type=Path, default=Path("benchmark"))
    build.add_argument("--v2-dataset", type=Path, required=True)
    build.add_argument("--detector-model", type=Path, default=Path("models/seal-detector-v1/best.onnx"))
    build.add_argument("--rec-model-dir", type=Path, default=Path("models/seal-ocr-rec-v2"))
    build.add_argument("--limit", type=int, default=0)
    status = commands.add_parser("status", parents=[common])
    args = parser.parse_args()
    if args.command == "pilot":
        result = (run_transport_smoke(args.v2_dataset, args.work_dir, args.agy_exe,
                                      args.smoke_count, args.transport_batch_size)
                  if args.transport_only else
                  run_pilot(args.v2_dataset, args.work_dir, args.rec_model_dir, args.agy_exe,
                            args.smoke_count, args.sample_count))
    elif args.command == "build":
        result = build_dataset(args.source_dir, args.benchmark_dir, args.work_dir, args.detector_model,
                               args.rec_model_dir, args.v2_dataset, args.agy_exe, args.limit)
    else:
        latest = load_latest_state(args.work_dir / ".pipeline_state.jsonl")
        result = write_views(args.work_dir, latest)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
