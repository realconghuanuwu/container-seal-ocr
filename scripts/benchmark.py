"""Freeze and benchmark the 300-image container-seal test set."""

import argparse
import csv
import hashlib
import json
import re
import statistics
import time
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
from PIL import Image, UnidentifiedImageError

from app.config import Settings
from app.postprocess import normalize, valid


MIMES = {".jpg": ("image/jpeg", "JPEG"), ".jpeg": ("image/jpeg", "JPEG"),
         ".png": ("image/png", "PNG"), ".webp": ("image/webp", "WEBP")}
EXPECTED_MODEL = "pp-ocrv6-medium-base"
REQUIRED_MODEL_FILES = ("inference.json", "inference.pdiparams", "inference.yml")
PROMOTION_MIN_EXACT_MATCHES = 128
PROMOTION_MAX_CER = 0.4465
PROMOTION_MAX_P95_MS = 10000
FIELDS = ["image", "ground_truth", "prediction", "raw_text", "confidence", "source",
          "status", "reason", "http_status", "latency_ms", "processing_time_ms", "exact_match"]
COMPARISON_FIELDS = ["image", "ground_truth", "base_prediction", "candidate_prediction",
                     "base_exact_match", "candidate_exact_match", "outcome", "base_status",
                     "candidate_status", "base_confidence", "candidate_confidence",
                     "base_latency_ms", "candidate_latency_ms"]


def model_artifact_sha256(model_dir: Path) -> str:
    if not model_dir.is_dir():
        raise ValueError(f"Model directory does not exist: {model_dir}")
    digest = hashlib.sha256()
    for name in REQUIRED_MODEL_FILES:
        path = model_dir / name
        if not path.is_file():
            raise ValueError(f"Model artifact is missing {name}")
        digest.update(name.encode() + b"\0")
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"Model artifact does not exist: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_slug(model_version: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", model_version.lower()).strip("_")
    if not slug:
        raise ValueError("Model version cannot be converted to a report name")
    return slug


def report_stem(model_version: str) -> str:
    if model_version == EXPECTED_MODEL:
        return "baseline_ppocrv6_medium_base"
    return f"benchmark_{model_slug(model_version)}"


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, 1):
        current = [i]
        for j, char_b in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (char_a != char_b)))
        previous = current
    return previous[-1]


def dataset(root: Path, settings: Settings) -> dict:
    images = (root / "images").resolve()
    labels_path = root / "labels.csv"
    labels_bytes = labels_path.read_bytes()
    with labels_path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != ["image", "ground_truth"]:
            raise ValueError("labels.csv must have image,ground_truth columns")
        rows = list(reader)
    if len(rows) != 300:
        raise ValueError(f"Expected exactly 300 labels, found {len(rows)}")
    listed, hashes, entries = set(), set(), []
    for row in rows:
        name, truth = row.get("image"), row.get("ground_truth")
        if not name or not truth or None in row:
            raise ValueError("Empty or malformed labels.csv row")
        if "/" in name or "\\" in name or name in {".", ".."} or ":" in name:
            raise ValueError(f"Image must be a filename under images/: {name}")
        if name in listed:
            raise ValueError(f"Duplicate image reference: {name}")
        listed.add(name)
        if not valid(truth, settings):
            raise ValueError(f"Invalid ground truth for {name}: {truth}")
        path = (images / name).resolve()
        if not path.is_relative_to(images) or not path.is_file():
            raise ValueError(f"Missing or unsafe image: {name}")
        expected_format = MIMES.get(path.suffix.lower())
        if expected_format is None:
            raise ValueError(f"Unsupported image type: {name}")
        try:
            with Image.open(path) as image:
                if image.format != expected_format[1]:
                    raise ValueError(f"Image extension/content mismatch: {name}")
                image.load()
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise ValueError(f"Invalid image: {name}") from exc
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in hashes:
            raise ValueError(f"Duplicate image bytes: {name}")
        hashes.add(digest)
        entries.append({"image": name, "ground_truth": truth, "sha256": digest})
    on_disk = {path.name for path in images.iterdir() if path.is_file()}
    if listed != on_disk:
        raise ValueError(f"Image inventory differs: unlisted={sorted(on_disk - listed)}, missing={sorted(listed - on_disk)}")
    return {"labelsSha256": hashlib.sha256(labels_bytes).hexdigest(),
            "images": sorted(entries, key=lambda entry: entry["image"])}


def freeze(root: Path, settings: Settings) -> Path:
    current = dataset(root, settings)
    path = root / "manifest.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != current:
            raise ValueError("Existing manifest differs from dataset; refusing to re-freeze")
        return path
    path.write_text(json.dumps(current, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def checked_dataset(root: Path, settings: Settings) -> tuple[list[dict], str]:
    current = dataset(root, settings)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Missing manifest.json; run --freeze after human review")
    manifest_bytes = manifest_path.read_bytes()
    if json.loads(manifest_bytes) != current:
        raise ValueError("Benchmark images or labels changed since freeze")
    return current["images"], hashlib.sha256(manifest_bytes).hexdigest()


def wait_ready(client: httpx.Client, url: str, expected_model: str = EXPECTED_MODEL,
               timeout_seconds: float = 180) -> None:
    deadline = time.monotonic() + timeout_seconds
    health_url = url.rstrip("/").rsplit("/api/v1/ocr/seal", 1)[0] + "/health/ready"
    while time.monotonic() < deadline:
        try:
            response = client.get(health_url)
            if response.status_code == 200:
                version = response.json().get("modelVersion")
                if version != expected_model:
                    raise ValueError(f"Service model version mismatch: expected {expected_model}, got {version}")
                return
        except httpx.HTTPError:
            pass
        time.sleep(1)
    raise TimeoutError("OCR service did not become ready within 180 seconds")


def run(root: Path, url: str, settings: Settings, expected_model: str = EXPECTED_MODEL,
        artifact_sha256: str | None = None,
        detector_artifact_sha256: str | None = None) -> tuple[dict, list[dict]]:
    entries, manifest_hash = checked_dataset(root, settings)
    rows = []
    with httpx.Client(timeout=15) as client:
        wait_ready(client, url, expected_model)
        for entry in entries:
            name, truth = entry["image"], entry["ground_truth"]
            started = time.monotonic()
            result, status_code = {}, None
            try:
                path = root / "images" / name
                with path.open("rb") as source:
                    response = client.post(url, files={"file": (name, source, MIMES[path.suffix.lower()][0])})
                status_code = response.status_code
                result = response.json()
                if not isinstance(result, dict):
                    result = {}
            except (httpx.HTTPError, ValueError):
                pass
            elapsed = round((time.monotonic() - started) * 1000, 1)
            version = result.get("modelVersion")
            if result and version != expected_model:
                raise ValueError(f"Model version mismatch for {name}: expected {expected_model}, got {version}")
            status = result.get("status") or "ERROR"
            failed = status == "ERROR" or status_code is None or status_code >= 400
            prediction = normalize(result.get("sealNumber") or "")
            rows.append({"image": name, "ground_truth": truth, "prediction": prediction,
                         "raw_text": result.get("rawText"), "confidence": result.get("confidence"),
                         "source": result.get("source"), "status": status, "reason": result.get("reason"),
                         "http_status": status_code, "latency_ms": elapsed,
                         "processing_time_ms": result.get("processingTimeMs"),
                         "exact_match": not failed and prediction == normalize(truth)})
    timings = [row["latency_ms"] for row in rows]
    exact = sum(row["exact_match"] for row in rows)
    failures = sum(row["status"] == "ERROR" or row["http_status"] is None or row["http_status"] >= 400 for row in rows)
    report = {
        "modelVersion": expected_model,
        "modelArtifactSha256": artifact_sha256,
        "sealDetectorArtifactSha256": detector_artifact_sha256,
        "runAtUtc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "manifestSha256": manifest_hash,
        "config": {"successThreshold": settings.success_threshold,
                   "reviewThreshold": settings.review_threshold,
                   "sealAllowedPattern": settings.seal_allowed_pattern,
                   "sealMinLength": settings.seal_min_length,
                   "sealMaxLength": settings.seal_max_length,
                   "ocrTimeoutSeconds": settings.ocr_timeout_seconds,
                   "maxUploadMb": settings.max_upload_mb,
                   "maxPixels": settings.max_pixels,
                   "maxSide": settings.max_side,
                   "minSide": settings.min_side,
                   "blurThreshold": settings.blur_threshold,
                   "darkThreshold": settings.dark_threshold,
                   "brightThreshold": settings.bright_threshold,
                   "detectionModelDir": settings.detection_model_dir,
                   "recognitionModelDir": settings.recognition_model_dir,
                   "sealDetectorModel": settings.seal_detector_model,
                   "sealDetectorInputSize": settings.seal_detector_input_size,
                   "sealDetectorConfidence": settings.seal_detector_confidence,
                   "sealDetectorIou": settings.seal_detector_iou,
                   "sealDetectorPadding": settings.seal_detector_padding,
                   "sealDetectorMaxRegions": settings.seal_detector_max_regions},
        "samples": len(rows), "exactMatches": exact,
        "exactMatchAccuracy": round(exact / len(rows), 4),
        "cer": round(sum(distance(row["ground_truth"], row["prediction"]) for row in rows) /
                     sum(len(row["ground_truth"]) for row in rows), 4),
        "averageLatencyMs": round(statistics.mean(timings), 1),
        "p50LatencyMs": round(percentile(timings, 0.5), 1),
        "p95LatencyMs": round(percentile(timings, 0.95), 1),
        "p99LatencyMs": round(percentile(timings, 0.99), 1),
        "ocrFailureCount": failures,
        "reviewCount": sum(row["status"] == "REVIEW" for row in rows),
        "recaptureCount": sum(row["status"] == "RECAPTURE" for row in rows),
        "over10Seconds": sum(value >= 10000 for value in timings),
    }
    return report, rows


def save(report: dict, rows: list[dict], reports: Path) -> str:
    reports.mkdir(parents=True, exist_ok=True)
    stamp = datetime.fromisoformat(report["runAtUtc"]).strftime("%Y%m%dT%H%M%SZ")
    stem = report_stem(report["modelVersion"])
    if any((reports / f"{stem}_{stamp}.{ext}").exists() for ext in ("json", "csv", "md")):
        raise FileExistsError(f"Timestamped benchmark already exists: {stamp}")
    markdown = f"# {report['modelVersion']} — frozen 300-image benchmark\n\n"
    markdown += f"Run: {report['runAtUtc']}  \nManifest SHA-256: `{report['manifestSha256']}`\n\n"
    markdown += "| Metric | Value |\n|---|---:|\n"
    markdown += "".join(f"| {key} | {value} |\n" for key, value in report.items()
                        if key not in {"runAtUtc", "manifestSha256", "config"})
    for suffix in (f"_{stamp}", ""):
        base = reports / f"{stem}{suffix}"
        base.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        with base.with_suffix(".csv").open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        base.with_suffix(".md").write_text(markdown, encoding="utf-8")
    return stem


def _truth(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    raise ValueError(f"Invalid exact_match value: {value!r}")


def load_saved(stem: Path) -> tuple[dict, list[dict]]:
    if stem.suffix:
        stem = stem.with_suffix("")
    summary_path, rows_path = stem.with_suffix(".json"), stem.with_suffix(".csv")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with rows_path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    return summary, rows


def _indexed_rows(report: dict, rows: list[dict], label: str) -> dict[str, dict]:
    indexed = {}
    for row in rows:
        image = row.get("image")
        if not image or image in indexed:
            raise ValueError(f"{label} report has a missing or duplicate image: {image!r}")
        indexed[image] = row
    exact = sum(_truth(row.get("exact_match")) for row in rows)
    if report.get("samples") != len(rows) or report.get("exactMatches") != exact:
        raise ValueError(f"{label} summary is inconsistent with its CSV")
    return indexed


def compare(base_report: dict, base_rows: list[dict], candidate_report: dict,
            candidate_rows: list[dict]) -> tuple[dict, list[dict]]:
    if base_report.get("manifestSha256") != candidate_report.get("manifestSha256"):
        raise ValueError("Base and candidate manifest hashes differ")
    base = _indexed_rows(base_report, base_rows, "Base")
    candidate = _indexed_rows(candidate_report, candidate_rows, "Candidate")
    if set(base) != set(candidate):
        raise ValueError("Base and candidate image inventories differ")

    rows = []
    outcome_counts = {name: 0 for name in ("FIXED", "REGRESSED", "BOTH_CORRECT", "BOTH_WRONG")}
    for image in sorted(base):
        old, new = base[image], candidate[image]
        if old.get("ground_truth") != new.get("ground_truth"):
            raise ValueError(f"Ground truth differs for {image}")
        old_match, new_match = _truth(old.get("exact_match")), _truth(new.get("exact_match"))
        if old_match and new_match:
            outcome = "BOTH_CORRECT"
        elif old_match:
            outcome = "REGRESSED"
        elif new_match:
            outcome = "FIXED"
        else:
            outcome = "BOTH_WRONG"
        outcome_counts[outcome] += 1
        rows.append({"image": image, "ground_truth": old["ground_truth"],
                     "base_prediction": old.get("prediction"),
                     "candidate_prediction": new.get("prediction"),
                     "base_exact_match": old_match, "candidate_exact_match": new_match,
                     "outcome": outcome, "base_status": old.get("status"),
                     "candidate_status": new.get("status"),
                     "base_confidence": old.get("confidence"),
                     "candidate_confidence": new.get("confidence"),
                     "base_latency_ms": old.get("latency_ms"),
                     "candidate_latency_ms": new.get("latency_ms")})

    reasons = []
    checks = [
        (candidate_report.get("samples") == 300, "candidate must contain exactly 300 samples"),
        (candidate_report.get("exactMatches", 0) >= PROMOTION_MIN_EXACT_MATCHES,
         f"exact matches must be at least {PROMOTION_MIN_EXACT_MATCHES}"),
        (candidate_report.get("cer", float("inf")) <= PROMOTION_MAX_CER,
         f"CER must be at most {PROMOTION_MAX_CER}"),
        (candidate_report.get("p95LatencyMs", float("inf")) < PROMOTION_MAX_P95_MS,
         f"P95 latency must be below {PROMOTION_MAX_P95_MS} ms"),
        (candidate_report.get("ocrFailureCount") == 0, "OCR failure count must be zero"),
        (candidate_report.get("over10Seconds") == 0, "requests at or above 10 seconds must be zero"),
    ]
    reasons.extend(message for passed, message in checks if not passed)
    metrics = ("samples", "exactMatches", "exactMatchAccuracy", "cer", "averageLatencyMs",
               "p50LatencyMs", "p95LatencyMs", "p99LatencyMs", "ocrFailureCount",
               "reviewCount", "recaptureCount", "over10Seconds")
    deltas = {key: round(candidate_report[key] - base_report[key], 4) for key in metrics}
    comparison = {
        "runAtUtc": candidate_report["runAtUtc"],
        "manifestSha256": candidate_report["manifestSha256"],
        "baseModelVersion": base_report["modelVersion"],
        "candidateModelVersion": candidate_report["modelVersion"],
        "candidateModelArtifactSha256": candidate_report.get("modelArtifactSha256"),
        "candidateSealDetectorArtifactSha256": candidate_report.get("sealDetectorArtifactSha256"),
        "baseMetrics": {key: base_report[key] for key in metrics},
        "candidateMetrics": {key: candidate_report[key] for key in metrics},
        "delta": deltas,
        "outcomes": outcome_counts,
        "promotionCriteria": {"minimumExactMatches": PROMOTION_MIN_EXACT_MATCHES,
                              "maximumCer": PROMOTION_MAX_CER,
                              "maximumP95LatencyMsExclusive": PROMOTION_MAX_P95_MS,
                              "maximumOcrFailures": 0, "maximumOver10Seconds": 0},
        "verdict": "PASS" if not reasons else "REJECT",
        "reasons": reasons,
    }
    return comparison, rows


def save_comparison(comparison: dict, rows: list[dict], reports: Path) -> str:
    reports.mkdir(parents=True, exist_ok=True)
    stamp = datetime.fromisoformat(comparison["runAtUtc"]).strftime("%Y%m%dT%H%M%SZ")
    stem = f"comparison_base_vs_{model_slug(comparison['candidateModelVersion'])}"
    if any((reports / f"{stem}_{stamp}.{ext}").exists() for ext in ("json", "csv", "md")):
        raise FileExistsError(f"Timestamped comparison already exists: {stamp}")
    markdown = f"# Base vs {comparison['candidateModelVersion']}\n\n"
    markdown += f"Verdict: **{comparison['verdict']}**  \nManifest SHA-256: `{comparison['manifestSha256']}`\n\n"
    markdown += "| Metric | Base | Candidate | Delta |\n|---|---:|---:|---:|\n"
    markdown += "".join(f"| {key} | {comparison['baseMetrics'][key]} | "
                        f"{comparison['candidateMetrics'][key]} | {comparison['delta'][key]} |\n"
                        for key in comparison["baseMetrics"])
    markdown += "\n| Outcome | Count |\n|---|---:|\n"
    markdown += "".join(f"| {key} | {value} |\n" for key, value in comparison["outcomes"].items())
    if comparison["reasons"]:
        markdown += "\n## Rejection reasons\n\n" + "".join(f"- {reason}\n" for reason in comparison["reasons"])
    for suffix in (f"_{stamp}", ""):
        base = reports / f"{stem}{suffix}"
        base.with_suffix(".json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
        with base.with_suffix(".csv").open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=COMPARISON_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        base.with_suffix(".md").write_text(markdown, encoding="utf-8")
    return stem


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("benchmark"))
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    parser.add_argument("--url", default="http://localhost:8000/api/v1/ocr/seal")
    parser.add_argument("--expected-model", default=EXPECTED_MODEL)
    parser.add_argument("--model-dir", type=Path,
                        help="Exported inference model directory to fingerprint")
    parser.add_argument("--detector-model", type=Path,
                        help="Exported seal detector ONNX file to fingerprint")
    parser.add_argument("--compare-with", type=Path,
                        help="Baseline report stem, for example reports/baseline_ppocrv6_medium_base")
    parser.add_argument("--freeze", action="store_true", help="Create or verify the frozen dataset manifest")
    args = parser.parse_args()
    settings = Settings()
    if args.freeze:
        print(freeze(args.dataset, settings))
    else:
        if args.expected_model != EXPECTED_MODEL and args.model_dir is None:
            parser.error("--model-dir is required for a non-base model")
        artifact_hash = model_artifact_sha256(args.model_dir) if args.model_dir else None
        detector_hash = file_sha256(args.detector_model) if args.detector_model else None
        summary, per_image = run(args.dataset, args.url, settings, args.expected_model,
                                 artifact_hash, detector_hash)
        save(summary, per_image, args.reports)
        if args.compare_with:
            base_summary, base_rows = load_saved(args.compare_with)
            comparison, comparison_rows = compare(base_summary, base_rows, summary, per_image)
            save_comparison(comparison, comparison_rows, args.reports)
            print(json.dumps({"benchmark": summary, "comparison": comparison}, indent=2))
        else:
            print(json.dumps(summary, indent=2))
