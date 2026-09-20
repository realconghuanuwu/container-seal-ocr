import csv
import json
from pathlib import Path

import httpx
import pytest
from PIL import Image

from app.config import Settings
from scripts import benchmark


@pytest.fixture
def frozen(tmp_path):
    root = tmp_path / "benchmark"
    images = root / "images"
    images.mkdir(parents=True)
    with (root / "labels.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["image", "ground_truth"])
        for index in range(300):
            name = f"seal_{index:03}.png"
            Image.new("RGB", (4, 4), (index % 256, index // 256, 0)).save(images / name)
            writer.writerow([name, "FX12345678"])
    benchmark.freeze(root, Settings())
    return root


def benchmark_result(model, exact_indexes, *, manifest="manifest", cer=0.4,
                     p95=1000.0, failures=0, over10=0):
    rows = []
    for index in range(300):
        exact = index in exact_indexes
        rows.append({"image": f"seal_{index:03}.png", "ground_truth": "FX12345678",
                     "prediction": "FX12345678" if exact else "FX12345679",
                     "raw_text": None, "confidence": 0.9, "source": "OCR",
                     "status": "SUCCESS", "reason": None, "http_status": 200,
                     "latency_ms": 100.0, "processing_time_ms": 90,
                     "exact_match": exact})
    matches = len(exact_indexes)
    report = {"modelVersion": model, "modelArtifactSha256": "a" * 64,
              "runAtUtc": "2026-09-18T00:00:00+00:00", "manifestSha256": manifest,
              "config": {}, "samples": 300, "exactMatches": matches,
              "exactMatchAccuracy": round(matches / 300, 4), "cer": cer,
              "averageLatencyMs": 100.0, "p50LatencyMs": 100.0,
              "p95LatencyMs": p95, "p99LatencyMs": p95,
              "ocrFailureCount": failures, "reviewCount": 0,
              "recaptureCount": 0, "over10Seconds": over10}
    return report, rows


def test_frozen_dataset_integrity(frozen):
    rows, digest = benchmark.checked_dataset(frozen, Settings())
    assert len(rows) == 300 and len(digest) == 64
    labels = frozen / "labels.csv"
    original = labels.read_bytes()
    for replacement in ("../seal_000.png", "seal_001.png", "seal_999.png"):
        labels.write_bytes(original.replace(b"seal_000.png", replacement.encode(), 1))
        with pytest.raises(ValueError):
            benchmark.dataset(frozen, Settings())
    labels.write_bytes(original.replace(b"FX12345678", b"INVALID-", 1))
    with pytest.raises(ValueError):
        benchmark.dataset(frozen, Settings())
    labels.write_bytes(original.replace(b"FX12345678", b"FX12345679", 1))
    with pytest.raises(ValueError, match="changed since freeze"):
        benchmark.checked_dataset(frozen, Settings())
    labels.write_bytes(original)
    image = frozen / "images" / "seal_000.png"
    image.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="Invalid image"):
        benchmark.dataset(frozen, Settings())
    Image.new("RGB", (4, 4), (99, 99, 99)).save(image)
    with pytest.raises(ValueError, match="changed since freeze"):
        benchmark.checked_dataset(frozen, Settings())
    with pytest.raises(ValueError, match="refusing to re-freeze"):
        benchmark.freeze(frozen, Settings())


def test_metrics_and_reports(frozen, tmp_path, monkeypatch):
    class Response:
        status_code = 200

        def __init__(self, payload, status_code=200):
            self.payload = payload
            self.status_code = status_code

        def json(self):
            return self.payload

    class Client:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get(self, url):
            return Response({"modelVersion": benchmark.EXPECTED_MODEL})

        def post(self, url, files):
            name = files["file"][0]
            index = int(Path(name).stem.split("_")[1])
            if index == 5:
                raise httpx.ConnectError("offline")
            status = "REVIEW" if index == 1 else "RECAPTURE" if index == 2 else "ERROR" if index == 3 else "SUCCESS"
            prediction = "FX12345679" if index == 1 else None if index == 3 else "fx12345678"
            return Response({"modelVersion": benchmark.EXPECTED_MODEL, "status": status,
                             "sealNumber": prediction, "rawText": prediction,
                             "confidence": None if index == 4 else 0.9,
                             "source": "BARCODE" if index == 4 else "OCR",
                             "processingTimeMs": 12}, 500 if index == 3 else 200)

    monkeypatch.setattr(benchmark.httpx, "Client", Client)
    report, rows = benchmark.run(frozen, "http://localhost:8000/api/v1/ocr/seal", Settings())
    assert len(rows) == report["samples"] == 300
    assert report["exactMatches"] == 297
    assert report["ocrFailureCount"] == 2
    assert report["reviewCount"] == report["recaptureCount"] == 1
    assert report["cer"] == round(21 / 3000, 4)
    assert rows[2]["exact_match"] is True
    assert rows[4]["source"] == "BARCODE" and rows[4]["confidence"] is None
    reports = tmp_path / "reports"
    benchmark.save(report, rows, reports)
    assert len(list(reports.iterdir())) == 6
    assert len(list(csv.DictReader((reports / "baseline_ppocrv6_medium_base.csv").open()))) == 300
    assert json.loads((reports / "baseline_ppocrv6_medium_base.json").read_text())["exactMatches"] == 297
    assert "297" in (reports / "baseline_ppocrv6_medium_base.md").read_text()
    with pytest.raises(FileExistsError):
        benchmark.save(report, rows, reports)


def test_distance_and_percentile():
    assert benchmark.distance("ABC", "AXC") == 1
    assert benchmark.distance("ABC", "") == 3
    assert benchmark.percentile([1, 2, 3, 4], 0.5) == 2.5


def test_wrong_model_is_rejected(frozen, monkeypatch):
    class Client:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get(self, url):
            return self

        @property
        def status_code(self):
            return 200

        def json(self):
            return {"modelVersion": "seal-ocr-rec-v1"}

    monkeypatch.setattr(benchmark.httpx, "Client", Client)
    with pytest.raises(ValueError, match="model version mismatch"):
        benchmark.run(frozen, "http://localhost:8000/api/v1/ocr/seal", Settings())


def test_model_artifact_hash_requires_and_hashes_all_inference_files(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    for index, name in enumerate(benchmark.REQUIRED_MODEL_FILES):
        (model / name).write_bytes(bytes([index]))
    digest = benchmark.model_artifact_sha256(model)
    assert len(digest) == 64
    assert len(benchmark.file_sha256(model / "inference.json")) == 64
    (model / "inference.yml").unlink()
    with pytest.raises(ValueError, match="missing inference.yml"):
        benchmark.model_artifact_sha256(model)
    with pytest.raises(ValueError, match="does not exist"):
        benchmark.file_sha256(model / "missing.onnx")


def test_candidate_reports_comparison_and_outcomes(tmp_path):
    base, base_rows = benchmark_result(benchmark.EXPECTED_MODEL, set(range(113)), cer=0.4465)
    candidate_exact = set(range(100)) | set(range(113, 141))
    candidate, candidate_rows = benchmark_result("seal-ocr-rec-v1", candidate_exact)
    comparison, rows = benchmark.compare(base, base_rows, candidate, candidate_rows)
    assert comparison["verdict"] == "PASS"
    assert comparison["outcomes"] == {"FIXED": 28, "REGRESSED": 13,
                                      "BOTH_CORRECT": 100, "BOTH_WRONG": 159}
    assert comparison["delta"]["exactMatches"] == 15
    reports = tmp_path / "reports"
    assert benchmark.save(candidate, candidate_rows, reports) == "benchmark_seal_ocr_rec_v1"
    assert benchmark.save_comparison(comparison, rows, reports) == "comparison_base_vs_seal_ocr_rec_v1"
    assert len(list(reports.iterdir())) == 12
    assert json.loads((reports / "comparison_base_vs_seal_ocr_rec_v1.json").read_text())["verdict"] == "PASS"
    assert len(list(csv.DictReader((reports / "comparison_base_vs_seal_ocr_rec_v1.csv").open()))) == 300


@pytest.mark.parametrize(
    ("matches", "cer", "p95", "failures", "over10", "verdict"),
    [(127, 0.4, 9999.9, 0, 0, "REJECT"),
     (128, 0.4466, 9999.9, 0, 0, "REJECT"),
     (128, 0.4, 10000.0, 0, 0, "REJECT"),
     (128, 0.4, 9999.9, 1, 0, "REJECT"),
     (128, 0.4, 9999.9, 0, 1, "REJECT"),
     (128, 0.4465, 9999.9, 0, 0, "PASS")],
)
def test_promotion_boundaries(matches, cer, p95, failures, over10, verdict):
    base, base_rows = benchmark_result(benchmark.EXPECTED_MODEL, set(range(113)), cer=0.4465)
    candidate, candidate_rows = benchmark_result(
        "seal-ocr-rec-v1", set(range(matches)), cer=cer, p95=p95,
        failures=failures, over10=over10)
    comparison, _ = benchmark.compare(base, base_rows, candidate, candidate_rows)
    assert comparison["verdict"] == verdict


def test_comparison_rejects_changed_benchmark():
    base, base_rows = benchmark_result(benchmark.EXPECTED_MODEL, set(range(113)))
    candidate, candidate_rows = benchmark_result("seal-ocr-rec-v1", set(range(128)))
    candidate["manifestSha256"] = "changed"
    with pytest.raises(ValueError, match="manifest hashes differ"):
        benchmark.compare(base, base_rows, candidate, candidate_rows)
    candidate["manifestSha256"] = base["manifestSha256"]
    candidate_rows.pop()
    candidate["samples"] = 299
    candidate["exactMatches"] = sum(row["exact_match"] for row in candidate_rows)
    with pytest.raises(ValueError, match="inventories differ"):
        benchmark.compare(base, base_rows, candidate, candidate_rows)


def test_response_model_version_is_rejected(frozen, monkeypatch):
    class Response:
        status_code = 200

        def __init__(self, payload):
            self.payload = payload

        def json(self):
            return self.payload

    class Client:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get(self, url):
            return Response({"modelVersion": "seal-ocr-rec-v1"})

        def post(self, url, files):
            return Response({"modelVersion": benchmark.EXPECTED_MODEL, "status": "SUCCESS"})

    monkeypatch.setattr(benchmark.httpx, "Client", Client)
    with pytest.raises(ValueError, match="Model version mismatch for"):
        benchmark.run(frozen, "http://localhost:8000/api/v1/ocr/seal", Settings(),
                      "seal-ocr-rec-v1")
