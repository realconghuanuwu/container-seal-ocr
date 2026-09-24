from io import BytesIO

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.main import HORIZONTAL_TTA_SCALES, create_app
from app.schemas import Candidate
from app.worker import WorkerBusy, WorkerTimeout


class FakeWorker:
    def __init__(self, settings):
        self.settings = settings
        self.candidates = [Candidate(text=" fx6i394i13 ", confidence=0.95)]
        self.exception = None
        self.is_ready = True
        self.startup_error = None

    def start(self):
        pass

    def stop(self):
        pass

    def ready(self):
        return self.is_ready

    def infer(self, image, deadline):
        if self.exception:
            raise self.exception
        return self.candidates

    def switch_model(self, model_version: str, recognition_model_dir: str | None = None):
        from dataclasses import replace
        self.settings = replace(
            self.settings,
            model_version=model_version,
            recognition_model_dir=recognition_model_dir,
        )


def image_bytes(format="PNG"):
    rng = np.random.default_rng(7)
    image = Image.fromarray(rng.integers(0, 256, (300, 600, 3), dtype=np.uint8))
    output = BytesIO()
    image.save(output, format=format)
    return output.getvalue()


def upload(client, data=None, mime="image/png"):
    return client.post("/api/v1/ocr/seal", files={"file": ("seal.png", image_bytes() if data is None else data, mime)})


def test_valid_image_and_readiness(monkeypatch):
    monkeypatch.setattr("app.main.barcode.decode", lambda image: [])
    with TestClient(create_app(worker_factory=FakeWorker)) as client:
        assert client.get("/health/live").status_code == 200
        test_page = client.get("/test")
        assert test_page.status_code == 200
        assert "Đọc số seal từ ảnh" in test_page.text
        assert test_page.headers["content-type"].startswith("text/html")
        assert client.get("/health/ready").status_code == 200
        result = upload(client).json()
        assert result["status"] == "SUCCESS"
        assert result["sealNumber"] in ("FX6I394I13", "FX61394113")
        assert result["rawText"] == " fx6i394i13 "
        assert result["source"] == "OCR"
        assert result["modelVersion"] in ("pp-ocrv6-medium-base", client.app.state.worker.settings.model_version)
        client.app.state.worker.is_ready = False
        assert client.get("/health/ready").status_code == 503


def test_upload_errors():
    with TestClient(create_app(worker_factory=FakeWorker)) as client:
        assert upload(client, mime="text/plain").status_code == 415
        assert upload(client, data=b"").json()["reason"] == "EMPTY_UPLOAD"
        assert upload(client, data=b"broken").json()["reason"] == "CORRUPTED_IMAGE"
        assert upload(client, data=image_bytes("JPEG"), mime="image/png").status_code == 415
        assert upload(client, data=image_bytes(), mime="image/png").status_code == 200


def test_size_and_quality():
    settings = Settings(max_upload_mb=1, min_side=400)
    with TestClient(create_app(settings, FakeWorker)) as client:
        assert upload(client).json()["reason"] == "IMAGE_TOO_SMALL"
        assert upload(client, data=b"x" * (1024 * 1024 + 1)).status_code == 413


def test_barcode_and_fallback(monkeypatch):
    with TestClient(create_app(worker_factory=FakeWorker)) as client:
        monkeypatch.setattr("app.main.barcode.decode", lambda image: ["FX12345678"])
        result = upload(client).json()
        assert result["source"] == "BARCODE" and result["confidence"] is None
        monkeypatch.setattr("app.main.barcode.decode", lambda image: ["invalid-"])
        assert upload(client).json()["source"] == "OCR"


def test_detector_crops_are_used(monkeypatch):
    class FakeDetector:
        def detect(self, image):
            return [
                {"x": 0, "y": 0, "width": 100, "height": 100, "confidence": 0.9},
                {"x": 0, "y": 0, "width": 120, "height": 120, "confidence": 0.8},
            ]

        def crop(self, image, detections):
            return [image[:item["height"], :item["width"]] for item in detections]

    decoded_shapes = []
    monkeypatch.setattr("app.main.barcode.decode", lambda image: decoded_shapes.append(image.shape) or [])
    with TestClient(create_app(worker_factory=FakeWorker,
                               detector_factory=lambda settings: FakeDetector())) as client:
        result = upload(client).json()
        assert result["status"] == "SUCCESS"
        assert result["imageWidth"] == 600
        assert result["imageHeight"] == 300
        assert result["detections"][0] == {
            "x": 0, "y": 0, "width": 100, "height": 100, "confidence": 0.9,
        }
    assert decoded_shapes[:2] == [(100, 100, 3), (120, 120, 3)]


def test_horizontal_tta_runs_only_for_registered_one_character_shortfall(monkeypatch):
    class SequenceWorker(FakeWorker):
        def __init__(self, settings):
            super().__init__(settings)
            self.responses = []
            self.shapes = []

        def infer(self, image, deadline):
            self.shapes.append(image.shape)
            return self.responses.pop(0)

    monkeypatch.setattr("app.main.barcode.decode", lambda image: [])
    with TestClient(create_app(worker_factory=SequenceWorker)) as client:
        worker = client.app.state.worker
        worker.responses = [
            [Candidate(text="FX3739524", confidence=0.99)],
            [Candidate(text="FX37395224", confidence=0.80)],
        ]
        result = upload(client).json()
        assert result["sealNumber"] == "FX37395224"
        assert worker.shapes == [(300, 600, 3), (300, int(600 * HORIZONTAL_TTA_SCALES[0]), 3)]

    with TestClient(create_app(worker_factory=SequenceWorker)) as client:
        worker = client.app.state.worker
        worker.responses = [
            [Candidate(text="FX3739524", confidence=0.99)],
            [Candidate(text="FX3739524", confidence=0.99)],
            [Candidate(text="FX37395224", confidence=0.80)],
        ]
        result = upload(client).json()
        assert result["sealNumber"] == "FX37395224"
        assert worker.shapes == [
            (300, 600, 3),
            (300, int(600 * HORIZONTAL_TTA_SCALES[0]), 3),
            (300, int(600 * HORIZONTAL_TTA_SCALES[1]), 3),
        ]

    with TestClient(create_app(worker_factory=SequenceWorker)) as client:
        worker = client.app.state.worker
        worker.responses = [
            [Candidate(text="NO", confidence=0.99)],
            [Candidate(text="YMAR89429", confidence=0.95)],
            [Candidate(text="YMAR889429", confidence=0.90)],
        ]
        result = upload(client).json()
        assert result["sealNumber"] == "YMAR889429"
        assert worker.shapes == [
            (300, 600, 3),
            (600, 300, 3),
            (600, int(300 * HORIZONTAL_TTA_SCALES[0]), 3),
        ]

    for text in ("FX37395224", "XY12345"):
        with TestClient(create_app(worker_factory=SequenceWorker)) as client:
            worker = client.app.state.worker
            worker.responses = [[Candidate(text=text, confidence=0.95)]]
            result = upload(client).json()
            assert result["sealNumber"] == text
            assert worker.shapes == [(300, 600, 3)]


def test_busy_and_timeout(monkeypatch):
    monkeypatch.setattr("app.main.barcode.decode", lambda image: [])
    with TestClient(create_app(worker_factory=FakeWorker)) as client:
        client.app.state.worker.exception = WorkerBusy("busy")
        response = upload(client)
        assert response.status_code == 503 and response.json()["reason"] == "SERVICE_BUSY"
        client.app.state.worker.exception = WorkerTimeout("timeout")
        response = upload(client)
        assert response.status_code == 504 and response.json()["reason"] == "OCR_TIMEOUT"


def test_models_endpoints():
    with TestClient(create_app(worker_factory=FakeWorker)) as client:
        res = client.get("/api/v1/models")
        assert res.status_code == 200
        data = res.json()
        assert "active_model" in data
        assert any(m["id"] == "pp-ocrv6-medium-base" for m in data["models"])

        # Switch to base
        res_switch = client.post("/api/v1/models/switch?model_id=pp-ocrv6-medium-base")
        assert res_switch.status_code == 200
        assert res_switch.json()["status"] == "switching"
        assert client.app.state.worker.settings.model_version == "pp-ocrv6-medium-base"

        # Ready reflects new model
        res_ready = client.get("/health/ready")
        assert res_ready.status_code == 200
        assert res_ready.json()["modelVersion"] == "pp-ocrv6-medium-base"

        # Switch via json body
        res_switch_json = client.post("/api/v1/models/switch", json={"model_id": "pp-ocrv6-medium-base"})
        assert res_switch_json.status_code == 200

        # Switch to invalid model
        res_bad = client.post("/api/v1/models/switch?model_id=non-existent-model")
        assert res_bad.status_code == 400

        # Switch without model_id
        res_missing = client.post("/api/v1/models/switch")
        assert res_missing.status_code == 400


def test_feature_toggles(monkeypatch):
    class FakeDetector:
        def __init__(self):
            self.detect_called = False

        def detect(self, image):
            self.detect_called = True
            return [{"x": 0, "y": 0, "width": 100, "height": 100, "confidence": 0.9}]

        def crop(self, image, detections):
            return [image[:item["height"], :item["width"]] for item in detections]

    detector = FakeDetector()
    barcode_called = []
    monkeypatch.setattr("app.main.barcode.decode", lambda image: barcode_called.append(True) or ["FX12345678"])

    with TestClient(create_app(worker_factory=FakeWorker, detector_factory=lambda s: detector)) as client:
        # Default: detector and barcode enabled
        res = client.post("/api/v1/ocr/seal", files={"file": ("seal.png", image_bytes(), "image/png")})
        assert res.status_code == 200
        assert res.json()["source"] == "BARCODE"
        assert detector.detect_called is True
        assert len(barcode_called) > 0

        # Disable barcode: should fallback to OCR
        detector.detect_called = False
        barcode_called.clear()
        res_no_barcode = client.post(
            "/api/v1/ocr/seal?enable_barcode=false",
            files={"file": ("seal.png", image_bytes(), "image/png")},
        )
        assert res_no_barcode.status_code == 200
        assert res_no_barcode.json()["source"] == "OCR"
        assert len(barcode_called) == 0

        # Disable detector: detector.detect should NOT be called
        detector.detect_called = False
        res_no_detector = client.post(
            "/api/v1/ocr/seal?enable_detector=false",
            files={"file": ("seal.png", image_bytes(), "image/png")},
        )
        assert res_no_detector.status_code == 200
        assert detector.detect_called is False
        assert len(res_no_detector.json()["detections"]) == 0


def test_environment_settings_and_readiness():
    settings = Settings()
    with TestClient(create_app(settings, FakeWorker)) as client:
        ready_res = client.get("/health/ready").json()
        assert ready_res["status"] == "healthy"
        models_res = client.get("/api/v1/models").json()
        assert "models" in models_res
        assert len(models_res["models"]) >= 1


def test_benchmark_endpoints():
    with TestClient(create_app(worker_factory=FakeWorker)) as client:
        # Test benchmark data endpoint
        data_res = client.get("/api/v1/benchmark/data")
        assert data_res.status_code == 200
        data = data_res.json()
        assert "items" in data
        assert data["total"] > 0
        assert "exactMatches" in data
        assert "accuracyPercent" in data

        first_img = data["items"][0]["image"]
        # Test benchmark image endpoint
        img_res = client.get(f"/api/v1/benchmark/images/{first_img}")
        assert img_res.status_code == 200
        assert img_res.headers["content-type"].startswith("image/")

        # Test invalid image name traversal
        bad_res = client.get("/api/v1/benchmark/images/../../etc/passwd")
        assert bad_res.status_code in (404, 422)

