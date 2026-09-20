import numpy as np
import pytest

from app.config import Settings
from app.seal_detector import SealDetector


class FakeNet:
    def setInput(self, blob):
        self.blob = blob

    def forward(self):
        return np.array([[
            [50, 51], [50, 50], [40, 40], [20, 20], [0.9, 0.8],
        ]], dtype=np.float32)


def test_detector_returns_nms_crop(tmp_path, monkeypatch):
    model = tmp_path / "best.onnx"
    model.write_bytes(b"model")
    fake = FakeNet()
    monkeypatch.setattr("app.seal_detector.cv2.dnn.readNetFromONNX", lambda path: fake)
    detector = SealDetector(Settings(
        seal_detector_model=str(model), seal_detector_input_size=100,
        seal_detector_confidence=0.5, seal_detector_padding=0,
    ))

    image = np.zeros((100, 200, 3), dtype=np.uint8)
    detections = detector.detect(image)
    crops = detector.crop(image, detections)

    assert detections == [{"x": 60, "y": 30, "width": 80, "height": 40,
                           "confidence": pytest.approx(0.9)}]
    assert len(crops) == 1
    assert crops[0].shape[:2] == (40, 80)
    assert fake.blob.shape == (1, 3, 100, 100)


def test_detector_rejects_missing_model(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        SealDetector(Settings(seal_detector_model=str(tmp_path / "missing.onnx")))
