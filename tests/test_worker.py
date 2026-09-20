import time
import types
import sys

import numpy as np
import pytest

from app.config import Settings
from app.worker import OcrWorker, WorkerTimeout, create_engine


class LiveProcess:
    def is_alive(self):
        return True


class TimedOutConnection:
    def __init__(self):
        self.messages = []

    def send(self, message):
        self.messages.append(message)

    def poll(self, timeout=0):
        return False


def test_worker_is_restarted_on_deadline(monkeypatch):
    worker = OcrWorker(Settings())
    worker._ready = True
    worker._process = LiveProcess()
    worker._conn = TimedOutConnection()
    restarted = []
    monkeypatch.setattr(worker, "_restart", lambda: restarted.append(True))
    with pytest.raises(WorkerTimeout):
        worker.infer(np.zeros((200, 200, 3), dtype=np.uint8), time.monotonic() + 0.01)
    assert restarted == [True]
    assert worker._conn.messages[0][0] == "predict"


def test_custom_recognizer_directory_is_forwarded_to_paddleocr(monkeypatch):
    captured = {}

    class PaddleOCR:
        def __init__(self, **options):
            captured.update(options)

    monkeypatch.setitem(sys.modules, "paddleocr", types.SimpleNamespace(PaddleOCR=PaddleOCR))
    create_engine(Settings(detection_model_dir="/models/det", recognition_model_dir="/models/seal-ocr-rec-v1",
                           model_version="seal-ocr-rec-v1"))
    assert captured["text_detection_model_dir"] == "/models/det"
    assert captured["text_recognition_model_dir"] == "/models/seal-ocr-rec-v1"
