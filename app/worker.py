from dataclasses import replace
import multiprocessing as mp
import threading
import time
from multiprocessing.connection import Connection

import numpy as np

from .config import Settings
from .schemas import Candidate


def create_engine(settings: Settings):
    from paddleocr import PaddleOCR

    options = dict(
        text_detection_model_name="PP-OCRv6_medium_det",
        text_recognition_model_name="PP-OCRv6_medium_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True,
        text_det_box_thresh=0.50,
        text_det_unclip_ratio=1.9,
        device="cpu",
    )
    if settings.detection_model_dir:
        options["text_detection_model_dir"] = settings.detection_model_dir
    if settings.recognition_model_dir:
        options["text_recognition_model_dir"] = settings.recognition_model_dir
    return PaddleOCR(**options)


def worker_main(conn: Connection, settings: Settings) -> None:
    try:
        engine = create_engine(settings)
        conn.send(("ready", None))
        while True:
            message, payload = conn.recv()
            if message == "stop":
                break
            if message != "predict":
                continue
            try:
                results = engine.predict(payload)
                candidates = []
                for result in results:
                    data = result.json["res"]
                    candidates.extend(
                        Candidate(text=text, confidence=float(score)).model_dump()
                        for text, score in zip(data["rec_texts"], data["rec_scores"])
                    )
                conn.send(("result", candidates))
            except Exception as exc:
                conn.send(("error", f"{type(exc).__name__}: {exc}"))
    except (EOFError, BrokenPipeError):
        pass
    except Exception as exc:
        try:
            conn.send(("startup_error", f"{type(exc).__name__}: {exc}"))
        except (EOFError, BrokenPipeError):
            pass
    finally:
        conn.close()


class WorkerError(RuntimeError):
    pass


class WorkerBusy(WorkerError):
    pass


class WorkerUnavailable(WorkerError):
    pass


class WorkerTimeout(WorkerError):
    pass


class OcrWorker:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._ctx = mp.get_context("spawn")
        self._lock = threading.Lock()
        self._conn: Connection | None = None
        self._process: mp.Process | None = None
        self._ready = False
        self._startup_error: str | None = None

    def start(self) -> None:
        parent, child = self._ctx.Pipe()
        self._conn = parent
        self._process = self._ctx.Process(target=worker_main, args=(child, self.settings), daemon=True)
        self._process.start()
        child.close()
        self._ready = False
        self._startup_error = None

    def _refresh(self) -> None:
        if self._ready or self._startup_error or self._conn is None:
            return
        if self._conn.poll():
            try:
                kind, payload = self._conn.recv()
            except EOFError:
                kind, payload = "startup_error", "Worker exited during startup"
            if kind == "ready":
                self._ready = True
            else:
                self._startup_error = str(payload)
        elif self._process is not None and not self._process.is_alive():
            self._startup_error = "Worker exited during startup"

    def ready(self) -> bool:
        if not self._lock.acquire(blocking=False):
            return self._ready and self._process is not None and self._process.is_alive()
        try:
            self._refresh()
            if self._ready and self._process is not None and not self._process.is_alive():
                self._restart()
            return self._ready and self._process is not None and self._process.is_alive()
        finally:
            self._lock.release()

    @property
    def startup_error(self) -> str | None:
        return self._startup_error

    def infer(self, image: np.ndarray, deadline: float) -> list[Candidate]:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise WorkerTimeout("OCR deadline elapsed")
        if not self._lock.acquire(blocking=True, timeout=remaining):
            raise WorkerBusy("Worker is busy")
        try:
            self._refresh()
            if self._ready and self._process is not None and not self._process.is_alive():
                self._restart()
            if not self._ready or self._process is None or not self._process.is_alive() or self._conn is None:
                raise WorkerUnavailable(self._startup_error or "Model is loading")
            if deadline <= time.monotonic():
                raise WorkerTimeout("OCR deadline elapsed")
            try:
                self._conn.send(("predict", image))
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not self._conn.poll(remaining):
                    self._restart()
                    raise WorkerTimeout("OCR deadline elapsed")
                kind, payload = self._conn.recv()
            except (EOFError, BrokenPipeError, OSError) as exc:
                self._restart()
                raise WorkerUnavailable("OCR worker exited") from exc
            if kind == "error":
                raise WorkerError(str(payload))
            if kind != "result":
                raise WorkerUnavailable("Unexpected worker response")
            return [Candidate.model_validate(item) for item in payload]
        finally:
            self._lock.release()

    def _restart(self) -> None:
        self.stop()
        self.start()

    def switch_model(self, model_version: str, recognition_model_dir: str | None = None) -> None:
        with self._lock:
            self.stop()
            self.settings = replace(
                self.settings,
                model_version=model_version,
                recognition_model_dir=recognition_model_dir,
            )
            self.start()

    def stop(self) -> None:
        if self._process is not None and self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=1)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=1)
        if self._conn is not None:
            self._conn.close()
        self._process = None
        self._conn = None
        self._ready = False
