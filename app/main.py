import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable

import cv2
from fastapi import FastAPI, File, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from . import barcode
from .config import Settings
from .image import ImageError, prepare, quality_reason, resize
from .model_manager import list_available_models, resolve_model
from .postprocess import classify, clean_seal_number, normalize, valid
from .schemas import Candidate, Detection, OcrResponse
from .seal_detector import create_detector
from .seal_registry import get_expected_seal_length
from .worker import OcrWorker, WorkerBusy, WorkerError, WorkerTimeout, WorkerUnavailable


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("seal_ocr")
TEST_PAGE = Path(__file__).parent / "static" / "test.html"
HORIZONTAL_TTA_SCALES = (1.20, 1.25)


def is_potential_ctc_collapse(candidates: list[Candidate]) -> bool:
    for candidate in candidates:
        text = clean_seal_number(candidate.text)
        expected_length = get_expected_seal_length(text)
        if expected_length and len(text) == expected_length - 1:
            return True
    return False


def create_app(settings: Settings | None = None, worker_factory: Callable | None = None,
               detector_factory: Callable | None = None) -> FastAPI:
    settings = settings or Settings()
    worker_factory = worker_factory or OcrWorker
    detector_factory = detector_factory or create_detector

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.detector = detector_factory(settings)
        app.state.worker = worker_factory(settings)
        app.state.worker.start()
        try:
            yield
        finally:
            app.state.worker.stop()

    app = FastAPI(title="Container Seal OCR", lifespan=lifespan)

    @app.get("/test", include_in_schema=False)
    def test_page():
        return FileResponse(TEST_PAGE, media_type="text/html")

    @app.get("/health/live")
    def live():
        return {"status": "healthy"}

    @app.get("/health/ready")
    def ready(request: Request):
        worker = request.app.state.worker
        worker_settings = getattr(worker, "settings", settings)
        if worker.ready():
            return {
                "status": "healthy",
                "modelVersion": worker_settings.model_version,
                "environment": worker_settings.app_env,
            }
        return JSONResponse(
            {"status": "unhealthy", "reason": worker.startup_error or "MODEL_LOADING"},
            status_code=503,
        )

    @app.get("/api/v1/models")
    def get_models(request: Request):
        worker = request.app.state.worker
        worker_settings = getattr(worker, "settings", settings)
        active_version = worker_settings.model_version
        models = list_available_models(worker_settings, active_version=active_version)
        return {
            "environment": worker_settings.app_env,
            "active_model": active_version,
            "models": models,
        }

    @app.post("/api/v1/models/switch")
    async def switch_model_endpoint(request: Request, model_id: str | None = None):
        target_id = model_id
        if not target_id:
            try:
                body = await request.json()
                if isinstance(body, dict):
                    target_id = body.get("model_id")
            except Exception:
                pass
        if not target_id:
            return JSONResponse(
                {"status": "error", "message": "Missing 'model_id' in query or JSON body"},
                status_code=400,
            )
        worker = request.app.state.worker
        worker_settings = getattr(worker, "settings", settings)
        try:
            version, rec_dir = resolve_model(target_id, worker_settings)
        except ValueError as exc:
            return JSONResponse({"status": "error", "message": str(exc)}, status_code=400)

        if hasattr(worker, "switch_model"):
            worker.switch_model(model_version=version, recognition_model_dir=rec_dir)
        elif hasattr(worker, "settings"):
            from dataclasses import replace
            worker.settings = replace(
                worker.settings,
                model_version=version,
                recognition_model_dir=rec_dir,
            )

        return {
            "status": "switching",
            "target_model": version,
            "recognition_model_dir": rec_dir,
            "message": f"Switched model to {version}. Model is initializing...",
        }

    @app.post("/api/v1/ocr/seal", response_model=OcrResponse)
    async def seal(
        request: Request,
        file: UploadFile = File(...),
        enable_detector: bool = Query(True),
        enable_barcode: bool = Query(True),
        enable_tta: bool = Query(True),
    ):
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        started = time.monotonic()
        worker = request.app.state.worker
        worker_settings = getattr(worker, "settings", settings)
        active_version = worker_settings.model_version
        max_bytes = worker_settings.max_upload_mb * 1024 * 1024
        chunks = bytearray()
        while chunk := await file.read(min(1024 * 1024, max_bytes + 1 - len(chunks))):
            chunks.extend(chunk)
            if len(chunks) > max_bytes:
                break
        deadline = time.monotonic() + worker_settings.ocr_timeout_seconds
        barcode_attempted = False
        barcode_success = False
        candidate_count = 0
        detector_region_count = 0
        detections = []
        image_width = None
        image_height = None
        http_status = 200
        async def bounded(call, *args):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise WorkerTimeout("OCR deadline elapsed")
            try:
                return await asyncio.wait_for(asyncio.to_thread(call, *args), timeout=remaining)
            except asyncio.TimeoutError as exc:
                raise WorkerTimeout("OCR deadline elapsed") from exc

        try:
            if len(chunks) > max_bytes:
                raise ImageError("FILE_TOO_LARGE", 413)
            prepared = await bounded(prepare, bytes(chunks), file.content_type, worker_settings)
            reason = await bounded(quality_reason, prepared.normalized, worker_settings)
            if reason:
                response = OcrResponse(
                    status="RECAPTURE", reason=reason, processingTimeMs=0,
                    modelVersion=active_version,
                )
            else:
                image = await bounded(resize, prepared.normalized, worker_settings)
                image_height, image_width = image.shape[:2]
                detector = request.app.state.detector if enable_detector else None
                if detector:
                    detections = await bounded(detector.detect, image)
                    regions = detector.crop(image, detections)
                else:
                    regions = [image]
                detector_region_count = len(regions)
                regions = regions or [image]

                if enable_barcode:
                    barcode_attempted = True
                    codes = []
                    for region in regions:
                        codes.extend(await bounded(barcode.decode, region))
                    if not codes and len(regions) > 0 and regions[0] is not image:
                        codes.extend(await bounded(barcode.decode, image))
                    selected = next(
                        ((raw, clean_seal_number(raw)) for raw in codes if valid(clean_seal_number(raw), worker_settings)),
                        None,
                    )
                else:
                    selected = None

                if selected:
                    barcode_success = True
                    response = OcrResponse(
                        status="SUCCESS", sealNumber=selected[1], rawText=selected[0],
                        source="BARCODE", processingTimeMs=0, modelVersion=active_version,
                    )
                else:
                    zero_degree_candidates = []
                    for region in regions:
                        zero_degree_candidates.extend(
                            await bounded(worker.infer, region, deadline)
                        )
                    candidates = list(zero_degree_candidates)
                    active_regions = list(regions)

                    if enable_tta:
                        # Multi-Angle TTA: if 0° orientation yielded no valid candidate (e.g. vertical/hanging seals)
                        if not any(valid(clean_seal_number(c.text), worker_settings) and c.confidence >= worker_settings.review_threshold for c in candidates):
                            for rot_code in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
                                rot_candidates = []
                                rot_regions = [cv2.rotate(region, rot_code) for region in regions]
                                for rot_region in rot_regions:
                                    rot_c = await bounded(worker.infer, rot_region, deadline)
                                    rot_candidates.extend(rot_c)
                                candidates.extend(rot_candidates)
                                if any(valid(clean_seal_number(c.text), worker_settings) and c.confidence >= worker_settings.review_threshold for c in rot_candidates):
                                    active_regions = rot_regions
                                    break

                        if is_potential_ctc_collapse(candidates):
                            for scale in HORIZONTAL_TTA_SCALES:
                                scale_candidates = []
                                for region in active_regions:
                                    height, width = region.shape[:2]
                                    stretched = cv2.resize(
                                        region,
                                        (int(width * scale), height),
                                        interpolation=cv2.INTER_CUBIC,
                                    )
                                    scale_candidates.extend(
                                        await bounded(worker.infer, stretched, deadline)
                                    )
                                candidates.extend(scale_candidates)
                                if any(
                                    (expected := get_expected_seal_length(clean_seal_number(c.text)))
                                    and len(clean_seal_number(c.text)) == expected
                                    and c.confidence >= worker_settings.review_threshold
                                    for c in scale_candidates
                                ):
                                    break

                    candidate_count = len(candidates)
                    response = classify(candidates, worker_settings)
        except ImageError as exc:
            http_status = exc.http_status
            response = OcrResponse(
                status="ERROR", reason=exc.code, processingTimeMs=0,
                modelVersion=active_version,
            )
        except WorkerTimeout:
            http_status = 504
            response = OcrResponse(
                status="ERROR", reason="OCR_TIMEOUT", processingTimeMs=0,
                modelVersion=active_version,
            )
        except WorkerBusy:
            http_status = 503
            response = OcrResponse(
                status="ERROR", reason="SERVICE_BUSY", processingTimeMs=0,
                modelVersion=active_version,
            )
        except WorkerUnavailable:
            http_status = 503
            response = OcrResponse(
                status="ERROR", reason="MODEL_UNAVAILABLE", processingTimeMs=0,
                modelVersion=active_version,
            )
        except WorkerError:
            logger.exception("OCR worker failed")
            http_status = 500
            response = OcrResponse(
                status="ERROR", reason="OCR_FAILED", processingTimeMs=0,
                modelVersion=active_version,
            )
        except Exception:
            logger.exception("OCR request failed")
            http_status = 500
            response = OcrResponse(
                status="ERROR", reason="INTERNAL_ERROR", processingTimeMs=0,
                modelVersion=active_version,
            )
        response.processingTimeMs = round((time.monotonic() - started) * 1000)
        response.imageWidth = image_width
        response.imageHeight = image_height
        response.detections = [Detection(**item) for item in detections]
        logger.info(json.dumps({
            "requestId": request_id, "processingTimeMs": response.processingTimeMs,
            "modelVersion": active_version, "barcodeAttempted": barcode_attempted,
            "barcodeSuccess": barcode_success, "ocrCandidateCount": candidate_count,
            "sealDetectorRegionCount": detector_region_count,
            "status": response.status, "confidence": response.confidence,
            "errorCode": response.reason,
        }))
        return JSONResponse(response.model_dump(), status_code=http_status, headers={"X-Request-ID": request_id})

    return app


app = create_app()
