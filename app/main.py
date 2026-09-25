import asyncio
import csv
import json
import logging
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable

import cv2
from fastapi import FastAPI, File, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from . import barcode
from .config import Settings
from .image import ImageError, prepare, quality_reason, resize
from .model_manager import list_available_models, resolve_model
from .postprocess import classify, clean_seal_number, normalize, valid
from .schemas import Candidate, Detection, OcrResponse
from .seal_detector import create_detector
from .seal_registry import get_expected_seal_length, SORTED_CARRIER_PREFIXES
from .worker import OcrWorker, WorkerBusy, WorkerError, WorkerTimeout, WorkerUnavailable


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("seal_ocr")
TEST_PAGE = Path(__file__).parent / "static" / "test.html"
HORIZONTAL_TTA_SCALES = (1.20, 1.25)


def is_potential_ctc_collapse(candidates: list[Candidate]) -> bool:
    for candidate in candidates:
        text = clean_seal_number(candidate.text)
        expected_length = get_expected_seal_length(text)
        if expected_length and (expected_length - 2 <= len(text) < expected_length):
            return True
    return False


def stitch_multiline_candidates(candidates: list[Candidate], settings: Settings) -> list[Candidate]:
    stitched = list(candidates)
    prefix_cands = []
    digits_cands = []
    for c in candidates:
        t = normalize(c.text)
        if re.fullmatch(r"^[0-9]{5,8}$", t):
            digits_cands.append(c)
        else:
            for pref in SORTED_CARRIER_PREFIXES:
                if t == pref or t.endswith(pref):
                    prefix_cands.append((pref, c))
                    break
    for pref, pc in prefix_cands:
        for dc in digits_cands:
            combined = pref + normalize(dc.text)
            if valid(combined, settings):
                stitched.append(Candidate(text=combined, confidence=(pc.confidence + dc.confidence) / 2))
    return stitched


async def run_ocr_pipeline(
    image_data: bytes,
    content_type: str | None,
    worker_settings: Settings,
    worker,
    detector,
    enable_detector: bool = True,
    enable_barcode: bool = True,
    enable_tta: bool = True,
    deadline: float | None = None,
) -> tuple[OcrResponse, int, dict]:
    started = time.monotonic()
    active_version = worker_settings.model_version
    max_bytes = worker_settings.max_upload_mb * 1024 * 1024
    effective_deadline = deadline or (time.monotonic() + worker_settings.ocr_timeout_seconds)
    barcode_attempted = False
    barcode_success = False
    candidate_count = 0
    detector_region_count = 0
    detections = []
    image_width = None
    image_height = None
    http_status = 200

    async def bounded(call, *args):
        remaining = effective_deadline - time.monotonic()
        if remaining <= 0:
            raise WorkerTimeout("OCR deadline elapsed")
        try:
            return await asyncio.wait_for(asyncio.to_thread(call, *args), timeout=remaining)
        except asyncio.TimeoutError as exc:
            raise WorkerTimeout("OCR deadline elapsed") from exc

    try:
        if len(image_data) > max_bytes:
            raise ImageError("FILE_TOO_LARGE", 413)
        prepared = await bounded(prepare, image_data, content_type or "image/jpeg", worker_settings)
        reason = await bounded(quality_reason, prepared.normalized, worker_settings)
        soft_blurry = (reason == "IMAGE_TOO_BLURRY")
        if reason and not soft_blurry:
            response = OcrResponse(
                status="RECAPTURE", reason=reason, processingTimeMs=0,
                modelVersion=active_version,
            )
        else:
            image = await bounded(resize, prepared.normalized, worker_settings)
            image_height, image_width = image.shape[:2]
            if enable_detector and detector:
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
                        await bounded(worker.infer, region, effective_deadline)
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
                                rot_c = await bounded(worker.infer, rot_region, effective_deadline)
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
                                    await bounded(worker.infer, stretched, effective_deadline)
                                )
                            candidates.extend(scale_candidates)
                            if any(
                                (expected := get_expected_seal_length(clean_seal_number(c.text)))
                                and len(clean_seal_number(c.text)) == expected
                                and c.confidence >= worker_settings.review_threshold
                                for c in scale_candidates
                            ):
                                break

                # Dual-Crop Fallback: if cropped regions didn't yield a high-confidence full seal
                valid_cands = [c for c in candidates if valid(clean_seal_number(c.text), worker_settings)]
                best_cand = max(valid_cands, key=lambda c: c.confidence, default=None) if valid_cands else None
                expected_len = get_expected_seal_length(clean_seal_number(best_cand.text)) if best_cand else None
                needs_fallback = (
                    not best_cand
                    or best_cand.confidence < 0.92
                    or (expected_len and len(clean_seal_number(best_cand.text)) != expected_len)
                    or (len(clean_seal_number(best_cand.text)) <= 7 and not (expected_len and len(clean_seal_number(best_cand.text)) == expected_len))
                )
                if enable_detector and detector and needs_fallback and bool(detections):
                    full_cands = await bounded(worker.infer, image, effective_deadline)
                    candidates.extend(full_cands)

                # Prefix-Serial Stitching: for two-tier / multi-line stamped seals
                candidates = stitch_multiline_candidates(candidates, worker_settings)

                candidate_count = len(candidates)
                response = classify(candidates, worker_settings)
                if soft_blurry and (response.status != "SUCCESS" or not response.sealNumber):
                    response = OcrResponse(
                        status="RECAPTURE", reason="IMAGE_TOO_BLURRY", processingTimeMs=0,
                        modelVersion=active_version,
                    )
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
    log_meta = {
        "processingTimeMs": response.processingTimeMs,
        "modelVersion": active_version,
        "barcodeAttempted": barcode_attempted,
        "barcodeSuccess": barcode_success,
        "ocrCandidateCount": candidate_count,
        "sealDetectorRegionCount": detector_region_count,
        "status": response.status,
        "confidence": response.confidence,
        "errorCode": response.reason,
    }
    return response, http_status, log_meta


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

    @app.get("/", include_in_schema=False)
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

    @app.get("/api/v1/benchmark/images/{filename}", include_in_schema=False)
    def get_benchmark_image(filename: str):
        root = Path(__file__).resolve().parent.parent / "benchmark" / "images"
        safe_name = Path(filename).name
        target = (root / safe_name).resolve()
        if not target.is_relative_to(root.resolve()) or not target.is_file():
            return JSONResponse({"status": "error", "message": "Image not found"}, status_code=404)
        return FileResponse(target, media_type="image/jpeg")

    @app.get("/api/v1/benchmark/data")
    def get_benchmark_data(request: Request, model_id: str | None = None):
        root = Path(__file__).resolve().parent.parent
        worker = request.app.state.worker
        worker_settings = getattr(worker, "settings", settings)
        active_version = model_id or worker_settings.model_version

        is_base = "base" in (active_version or "").lower()
        items = []

        if is_base:
            comp_csv = root / "reports" / "comparison_base_vs_seal_ocr_det_v1_rec_v3.csv"
            if comp_csv.is_file():
                with comp_csv.open(newline="", encoding="utf-8-sig") as f:
                    for row in csv.DictReader(f):
                        exact = row.get("base_exact_match", "").lower() == "true"
                        items.append({
                            "image": row["image"],
                            "ground_truth": row["ground_truth"],
                            "prediction": row.get("base_prediction", ""),
                            "exact_match": exact,
                            "confidence": float(row["base_confidence"]) if row.get("base_confidence") else None,
                            "status": row.get("base_status", ""),
                            "latency_ms": float(row["base_latency_ms"]) if row.get("base_latency_ms") else None,
                            "image_url": f"/api/v1/benchmark/images/{row['image']}",
                        })

        if not items:
            # Production v1 baseline
            v1_csv = root / "baselines" / "seal-ocr-det-v1-rec-v1.csv"
            if v1_csv.is_file():
                with v1_csv.open(newline="", encoding="utf-8-sig") as f:
                    for row in csv.DictReader(f):
                        exact = row.get("exact_match", "").lower() == "true"
                        items.append({
                            "image": row["image"],
                            "ground_truth": row["ground_truth"],
                            "prediction": row.get("prediction", ""),
                            "exact_match": exact,
                            "confidence": float(row["confidence"]) if row.get("confidence") else None,
                            "status": row.get("status", ""),
                            "latency_ms": float(row["latency_ms"]) if row.get("latency_ms") else None,
                            "image_url": f"/api/v1/benchmark/images/{row['image']}",
                        })

        if not items:
            # Fallback to labels.csv
            lbl_csv = root / "benchmark" / "labels.csv"
            if lbl_csv.is_file():
                with lbl_csv.open(newline="", encoding="utf-8-sig") as f:
                    for row in csv.DictReader(f):
                        items.append({
                            "image": row["image"],
                            "ground_truth": row["ground_truth"],
                            "prediction": "",
                            "exact_match": False,
                            "confidence": None,
                            "status": "",
                            "latency_ms": None,
                            "image_url": f"/api/v1/benchmark/images/{row['image']}",
                        })

        total = len(items)
        exact_matches = sum(1 for item in items if item["exact_match"])
        accuracy = round((exact_matches / total) * 100, 2) if total else 0.0

        return {
            "modelVersion": active_version,
            "total": total,
            "exactMatches": exact_matches,
            "mismatches": total - exact_matches,
            "accuracyPercent": accuracy,
            "items": items,
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
        worker = request.app.state.worker
        worker_settings = getattr(worker, "settings", settings)
        max_bytes = worker_settings.max_upload_mb * 1024 * 1024
        chunks = bytearray()
        while chunk := await file.read(min(1024 * 1024, max_bytes + 1 - len(chunks))):
            chunks.extend(chunk)
            if len(chunks) > max_bytes:
                break

        detector = request.app.state.detector if enable_detector else None
        response, http_status, log_meta = await run_ocr_pipeline(
            image_data=bytes(chunks),
            content_type=file.content_type,
            worker_settings=worker_settings,
            worker=worker,
            detector=detector,
            enable_detector=enable_detector,
            enable_barcode=enable_barcode,
            enable_tta=enable_tta,
        )
        log_meta["requestId"] = request_id
        logger.info(json.dumps(log_meta))
        return JSONResponse(response.model_dump(), status_code=http_status, headers={"X-Request-ID": request_id})

    @app.post("/api/v1/ocr/batch")
    async def batch_ocr(
        request: Request,
        files: list[UploadFile] = File(...),
        enable_detector: bool = Query(True),
        enable_barcode: bool = Query(True),
        enable_tta: bool = Query(True),
    ):
        worker = request.app.state.worker
        worker_settings = getattr(worker, "settings", settings)
        detector = request.app.state.detector if enable_detector else None
        max_bytes = worker_settings.max_upload_mb * 1024 * 1024

        results = []
        for file in files:
            chunks = bytearray()
            while chunk := await file.read(min(1024 * 1024, max_bytes + 1 - len(chunks))):
                chunks.extend(chunk)
                if len(chunks) > max_bytes:
                    break
            res, _, _ = await run_ocr_pipeline(
                image_data=bytes(chunks),
                content_type=file.content_type,
                worker_settings=worker_settings,
                worker=worker,
                detector=detector,
                enable_detector=enable_detector,
                enable_barcode=enable_barcode,
                enable_tta=enable_tta,
            )
            results.append({"filename": file.filename, "result": res.model_dump()})
        return {"total": len(results), "items": results}

    @app.get("/api/v1/benchmark/stream")
    async def stream_benchmark(
        request: Request,
        enable_detector: bool = Query(True),
        enable_barcode: bool = Query(True),
        enable_tta: bool = Query(True),
        limit: int | None = Query(None),
    ):
        worker = request.app.state.worker
        worker_settings = getattr(worker, "settings", settings)
        detector = request.app.state.detector if enable_detector else None
        root = Path(__file__).resolve().parent.parent
        images_dir = root / "benchmark" / "images"
        lbl_csv = root / "benchmark" / "labels.csv"

        if not lbl_csv.is_file():
            return JSONResponse({"status": "error", "message": "labels.csv not found"}, status_code=404)

        rows = []
        with lbl_csv.open(newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                rows.append({"image": r["image"], "ground_truth": r["ground_truth"]})

        if limit is not None and limit > 0:
            rows = rows[:limit]

        async def event_generator():
            total = len(rows)
            exact_matches = 0
            start_all = time.monotonic()

            for idx, item in enumerate(rows):
                if await request.is_disconnected():
                    logger.info("Benchmark client disconnected, stopping stream")
                    break

                img_path = images_dir / item["image"]
                if not img_path.is_file():
                    event_data = {
                        "type": "item",
                        "index": idx + 1,
                        "image": item["image"],
                        "ground_truth": item["ground_truth"],
                        "prediction": "",
                        "exact_match": False,
                        "confidence": None,
                        "status": "ERROR",
                        "latency_ms": 0,
                        "error": "File not found",
                    }
                    yield f"data: {json.dumps(event_data)}\n\n"
                    continue

                try:
                    img_bytes = img_path.read_bytes()
                    ocr_res, _, _ = await run_ocr_pipeline(
                        image_data=img_bytes,
                        content_type="image/jpeg",
                        worker_settings=worker_settings,
                        worker=worker,
                        detector=detector,
                        enable_detector=enable_detector,
                        enable_barcode=enable_barcode,
                        enable_tta=enable_tta,
                    )
                    pred = ocr_res.sealNumber or ""
                    is_match = pred == item["ground_truth"]
                    if is_match:
                        exact_matches += 1

                    event_data = {
                        "type": "item",
                        "index": idx + 1,
                        "image": item["image"],
                        "ground_truth": item["ground_truth"],
                        "prediction": pred,
                        "exact_match": is_match,
                        "confidence": ocr_res.confidence,
                        "status": ocr_res.status,
                        "latency_ms": ocr_res.processingTimeMs,
                    }
                except Exception as exc:
                    event_data = {
                        "type": "item",
                        "index": idx + 1,
                        "image": item["image"],
                        "ground_truth": item["ground_truth"],
                        "prediction": "",
                        "exact_match": False,
                        "confidence": None,
                        "status": "ERROR",
                        "latency_ms": 0,
                        "error": str(exc),
                    }

                yield f"data: {json.dumps(event_data)}\n\n"

            elapsed_total = round((time.monotonic() - start_all) * 1000)
            summary = {
                "type": "done",
                "total": total,
                "exact_matches": exact_matches,
                "mismatches": total - exact_matches,
                "accuracy": round((exact_matches / total) * 100, 2) if total else 0.0,
                "total_time_ms": elapsed_total,
                "modelVersion": worker_settings.model_version,
            }
            yield f"data: {json.dumps(summary)}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return app


app = create_app()
