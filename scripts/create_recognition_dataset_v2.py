"""Generate PaddleOCR container seal recognition dataset v2 with strict benchmark protection and quality gates."""

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
SEAL_ALLOWED_PATTERN = re.compile(r"^[A-Z0-9]{5,20}$")

BRAND_NAMES = {
    "WANHAI", "WHL", "YANGMING", "YML", "MAERSK", "MSK", "COSCO", "ONE",
    "HAPAG", "LLOYD", "HAPAGLLOYD", "EVERGREEN", "EMC", "CMA", "CGM", "CMACGM",
    "MSC", "PIL", "SITC", "ZIM", "OOCL", "HMM", "KMTC", "SEAL", "SECURITY",
    "CONTAINER", "HIGH", "BOLT", "LOCK", "MEGA", "CUSTOMS", "LINE", "SHIPPING",
    "CHINA", "SHANGHAI", "GENSTAR", "INTERPOOL", "TEX", "BEACON", "TRITON", "CAI", "SEACUBE"
}

CONFUSING_PAIRS = [
    ("O", "0"),
    ("I", "1"),
    ("B", "8"),
    ("S", "5"),
    ("Z", "2"),
    ("G", "6"),
]


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_phash(image_or_path) -> list[int]:
    if isinstance(image_or_path, (str, Path)):
        image = cv2.imread(str(image_or_path), cv2.IMREAD_GRAYSCALE)
    else:
        if len(image_or_path.shape) == 3:
            image = cv2.cvtColor(image_or_path, cv2.COLOR_BGR2GRAY)
        else:
            image = image_or_path
    if image is None:
        raise ValueError("Invalid image for pHash")
    hashes = []
    for rotation in range(4):
        rotated = np.rot90(image, rotation).copy()
        resized = cv2.resize(rotated, (32, 32), interpolation=cv2.INTER_AREA)
        values = cv2.dct(np.float32(resized))[:8, :8].flatten()
        median = float(np.median(values[1:]))
        h = sum(int(val > median) << idx for idx, val in enumerate(values))
        hashes.append(h)
    return hashes


def hamming_distance(left: list[int], right: list[int]) -> int:
    return min((a ^ b).bit_count() for a in left for b in right)


def load_benchmark_index(benchmark_dir: Path) -> tuple[dict[str, str], list[tuple[str, list[int]]]]:
    manifest_path = benchmark_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Benchmark manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sha_to_name = {item["sha256"]: item["image"] for item in manifest.get("images", [])}

    phash_cache_path = benchmark_dir / ".phash_cache.json"
    phashes = []
    if phash_cache_path.is_file():
        try:
            cached = json.loads(phash_cache_path.read_text(encoding="utf-8"))
            phashes = [(k, v) for k, v in cached.items()]
        except Exception:
            phashes = []

    if len(phashes) != len(manifest.get("images", [])):
        phashes = []
        cache_dict = {}
        for item in manifest.get("images", []):
            img_path = benchmark_dir / "images" / item["image"]
            if img_path.is_file():
                h = compute_phash(img_path)
                phashes.append((item["image"], h))
                cache_dict[item["image"]] = h
        try:
            phash_cache_path.write_text(json.dumps(cache_dict, indent=2), encoding="utf-8")
        except Exception:
            pass

    return sha_to_name, phashes


class SealDetectorModel:
    def __init__(self, model_path: Path, input_size: int = 960, confidence: float = 0.45,
                 iou: float = 0.45, padding: float = 0.12, max_regions: int = 3):
        if not model_path.is_file():
            raise FileNotFoundError(f"Detector model not found: {model_path}")
        self.net = cv2.dnn.readNetFromONNX(str(model_path))
        self.input_size = input_size
        self.confidence = confidence
        self.iou = iou
        self.padding = padding
        self.max_regions = max_regions

    def detect(self, image: np.ndarray) -> list[dict]:
        height, width = image.shape[:2]
        scale = min(self.input_size / width, self.input_size / height)
        resized_w, resized_h = round(width * scale), round(height * scale)
        resized = cv2.resize(image, (resized_w, resized_h))
        left = (self.input_size - resized_w) // 2
        top = (self.input_size - resized_h) // 2
        canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        canvas[top:top + resized_h, left:left + resized_w] = resized
        blob = cv2.dnn.blobFromImage(canvas, 1 / 255.0, (self.input_size, self.input_size),
                                     swapRB=True, crop=False)
        self.net.setInput(blob)
        output = self.net.forward()
        predictions = output[0].T if output.shape[1] == 5 else output[0]
        predictions = predictions[predictions[:, 4] >= self.confidence]
        if not len(predictions):
            return []
        boxes = [[float(cx - bw / 2), float(cy - bh / 2), float(bw), float(bh)]
                 for cx, cy, bw, bh, _ in predictions]
        scores = [float(row[4]) for row in predictions]
        indexes = cv2.dnn.NMSBoxes(boxes, scores, self.confidence, self.iou)
        ordered = sorted((int(i) for i in indexes), key=scores.__getitem__, reverse=True)
        detections = []
        for i in ordered[:self.max_regions]:
            bx, by, bw, bh = boxes[i]
            x1 = (bx - left) / scale
            y1 = (by - top) / scale
            x2 = (bx + bw - left) / scale
            y2 = (by + bh - top) / scale
            pad_x = bw / scale * self.padding
            pad_y = bh / scale * self.padding
            rx1 = max(0, round(x1 - pad_x))
            ry1 = max(0, round(y1 - pad_y))
            rx2 = min(width, round(x2 + pad_x))
            ry2 = min(height, round(y2 + pad_y))
            if rx2 > rx1 and ry2 > ry1:
                detections.append({
                    "x": rx1, "y": ry1, "width": rx2 - rx1, "height": ry2 - ry1,
                    "confidence": scores[i]
                })
        return detections

    def crop_detections(self, image: np.ndarray, detections: list[dict]) -> list[tuple[np.ndarray, dict]]:
        return [
            (image[d["y"]:d["y"] + d["height"], d["x"]:d["x"] + d["width"]].copy(), d)
            for d in detections
        ]


def crop_polygon(image: np.ndarray, polygon: list[list[float]]) -> np.ndarray:
    points = np.asarray(polygon, dtype=np.float32)
    w = max(1, round(max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3]))))
    h = max(1, round(max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2]))))
    target = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(points, target)
    crop = cv2.warpPerspective(image, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if crop.shape[0] / crop.shape[1] >= 1.4:
        crop = np.rot90(crop).copy()
    return crop


def normalize_text(text: str) -> str:
    return "".join(re.sub(r"[^A-Za-z0-9]", "", text).upper().split())


def is_brand_name(text: str) -> bool:
    norm = normalize_text(text)
    if norm in BRAND_NAMES:
        return True
    for brand in BRAND_NAMES:
        if len(brand) >= 4 and brand in norm and not any(ch.isdigit() for ch in norm):
            return True
    return False


def check_character_ambiguity(text: str, confidence: float) -> list[str]:
    ambiguous = []
    norm = normalize_text(text)
    for pair_a, pair_b in CONFUSING_PAIRS:
        if pair_a in norm or pair_b in norm:
            has_letters = any(c.isalpha() for c in norm)
            has_digits = any(c.isdigit() for c in norm)
            if (pair_a == "O" and pair_b == "0"):
                if "O" in norm and has_digits:
                    ambiguous.append(f"{pair_a}/{pair_b}")
                elif confidence < 0.96:
                    ambiguous.append(f"{pair_a}/{pair_b}")
            elif (pair_a == "I" and pair_b == "1"):
                if "I" in norm and has_digits:
                    ambiguous.append(f"{pair_a}/{pair_b}")
                elif confidence < 0.96:
                    ambiguous.append(f"{pair_a}/{pair_b}")
            else:
                if confidence < 0.94:
                    ambiguous.append(f"{pair_a}/{pair_b}")
    return sorted(list(set(ambiguous)))


def is_clipped(polygon: list[list[float]], image_shape: tuple[int, int]) -> bool:
    h, w = image_shape[:2]
    points = np.asarray(polygon)
    min_x, max_x = points[:, 0].min(), points[:, 0].max()
    min_y, max_y = points[:, 1].min(), points[:, 1].max()
    margin_x = max(2, w * 0.02)
    margin_y = max(2, h * 0.02)
    return min_x <= margin_x or max_x >= (w - margin_x) or min_y <= margin_y or max_y >= (h - margin_y)


def evaluate_image_quality(crop: np.ndarray) -> tuple[bool, str | None]:
    if crop.shape[0] < 8 or crop.shape[1] < 20:
        return False, "CROP_TOO_SMALL"
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    mean_val = float(np.mean(gray))
    if mean_val < 25:
        return False, "IMAGE_TOO_DARK"
    if mean_val > 235:
        return False, "IMAGE_TOO_BRIGHT"
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if lap_var < 20:
        return False, "IMAGE_TOO_BLURRY"
    return True, None


def extract_best_candidate(ocr_engine, seal_image: np.ndarray) -> dict[str, Any]:
    rotations_to_try = [0, 180]
    candidates = []
    orientation_results = {}

    for rot in rotations_to_try:
        rotated = np.rot90(seal_image, rot // 90).copy()
        try:
            results = list(ocr_engine.predict(rotated))
        except Exception:
            continue
        if not results:
            continue
        data = results[0].json.get("res", {})
        rec_texts = data.get("rec_texts", [])
        rec_scores = data.get("rec_scores", [])
        rec_polys = data.get("rec_polys", data.get("dt_polys", []))

        orientation_candidates = []
        for text, score, poly in zip(rec_texts, rec_scores, rec_polys):
            norm = normalize_text(text)
            if not norm or is_brand_name(norm):
                continue
            orientation_candidates.append({
                "text": text,
                "normalized": norm,
                "score": float(score),
                "poly": poly,
                "rotation": rot,
                "rotated_image": rotated
            })
        orientation_results[rot] = orientation_candidates
        candidates.extend(orientation_candidates)

    has_high_valid = any(SEAL_ALLOWED_PATTERN.match(c["normalized"]) and c["score"] >= 0.85 for c in candidates)
    if not has_high_valid:
        for rot in [90, 270]:
            rotated = np.rot90(seal_image, rot // 90).copy()
            try:
                results = list(ocr_engine.predict(rotated))
            except Exception:
                continue
            if not results:
                continue
            data = results[0].json.get("res", {})
            rec_texts = data.get("rec_texts", [])
            rec_scores = data.get("rec_scores", [])
            rec_polys = data.get("rec_polys", data.get("dt_polys", []))

            orientation_candidates = []
            for text, score, poly in zip(rec_texts, rec_scores, rec_polys):
                norm = normalize_text(text)
                if not norm or is_brand_name(norm):
                    continue
                orientation_candidates.append({
                    "text": text,
                    "normalized": norm,
                    "score": float(score),
                    "poly": poly,
                    "rotation": rot,
                    "rotated_image": rotated
                })
            orientation_results[rot] = orientation_candidates
            candidates.extend(orientation_candidates)

    if not candidates:
        return {"status": "FAILED", "reason": "NO_TEXT_DETECTED"}

    valid_candidates = [c for c in candidates if SEAL_ALLOWED_PATTERN.match(c["normalized"])]
    if not valid_candidates:
        longest = max(candidates, key=lambda c: len(c["normalized"]))
        return {
            "status": "FAILED",
            "reason": "INVALID_CHARACTERS" if re.search(r"[^A-Z0-9]", longest["text"].upper()) else "UNTRUSTED_CROP",
            "proposed_label": longest["normalized"],
            "rotation": longest["rotation"],
            "raw_candidate": longest
        }

    valid_candidates.sort(key=lambda c: (
        1 if any(ch.isdigit() for ch in c["normalized"]) else 0,
        c["score"],
        len(c["normalized"])
    ), reverse=True)

    best = valid_candidates[0]
    cropped_line = crop_polygon(best["rotated_image"], best["poly"])
    quality_ok, quality_reason = evaluate_image_quality(cropped_line)

    ambiguous = check_character_ambiguity(best["normalized"], best["score"])
    clipped = is_clipped(best["poly"], best["rotated_image"].shape)

    rot_conflicts = []
    opp_rot = (best["rotation"] + 180) % 360
    for c in orientation_results.get(opp_rot, []):
        if c["normalized"] != best["normalized"] and c["score"] >= 0.70 and SEAL_ALLOWED_PATTERN.match(c["normalized"]):
            rot_conflicts.append(c["normalized"])

    distinct_candidates = {c["normalized"] for c in valid_candidates if c["score"] >= 0.75}

    gate_reasons = []
    if not quality_ok:
        gate_reasons.append(quality_reason)
    if clipped:
        gate_reasons.append("CLIPPED_CROP")
    if ambiguous:
        gate_reasons.append(f"AMBIGUOUS_CHARACTER: {', '.join(ambiguous)}")
    if rot_conflicts:
        gate_reasons.append(f"ORIENTATION_CONFLICT: {best['normalized']} vs {rot_conflicts[0]}")
    if len(distinct_candidates) > 1:
        gate_reasons.append(f"MULTIPLE_SEAL_CANDIDATES: {sorted(list(distinct_candidates))}")
    if best["score"] < 0.85:
        gate_reasons.append("LOW_CONFIDENCE")

    if gate_reasons:
        return {
            "status": "NEEDS_REVIEW",
            "reason": "; ".join(gate_reasons),
            "proposed_label": best["normalized"],
            "confidence": best["score"],
            "rotation": best["rotation"],
            "crop": cropped_line,
            "ambiguous_characters": ambiguous,
            "source_box": best["poly"]
        }

    return {
        "status": "AUTO_LABELED",
        "reason": "HIGH_CONFIDENCE_MATCH",
        "proposed_label": best["normalized"],
        "confidence": best["score"],
        "rotation": best["rotation"],
        "crop": cropped_line,
        "ambiguous_characters": [],
        "source_box": best["poly"]
    }


def process_dataset(
    source_dir: Path,
    benchmark_dir: Path,
    work_dir: Path,
    detector_model: Path,
    rec_model_dir: str | None = None,
    file_list: Path | None = None,
    limit: int = 0,
    target_crops: int = 0,
    seed: int = 20260918,
) -> dict[str, Any]:
    work_dir.mkdir(parents=True, exist_ok=True)
    images_dir = work_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    review_images_dir = work_dir / "review_images"
    review_images_dir.mkdir(parents=True, exist_ok=True)

    state_file = work_dir / ".pipeline_state.jsonl"
    prelabels_file = work_dir / "prelabels.csv"
    review_queue_file = work_dir / "review_queue.csv"
    rejected_file = work_dir / "rejected.csv"

    print("Loading benchmark index...")
    bm_sha256, bm_phashes = load_benchmark_index(benchmark_dir)
    print(f"Benchmark indexed: {len(bm_sha256)} images.")

    print(f"Loading seal detector from {detector_model}...")
    detector = SealDetectorModel(detector_model)

    print("Loading PaddleOCR engine...")
    from paddleocr import PaddleOCR
    ocr_kwargs: dict[str, Any] = {
        "text_detection_model_name": "PP-OCRv6_medium_det",
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
        "device": "cpu"
    }
    if rec_model_dir and Path(rec_model_dir).exists():
        ocr_kwargs["text_recognition_model_dir"] = rec_model_dir
    else:
        ocr_kwargs["text_recognition_model_name"] = "PP-OCRv6_medium_rec"
    ocr_engine = PaddleOCR(**ocr_kwargs)
    print("Models loaded successfully.")

    processed_records = {}
    if state_file.is_file():
        with state_file.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    processed_records[item["source_image"]] = item
        print(f"Resuming: found {len(processed_records)} existing processed records.")

    print(f"Scanning source images in {source_dir}...")
    all_files = []
    if file_list and Path(file_list).is_file():
        lines = Path(file_list).read_text(encoding="utf-8").splitlines()
        for line in lines:
            if line.strip():
                p = Path(line.strip())
                if not p.is_absolute():
                    p = source_dir / p
                all_files.append(p)
        print(f"Loaded {len(all_files)} images from file list: {file_list}")
    else:
        for root, _, files in os.walk(source_dir):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in SUPPORTED_EXTENSIONS:
                    all_files.append(Path(root) / file)
        all_files.sort(key=lambda p: p.name)
        print(f"Found {len(all_files)} total image files.")

    seen_dataset_hashes = set()
    prelabels_rows = []
    review_queue_rows = []
    rejected_rows = []

    stats = {
        "total_scanned": 0,
        "exact_duplicates": 0,
        "benchmark_exact_matches": 0,
        "benchmark_near_matches": 0,
        "corrupted_images": 0,
        "no_seal_detected": 0,
        "auto_labeled": 0,
        "needs_review": 0,
        "rejected": 0,
    }

    target_count = limit if limit > 0 else len(all_files)
    sample_index = len(processed_records)

    for file_path in all_files:
        if limit > 0 and sample_index >= target_count:
            break
        if target_crops > 0 and (stats["auto_labeled"] + stats["needs_review"]) >= target_crops:
            print(f"Reached target crop count ({stats['auto_labeled'] + stats['needs_review']} >= {target_crops}). Finishing.")
            break

        rel_source = file_path.name
        stats["total_scanned"] += 1

        if rel_source in processed_records:
            record = processed_records[rel_source]
            sample_index += 1
            if record["status"] == "AUTO_LABELED":
                stats["auto_labeled"] += 1
                prelabels_rows.append(record)
            elif record["status"] == "NEEDS_REVIEW":
                stats["needs_review"] += 1
                prelabels_rows.append(record)
                review_queue_rows.append({
                    "image": record["image"],
                    "proposed_label": record["proposed_label"],
                    "reason": record["reason"],
                    "source_image": record["source_image"]
                })
            elif record["status"] == "REJECTED":
                stats["rejected"] += 1
                prelabels_rows.append(record)
                rejected_rows.append({
                    "source_image": record["source_image"],
                    "reason": record["reason"],
                    "details": record.get("details", "")
                })
            continue

        try:
            with Image.open(file_path) as img:
                img.verify()
            image = cv2.imread(str(file_path))
            if image is None:
                raise ValueError("cv2.imread failed")
        except Exception as exc:
            stats["corrupted_images"] += 1
            stats["rejected"] += 1
            rec = {
                "source_image": rel_source,
                "reason": "CORRUPTED_IMAGE",
                "details": str(exc),
                "status": "REJECTED",
                "image": "",
                "proposed_label": "",
                "rotation": 0
            }
            rejected_rows.append({"source_image": rel_source, "reason": rec["reason"], "details": rec["details"]})
            prelabels_rows.append(rec)
            with state_file.open("a", encoding="utf-8") as sf:
                sf.write(json.dumps(rec) + "\n")
            sample_index += 1
            continue

        sha256 = compute_sha256(file_path)
        if sha256 in bm_sha256:
            stats["benchmark_exact_matches"] += 1
            stats["rejected"] += 1
            rec = {
                "source_image": rel_source,
                "reason": "BENCHMARK_EXACT_MATCH",
                "details": f"Matches benchmark image {bm_sha256[sha256]}",
                "status": "REJECTED",
                "image": "",
                "proposed_label": "",
                "rotation": 0
            }
            rejected_rows.append({"source_image": rel_source, "reason": rec["reason"], "details": rec["details"]})
            prelabels_rows.append(rec)
            with state_file.open("a", encoding="utf-8") as sf:
                sf.write(json.dumps(rec) + "\n")
            sample_index += 1
            continue

        ph = compute_phash(image)
        near_bm = [(bm_img, hamming_distance(ph, bm_h)) for bm_img, bm_h in bm_phashes]
        min_bm_dist, closest_bm = min(((d, img) for img, d in near_bm), default=(999, ""))
        if min_bm_dist <= 6:
            stats["benchmark_near_matches"] += 1
            stats["rejected"] += 1
            rec = {
                "source_image": rel_source,
                "reason": "BENCHMARK_NEAR_MATCH",
                "details": f"Hamming distance {min_bm_dist} to benchmark image {closest_bm}",
                "status": "REJECTED",
                "image": "",
                "proposed_label": "",
                "rotation": 0
            }
            rejected_rows.append({"source_image": rel_source, "reason": rec["reason"], "details": rec["details"]})
            prelabels_rows.append(rec)
            with state_file.open("a", encoding="utf-8") as sf:
                sf.write(json.dumps(rec) + "\n")
            sample_index += 1
            continue

        if sha256 in seen_dataset_hashes:
            stats["exact_duplicates"] += 1
            stats["rejected"] += 1
            rec = {
                "source_image": rel_source,
                "reason": "EXACT_DUPLICATE_IMAGE",
                "details": f"SHA256 {sha256} already seen in dataset",
                "status": "REJECTED",
                "image": "",
                "proposed_label": "",
                "rotation": 0
            }
            rejected_rows.append({"source_image": rel_source, "reason": rec["reason"], "details": rec["details"]})
            prelabels_rows.append(rec)
            with state_file.open("a", encoding="utf-8") as sf:
                sf.write(json.dumps(rec) + "\n")
            sample_index += 1
            continue
        seen_dataset_hashes.add(sha256)

        detections = detector.detect(image)
        if not detections:
            stats["no_seal_detected"] += 1
            stats["needs_review"] += 1
            rec = {
                "source_image": rel_source,
                "status": "NEEDS_REVIEW",
                "reason": "NO_SEAL_DETECTED",
                "image": "",
                "proposed_label": "",
                "rotation": 0
            }
            review_queue_rows.append({
                "image": "",
                "proposed_label": "",
                "reason": rec["reason"],
                "source_image": rel_source
            })
            prelabels_rows.append(rec)
            with state_file.open("a", encoding="utf-8") as sf:
                sf.write(json.dumps(rec) + "\n")
            sample_index += 1
            continue

        seal_crops = detector.crop_detections(image, detections)
        seal_crop_img, seal_det = seal_crops[0]

        eval_result = extract_best_candidate(ocr_engine, seal_crop_img)

        sample_index += 1
        crop_filename = f"crop_{sample_index:06d}.jpg"
        crop_rel_path = f"images/{crop_filename}"
        crop_disk_path = images_dir / crop_filename

        if "crop" in eval_result:
            cv2.imwrite(str(crop_disk_path), eval_result["crop"], [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:
            cv2.imwrite(str(crop_disk_path), seal_crop_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

        status = eval_result["status"]
        reason = eval_result.get("reason", "")
        proposed = eval_result.get("proposed_label", "")
        rotation = eval_result.get("rotation", 0)

        if status == "AUTO_LABELED":
            stats["auto_labeled"] += 1
            rec = {
                "image": crop_rel_path,
                "proposed_label": proposed,
                "status": "AUTO_LABELED",
                "reason": reason,
                "source_image": rel_source,
                "rotation": rotation
            }
            prelabels_rows.append(rec)
        elif status == "NEEDS_REVIEW":
            stats["needs_review"] += 1
            shutil.copy2(crop_disk_path, review_images_dir / crop_filename)
            rec = {
                "image": crop_rel_path,
                "proposed_label": proposed,
                "status": "NEEDS_REVIEW",
                "reason": reason,
                "source_image": rel_source,
                "rotation": rotation
            }
            prelabels_rows.append(rec)
            review_queue_rows.append({
                "image": crop_rel_path,
                "proposed_label": proposed,
                "reason": reason,
                "source_image": rel_source
            })
        else:
            stats["rejected"] += 1
            rec = {
                "image": crop_rel_path,
                "proposed_label": proposed,
                "status": "REJECTED",
                "reason": reason,
                "source_image": rel_source,
                "rotation": rotation
            }
            prelabels_rows.append(rec)
            rejected_rows.append({
                "source_image": rel_source,
                "reason": reason,
                "details": f"OCR failed or text invalid: {proposed}"
            })

        with state_file.open("a", encoding="utf-8") as sf:
            sf.write(json.dumps(rec) + "\n")

        if stats["total_scanned"] % 50 == 0:
            total_crops = stats["auto_labeled"] + stats["needs_review"]
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Scanned: {stats['total_scanned']}, Crops: {total_crops} (Auto: {stats['auto_labeled']}, Review: {stats['needs_review']}), Rejected: {stats['rejected']}", flush=True)
            with prelabels_file.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["image", "proposed_label", "status", "reason", "source_image", "rotation"], extrasaction="ignore")
                writer.writeheader()
                writer.writerows(prelabels_rows)
            with review_queue_file.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["image", "proposed_label", "reason", "source_image"], extrasaction="ignore")
                writer.writeheader()
                writer.writerows(review_queue_rows)
            with rejected_file.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["source_image", "reason", "details"], extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rejected_rows)

    with prelabels_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "proposed_label", "status", "reason", "source_image", "rotation"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(prelabels_rows)

    with review_queue_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "proposed_label", "reason", "source_image"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(review_queue_rows)

    with rejected_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["source_image", "reason", "details"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rejected_rows)

    summary = {
        "stats": stats,
        "counts": {
            "prelabels": len(prelabels_rows),
            "review_queue": len(review_queue_rows),
            "rejected": len(rejected_rows),
        }
    }
    summary_path = work_dir / "pilot_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n--- Processing Finished ---")
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("/workspace/ocr/archive/new_dataset"))
    parser.add_argument("--benchmark-dir", type=Path, default=Path("benchmark"))
    parser.add_argument("--work-dir", type=Path, default=Path("recognition_dataset_v2_work"))
    parser.add_argument("--detector-model", type=Path, default=Path("models/seal-detector-v1/best.onnx"))
    parser.add_argument("--rec-model-dir", type=str, default="models/seal-ocr-rec-v1")
    parser.add_argument("--file-list", type=Path, default=None, help="Optional text file listing image paths")
    parser.add_argument("--limit", type=int, default=0, help="Number of images to process (0 = all)")
    parser.add_argument("--target-crops", type=int, default=0, help="Target number of valid crops to stop at (0 = no limit)")
    parser.add_argument("--seed", type=int, default=20260918)
    args = parser.parse_args()

    process_dataset(
        source_dir=args.source_dir,
        benchmark_dir=args.benchmark_dir,
        work_dir=args.work_dir,
        detector_model=args.detector_model,
        rec_model_dir=args.rec_model_dir,
        file_list=args.file_list,
        limit=args.limit,
        target_crops=args.target_crops,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
