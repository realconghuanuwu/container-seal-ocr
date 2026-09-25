import threading
from pathlib import Path

import cv2
import numpy as np

from .config import Settings


class SealDetector:
    def __init__(self, settings: Settings):
        model = Path(settings.seal_detector_model or "")
        if not model.is_file():
            raise ValueError(f"Seal detector model does not exist: {model}")
        self.net = cv2.dnn.readNetFromONNX(str(model))
        self.input_size = settings.seal_detector_input_size
        self.confidence = settings.seal_detector_confidence
        self.iou = settings.seal_detector_iou
        self.padding = settings.seal_detector_padding
        if self.padding == 0:
            self.padding_x = 0.0
            self.padding_y = 0.0
        else:
            self.padding_x = getattr(settings, "seal_detector_padding_x", max(self.padding * 1.95, 0.35))
            self.padding_y = getattr(settings, "seal_detector_padding_y", max(self.padding * 1.15, 0.20))
        self.max_regions = settings.seal_detector_max_regions
        self._lock = threading.Lock()

    def detect(self, image: np.ndarray) -> list[dict]:
        height, width = image.shape[:2]
        scale = min(self.input_size / width, self.input_size / height)
        resized_width, resized_height = round(width * scale), round(height * scale)
        resized = cv2.resize(image, (resized_width, resized_height))
        left = (self.input_size - resized_width) // 2
        top = (self.input_size - resized_height) // 2
        canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        canvas[top:top + resized_height, left:left + resized_width] = resized
        blob = cv2.dnn.blobFromImage(canvas, 1 / 255, (self.input_size, self.input_size),
                                     swapRB=True, crop=False)
        with self._lock:
            self.net.setInput(blob)
            output = self.net.forward()
        predictions = output[0].T if output.shape[1] == 5 else output[0]
        predictions = predictions[predictions[:, 4] >= self.confidence]
        if not len(predictions):
            return []

        boxes = [[float(cx - box_width / 2), float(cy - box_height / 2),
                  float(box_width), float(box_height)]
                 for cx, cy, box_width, box_height, _ in predictions]
        scores = [float(row[4]) for row in predictions]
        indexes = cv2.dnn.NMSBoxes(boxes, scores, self.confidence, self.iou)
        ordered = sorted((int(index) for index in indexes), key=scores.__getitem__, reverse=True)

        detections = []
        for index in ordered[:self.max_regions]:
            x, y, box_width, box_height = boxes[index]
            x1, y1 = (x - left) / scale, (y - top) / scale
            x2, y2 = (x + box_width - left) / scale, (y + box_height - top) / scale
            pad_x = box_width / scale * self.padding_x
            pad_y = box_height / scale * self.padding_y
            x1, y1 = max(0, round(x1 - pad_x)), max(0, round(y1 - pad_y))
            x2, y2 = min(width, round(x2 + pad_x)), min(height, round(y2 + pad_y))
            if x2 > x1 and y2 > y1:
                detections.append({
                    "x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1,
                    "confidence": scores[index],
                })
        return detections

    @staticmethod
    def crop(image: np.ndarray, detections: list[dict]) -> list[np.ndarray]:
        return [image[item["y"]:item["y"] + item["height"],
                      item["x"]:item["x"] + item["width"]].copy()
                for item in detections]

    def crops(self, image: np.ndarray) -> list[np.ndarray]:
        return self.crop(image, self.detect(image))


def create_detector(settings: Settings) -> SealDetector | None:
    return SealDetector(settings) if settings.seal_detector_model else None
