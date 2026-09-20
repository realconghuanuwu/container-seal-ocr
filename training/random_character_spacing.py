"""PaddleOCR train-only horizontal spacing augmentation."""

import random

import cv2

try:
    from .text_image_aug import tia_stretch
except ImportError:  # allows the repository test to inject a tiny stand-in
    tia_stretch = None


class RandomCharacterSpacing:
    def __init__(self, prob=0.35, min_segments=3, max_segments=6, **kwargs):
        self.prob = float(prob)
        self.min_segments = int(min_segments)
        self.max_segments = int(max_segments)

    def __call__(self, data):
        image = data["image"]
        if random.random() > self.prob or image.shape[0] < 20 or image.shape[1] < 20:
            return data
        if tia_stretch is None:
            raise RuntimeError("RandomCharacterSpacing must run inside the PaddleOCR imaug package")
        height = image.shape[0]
        stretched = tia_stretch(image, random.randint(self.min_segments, self.max_segments))
        if stretched.shape[0] != height:
            stretched = cv2.resize(stretched, (stretched.shape[1], height), interpolation=cv2.INTER_CUBIC)
        data["image"] = stretched
        return data
