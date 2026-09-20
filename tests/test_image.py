from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw

from app.config import Settings
from app.image import prepare, quality_reason, resize


def test_exif_orientation_and_resize():
    image = Image.new("RGB", (300, 600), "white")
    exif = Image.Exif()
    exif[274] = 6
    output = BytesIO()
    image.save(output, format="JPEG", exif=exif)
    prepared = prepare(output.getvalue(), "image/jpeg", Settings())
    assert prepared.original.shape[:2] == (600, 300)
    assert prepared.normalized.shape[:2] == (300, 600)
    assert resize(prepared.normalized, Settings(max_side=300)).shape[:2] == (150, 300)


def test_extreme_brightness_is_recapture():
    assert quality_reason(np.zeros((200, 200, 3), dtype=np.uint8), Settings()) == "IMAGE_TOO_DARK"
    assert quality_reason(np.full((200, 200, 3), 255, dtype=np.uint8), Settings()) == "IMAGE_TOO_BRIGHT"
    white_seal = Image.new("RGB", (400, 200), "white")
    ImageDraw.Draw(white_seal).rectangle((100, 80, 300, 110), fill="black")
    assert quality_reason(np.asarray(white_seal), Settings()) is None
