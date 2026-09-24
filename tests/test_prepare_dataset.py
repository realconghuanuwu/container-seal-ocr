import json
import numpy as np
import pytest
from PIL import Image

from scripts.prepare_dataset import (
    extract_label_from_filename,
    is_valid_seal_label,
    load_dataset_samples,
    prepare_dataset,
)


def test_extract_label_from_filename():
    assert extract_label_from_filename("FX40295831.jpg") == "FX40295831"
    assert extract_label_from_filename("FX40295831_1.jpg") == "FX40295831"
    assert extract_label_from_filename("sitr722123 (1).png") == "SITR722123"
    assert extract_label_from_filename("OOLKCK28059__r90.webp") == "OOLKCK28059"
    assert extract_label_from_filename("abc-12345.jpg") == "ABC12345"


def test_is_valid_seal_label():
    assert is_valid_seal_label("FX40295831") is True
    assert is_valid_seal_label("SITR722123") is True
    assert is_valid_seal_label("12345") is True
    # Too short (< 5 chars)
    assert is_valid_seal_label("ABC1") is False
    # No digits
    assert is_valid_seal_label("SEALNUMBER") is False
    # Special characters
    assert is_valid_seal_label("FX@12345") is False


def test_prepare_dataset_filename_as_label(tmp_path):
    input_dir = tmp_path / "raw_images"
    input_dir.mkdir()

    # Create dummy images where filename is label
    labels = ["FX40295831", "SITR722123", "OOLKCK28059", "OOLKCK28059_2", "HJLU910283"]
    rng = np.random.default_rng(42)
    for lbl in labels:
        img_arr = rng.integers(0, 256, (32, 64, 3), dtype=np.uint8)
        Image.fromarray(img_arr).save(input_dir / f"{lbl}.jpg")

    output_dir = tmp_path / "output_dataset"
    meta = prepare_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        val_ratio=0.25,
        seed=123,
    )

    assert meta["totalSamples"] == len(labels)
    assert (output_dir / "train.txt").is_file()
    assert (output_dir / "val.txt").is_file()
    assert (output_dir / "manifest.csv").is_file()
    assert (output_dir / "metadata.json").is_file()

    train_lines = (output_dir / "train.txt").read_text(encoding="utf-8").splitlines()
    val_lines = (output_dir / "val.txt").read_text(encoding="utf-8").splitlines()
    assert len(train_lines) + len(val_lines) == len(labels)

    # Check line format: images/xxx.jpg\tLABEL
    for line in train_lines + val_lines:
        parts = line.split("\t")
        assert len(parts) == 2
        assert parts[0].startswith("images/")
        assert is_valid_seal_label(parts[1])


def test_prepare_dataset_with_labels_csv(tmp_path):
    input_dir = tmp_path / "dataset_with_csv"
    images_dir = input_dir / "images"
    images_dir.mkdir(parents=True)

    rng = np.random.default_rng(99)
    csv_rows = ["image,label\n"]
    for i in range(5):
        name = f"photo_{i:03d}.jpg"
        lbl = f"SEAL{i:04d}"
        img_arr = rng.integers(0, 256, (32, 64, 3), dtype=np.uint8)
        Image.fromarray(img_arr).save(images_dir / name)
        csv_rows.append(f"{name},{lbl}\n")

    (input_dir / "labels.csv").write_text("".join(csv_rows), encoding="utf-8")

    output_dir = tmp_path / "output_csv_dataset"
    meta = prepare_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        val_ratio=0.4,
        seed=77,
    )

    assert meta["totalSamples"] == 5
    assert (output_dir / "train.txt").is_file()
    assert (output_dir / "val.txt").is_file()
