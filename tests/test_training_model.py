import csv
import hashlib
import json

import pytest
from PIL import Image

from scripts import training_model


def dataset(root):
    (root / "images").mkdir(parents=True)
    rows = []
    for index, split in enumerate(("train", "train", "val")):
        crop = f"images/{index}.jpg"
        Image.new("RGB", (40, 16), "white").save(root / crop)
        rows.append({"crop": crop, "ground_truth": f"FX0000000{index}", "split": split})
    (root / "metadata.json").write_text(json.dumps({"datasetVersion": "test-v1", "sourceDatasetSha256": "hash",
        "sourceSamples": 3, "acceptedSamples": 3, "trainSamples": 2, "validationSamples": 1, "seed": 1}), encoding="utf-8")
    with (root / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["crop", "ground_truth", "split"])
        writer.writeheader()
        writer.writerows(rows)
    for name, selected in (("train.txt", rows[:2]), ("val.txt", rows[2:])):
        (root / name).write_text("\n".join(f"{row['crop']}\t{row['ground_truth']}" for row in selected) + "\n", encoding="utf-8")
    return {"datasetVersion": "test-v1", "sourceDatasetSha256": "hash", "sourceSamples": 3,
            "acceptedSamples": 3, "trainSamples": 2, "validationSamples": 1, "seed": 1}


def test_validate_and_archive_prepared_dataset(tmp_path):
    expected = dataset(tmp_path / "dataset")
    assert training_model.validate_prepared_dataset(tmp_path / "dataset", expected)["acceptedSamples"] == 3
    archive = tmp_path / "recognition_dataset.zip"
    result = training_model.archive_dataset(tmp_path / "dataset", archive, expected)
    assert archive.is_file() and result["archiveSha256"]


def test_validate_rejects_split_disagreement(tmp_path):
    root = tmp_path / "dataset"
    expected = dataset(root)
    (root / "val.txt").write_text("images/0.jpg\tFX00000000\n", encoding="utf-8")
    with pytest.raises(ValueError, match="overlap|Split count|different crops"):
        training_model.validate_prepared_dataset(root, expected)


def test_validate_rejects_changed_crop_when_content_hash_is_frozen(tmp_path):
    root = tmp_path / "dataset"
    expected = dataset(root)
    expected["datasetContentSha256"] = training_model.content_sha256(root)
    Image.new("RGB", (40, 16), "black").save(root / "images" / "0.jpg")
    with pytest.raises(ValueError, match="content hash"):
        training_model.validate_prepared_dataset(root, expected)


def test_validate_v2_dataset(tmp_path):
    root = tmp_path / "dataset"
    images = root / "images"
    images.mkdir(parents=True)
    image = images / "crop_000001.jpg"
    Image.new("RGB", (20, 10), "white").save(image)
    image_hash = training_model.sha256(image)
    dataset_hash = hashlib.sha256(
        f"images/crop_000001.jpg\tABC123\t{image_hash}".encode()
    ).hexdigest()
    (root / "train.txt").write_text("images/crop_000001.jpg\tABC123\n", encoding="utf-8")
    (root / "val.txt").write_text("", encoding="utf-8")
    row = {
        "image": "images/crop_000001.jpg", "ground_truth": "ABC123",
        "source_image": "source.jpg", "split": "train", "sha256": image_hash,
        "original_crop": "images/source.jpg",
    }
    for name in ("labels.csv", "manifest.csv"):
        with (root / name).open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=row)
            writer.writeheader()
            writer.writerow(row)
    (root / "metadata.json").write_text(json.dumps({
        "datasetVersion": "seal-recognition-v2", "totalVerified": 1,
        "trainSamples": 1, "valSamples": 0, "datasetSha256": dataset_hash,
    }), encoding="utf-8")
    (root / "validation_report.json").write_text(json.dumps({
        "totalImages": 1, "trainCount": 1, "valCount": 0,
        "datasetSha256": dataset_hash, "benchmarkOverlapCount": 0,
        "status": "PASS",
    }), encoding="utf-8")

    result = training_model.validate_prepared_dataset(root)

    assert result["datasetVersion"] == "seal-recognition-v2"
    assert result["totalVerified"] == 1
