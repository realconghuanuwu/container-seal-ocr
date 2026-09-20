import csv
import hashlib
import json

import numpy as np
import pytest
from PIL import Image

from scripts import training_data


def write_dataset(root, count=4, start=0):
    images = root / "images"
    images.mkdir(parents=True)
    rng = np.random.default_rng(start + 1)
    rows = []
    for index in range(count):
        name = f"image-{start + index}.png"
        Image.fromarray(rng.integers(0, 256, (40, 80, 3), dtype=np.uint8)).save(images / name)
        rows.append((name, f"FX{start + index:08d}"))
    with (root / "labels.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "ground_truth"])
        writer.writerows(rows)
    return rows


def write_benchmark(root):
    rows = write_dataset(root, 2, 100)
    images = []
    for name, truth in rows:
        path = root / "images" / name
        images.append({"image": name, "ground_truth": truth,
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (root / "manifest.json").write_text(json.dumps({"images": images}), encoding="utf-8")


class Result:
    def __init__(self, text, polygon=None):
        polygon = polygon or [[2, 2], [30, 2], [30, 10], [2, 10]]
        self.json = {"res": {"rec_texts": [text], "rec_scores": [0.99], "rec_polys": [polygon]}}


class ExactEngine:
    def __init__(self, labels):
        self.labels = labels

    def predict(self, image, **kwargs):
        return [Result(self.labels[int(image.sum())])]


def test_validate_excludes_same_label_duplicate_and_rejects_leakage(tmp_path):
    source, benchmark = tmp_path / "source", tmp_path / "benchmark"
    rows = write_dataset(source, 2)
    write_benchmark(benchmark)
    duplicate = source / "images" / rows[1][0]
    duplicate.write_bytes((source / "images" / rows[0][0]).read_bytes())
    labels = source / "labels.csv"
    labels.write_text(labels.read_text().replace(rows[1][1], rows[0][1]), encoding="utf-8")
    samples, report = training_data.validate_dataset(source, benchmark)
    assert len(samples) == 1 and len(report["exactDuplicateImages"]) == 1

    source, benchmark = tmp_path / "leaked", tmp_path / "benchmark2"
    rows = write_dataset(source, 2)
    write_benchmark(benchmark)
    (source / "images" / rows[0][0]).write_bytes((benchmark / "images" / "image-100.png").read_bytes())
    with pytest.raises(ValueError, match="Benchmark leakage"):
        training_data.validate_dataset(source, benchmark)


def test_validate_rejects_unsafe_invalid_and_conflicting_input(tmp_path):
    source, benchmark = tmp_path / "source", tmp_path / "benchmark"
    rows = write_dataset(source, 2)
    write_benchmark(benchmark)
    labels = source / "labels.csv"
    labels.write_text("image,ground_truth\n../outside.png,FX00000000\n", encoding="utf-8")
    with pytest.raises(ValueError, match="filename under images"):
        training_data.validate_dataset(source, benchmark)

    labels.write_text(f"image,ground_truth\n{rows[0][0]},not-a-seal\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid ground truth"):
        training_data.validate_dataset(source, benchmark)

    (source / "images" / rows[1][0]).write_bytes((source / "images" / rows[0][0]).read_bytes())
    labels.write_text("image,ground_truth\n" + f"{rows[0][0]},FX00000000\n" +
                      f"{rows[1][0]},FX99999999\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Conflicting labels"):
        training_data.validate_dataset(source, benchmark)


def test_candidate_rotation_fallback_and_shape_gate():
    image = np.zeros((20, 40, 3), dtype=np.uint8)
    image[0, 0, 0], image[0, -1, 0] = 1, 2

    class RotationEngine:
        def predict(self, value, **kwargs):
            return [Result("FX12345678" if value[0, 0, 0] == 2 else "BAD")]

    candidate, _, rotation = training_data.best_candidate(RotationEngine(), image, "FX12345678")
    assert candidate["prediction"] == "FX12345678" and rotation == 90

    class FallbackEngine:
        def predict(self, value, **kwargs):
            return [Result("FX12345678")] if kwargs else []

    candidate, _, _ = training_data.best_candidate(FallbackEngine(), image, "FX12345678")
    assert candidate["prediction"] == "FX12345678"
    assert not training_data.accepted({"edit_ratio": 0, "aspect": 1, "area_ratio": 0.01})
    assert not training_data.accepted({"edit_ratio": 0.8, "aspect": 3, "area_ratio": 0.01})
    assert not training_data.accepted({"edit_ratio": 0, "aspect": 3, "area_ratio": 0.5})


def test_prepare_writes_only_complete_output(tmp_path):
    source, benchmark, output = tmp_path / "source", tmp_path / "benchmark", tmp_path / "dataset"
    rows = write_dataset(source, 4)
    write_benchmark(benchmark)
    labels = {int(np.asarray(Image.open(source / "images" / name)).sum()): truth for name, truth in rows}
    metadata = training_data.prepare(source, benchmark, output, engine=ExactEngine(labels))
    assert metadata["acceptedSamples"] == 4
    assert len((output / "train.txt").read_text().splitlines()) + len((output / "val.txt").read_text().splitlines()) == 4
    assert len(list((output / "images").iterdir())) == 4

    class BrokenEngine:
        def predict(self, value, **kwargs):
            raise RuntimeError("broken")

    failed = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="broken"):
        training_data.prepare(source, benchmark, failed, engine=BrokenEngine())
    assert not failed.exists()


def test_split_is_reproducible_and_keeps_linked_samples_together():
    samples = [
        {"image": "a", "ground_truth": "FX00000000", "phash": [0]},
        {"image": "b", "ground_truth": "FX00000000", "phash": [255]},
        {"image": "c", "ground_truth": "FX00000001", "phash": [65535]},
        {"image": "d", "ground_truth": "FX00000002", "phash": [2 ** 63 - 1]},
    ]
    train, val = training_data.split_groups(samples, 20260917)
    assert ("a" in train) == ("b" in train)
    assert (train, val) == training_data.split_groups(samples, 20260917)
