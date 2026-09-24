import json
import random
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

CUR_DIR = Path(__file__).resolve().parent
if str(CUR_DIR) not in sys.path:
    sys.path.insert(0, str(CUR_DIR))

from create_recognition_dataset_v3 import (
    PHashIndex,
    load_latest_state,
    parse_agy_output,
    route_decision,
    run_agy,
    valid_label,
)
from finalize_dataset_v3 import choose_validation_groups
from review_tool_v2 import resolve_source_path
from training import random_character_spacing


def test_agy_parser_validation_and_two_stage_routing():
    response = {"status": "SUCCESS", "response": json.dumps({
        "results": [{"id": "a", "label": " fx37-407344 ", "readable": True, "alternatives": []}]
    })}
    assert parse_agy_output(json.dumps(response))[0]["label"] == "FX37407344"
    assert valid_label("FX37407344")
    assert not valid_label("FX3740734")
    assert route_decision("FX37407344", {"label": "FX37407344", "readable": True})[0] == "AUTO_ACCEPTED"
    assert route_decision("FX37407344", {"label": "FX37407345", "readable": True})[0] == "SECOND_PASS"
    assert route_decision(
        "FX37407344", {"label": "FX37407345", "readable": True},
        {"label": "FX37407345", "readable": True},
    )[:2] == ("AUTO_ACCEPTED", "FX37407345")


def test_agy_retries_and_never_invokes_a_real_process(tmp_path):
    image = tmp_path / "crop.jpg"
    image.write_bytes(b"not-decoded-by-the-fake-runner")
    calls = []

    def fake_runner(command, **kwargs):
        calls.append((command, kwargs))
        if len(calls) < 3:
            return subprocess.CompletedProcess(command, 1, "", "temporary")
        payload = {"results": [{"id": "crop", "label": "SJJA123456", "readable": True,
                                "alternatives": []}]}
        return subprocess.CompletedProcess(command, 0, json.dumps({"status": "SUCCESS", "response": payload}), "")

    delays = []
    result = run_agy([("crop", image)], tmp_path / "staging", "agy-test", fake_runner, delays.append)
    assert result["crop"]["label"] == "SJJA123456"
    assert delays == [2, 5]
    assert all("--sandbox" in command and "--dangerously-skip-permissions" not in command
               for command, _ in calls)


def test_resume_phash_index_and_safe_source_lookup(tmp_path):
    state = tmp_path / "state.jsonl"
    state.write_text('{"record_key":"a","status":"PENDING"}\n'
                     '{"record_key":"a","status":"AUTO_ACCEPTED"}\n{"broken"', encoding="utf-8")
    assert load_latest_state(state)["a"]["status"] == "AUTO_ACCEPTED"

    index = PHashIndex()
    index.add([0, 0, 0, 0], "first")
    assert index.near([1, 1, 1, 1]) == ("first", 1)
    assert index.near([(1 << 20) - 1] * 4) is None

    source = tmp_path / "source"
    (source / "train/images").mkdir(parents=True)
    expected = source / "train/images/seal.jpg"
    expected.write_bytes(b"image")
    assert resolve_source_path(source, "train/images/seal.jpg") == expected.resolve()
    assert resolve_source_path(source, "seal.jpg") == expected.resolve()
    assert resolve_source_path(source, "../seal.jpg") is None


def test_exact_group_split_and_spacing_transform(monkeypatch):
    groups = [[0, 1], [2], [3, 4, 5], [6]]
    selected = choose_validation_groups(groups, 3, seed=7)
    assert len(selected) == 3
    assert all(not (set(group) & selected) or set(group) <= selected for group in groups)

    monkeypatch.setattr(
        random_character_spacing,
        "tia_stretch",
        lambda image, segments: cv2.resize(image, (image.shape[1] + segments, image.shape[0])),
    )
    transform = random_character_spacing.RandomCharacterSpacing(prob=1.0)
    original = np.zeros((24, 60, 3), dtype=np.uint8)
    random.seed(11)
    first = transform({"image": original.copy(), "label": "SJJA123456"})
    random.seed(11)
    second = transform({"image": original.copy(), "label": "SJJA123456"})
    assert first["label"] == "SJJA123456"
    assert first["image"].shape[0] == original.shape[0]
    assert np.array_equal(first["image"], second["image"])


def test_batch_labeling_export_and_evaluate_calibration(tmp_path):
    from batch_labeling_v3 import evaluate_calibration, export_calibration, parse_labeling_json

    # Setup fake v2 dataset
    v2_dir = tmp_path / "v2"
    (v2_dir / "images").mkdir(parents=True)
    manifest = v2_dir / "manifest.csv"
    manifest_rows = ["image,ground_truth,source_image,split,sha256,original_crop\n"]
    for i in range(10):
        img_name = f"images/crop_{i:04d}.jpg"
        (v2_dir / img_name).write_bytes(b"dummy")
        sha = f"sha_{i:04d}"
        manifest_rows.append(f"{img_name},FX3740{i:04d},src_{i}.jpg,train,{sha},{img_name}\n")
    manifest.write_text("".join(manifest_rows), encoding="utf-8")

    work_dir = tmp_path / "work"
    export_res = export_calibration(v2_dir, work_dir, sample_count=10, seed=42)
    assert export_res["status"] == "CALIBRATION_EXPORTED"
    calib_dir = work_dir / "calibration"
    assert (calib_dir / "batches" / "smoke_5" / "manifest.csv").is_file()

    # Simulate response for smoke_5
    resp_payload = {
        "results": [
            {"id": f"sha_{i:04d}", "label": f"FX3740{i:04d}", "readable": True, "alternatives": []}
            for i in range(5)
        ]
    }
    (calib_dir / "responses" / "smoke_5.json").write_text(json.dumps(resp_payload), encoding="utf-8")

    eval_res = evaluate_calibration(calib_dir)
    assert eval_res["status"] == "PASS"
    assert eval_res["precision"] == 1.0
    assert eval_res["autoAcceptRate"] == 1.0
    assert (calib_dir / "v2_calibration_report.json").is_file()
    assert (calib_dir / "v2_calibration_report.md").is_file()


def test_batch_labeling_import_and_label(tmp_path, monkeypatch):
    import subprocess
    from batch_labeling_v3 import import_batch, label_batch

    work_dir = tmp_path / "work"
    work_dir.mkdir(parents=True)
    batch_dir = work_dir / "candidate_batches" / "batch_001"
    (batch_dir / "images").mkdir(parents=True)

    manifest_lines = [
        "id,filename,crop_path,source_image,local_v2,rotation\n",
        "sha_001,sha_001.jpg,images/c1.jpg,src1.jpg,FX37401111,0\n",
        "sha_002,sha_002.jpg,images/c2.jpg,src2.jpg,SJJA701234,0\n",
    ]
    (batch_dir / "manifest.csv").write_text("".join(manifest_lines), encoding="utf-8")
    (batch_dir / "images" / "sha_001.jpg").write_bytes(b"dummy1")
    (batch_dir / "images" / "sha_002.jpg").write_bytes(b"dummy2")

    # Mock subprocess.run for label_batch
    def fake_run(*args, **kwargs):
        payload = {
            "results": [
                {"id": "sha_001", "label": "FX37401111", "readable": True, "alternatives": []},
                {"id": "sha_002", "label": "SJJA709999", "readable": True, "alternatives": []},
            ]
        }
        return subprocess.CompletedProcess(
            args=args[0], returncode=0, stdout=json.dumps(payload), stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    # Initialize state
    state_file = work_dir / ".pipeline_state.jsonl"
    for r_id, loc in [("sha_001", "FX37401111"), ("sha_002", "SJJA701234")]:
        state_file.open("a", encoding="utf-8").write(
            json.dumps({
                "record_key": r_id, "crop_sha256": r_id, "source_image": f"{r_id}.jpg",
                "source_sha256": f"src_{r_id}", "image": f"images/{r_id}.jpg",
                "local_v2": loc, "status": "PENDING", "reason": "AWAITING_LABEL",
            }) + "\n"
        )

    lbl_res = label_batch(batch_dir)
    assert lbl_res["count"] == 2
    assert (batch_dir / "response.json").is_file()

    imp_res = import_batch(batch_dir, batch_dir / "response.json", work_dir)
    assert imp_res["autoAccepted"] == 1  # sha_001 agreed
    assert imp_res["needsReview"] == 1   # sha_002 disagreed
    assert imp_res["currentQuotas"]["hard"] == 1

