"""Regressions for the 3e8c69c review.

Fixture weights and 28px tensors are not official DINOv2 and not an MRI pilot.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from rsna_knee.cli import experiment_run
from rsna_knee.data.cache import write_shard
from rsna_knee.data.cache_build import build_cache
from rsna_knee.errors import LeakageError
from rsna_knee.hashing import sha256_file, sha256_json
from rsna_knee.labels.canonical import publish_training_labels
from rsna_knee.models.dinov2 import Dinov2ViTS14
from rsna_knee.models.study import StudyModel
from rsna_knee.ontology import TARGET_COLUMNS
from rsna_knee.pipeline import resolve_train_config, stage_labels, stage_runtime, stage_train
from rsna_knee.roots import roots_for
from rsna_knee.runtime.jobs import build_job, job_spec, refresh_spec, resolve_checkpoint_out, validate_job
from rsna_knee.runtime.train_payload import assert_split_disjoint, load_label_table, prepare_training
from rsna_knee.runtime.transport import FileTransport
from rsna_knee.runtime.worker import load_job_bundle, run_job
from rsna_knee.submission.package import _checkpoint_errors, package_run
from rsna_knee.training.checkpoint import torch_save
from rsna_knee.training.loop import train_study_model
from rsna_knee.workflow.graph import sequential_run
from rsna_knee.workflow.locks import ControllerLock
from rsna_knee.workflow.registry import Registry
from rsna_knee.workflow.state import PipelineState, Stage


def _model() -> StudyModel:
    torch.manual_seed(0)
    return StudyModel(Dinov2ViTS14(img_size=28), freeze_encoder=True)


def _batch(n: int, *, mask: float = 1.0, seed: int = 1) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    y = torch.zeros(n, 12)
    y[:, 0] = torch.tensor([float(i % 2) for i in range(n)])
    return {
        "images": torch.randn(n, 1, 1, 3, 28, 28, generator=gen),
        "slot_mask": torch.ones(n, 1),
        "center_mask": torch.ones(n, 1, 1),
        "y": y,
        "y_mask": torch.full((n, 12), mask),
        "y_weight": torch.full((n, 12), mask),
    }


def _envelope(uid: str, *, positive: bool) -> dict:
    targets = {}
    for name in TARGET_COLUMNS:
        if positive and name == "ACL":
            targets[name] = {"state": "criterion_positive", "evidence_span": "ACL tear", "criterion_mapping": "high-grade ACL"}
        else:
            targets[name] = {"state": "not_mentioned", "evidence_span": "", "criterion_mapping": "unmentioned"}
    return {"status": "ok", "payload": {"StudyInstanceUID": uid, "targets": targets}}


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _shards(root: Path, uids: list[str]) -> Path:
    studies = {}
    for i, uid in enumerate(uids):
        volume = np.random.default_rng(i).random((1, 1, 3, 28, 28)).astype(np.float32)
        meta = {"uid": uid, "shape": [1, 1, 3, 28, 28], "dtype": "uint8", "slot_mask": [1.0], "synthetic": False}
        shard = write_shard(uid, volume, meta, root=root, dtype="uint8")
        meta["sha256"] = sha256_file(shard)
        studies[uid] = meta
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps({"scope": "pilot", "studies": studies, "n_studies": len(uids), "n_requested": len(uids), "requested_uids": uids}),
        encoding="utf-8",
    )
    return manifest


def _weights(path: Path, allow: Path) -> None:
    torch.manual_seed(0)
    torch.save(Dinov2ViTS14(img_size=28).state_dict(), path)
    allow.write_text(json.dumps({"sha256": [sha256_file(path)]}), encoding="utf-8")


def _folds(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["StudyInstanceUID", "fold", "is_gold", "gold_split", "group_id"])
        writer.writeheader()
        writer.writerows(rows)


def test_envelope_uses_states_and_gold_overrides(tmp_path):
    uid = "1.2.20"
    raw = tmp_path / "raw.jsonl"
    _write_rows(raw, [_envelope(uid, positive=True), {"uid": "other", "payload": {"StudyInstanceUID": uid}}])
    gold = tmp_path / "train.csv"
    header = ["StudyInstanceUID", *TARGET_COLUMNS]
    with gold.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerow({"StudyInstanceUID": uid, "ACL": "-1", **{name: "" for name in TARGET_COLUMNS if name != "ACL"}})
    published = publish_training_labels([json.loads(line) for line in raw.read_text().splitlines()], tmp_path / "training.jsonl", train_csv=gold)
    assert published["rows"] == 1
    assert published["rejected"]
    table = load_label_table(raw, gold_csv=gold)
    assert uid in table
    acl = TARGET_COLUMNS.index("ACL")
    assert table[uid]["y_mask"][acl] == 0.0
    canonical = load_label_table(Path(published["path"]))
    assert canonical[uid]["y_mask"][acl] == 0.0
    assert canonical[uid]["source"] == "gold"
    positive = tmp_path / "gold1.csv"
    with positive.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerow({"StudyInstanceUID": uid, "ACL": "1", **{name: "" for name in TARGET_COLUMNS if name != "ACL"}})
    overridden = load_label_table(raw, gold_csv=positive)
    assert overridden[uid]["y"][acl] == 1.0
    assert overridden[uid]["y_weight"][acl] == 1.0


def test_stage_labels_format_reaches_the_worker(tmp_path, monkeypatch):
    monkeypatch.setattr("rsna_knee.pipeline.roots_for", lambda synthetic: roots_for(False))
    state = PipelineState(run_id="syn-format", synthetic=True, profile="smoke")
    out = stage_labels(state, Registry(tmp_path / "reg.sqlite"), live=False)
    raw = Path(out.artifacts["labels_raw"])
    first = json.loads(raw.read_text(encoding="utf-8").splitlines()[0])
    assert first["payload"]["targets"]["ACL"]["state"]
    table = load_label_table(raw)
    assert sum(sum(row["y_mask"]) for row in table.values()) > 0
    uid = "1.2.21"
    labels = tmp_path / "labels.jsonl"
    _write_rows(labels, [_envelope(uid, positive=True)])
    folds = tmp_path / "folds.csv"
    _folds(folds, [{"StudyInstanceUID": uid, "fold": "1", "is_gold": "0", "gold_split": "weak", "group_id": "g"}])
    manifest = _shards(tmp_path / "cache", [uid])
    weights = tmp_path / "weights.pt"
    allow = tmp_path / "allow.json"
    _weights(weights, allow)
    monkeypatch.setenv("RSNA_DINOV2_ALLOWLIST", str(allow))
    job = build_job(
        run_id="envelope-worker",
        stage="TRAINING",
        inputs={},
        dest=tmp_path / "job",
        synthetic=False,
        resources={"folds": str(folds), "labels": str(labels), "cache": str(manifest), "weights": str(weights)},
        train={"epochs": 1, "effective_batch": 1, "microbatch": 1, "seed": 0, "profile": "pilot"},
        checkpoint_out="checkpoint.pt",
    )
    rec = run_job(job, FileTransport(tmp_path / "tr"), expected_token=job["fencing_token"])
    assert rec.exit_reason == "ok"
    assert rec.metrics.weights_official is False
    assert float(rec.metrics.supervised_weight) > 0


def test_zero_supervision_is_not_a_successful_step(tmp_path, monkeypatch):
    model = _model()
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    trained = train_study_model(model, [_batch(2, mask=0.0)], epochs=1, effective_batch=2, microbatch=1, seed=0)
    assert trained["step"] == 0
    assert trained["supervised_weight"] == 0
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key])
    labels = tmp_path / "labels.jsonl"
    uid = "1.2.22"
    _write_rows(labels, [_envelope(uid, positive=False)])
    folds = tmp_path / "folds.csv"
    _folds(folds, [{"StudyInstanceUID": uid, "fold": "1", "is_gold": "0", "gold_split": "weak", "group_id": "g"}])
    manifest = _shards(tmp_path / "cache", [uid])
    weights = tmp_path / "weights.pt"
    allow = tmp_path / "allow.json"
    _weights(weights, allow)
    monkeypatch.setenv("RSNA_DINOV2_ALLOWLIST", str(allow))
    job = build_job(
        run_id="zero",
        stage="TRAINING",
        inputs={},
        dest=tmp_path / "job",
        synthetic=False,
        resources={"folds": str(folds), "labels": str(labels), "cache": str(manifest), "weights": str(weights)},
        train={"epochs": 1, "effective_batch": 1, "microbatch": 1, "seed": 0, "profile": "pilot"},
    )
    rec = run_job(job, FileTransport(tmp_path / "tr"), expected_token=job["fencing_token"])
    assert rec.exit_reason == "no_supervision"


def test_group_capacity_limits_take_and_matches_microbatch():
    batch = _batch(6, seed=3)
    trained_left = _model()
    trained_right = StudyModel(Dinov2ViTS14(img_size=28), freeze_encoder=True)
    trained_right.load_state_dict(trained_left.state_dict())
    wide = train_study_model(trained_left, [batch], epochs=1, effective_batch=4, microbatch=3, seed=4)
    fine = train_study_model(trained_right, [batch], epochs=1, effective_batch=4, microbatch=1, seed=4)
    assert wide["step"] == 2
    assert fine["step"] == 2
    assert wide["updates_planned"] == 2
    for key, value in trained_left.state_dict().items():
        assert torch.allclose(value, trained_right.state_dict()[key], atol=1e-5), key


def test_split_intersections_are_rejected():
    folds = [
        {"StudyInstanceUID": "1.2.30", "fold": "1", "group_id": "g", "gold_split": "weak"},
        {"StudyInstanceUID": "1.2.31", "fold": "0", "group_id": "g", "gold_split": "weak"},
        {"StudyInstanceUID": "1.2.32", "fold": "2", "group_id": "gold", "gold_split": "eval"},
    ]
    with pytest.raises(LeakageError, match="group_crosses_splits"):
        assert_split_disjoint(["1.2.30"], ["1.2.31"], [], folds)
    with pytest.raises(LeakageError, match="train_weak_overlap"):
        assert_split_disjoint(["1.2.30"], ["1.2.30"], [], folds)
    with pytest.raises(LeakageError, match="train_gold_overlap"):
        assert_split_disjoint(["1.2.32"], [], ["1.2.32"], folds)


def test_pixel_or_root_change_does_not_reuse_the_shard(tmp_path):
    from tests.test_cache_manifest import _write_slice

    study = "1.2.840.113619.2.9"
    sid = "1.2.840.113619.2.10"
    root = tmp_path / "dicom"
    folder = root / study / sid
    folder.mkdir(parents=True)
    for instance, z in ((1, 1.0), (2, 2.0), (3, 3.0)):
        _write_slice(folder / f"{instance}.dcm", study, sid, instance, z)
    rows = [{
        "StudyInstanceUID": study,
        "SeriesInstanceUID": sid,
        "Anatomical_Plane": "Sagittal",
        "Fluid_Sensitive": "1",
        "Fat_Suppression": "1",
    }]
    csv_path = tmp_path / "series.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    dest = tmp_path / "cache"
    first = build_cache(series_csv=csv_path, dicom_root=root, dest=dest, size=28, n_centers=1)
    first_sha = first["studies"][study]["sha256"]
    moved = tmp_path / "dicom-moved"
    shutil.copytree(root, moved)
    second = build_cache(series_csv=csv_path, dicom_root=moved, dest=dest, size=28, n_centers=1)
    assert second["studies"][study].get("resumed") is not True
    assert second["studies"][study]["source_hashes"]["dicom_root"] == str(moved.resolve())
    _write_slice(folder / "1.dcm", study, sid, 1, 1.0)
    # Pixel bytes differ from the first file even when the geometry matches.
    ds_path = folder / "1.dcm"
    import pydicom

    ds = pydicom.dcmread(ds_path)
    ds.PixelData = (np.zeros((8, 8), dtype=np.uint16) + 9).tobytes()
    ds.save_as(str(ds_path), enforce_file_format=True)
    third = build_cache(series_csv=csv_path, dicom_root=root, dest=dest, size=28, n_centers=1)
    assert third["studies"][study].get("resumed") is not True
    assert third["studies"][study]["sha256"] != first_sha


def test_train_loader_does_not_materialize_the_dataset(tmp_path, monkeypatch):
    uids = ["1.2.40", "1.2.41"]
    folds = tmp_path / "folds.csv"
    _folds(
        folds,
        [
            {"StudyInstanceUID": uids[0], "fold": "1", "is_gold": "0", "gold_split": "weak", "group_id": "a"},
            {"StudyInstanceUID": uids[1], "fold": "1", "is_gold": "0", "gold_split": "weak", "group_id": "b"},
        ],
    )
    labels = tmp_path / "labels.jsonl"
    _write_rows(labels, [_envelope(uid, positive=True) for uid in uids])
    manifest = _shards(tmp_path / "cache", uids)
    weights = tmp_path / "weights.pt"
    allow = tmp_path / "allow.json"
    _weights(weights, allow)
    monkeypatch.setenv("RSNA_DINOV2_ALLOWLIST", str(allow))
    calls = {"n": 0}
    real_load = np.load

    def counting_load(*args, **kwargs):
        calls["n"] += 1
        return real_load(*args, **kwargs)

    monkeypatch.setattr(np, "load", counting_load)
    job = build_job(
        run_id="lazy",
        stage="TRAINING",
        inputs={},
        dest=tmp_path / "job",
        synthetic=False,
        resources={"folds": str(folds), "labels": str(labels), "cache": str(manifest), "weights": str(weights)},
        train={"epochs": 1, "effective_batch": 1, "microbatch": 1, "seed": 0, "profile": "pilot"},
    )
    payload, reason = prepare_training(job, tmp_path / "job")
    assert reason is None and payload is not None
    assert calls["n"] == 0
    first = payload["batches"][0]["images"]
    second = payload["batches"][0]["images"]
    assert first.data_ptr() != second.data_ptr()
    assert calls["n"] == 2


def test_train_edit_breaks_the_spec_and_resume_covers_weights(tmp_path):
    job = build_job(
        run_id="spec",
        stage="TRAINING",
        inputs={},
        dest=tmp_path / "job",
        synthetic=False,
        train={"epochs": 2, "effective_batch": 8, "microbatch": 1, "seed": 2026, "profile": "pilot"},
        checkpoint_out="/Users/mac/does-not-exist/checkpoint.pt",
    )
    assert job["checkpoint_out"] == "checkpoint.pt"
    assert validate_job(job) == []
    edited = json.loads((tmp_path / "job" / "job.json").read_text(encoding="utf-8"))
    edited["train"]["epochs"] = 999
    assert "spec_hash" in validate_job(edited)
    assert sha256_json(job_spec(job)) == job["spec_sha256"]


def test_job_is_written_before_handoff_and_settings_come_from_config(tmp_path, monkeypatch):
    monkeypatch.setattr("rsna_knee.pipeline.write_handoff", lambda result, run_id: tmp_path / "handoff.md")
    state = PipelineState(run_id="before-handoff", synthetic=False, profile="pilot", fencing_token="token")
    out = stage_runtime(state, Registry(tmp_path / "reg.sqlite"))
    assert out.stage == Stage.NEEDS_RUNTIME
    job_path = Path(out.artifacts["job"])
    assert job_path.is_file()
    job = json.loads(job_path.read_text(encoding="utf-8"))
    expected = resolve_train_config("pilot")
    assert job["train"]["epochs"] == expected["epochs"]
    assert job["train"]["effective_batch"] == expected["effective_batch"]
    assert job["train"]["seed"] == expected["seed"]
    assert job["train"] != {"epochs": 1, "effective_batch": 1, "microbatch": 1, "seed": 0}
    cfg = tmp_path / "exp.yaml"
    cfg.write_text(
        "experiment_id: T\nstatus: defined_not_run\ntrain:\n  epochs: 3\n  effective_batch: 4\n  microbatch: 2\n  seed: 7\n",
        encoding="utf-8",
    )
    experiment_run(cfg)
    created = list(roots_for(False).runs.glob("*/job/job.json"))
    stored = next(json.loads(path.read_text(encoding="utf-8")) for path in created if json.loads(path.read_text(encoding="utf-8"))["train"]["epochs"] == 3)
    assert stored["train"]["epochs"] == 3
    assert stored["train"]["effective_batch"] == 4
    assert stored["train"]["seed"] == 7
    assert stored["resources"]["config"]["uri"].endswith("exp.yaml")


def test_campaign_cpu_does_not_train(tmp_path):
    uid = "1.2.50"
    folds = tmp_path / "folds.csv"
    _folds(folds, [{"StudyInstanceUID": uid, "fold": "1", "is_gold": "0", "gold_split": "weak", "group_id": "g"}])
    labels = tmp_path / "labels.jsonl"
    _write_rows(labels, [_envelope(uid, positive=True)])
    manifest = _shards(tmp_path / "cache", [uid])
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"not-opened")
    state = PipelineState(
        run_id="campaign-cpu",
        synthetic=False,
        profile="campaign",
        fencing_token="token",
        artifacts={"folds": str(folds), "labels": str(labels), "cache_manifest": str(manifest), "weights": str(weights)},
    )
    out = stage_train(state, Registry(tmp_path / "reg.sqlite"))
    assert out.stage == Stage.NEEDS_RUNTIME
    assert out.blocked_reason == "campaign_requires_gpu"
    assert Path(out.artifacts["job"]).is_file()
    assert "checkpoint" not in out.artifacts


def test_relocation_uses_input_root_and_not_a_mac_path(tmp_path, monkeypatch):
    root = tmp_path / "bundle"
    root.mkdir()
    uid = "1.2.60"
    folds = root / "folds.csv"
    _folds(folds, [{"StudyInstanceUID": uid, "fold": "1", "is_gold": "0", "gold_split": "weak", "group_id": "g"}])
    labels = root / "labels.jsonl"
    _write_rows(labels, [_envelope(uid, positive=True)])
    manifest = _shards(root / "cache", [uid])
    weights = root / "weights.pt"
    allow = root / "allow.json"
    _weights(weights, allow)
    build_job(
        run_id="move",
        stage="TRAINING",
        inputs={},
        dest=root / "job",
        synthetic=False,
        resources={"folds": str(folds), "labels": str(labels), "cache": str(manifest), "weights": str(weights)},
        input_root=root,
        train={"epochs": 1, "effective_batch": 1, "microbatch": 1, "seed": 0, "profile": "pilot"},
        checkpoint_out="/Users/mac/checkpoint.pt",
    )
    moved = tmp_path / "moved"
    shutil.copytree(root, moved)
    shutil.rmtree(root)
    monkeypatch.setenv("RSNA_INPUT_ROOT", str(moved))
    monkeypatch.setenv("RSNA_DINOV2_ALLOWLIST", str(moved / "allow.json"))
    loaded = load_job_bundle(moved / "job" / "job.json")
    assert validate_job(loaded) == []
    assert resolve_checkpoint_out(loaded, moved / "job") == moved / "checkpoint.pt"
    payload, reason = prepare_training(loaded, moved / "job")
    assert reason is None and payload is not None
    rec = run_job(loaded, FileTransport(tmp_path / "tr"), expected_token=loaded["fencing_token"])
    assert rec.exit_reason == "ok"
    assert Path(rec.checkpoint_uris[0]) == moved / "checkpoint.pt"
    assert not Path("/Users/mac/checkpoint.pt").exists()


def test_output_file_loss_reruns_the_stage(tmp_path):
    calls = []
    target = tmp_path / "marker.txt"

    def labels(state, registry):
        calls.append("labels")
        target.write_text("v1", encoding="utf-8")
        state.artifacts["marker"] = str(target)
        state.stage = Stage.LABELS_READY
        return state

    registry = Registry(tmp_path / "reg.sqlite")
    state = PipelineState(run_id="out", synthetic=False, artifacts={})
    sequential_run(state, registry, [("labels", labels)], lock=ControllerLock(tmp_path / "lock"))
    assert calls == ["labels"]
    target.unlink()
    again = PipelineState.model_validate(registry.get_run("out"))
    again.resume = True
    sequential_run(again, registry, [("labels", labels)], lock=ControllerLock(tmp_path / "lock2"))
    assert calls == ["labels", "labels"]


def test_empty_checkpoint_fails_strict_load_and_freeze_keeps_the_tar(tmp_path):
    empty = tmp_path / "empty.pt"
    torch_save(empty, {"encoder_name": "dinov2_vits14", "model": {}, "step": 1, "img_size": 28})
    assert "checkpoint_state_mismatch" in _checkpoint_errors(empty)
    real = tmp_path / "real.pt"
    model = _model()
    torch_save(real, {"encoder_name": "dinov2_vits14", "model": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "step": 1, "img_size": 28})
    first = package_run("freeze-me", dest_root=tmp_path / "pkg", synthetic=True, checkpoint=real)
    assert "offline_manifest_missing" not in first["identity_errors"]
    frozen = Path(first["dir"]) / "FROZEN"
    frozen.write_text(first["sha256"], encoding="utf-8")
    (Path(first["dir"]) / "extra.txt").write_text("should not be packed", encoding="utf-8")
    second = package_run("freeze-me", dest_root=tmp_path / "pkg", synthetic=True, checkpoint=real)
    assert second["frozen"] is True
    assert second["sha256"] == first["sha256"]


def test_cache_kernel_bootstraps_and_limits_the_pilot(tmp_path):
    from rsna_knee.runtime.kaggle_kernel import prepare_cache_kernel

    prepared = prepare_cache_kernel(tmp_path / "kernel")
    text = (tmp_path / "kernel" / "cache_entry.py").read_text(encoding="utf-8")
    assert "bootstrap.py" in text
    assert "RSNA_CACHE_LIMIT" in text
    assert "limit=int(limit)" in text
    assert "else None" not in text
    assert prepared["executed"] is False


def test_runbook_order_and_connected_example_pilot(tmp_path, monkeypatch):
    runbook = Path("docs/USER_RUNBOOK_TR.md").read_text(encoding="utf-8")
    section = runbook.split("Küçük gerçek pilot", 1)[1]
    positions = [section.index(token) for token in ("rsna doctor", "rsna labels pilot", "rsna cache build", "rsna pipeline run", "rsna worker", "rsna evaluate", "rsna package", "rsna audit")]
    assert positions == sorted(positions)
    root = tmp_path / "pilot"
    root.mkdir()
    train_uid, weak_uid, gold_uid = "1.2.70", "1.2.71", "1.2.72"
    _folds(
        root / "folds.csv",
        [
            {"StudyInstanceUID": train_uid, "fold": "1", "is_gold": "0", "gold_split": "weak", "group_id": "train"},
            {"StudyInstanceUID": weak_uid, "fold": "0", "is_gold": "0", "gold_split": "weak", "group_id": "weak"},
            {"StudyInstanceUID": gold_uid, "fold": "2", "is_gold": "1", "gold_split": "eval", "group_id": "gold"},
        ],
    )
    envelopes = [_envelope(train_uid, positive=True), _envelope(weak_uid, positive=True), _envelope(gold_uid, positive=False)]
    _write_rows(root / "raw.jsonl", envelopes)
    header = ["StudyInstanceUID", *TARGET_COLUMNS]
    gold_csv = root / "train.csv"
    with gold_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerow({"StudyInstanceUID": train_uid, "ACL": "1", **{name: "" for name in TARGET_COLUMNS if name != "ACL"}})
    published = publish_training_labels(envelopes, root / "labels.jsonl", train_csv=gold_csv)
    assert published["supervised_targets"] > 0
    manifest = _shards(root / "cache", [train_uid, weak_uid, gold_uid])
    weights = root / "weights.pt"
    allow = root / "allow.json"
    _weights(weights, allow)
    train = resolve_train_config("pilot")
    train["interrupt_after_steps"] = 1
    train["img_size"] = 28
    job = build_job(
        run_id="example-pilot",
        stage="TRAINING",
        inputs={},
        dest=root / "job",
        synthetic=False,
        resources={
            "folds": str(root / "folds.csv"),
            "labels": str(root / "labels.jsonl"),
            "cache": str(manifest),
            "weights": str(weights),
            "train_csv": str(gold_csv),
        },
        input_root=root,
        train=train,
        checkpoint_out="checkpoint.pt",
    )
    moved = tmp_path / "relocated"
    shutil.copytree(root, moved)
    shutil.rmtree(root)
    monkeypatch.setenv("RSNA_INPUT_ROOT", str(moved))
    monkeypatch.setenv("RSNA_DINOV2_ALLOWLIST", str(moved / "allow.json"))
    monkeypatch.setenv("RSNA_ROOTS", str(tmp_path / "roots"))
    monkeypatch.setenv("RSNA_REGISTRY", str(tmp_path / "roots" / "registry.sqlite"))
    rsna = Path(sys.executable).with_name("rsna")
    env = os.environ.copy()
    doctor = subprocess.run([str(rsna), "doctor"], env=env, capture_output=True, text=True, check=False, timeout=90)
    assert doctor.returncode == 0
    job_path = moved / "job" / "job.json"
    first = subprocess.run([str(rsna), "worker", "--job-bundle", str(job_path)], env=env, capture_output=True, text=True, check=False)
    assert first.returncode == 0, first.stderr
    loaded = json.loads(job_path.read_text(encoding="utf-8"))
    loaded["train"]["resume"] = True
    loaded["train"]["phase"] = "resume"
    loaded["train"].pop("interrupt_after_steps", None)
    manifest_path = job_path.parent / "input_manifest.json"
    manifest_body = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_body["train"] = loaded["train"]
    manifest_path.write_text(json.dumps(manifest_body, indent=2), encoding="utf-8")
    loaded["bundle_sha256"] = sha256_file(manifest_path)
    refresh_spec(loaded)
    job_path.write_text(json.dumps(loaded, indent=2), encoding="utf-8")
    second = subprocess.run([str(rsna), "worker", "--job-bundle", str(job_path)], env=env, capture_output=True, text=True, check=False)
    assert second.returncode == 0, second.stderr
    evaluate = subprocess.run([str(rsna), "evaluate", "--run-id", "example-pilot"], env=env, capture_output=True, text=True, check=False)
    assert evaluate.returncode == 0, evaluate.stderr
    assert "evaluate_requires_saved_checkpoint" not in evaluate.stdout
    packaged = subprocess.run([str(rsna), "package", "--run-id", "example-pilot"], env=env, capture_output=True, text=True, check=False)
    assert packaged.returncode == 0, packaged.stderr
    audited = subprocess.run([str(rsna), "audit", "--run-id", "example-pilot"], env=env, capture_output=True, text=True, check=False)
    assert audited.returncode == 0, audited.stderr
    run = Registry().get_run("example-pilot")
    assert run is not None
    assert run["run_id"] == "example-pilot"
    checkpoint = Path(run["artifacts"]["checkpoint"])
    assert checkpoint.is_file()
    assert str(moved) in str(checkpoint)
    assert sha256_file(checkpoint) == run["hashes"]["train"]
    worker_metrics = json.loads((moved / "job" / "metrics.json").read_text(encoding="utf-8"))
    assert worker_metrics["weights_official"] is False
    assert float(worker_metrics["supervised_weight"]) > 0
    assert train_uid in worker_metrics["train_uids"]
    assert gold_uid not in worker_metrics["train_uids"]
    assert weak_uid not in worker_metrics["train_uids"]
    payload, reason = prepare_training(load_job_bundle(job_path), job_path.parent)
    assert reason is None and payload is not None
    assert set(payload["train_uids"]).isdisjoint(payload["weak_uids"])
    assert set(payload["train_uids"]).isdisjoint(payload["gold_uids"])
    assert train_uid in payload["train_uids"]
    assert gold_uid not in payload["train_uids"]
    audit = json.loads(Path(run["artifacts"]["audit"]).read_text(encoding="utf-8"))
    assert audit["run_id"] == "example-pilot"
    assert audit["checkpoint_sha256"] == run["hashes"]["train"]
    assert audit["package_sha256"]
    assert "revalidated" in audited.stdout or "example-pilot" in audited.stdout
    assert job["run_id"] == loaded["run_id"] == "example-pilot"
