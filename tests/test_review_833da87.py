"""Regressions from the 833da87 review. Example tensors are not an MRI pilot."""

from __future__ import annotations

import csv
import json
import subprocess
import sys

import numpy as np
import torch

from rsna_knee.data.cache import write_shard
from rsna_knee.data.cache_build import build_cache, manifest_ready
from rsna_knee.hashing import sha256_file
from rsna_knee.models.dinov2 import Dinov2ViTS14
from rsna_knee.models.study import StudyModel
from rsna_knee.ontology import SERIES_ID_COL, STUDY_ID_COL, TARGET_COLUMNS
from rsna_knee.pipeline import stage_cache, stage_package
from rsna_knee.runtime.jobs import build_job
from rsna_knee.runtime.transport import FileTransport
from rsna_knee.runtime.worker import run_job
from rsna_knee.training.checkpoint import torch_save
from rsna_knee.training.loop import train_study_model
from rsna_knee.workflow.graph import sequential_run
from rsna_knee.workflow.locks import ControllerLock
from rsna_knee.workflow.registry import Registry
from rsna_knee.workflow.state import PipelineState, Stage


def _model() -> StudyModel:
    torch.manual_seed(0)
    return StudyModel(Dinov2ViTS14(img_size=28), freeze_encoder=True)


def _batch(n: int, *, seed: int = 1) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    y = torch.zeros(n, 12)
    y[:, 0] = torch.tensor([float(i % 2) for i in range(n)])
    return {
        "images": torch.randn(n, 1, 1, 3, 28, 28, generator=gen),
        "slot_mask": torch.ones(n, 1),
        "center_mask": torch.ones(n, 1, 1),
        "y": y,
        "y_mask": torch.ones(n, 12),
        "y_weight": torch.ones(n, 12),
    }


def _copy(model: StudyModel) -> StudyModel:
    fresh = _model()
    fresh.load_state_dict(model.state_dict())
    return fresh


def test_partial_group_still_updates(tmp_path):
    model = _model()
    before = model.head.weight.detach().clone()
    out = train_study_model(
        model,
        [_batch(1)],
        epochs=1,
        effective_batch=4,
        microbatch=1,
        seed=0,
        ckpt_path=tmp_path / "partial.pt",
    )
    assert out["step"] == 1
    assert not torch.allclose(model.head.weight.detach(), before)


def test_microbatch_slices_the_study_axis():
    model = _model()
    seen: list[int] = []
    original = model.forward

    def wrapped(images, slot, center):
        seen.append(int(images.shape[0]))
        return original(images, slot, center)

    model.forward = wrapped
    train_study_model(model, [_batch(3)], epochs=1, effective_batch=3, microbatch=1, seed=0)
    assert seen
    assert max(seen) == 1


def test_oom_retry_shrinks_the_forward(tmp_path):
    model = _model()
    seen: list[int] = []
    original = model.forward

    def wrapped(images, slot, center):
        seen.append(int(images.shape[0]))
        return original(images, slot, center)

    model.forward = wrapped
    out = train_study_model(
        model,
        [_batch(4)],
        epochs=1,
        effective_batch=4,
        microbatch=4,
        seed=0,
        oob_once=True,
        ckpt_path=tmp_path / "oom.pt",
    )
    assert out["step"] == 1
    assert out["microbatch"] == 2
    assert seen == [2, 2]
    assert out["img_size"] == 28


def test_unfreeze_resume_matches_with_a_smaller_microbatch(tmp_path):
    batches = [_batch(4)]
    base = _model()
    full = _copy(base)
    train_study_model(
        full,
        batches,
        epochs=1,
        effective_batch=2,
        microbatch=2,
        seed=0,
        unfreeze_after_steps=1,
        ckpt_path=tmp_path / "full.pt",
    )
    part = _copy(base)
    train_study_model(
        part,
        batches,
        epochs=1,
        effective_batch=2,
        microbatch=2,
        seed=0,
        unfreeze_after_steps=1,
        interrupt_after_steps=1,
        ckpt_path=tmp_path / "part.pt",
    )
    resumed = _model()
    train_study_model(
        resumed,
        batches,
        epochs=1,
        effective_batch=2,
        microbatch=1,
        seed=0,
        unfreeze_after_steps=1,
        ckpt_path=tmp_path / "part.pt",
        resume=True,
    )
    assert any(p.requires_grad for p in resumed.encoder.blocks[-1].parameters())
    for key, value in full.state_dict().items():
        assert torch.allclose(value, resumed.state_dict()[key], atol=1e-5), key


def test_cache_resume_rejects_a_different_shape(tmp_path):
    from tests.test_cache_manifest import _write_slice

    study = "1.2.840.113619.2.1"
    sid = "1.2.840.113619.2.2"
    folder = tmp_path / "dicom" / study / sid
    folder.mkdir(parents=True)
    for instance, z in ((1, 1.0), (2, 2.0), (3, 3.0)):
        _write_slice(folder / f"{instance}.dcm", study, sid, instance, z)
    rows = [
        {
            STUDY_ID_COL: study,
            SERIES_ID_COL: sid,
            "Anatomical_Plane": "Sagittal",
            "Fluid_Sensitive": "1",
            "Fat_Suppression": "1",
        }
    ]
    csv_path = tmp_path / "series.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    first = build_cache(series_csv=csv_path, dicom_root=tmp_path / "dicom", dest=tmp_path / "cache", size=28, n_centers=3)
    assert first["studies"][study]["shape"] == [3, 3, 3, 28, 28]
    second = build_cache(
        series_csv=csv_path,
        dicom_root=tmp_path / "dicom",
        dest=tmp_path / "cache",
        size=42,
        n_centers=1,
        limit=1,
        resume=True,
    )
    assert second["studies"][study]["shape"] == [3, 1, 3, 42, 42]
    assert second["studies"][study]["preprocess_hash"] == second["preprocess_hash"]
    assert second["scope"] == "pilot"


def test_manifest_without_shard_is_not_cache_ready(tmp_path):
    root = tmp_path / "cache"
    root.mkdir()
    manifest = {
        "scope": "pilot",
        "studies": {"1.2.3": {"sha256": "abc", "shape": [3, 3, 3, 28, 28]}},
        "n_studies": 1,
        "n_requested": 1,
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    ready, reason = manifest_ready(path)
    assert ready is False
    assert reason == "shard_missing"
    state = PipelineState(run_id="cache", synthetic=False, artifacts={})
    from rsna_knee.roots import roots_for

    real_cache = roots_for(False).cache
    real_cache.mkdir(parents=True, exist_ok=True)
    (real_cache / "manifest.json").write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    out = stage_cache(state, Registry(tmp_path / "reg.sqlite"))
    assert out.stage == Stage.BLOCKED
    assert out.blocked_reason == "shard_missing"


def _labels(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _pilot_files(tmp_path, *, allow: bool):
    uids = ["1.2.10", "1.2.11", "1.2.12", "1.2.13"]
    folds = tmp_path / "folds.csv"
    with folds.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["StudyInstanceUID", "fold", "is_gold", "gold_split", "group_id"])
        writer.writeheader()
        writer.writerows(
            [
                {"StudyInstanceUID": "1.2.10", "fold": "1", "is_gold": "0", "gold_split": "train", "group_id": "g10"},
                {"StudyInstanceUID": "1.2.11", "fold": "1", "is_gold": "0", "gold_split": "train", "group_id": "g11"},
                {"StudyInstanceUID": "1.2.12", "fold": "0", "is_gold": "0", "gold_split": "train", "group_id": "g12"},
                {"StudyInstanceUID": "1.2.13", "fold": "0", "is_gold": "1", "gold_split": "eval", "group_id": "g13"},
            ]
        )
    labels = tmp_path / "labels.jsonl"
    _labels(
        labels,
        [
            {
                "StudyInstanceUID": uid,
                "source": "gold" if uid == "1.2.13" else "weak",
                "targets": {name: float(i % 2) if name == "ACL" else 0.0 for name in TARGET_COLUMNS},
            }
            for i, uid in enumerate(uids)
        ],
    )
    cache = tmp_path / "cache"
    studies = {}
    for i, uid in enumerate(uids):
        volume = np.random.default_rng(i).random((1, 1, 3, 28, 28)).astype(np.float32)
        meta = {"uid": uid, "shape": [1, 1, 3, 28, 28], "dtype": "uint8", "slot_mask": [1.0], "synthetic": False}
        shard = write_shard(uid, volume, meta, root=cache, dtype="uint8")
        meta["sha256"] = sha256_file(shard)
        meta["shape"] = [1, 1, 3, 28, 28]
        studies[uid] = meta
    manifest = cache / "manifest.json"
    manifest.write_text(
        json.dumps({"scope": "pilot", "studies": studies, "n_studies": 4, "n_requested": 4, "requested_uids": uids}),
        encoding="utf-8",
    )
    torch.manual_seed(0)
    encoder = Dinov2ViTS14(img_size=28)
    weights = tmp_path / "weights.pt"
    torch.save(encoder.state_dict(), weights)
    allowlist = tmp_path / "allow.json"
    allowlist.write_text(json.dumps({"sha256": [sha256_file(weights)] if allow else []}), encoding="utf-8")
    return folds, labels, manifest, weights, allowlist


def test_worker_rejects_dummy_checkpoint_and_tiny_encoder(tmp_path, monkeypatch):
    monkeypatch.delenv("RSNA_DINOV2_ALLOWLIST", raising=False)
    dummy = tmp_path / "dummy.pt"
    dummy.write_text("not a model", encoding="utf-8")
    job = build_job(run_id="dummy", stage="TRAINING", inputs={"config": "x"}, dest=tmp_path / "job", synthetic=False)
    job["checkpoint"] = str(dummy)
    rec = run_job(job, FileTransport(tmp_path / "tr"), items=[{"uid": "nope"}], expected_token=job["fencing_token"])
    assert rec.exit_reason == "rejected"
    assert rec.metrics.kind == "unavailable"
    tiny = dict(job)
    tiny["encoder"] = "tiny_test_encoder"
    tiny["synthetic"] = False
    rec_tiny = run_job(tiny, FileTransport(tmp_path / "tr2"), expected_token=tiny["fencing_token"])
    assert rec_tiny.exit_reason == "rejected_encoder"


def test_worker_trains_from_files_and_writes_heldout(tmp_path, monkeypatch):
    folds, labels, manifest, weights, allowlist = _pilot_files(tmp_path, allow=True)
    monkeypatch.setenv("RSNA_DINOV2_ALLOWLIST", str(allowlist))
    dest = tmp_path / "bundle"
    job = build_job(
        run_id="pilot-example",
        stage="TRAINING",
        inputs={},
        dest=dest,
        synthetic=False,
        resources={"folds": str(folds), "labels": str(labels), "cache": str(manifest), "weights": str(weights)},
        checkpoint_out=str(dest / "checkpoint.pt"),
        train={"epochs": 1, "effective_batch": 1, "microbatch": 1, "seed": 0},
    )
    leaked = dict(job)
    leaked["train_uids"] = ["1.2.10", "1.2.13"]
    leaked_rec = run_job(leaked, FileTransport(tmp_path / "leak"), expected_token=job["fencing_token"])
    assert leaked_rec.exit_reason == "leakage"
    rec = run_job(job, FileTransport(tmp_path / "ok"), expected_token=job["fencing_token"])
    assert rec.exit_reason == "ok"
    assert rec.metrics.kind == "real"
    assert rec.metrics.weights_official is False
    assert int(rec.metrics.steps) >= 1
    ckpt = dest / "checkpoint.pt"
    assert ckpt.is_file()
    preds = dest / "predictions.jsonl"
    assert preds.is_file()
    rows = [json.loads(line) for line in preds.read_text(encoding="utf-8").splitlines()]
    splits = {row["split"] for row in rows}
    assert "gold" in splits
    assert "weak" in splits
    assert "1.2.13" in {row["StudyInstanceUID"] for row in rows if row["split"] == "gold"}
    assert "1.2.10" not in {row["StudyInstanceUID"] for row in rows}


def test_unverified_weights_do_not_count_as_pretrained(tmp_path, monkeypatch):
    folds, labels, manifest, weights, allowlist = _pilot_files(tmp_path, allow=False)
    monkeypatch.setenv("RSNA_DINOV2_ALLOWLIST", str(allowlist))
    dest = tmp_path / "bundle"
    job = build_job(
        run_id="random",
        stage="TRAINING",
        inputs={},
        dest=dest,
        synthetic=False,
        resources={"folds": str(folds), "labels": str(labels), "cache": str(manifest), "weights": str(weights)},
        checkpoint_out=str(dest / "checkpoint.pt"),
    )
    rec = run_job(job, FileTransport(tmp_path / "tr"), expected_token=job["fencing_token"])
    assert rec.exit_reason == "unverified_weights"
    assert not (dest / "checkpoint.pt").exists()


def test_pause_is_consumed_and_resume_continues(tmp_path):
    calls = []

    def cache(state, registry):
        calls.append("cache")
        state.stage = Stage.CACHE_READY
        state.artifacts["marker"] = "kept"
        return state

    registry = Registry(tmp_path / "reg.sqlite")
    state = PipelineState(run_id="pause-run", synthetic=False, hashes={"config": "c"}, artifacts={"marker": "kept"})
    from rsna_knee.roots import roots_for

    path = roots_for(False).control
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"run_id": "pause-run", "command": "pause"}), encoding="utf-8")
    lock = ControllerLock(tmp_path / "lock")
    first = sequential_run(state, registry, [("cache", cache)], lock=lock)
    assert first.stage == Stage.PAUSED
    assert json.loads(path.read_text(encoding="utf-8"))["consumed"] is True
    assert calls == []
    second = PipelineState.model_validate(registry.get_run("pause-run"))
    second.resume = True
    out = sequential_run(second, registry, [("cache", cache)], lock=ControllerLock(tmp_path / "lock2"))
    assert calls == ["cache"]
    assert out.stage == Stage.CACHE_READY
    assert out.artifacts["marker"] == "kept"


def test_resume_sees_a_changed_artifact_file(tmp_path):
    calls = []

    def labels(state, registry):
        calls.append("labels")
        state.stage = Stage.LABELS_READY
        return state

    registry = Registry(tmp_path / "reg.sqlite")
    folds = tmp_path / "folds.csv"
    folds.write_text("StudyInstanceUID,fold\n1,0\n", encoding="utf-8")
    state = PipelineState(
        run_id="hash-run",
        synthetic=False,
        hashes={"folds": "stored", "config": "c"},
        artifacts={"folds": str(folds)},
    )
    sequential_run(state, registry, [("labels", labels)], lock=ControllerLock(tmp_path / "lock"))
    assert calls == ["labels"]
    calls.clear()
    again = PipelineState.model_validate(registry.get_run("hash-run"))
    again.resume = True
    sequential_run(again, registry, [("labels", labels)], lock=ControllerLock(tmp_path / "lockb"))
    assert calls == []
    folds.write_text("StudyInstanceUID,fold\n1,1\n", encoding="utf-8")
    changed = PipelineState.model_validate(registry.get_run("hash-run"))
    changed.resume = True
    changed.artifacts["folds"] = str(folds)
    sequential_run(changed, registry, [("labels", labels)], lock=ControllerLock(tmp_path / "lockc"))
    assert calls == ["labels"]


def test_package_and_plain_checkpoint(tmp_path):
    from rsna_knee.submission.package import package_run

    text = tmp_path / "notes.txt"
    text.write_text("not-a-weight-file", encoding="utf-8")
    plain = package_run("plain", dest_root=tmp_path / "out", synthetic=False, checkpoint=text)
    assert "checkpoint_unreadable" in plain["identity_errors"]
    assert plain["manifest"]["checkpoint"] == "asset/assets/checkpoint.pt"
    model = _model()
    ckpt = tmp_path / "real.pt"
    torch_save(
        ckpt,
        {"encoder_name": "dinov2_vits14", "model": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "step": 1, "img_size": 28},
    )
    good = package_run("good", dest_root=tmp_path / "out2", synthetic=False, checkpoint=ckpt)
    assert "checkpoint_unreadable" not in good["identity_errors"]
    assert (tmp_path / "out2" / "good" / "asset" / "assets" / "checkpoint.pt").is_file()
    assert "offline_wheels_not_vendored" in good["identity_errors"]
    asset_src = tmp_path / "out2" / "good" / "asset" / "src"
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from pathlib import Path; import torch; "
            "sys.path.insert(0, sys.argv[1]); "
            "from rsna_knee.models.dinov2 import Dinov2ViTS14; "
            "from rsna_knee.models.study import StudyModel; "
            "from rsna_knee.training.checkpoint import torch_load; "
            "blob=torch_load(sys.argv[2]); model=StudyModel(Dinov2ViTS14(img_size=28)); "
            "model.load_state_dict(blob['model']); "
            "y=model(torch.zeros(1,1,1,3,28,28), torch.ones(1,1), torch.ones(1,1,1)); "
            "assert tuple(y.shape)==(1,12)",
            str(asset_src),
            str(tmp_path / "out2" / "good" / "asset" / "assets" / "checkpoint.pt"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"overall": "PASS", "run_id": "pkg"}), encoding="utf-8")
    state = PipelineState(run_id="pkg", synthetic=False, artifacts={"audit": str(audit)})
    packed = stage_package(state, Registry(tmp_path / "reg.sqlite"))
    assert packed.stage == Stage.BLOCKED
    assert packed.blocked_reason == "checkpoint_missing"
