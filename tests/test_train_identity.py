"""Experiment identity, init source, measurement, soft targets, anatomical TTA and the worker path."""
import json

import numpy as np
import pytest
import torch

from rsna_knee.data.cache import write_shard
from rsna_knee.evaluation.heldout import anatomical_mirror, evaluate_checkpoint
from rsna_knee.hashing import sha256_file, sha256_json
from rsna_knee.models.dinov2 import Dinov2ViTS14
from rsna_knee.models.study import StudyModel
from rsna_knee.ontology import TARGET_COLUMNS
from rsna_knee.runtime.jobs import build_job
from rsna_knee.runtime.train_payload import (
    experiment_identity,
    legacy_config_identity,
    load_soft_label_table,
    prepare_training,
)
from rsna_knee.training.loop import NUMERICAL_POLICY_VERSION, train_study_model
from rsna_knee.training.losses import torch_masked_bce_parts
from tests.test_resume_equivalence import _batches, _fresh
from tests.test_review_3e8c69c import _envelope, _folds, _weights, _write_rows

FIVE = [["Sagittal", "fluid"], ["Coronal", "fluid"], ["Axial", "fluid"],
        ["Sagittal", "structural"], ["Coronal", "structural"]]


def test_init_source_is_recorded_and_refused_when_changed(tmp_path):
    path = tmp_path / "ck.pt"
    args = dict(epochs=2, effective_batch=1, seed=0)
    train_study_model(_fresh(), _batches(), ckpt_path=path, interrupt_after_steps=1,
                      init_source="official_pretrained", **args)
    assert torch.load(path, weights_only=False)["init_source"] == "official_pretrained"
    with pytest.raises(RuntimeError, match="initialization source changed"):
        train_study_model(_fresh(), _batches(), ckpt_path=path, resume=True,
                          init_source="supervised_checkpoint:" + "a" * 64, **args)
    out = train_study_model(_fresh(), _batches(), ckpt_path=path, resume=True, **args)
    assert out["init_source"] == "official_pretrained"
    blob = torch.load(path, weights_only=False)
    blob.pop("init_source")
    torch.save(blob, path)
    out = train_study_model(_fresh(), _batches(), ckpt_path=path, resume=True, init_source="official_pretrained", **args)
    assert out["init_source"] == "unrecorded_legacy"  # never relabelled as official


def test_supervised_initialization_is_model_only_and_checked(tmp_path):
    train_study_model(_fresh(), _batches(), epochs=1, effective_batch=1, ckpt_path=tmp_path / "sup.pt")
    blob = torch.load(tmp_path / "sup.pt", weights_only=False)
    model = StudyModel(Dinov2ViTS14(img_size=28))
    model.initialize_from_checkpoint(blob)
    assert torch.equal(model.head.weight, blob["model"]["head.weight"])
    attention = StudyModel(Dinov2ViTS14(img_size=28), pooling="label_attention")
    attention.initialize_from_checkpoint(blob)  # mean -> attention keeps the mean function
    with pytest.raises(ValueError, match="another encoder family"):
        model.initialize_from_checkpoint({**blob, "encoder_name": "convnext_tiny"})
    with pytest.raises(ValueError, match="no optimizer update"):
        model.initialize_from_checkpoint({**blob, "step": 0})


def test_legacy_checkpoint_resumes_against_legacy_config_id(tmp_path):
    path = tmp_path / "legacy.pt"
    args = dict(epochs=2, effective_batch=1, seed=0, input_id="in")
    train_study_model(_fresh(), _batches(), ckpt_path=path, interrupt_after_steps=1, config_id="old-id", **args)
    blob = torch.load(path, weights_only=False)
    blob.pop("identity_version")
    torch.save(blob, path)
    out = train_study_model(_fresh(), _batches(), ckpt_path=path, resume=True, config_id="new-id",
                            legacy_config_id="old-id", **args)
    assert out["step"] == 4 and out["validated_completed_resume"] is False
    again = train_study_model(_fresh(), _batches(), ckpt_path=path, resume=True, config_id="new-id",
                              legacy_config_id="old-id", **args)
    assert again["resume_validation"]["config_identity_matches"] is True
    blob = torch.load(path, weights_only=False)
    assert "identity_version" not in blob or blob["identity_version"] == 1
    blob["identity_version"] = 2
    torch.save(blob, path)
    with pytest.raises(RuntimeError, match="config identity changed"):
        train_study_model(_fresh(), _batches(), ckpt_path=path, resume=True, config_id="new-id",
                          legacy_config_id="old-id", **args)


def test_legacy_identity_formula_is_unchanged():
    train = {"epochs": 8, "effective_batch": 8, "microbatch": 1, "seed": 2026, "lr_head": 1e-4,
             "lr_backbone": 1e-5, "profile": "campaign", "img_size": 224, "unfreeze_blocks": 2,
             "pooling": "label_attention", "resume": True}
    old = {"train": {"epochs": 8, "effective_batch": 8, "microbatch": 1, "seed": 2026, "lr_backbone": 1e-5,
                     "lr_head": 1e-4, "profile": "campaign", "img_size": 224,
                     "numerical_policy": NUMERICAL_POLICY_VERSION, "pooling": "label_attention",
                     "pooling_version": "plane-global-denominator.v1"}, "config_sha": "cfg"}
    assert sha256_json(legacy_config_identity(train, "cfg")) == sha256_json(old)


def test_experiment_identity_covers_every_controlled_axis():
    base_train = {"encoder": "dinov2_vits14", "pooling": "mean", "unfreeze_blocks": 2, "epochs": 8}
    base = dict(config_sha="cfg", init_source="official_pretrained", preprocess_hash="pp", slots=3, centers=12,
                img_size=224, label_source={"name": "hard_masked", "sha256": "l"}, model_img_size=224)
    reference = sha256_json(experiment_identity(base_train, **base))
    train_changes = [{"encoder": "convnext_tiny"}, {"pooling": "label_attention"}, {"aggregator": "window_transformer",
                     "pooling": None}, {"unfreeze_blocks": "all"}, {"unfreeze_blocks": 4}, {"layer_decay": 0.75},
                     {"train_centers_per_slot": 6}]
    other_changes = [{"init_source": "supervised_checkpoint:abc"}, {"preprocess_hash": "pp2"}, {"slots": 5},
                     {"centers": 6}, {"label_source": {"name": "soft_v1", "sha256": "s"}}, {"model_img_size": 308}]
    ids = {reference}
    for change in train_changes:
        ids.add(sha256_json(experiment_identity({**base_train, **change}, **base)))
    for change in other_changes:
        ids.add(sha256_json(experiment_identity(base_train, **{**base, **change})))
    assert len(ids) == 1 + len(train_changes) + len(other_changes)
    assert experiment_identity({"unfreeze_blocks": "2"}, **base)["train"]["unfreeze_blocks"] == 2
    assert experiment_identity({}, **base)["train"]["unfreeze_blocks"] == 1


def test_measurements_are_reported():
    out = train_study_model(_fresh(), _batches(), epochs=2, effective_batch=1, seed=0)
    assert len(out["epoch_wall_seconds"]) == 2 and all(s >= 0 for s in out["epoch_wall_seconds"])
    assert out["train_studies_seen"] == 4 and out["seconds_per_train_study"] > 0
    assert out["peak_cuda_memory_bytes"] is None and out["resumed_mid_epoch"] is False


def test_soft_targets_are_plain_bce_and_masks_still_drop_terms():
    logits = torch.tensor([[0.3, -1.0, 2.0]])
    y = torch.tensor([[0.8, 0.25, 0.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    total, weight = torch_masked_bce_parts(logits, y, mask)
    p = torch.sigmoid(logits[0, :2])
    manual = -(y[0, :2] * p.log() + (1 - y[0, :2]) * (1 - p).log()).sum()
    assert torch.allclose(total, manual) and weight == 2
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        torch_masked_bce_parts(logits, torch.tensor([[1.5, 0.0, 0.0]]), mask)
    with pytest.raises(ValueError, match="sentinel"):
        torch_masked_bce_parts(logits, torch.tensor([[-1.0, 0.0, 0.0]]), mask)


def test_soft_label_table_validation(tmp_path):
    path = tmp_path / "soft.jsonl"
    _write_rows(path, [{"StudyInstanceUID": "a", "soft_targets": {TARGET_COLUMNS[0]: 0.8, TARGET_COLUMNS[1]: None},
                        "target_weights": {TARGET_COLUMNS[0]: 0.5}}])
    row = load_soft_label_table(path)["a"]
    assert row["y"][0] == 0.8 and row["y_mask"][:3] == [1.0, 0.0, 0.0] and row["y_weight"][0] == 0.5
    for bad in ({"StudyInstanceUID": "a", "soft_targets": {TARGET_COLUMNS[0]: 1.2}},
                {"StudyInstanceUID": "a", "soft_targets": {"Unknown": 0.1}},
                {"StudyInstanceUID": "a", "soft_targets": {TARGET_COLUMNS[0]: 0.1}, "target_weights": {TARGET_COLUMNS[0]: -1}}):
        _write_rows(path, [bad])
        with pytest.raises(ValueError):
            load_soft_label_table(path)
    _write_rows(path, [{"StudyInstanceUID": "a", "soft_targets": {}}] * 2)
    with pytest.raises(ValueError, match="duplicated"):
        load_soft_label_table(path)


def test_anatomical_mirror_per_plane():
    images = torch.rand(1, 3, 2, 3, 4, 5)
    out = anatomical_mirror(images, ["Sagittal", "Coronal", "Axial"])
    assert torch.equal(out[:, 0], images[:, 0].flip(dims=[2]))  # slice channels reversed
    assert torch.equal(out[:, 1], images[:, 1].flip(dims=[-1]))  # width flipped
    assert torch.equal(out[:, 2], images[:, 2].flip(dims=[-1]))
    assert torch.equal(anatomical_mirror(out, ["Sagittal", "Coronal", "Axial"]), images)
    with pytest.raises(ValueError, match="one plane name per slot"):
        anatomical_mirror(images, ["Sagittal"])
    with pytest.raises(ValueError, match="unknown plane"):
        anatomical_mirror(images, ["Sagittal", "Oblique", "Axial"])


def test_tta_evaluation_averages_with_the_plain_pass(tmp_path):
    train_study_model(_fresh(), _batches(), epochs=1, effective_batch=1, ckpt_path=tmp_path / "ck.pt")
    batch = {"images": torch.rand(1, 1, 1, 3, 28, 28), "slot_mask": torch.ones(1, 1), "center_mask": torch.ones(1, 1, 1),
             "y": torch.ones(1, 12), "y_mask": torch.ones(1, 12), "y_weight": torch.ones(1, 12), "uid": "u"}
    metrics = evaluate_checkpoint(tmp_path / "ck.pt", train_batches=None, weak_batches=[batch], gold_batches=[],
                                  dest=tmp_path / "m.json", img_size=28, device="cpu", tta="anatomical",
                                  slot_planes=["Coronal"])
    assert metrics["evaluation_tta"] == "anatomical-mirror.v1" and "plain_weak_metrics" in metrics
    row = json.loads((tmp_path / "predictions.jsonl").read_text())
    model = _fresh()
    model.load_state_dict(torch.load(tmp_path / "ck.pt", weights_only=False)["model"])
    model.eval()
    with torch.no_grad():
        plain = torch.sigmoid(model(batch["images"], batch["slot_mask"], batch["center_mask"]))
        mirror = torch.sigmoid(model(batch["images"].flip(dims=[-1]), batch["slot_mask"], batch["center_mask"]))
    assert torch.allclose(torch.tensor([row[name] for name in TARGET_COLUMNS]), ((plain + mirror) / 2)[0], atol=1e-5)
    with pytest.raises(ValueError, match="slot planes"):
        evaluate_checkpoint(tmp_path / "ck.pt", train_batches=None, weak_batches=[batch], gold_batches=[],
                            dest=tmp_path / "m.json", img_size=28, device="cpu", tta="anatomical")


def _five_slot_job(tmp_path, monkeypatch, train, *, soft=True):
    uids = ["1.2.50", "1.2.51", "1.2.52", "1.2.53"]
    tmp_path.mkdir(parents=True, exist_ok=True)
    folds = tmp_path / "folds.csv"
    _folds(folds, [
        {"StudyInstanceUID": uids[0], "fold": "1", "is_gold": "0", "gold_split": "weak", "group_id": "a"},
        {"StudyInstanceUID": uids[1], "fold": "2", "is_gold": "0", "gold_split": "weak", "group_id": "b"},
        {"StudyInstanceUID": uids[2], "fold": "0", "is_gold": "0", "gold_split": "weak", "group_id": "c"},
        {"StudyInstanceUID": uids[3], "fold": "3", "is_gold": "1", "gold_split": "eval", "group_id": "d"},
    ])
    labels = tmp_path / "labels.jsonl"
    _write_rows(labels, [_envelope(uid, positive=i % 2 == 0) for i, uid in enumerate(uids)])
    root = tmp_path / "cache"
    studies = {}
    for i, uid in enumerate(uids):
        volume = np.random.default_rng(i).random((5, 4, 3, 28, 28)).astype(np.float32)
        meta = {"uid": uid, "shape": [5, 4, 3, 28, 28], "dtype": "uint8", "slot_mask": [1.0, 1.0, 1.0, float(i != 1), 1.0],
                "slots": FIVE, "preprocess_hash": "pp-five", "synthetic": False}
        meta["sha256"] = sha256_file(write_shard(uid, volume, meta, root=root, dtype="uint8"))
        studies[uid] = meta
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"scope": "pilot", "studies": studies, "n_studies": 4, "n_requested": 4,
                                    "requested_uids": uids, "preprocess_hash": "pp-five"}))
    weights, allow = tmp_path / "weights.pt", tmp_path / "allow.json"
    _weights(weights, allow)
    monkeypatch.setenv("RSNA_DINOV2_ALLOWLIST", str(allow))
    resources = {"folds": str(folds), "labels": str(labels), "cache": str(manifest), "weights": str(weights)}
    if soft:
        soft_path = tmp_path / "soft.jsonl"
        _write_rows(soft_path, [{"StudyInstanceUID": uid, "soft_targets": {name: 0.7 for name in TARGET_COLUMNS[:4]}}
                                for uid in uids[:2]])
        resources["soft_labels"] = str(soft_path)
    job = build_job(run_id="five", stage="TRAINING", inputs={}, dest=tmp_path / "job", resources=resources,
                    train={"epochs": 1, "effective_batch": 1, "microbatch": 1, "seed": 0, "profile": "pilot", **train})
    return job, uids


def test_payload_uses_soft_labels_for_training_and_hard_labels_for_evaluation(tmp_path, monkeypatch):
    job, uids = _five_slot_job(tmp_path, monkeypatch, {"label_source": "soft_v1"})
    payload, reason = prepare_training(job, tmp_path / "job")
    assert reason is None and payload["slots"] == 5 and payload["slot_planes"][3] == "Sagittal"
    assert payload["label_source"]["name"] == "soft_v1" and payload["soft_label_coverage"] == 2
    train_y = payload["batches"][0]["y"]
    assert train_y[0, 0] == pytest.approx(0.7) and payload["batches"][0]["y_mask"][0, 5] == 0
    from rsna_knee.runtime.train_payload import heldout_batches
    gold = heldout_batches(payload, payload["gold_uids"])
    assert set(gold[0]["y"].unique().tolist()) <= {0.0, 1.0}
    hard_job, _ = _five_slot_job(tmp_path / "h", monkeypatch, {}, soft=False)
    hard, _ = prepare_training(hard_job, tmp_path / "h" / "job")
    assert hard["config_id"] != payload["config_id"] and hard["label_source"]["name"] == "hard_masked"


@pytest.mark.parametrize("train, reason", [
    ({"label_source": "soft_v1"}, "soft_labels_missing"),
    ({"img_size": 42}, "img_size_mismatch"),
    ({"img_size": 40, "resize_from_cache": True}, "invalid_train_config"),
    ({"pooling": "plane_concat"}, "plane_concat_requires_three_slots"),
    ({"train_centers_per_slot": 5}, "invalid_train_centers_per_slot"),
    ({"init": "supervised_checkpoint"}, "init_checkpoint_missing"),
    ({"aggregator": "window_transformer", "pooling": "mean"}, "invalid_train_config"),
])
def test_payload_refuses_inconsistent_configs(tmp_path, monkeypatch, train, reason):
    job, _ = _five_slot_job(tmp_path, monkeypatch, train, soft=False)
    assert prepare_training(job, tmp_path / "job") == (None, reason)


def test_worker_trains_five_slots_with_every_new_option(tmp_path, monkeypatch):
    from rsna_knee.runtime.worker import _finish_training

    train = {"label_source": "soft_v1", "aggregator": "window_transformer", "aggregator_layers": 1,
             "train_centers_per_slot": 2, "img_size": 42, "resize_from_cache": True, "tta": "anatomical",
             "unfreeze_blocks": "all", "layer_decay": 0.8, "unfreeze_after_epochs": 0}
    job, _ = _five_slot_job(tmp_path, monkeypatch, train)
    reason, metrics, uris = _finish_training(job, tmp_path / "job")
    assert reason == "ok", metrics
    assert metrics["kind"] == "real" and metrics["slots"] == 5 and metrics["model_img_size"] == 42
    assert metrics["init_source"] == "official_pretrained" and metrics["label_source"]["name"] == "soft_v1"
    assert metrics["train_centers_per_slot"] == 2 and metrics["unfreeze_blocks"] == "all"
    assert metrics["layer_decay"] == 0.8 and metrics["aggregator_config"]["layers"] == 1
    assert metrics["input_resize"]["from"] == 28 and metrics["evaluation_tta"] == "anatomical-mirror.v1"
    for key in ("epoch_wall_seconds", "seconds_per_train_study", "evaluation_seconds", "config_id", "preprocess_hash"):
        assert metrics[key] is not None, key
    assert "peak_cuda_memory_bytes" in metrics
    blob = torch.load(uris[0], weights_only=False)
    assert blob["pooling"] == "window_transformer" and blob["train_centers_per_slot"] == 2
