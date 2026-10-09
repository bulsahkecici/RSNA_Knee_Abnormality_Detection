"""Held-out predictions from a saved StudyModel checkpoint. Gold and weak stay separate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rsna_knee.evaluation.metrics import per_class_auc
from rsna_knee.hashing import sha256_file
from rsna_knee.models.dinov2 import ENCODER_NAME, Dinov2ViTS14
from rsna_knee.models.study import StudyModel
from rsna_knee.ontology import TARGET_COLUMNS
from rsna_knee.training.checkpoint import torch_load


def select_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_trained_study(checkpoint: Path, img_size: int, device: torch.device | None = None) -> tuple[StudyModel, dict[str, Any]]:
    blob = torch_load(checkpoint, map_location="cpu")
    if blob.get("encoder_name") != ENCODER_NAME:
        raise RuntimeError(f"checkpoint encoder is {blob.get('encoder_name')}")
    if "model" not in blob or not isinstance(blob["model"], dict):
        raise RuntimeError("checkpoint has no model state")
    if int(blob.get("step") or 0) < 1:
        raise RuntimeError("checkpoint has no optimizer update")
    if int(blob.get("img_size", img_size)) != img_size:
        raise RuntimeError("checkpoint image size does not match the cache")
    model = StudyModel(Dinov2ViTS14(img_size=img_size), freeze_encoder=True,
                       pooling=blob.get("pooling", "mean"))
    if blob.get("pooling_version", model.pooling_version) != model.pooling_version:
        raise RuntimeError("checkpoint pooling version is unsupported")
    model.load_state_dict(blob["model"])
    saved_head = blob["model"]["head.weight"]
    if not torch.equal(model.head.weight.detach().cpu(), saved_head.detach().cpu()):
        raise RuntimeError("evaluate did not keep the saved head weights")
    dev = device or select_device()
    model = model.to(dev)
    model.eval()
    return model, blob


def _predict(model: StudyModel, batches: list[dict[str, Any]], device: torch.device) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    uids: list[str] = []
    ys: list[np.ndarray] = []
    ps: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    with torch.no_grad():
        for batch in batches:
            images = batch["images"].to(device)
            slot = batch["slot_mask"].to(device)
            center = batch["center_mask"].to(device)
            logits = model(images, slot, center)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            uids.append(str(batch["uid"]))
            ys.append(batch["y"].detach().cpu().numpy()[0])
            masks.append(batch["y_mask"].detach().cpu().numpy()[0])
            ps.append(probs[0])
    if not uids:
        empty = np.zeros((0, len(TARGET_COLUMNS)))
        return [], empty, empty, empty
    return uids, np.stack(ys), np.stack(ps), np.stack(masks)


def _pack(name: str, y: np.ndarray, p: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    metrics = per_class_auc(y, p, mask) if len(y) else {"per_class": {}, "macro_auc": None, "n_defined": 0, "n_targets": 12}
    metrics["split"] = name
    metrics["n_studies"] = int(len(y))
    return metrics


def evaluate_checkpoint(
    checkpoint: Path,
    *,
    train_batches: list[dict[str, Any]] | None,
    weak_batches: list[dict[str, Any]],
    gold_batches: list[dict[str, Any]],
    dest: Path,
    img_size: int,
    device: str | None = None,
    scope: str | None = None,
) -> dict[str, Any]:
    dev = select_device(device)
    model, blob = load_trained_study(checkpoint, img_size, dev)
    del train_batches
    gold_uids, gold_y, gold_p, gold_m = _predict(model, gold_batches, dev)
    weak_uids, weak_y, weak_p, weak_m = _predict(model, weak_batches, dev)
    rows = []
    for split, uids, probs in (
        ("gold", gold_uids, gold_p),
        ("weak", weak_uids, weak_p),
    ):
        for uid, prob in zip(uids, probs, strict=True):
            row = {"StudyInstanceUID": uid, "split": split}
            row.update({name: float(value) for name, value in zip(TARGET_COLUMNS, prob, strict=True)})
            rows.append(row)
    dest.parent.mkdir(parents=True, exist_ok=True)
    pred_path = dest.parent / "predictions.jsonl"
    with pred_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    gold_metrics = _pack("gold", gold_y, gold_p, gold_m)
    weak_metrics = _pack("weak", weak_y, weak_p, weak_m)
    metrics = {
        "kind": "real",
        "macro_auc": gold_metrics.get("macro_auc"),
        "gold_metrics": gold_metrics,
        "weak_metrics": weak_metrics,
        "per_class": gold_metrics.get("per_class") or {},
        "n_defined": gold_metrics.get("n_defined"),
        "oof_provenance": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "encoder": ENCODER_NAME,
            "step": int(blob.get("step") or 0),
            "img_size": img_size,
            "device": str(dev),
        },
        "note": "Pilot holdout metrics are not a competition generalization score.",
        "cache_scope": scope,
        "predictions": str(pred_path),
    }
    dest.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
