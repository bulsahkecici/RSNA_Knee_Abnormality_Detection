"""Held-out predictions from a saved StudyModel checkpoint. Gold and weak stay separate.

Evaluation targets are always the fixed hard masked gold/weak labels, whatever
label source trained the model. Optional anatomical TTA (evaluation only):
sagittal windows reverse their 3 slice channels, coronal/axial windows flip
horizontally; probabilities are averaged with the plain pass.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rsna_knee.evaluation.metrics import per_class_auc
from rsna_knee.hashing import sha256_file
from rsna_knee.models.families import build_from_config
from rsna_knee.models.study import StudyModel
from rsna_knee.ontology import TARGET_COLUMNS
from rsna_knee.training.checkpoint import torch_load

TTA_MODES = {None, "anatomical"}
TTA_VERSION = "anatomical-mirror.v1"


def anatomical_mirror(images: torch.Tensor, slot_planes: list[str]) -> torch.Tensor:
    """images [B,slots,centers,3,H,W]. Applied before encoder normalization."""
    if images.ndim != 6 or len(slot_planes) != images.shape[1]:
        raise ValueError("anatomical TTA needs one plane name per slot")
    parts = []
    for slot, plane in enumerate(slot_planes):
        window = images[:, slot]
        if plane == "Sagittal":
            parts.append(window.flip(dims=[2]))
        elif plane in {"Coronal", "Axial"}:
            parts.append(window.flip(dims=[-1]))
        else:
            raise ValueError(f"unknown plane for anatomical TTA: {plane!r}")
    return torch.stack(parts, dim=1)


def select_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_trained_study(checkpoint: Path, img_size: int, device: torch.device | None = None,
                       *, allow_test_encoder: bool = False) -> tuple[StudyModel, dict[str, Any]]:
    """Rebuild the checkpoint's registered encoder family without pretrained weights, then load all state."""
    blob = torch_load(checkpoint, map_location="cpu")
    encoder_name = blob.get("encoder_name")
    resize = blob.get("input_resize")
    if resize is not None:
        if int(resize["from"]) != img_size:
            raise RuntimeError("checkpoint input resize does not start from the cache image size")
        img_size = int(resize["to"])
    try:
        encoder = build_from_config(blob.get("encoder_config"), encoder_name, img_size,
                                    allow_test_encoder=allow_test_encoder)
    except ValueError as exc:
        raise RuntimeError(f"checkpoint encoder is {encoder_name}") from exc
    if "model" not in blob or not isinstance(blob["model"], dict):
        raise RuntimeError("checkpoint has no model state")
    if int(blob.get("step") or 0) < 1:
        raise RuntimeError("checkpoint has no optimizer update")
    if int(blob.get("img_size", img_size)) != img_size:
        raise RuntimeError("checkpoint image size does not match the cache")
    aggregator = blob.get("aggregator_config") or {}
    model = StudyModel(encoder, freeze_encoder=True, pooling=blob.get("pooling", "mean"),
                       allow_test_encoder=allow_test_encoder,
                       input_size_from=None if resize is None else int(resize["from"]),
                       aggregator_layers=int(aggregator.get("layers", 2)),
                       aggregator_heads=int(aggregator.get("heads", 4)),
                       max_slots=int(aggregator.get("max_slots", 8)),
                       max_centers=int(aggregator.get("max_centers", 128)))
    if model.aggregator_config != blob.get("aggregator_config"):
        raise RuntimeError("checkpoint aggregator configuration is unsupported")
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


def _predict(model: StudyModel, batches: list[dict[str, Any]], device: torch.device,
             slot_planes: list[str] | None = None, plain: list[np.ndarray] | None = None,
             ) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """All centers, no window sampling. With slot_planes, average plain and anatomical-mirror probabilities;
    the plain-only probabilities are appended to `plain` for comparison."""
    uids: list[str] = []
    ys: list[np.ndarray] = []
    ps: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    with torch.no_grad():
        for batch in batches:
            images = batch["images"].to(device)
            slot = batch["slot_mask"].to(device)
            center = batch["center_mask"].to(device)
            probs = torch.sigmoid(model(images, slot, center)).float()
            if slot_planes is not None:
                if plain is not None:
                    plain.append(probs.detach().cpu().numpy()[0])
                mirrored = torch.sigmoid(model(anatomical_mirror(images, slot_planes), slot, center)).float()
                probs = (probs + mirrored) / 2
            probs = probs.detach().cpu().numpy()
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
    tta: str | None = None,
    slot_planes: list[str] | None = None,
) -> dict[str, Any]:
    if tta not in TTA_MODES:
        raise ValueError(f"unsupported evaluation TTA: {tta}")
    if tta == "anatomical" and not slot_planes:
        raise ValueError("anatomical TTA requires the cache slot planes")
    planes = slot_planes if tta == "anatomical" else None
    started = time.perf_counter()
    dev = select_device(device)
    model, blob = load_trained_study(checkpoint, img_size, dev)
    del train_batches
    gold_plain: list[np.ndarray] = []
    weak_plain: list[np.ndarray] = []
    gold_uids, gold_y, gold_p, gold_m = _predict(model, gold_batches, dev, planes, gold_plain)
    weak_uids, weak_y, weak_p, weak_m = _predict(model, weak_batches, dev, planes, weak_plain)
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
            "encoder": model.encoder_name,
            "step": int(blob.get("step") or 0),
            "img_size": img_size,
            "device": str(dev),
        },
        "note": "Pilot holdout metrics are not a competition generalization score.",
        "cache_scope": scope,
        "predictions": str(pred_path),
        # Evaluation always encodes every cached center; training window sampling never applies here.
        "evaluation_windows": "all_centers",
        "evaluation_tta": None if tta is None else TTA_VERSION,
        "evaluation_seconds": round(time.perf_counter() - started, 3),
    }
    if tta is not None:
        metrics["plain_gold_metrics"] = _pack("gold", gold_y, np.stack(gold_plain) if gold_plain else gold_p, gold_m)
        metrics["plain_weak_metrics"] = _pack("weak", weak_y, np.stack(weak_plain) if weak_plain else weak_p, weak_m)
    dest.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
