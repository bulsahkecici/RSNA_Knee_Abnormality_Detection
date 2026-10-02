"""Load folds, labels, and cache shards named by a job bundle."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rsna_knee.data.folds import assert_train_uids, image_training_uids, load_folds
from rsna_knee.hashing import sha256_file
from rsna_knee.ontology import TARGET_COLUMNS


def resolve_resource(spec: dict[str, Any], job_dir: Path) -> Path | None:
    relative = spec.get("relative")
    uri = spec.get("uri")
    roots: list[Path] = []
    env_root = os.environ.get("RSNA_INPUT_ROOT")
    if env_root:
        roots.append(Path(env_root))
    if spec.get("input_root"):
        roots.append(Path(str(spec["input_root"])))
    if uri:
        direct = Path(str(uri))
        if direct.is_file():
            return direct
    for root in roots:
        if relative:
            candidate = root / relative
            if candidate.is_file():
                return candidate
        if uri:
            direct = Path(str(uri))
            if direct.is_absolute():
                try:
                    candidate = root / direct.relative_to(direct.anchor)
                except ValueError:
                    candidate = root / direct.name
                if candidate.is_file():
                    return candidate
    if relative and (job_dir / relative).is_file():
        return job_dir / relative
    return None


def _check_hash(path: Path, expected: str | None) -> str | None:
    if not expected:
        return "hash_missing"
    if sha256_file(path) != expected:
        return "hash_mismatch"
    return None


def load_label_table(path: Path) -> dict[str, dict[str, Any]]:
    table: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            uid = str(row.get("StudyInstanceUID") or row.get("uid") or "")
            if not uid:
                continue
            targets = row.get("targets") or row.get("y") or {}
            masks = row.get("y_mask") or {}
            weights = row.get("y_weight") or {}
            y = []
            mask = []
            weight = []
            for name in TARGET_COLUMNS:
                if name in masks:
                    present = float(masks[name]) > 0 and targets.get(name) is not None
                else:
                    present = targets.get(name) is not None and str(targets.get(name)) not in {"", "nan", "None"}
                if not present:
                    y.append(0.0)
                    mask.append(0.0)
                    weight.append(0.0)
                    continue
                value = float(targets[name])
                if value not in (0.0, 1.0):
                    y.append(0.0)
                    mask.append(0.0)
                    weight.append(0.0)
                    continue
                y.append(value)
                mask.append(1.0)
                weight.append(float(weights.get(name, 1.0)))
            table[uid] = {
                "y": y,
                "y_mask": mask,
                "y_weight": weight,
                "source": row.get("source") or "weak",
            }
    return table


def pilot_splits(folds: list[dict[str, str]]) -> tuple[list[str], list[str], list[str]]:
    """Train on image-training UIDs outside fold 0. Fold 0 is the weak holdout. Gold eval stays out."""
    trainable = image_training_uids(folds)
    by_uid = {row["StudyInstanceUID"]: row for row in folds}
    train = [uid for uid in trainable if str(by_uid[uid].get("fold")) != "0"]
    weak = [uid for uid in trainable if str(by_uid[uid].get("fold")) == "0"]
    gold = [row["StudyInstanceUID"] for row in folds if row.get("gold_split") == "eval"]
    if not train:
        train = list(trainable)
        weak = []
    return train, weak, gold


def _batch_from_shard(uid: str, meta: dict[str, Any], labels: dict[str, dict[str, Any]], root: Path) -> dict[str, torch.Tensor]:
    shard = root / f"{uid}.npy"
    volume = np.load(shard)
    if volume.dtype == np.uint8:
        volume = volume.astype(np.float32) / 255.0
    else:
        volume = volume.astype(np.float32)
    if volume.ndim != 5:
        raise ValueError(f"{uid} shard rank {volume.ndim} is not [slots,centers,3,H,W]")
    slots, centers = int(volume.shape[0]), int(volume.shape[1])
    slot_mask = meta.get("slot_mask") or [1.0] * slots
    images = torch.from_numpy(volume).unsqueeze(0)
    slot = torch.tensor(slot_mask, dtype=torch.float32).view(1, slots)
    center = torch.ones((1, slots, centers), dtype=torch.float32)
    lab = labels.get(uid) or {"y": [0.0] * 12, "y_mask": [0.0] * 12, "y_weight": [0.0] * 12}
    return {
        "images": images,
        "slot_mask": slot,
        "center_mask": center,
        "y": torch.tensor([lab["y"]], dtype=torch.float32),
        "y_mask": torch.tensor([lab["y_mask"]], dtype=torch.float32),
        "y_weight": torch.tensor([lab["y_weight"]], dtype=torch.float32),
        "uid": uid,  # type: ignore[dict-item]
    }


def prepare_training(job: dict[str, Any], job_dir: Path) -> tuple[dict[str, Any] | None, str | None]:
    resources = job.get("resources") or {}
    required = ("folds", "labels", "cache", "weights")
    paths: dict[str, Path] = {}
    for key in required:
        spec = resources.get(key)
        if not isinstance(spec, dict):
            return None, "no_items"
        path = resolve_resource(spec, job_dir)
        if path is None:
            return None, "no_items"
        mismatch = _check_hash(path, spec.get("sha256"))
        if mismatch:
            return None, f"{key}_{mismatch}"
        paths[key] = path
    folds = load_folds(paths["folds"])
    labels = load_label_table(paths["labels"])
    manifest = json.loads(paths["cache"].read_text(encoding="utf-8"))
    if manifest.get("scope") not in {"pilot", "full"}:
        return None, "cache_scope_missing"
    studies = manifest.get("studies") or {}
    cache_root = paths["cache"].parent
    missing = [uid for uid, meta in studies.items() if not (cache_root / f"{uid}.npy").is_file() or sha256_file(cache_root / f"{uid}.npy") != meta.get("sha256")]
    if missing:
        return None, "shard_missing"
    requested = job.get("train_uids")
    train_uids, weak_uids, gold_uids = pilot_splits(folds)
    if requested is not None:
        train_uids = [str(uid) for uid in requested]
    try:
        assert_train_uids(set(train_uids), folds)
    except Exception as exc:  # noqa: BLE001
        return None, f"leakage:{exc.__class__.__name__}"
    present = [uid for uid in train_uids if uid in studies]
    if not present:
        return None, "no_items"
    batches = []
    for uid in present:
        try:
            batches.append(_batch_from_shard(uid, studies[uid], labels, cache_root))
        except (OSError, ValueError):
            return None, "shard_unreadable"
    shape = batches[0]["images"].shape
    return {
        "batches": batches,
        "train_uids": present,
        "weak_uids": [uid for uid in weak_uids if uid in studies],
        "gold_uids": [uid for uid in gold_uids if uid in studies],
        "labels": labels,
        "studies": studies,
        "cache_root": cache_root,
        "folds": folds,
        "weights": paths["weights"],
        "img_size": int(shape[-1]),
        "scope": manifest.get("scope"),
        "input_id": sha256_file(paths["cache"]) + sha256_file(paths["folds"]) + sha256_file(paths["labels"]),
        "config_id": (resources.get("config") or {}).get("sha256"),
    }, None


def heldout_batches(payload: dict[str, Any], uids: list[str]) -> list[dict[str, Any]]:
    batches = []
    for uid in uids:
        if uid not in payload["studies"]:
            continue
        batches.append(_batch_from_shard(uid, payload["studies"][uid], payload["labels"], payload["cache_root"]))
    return batches
