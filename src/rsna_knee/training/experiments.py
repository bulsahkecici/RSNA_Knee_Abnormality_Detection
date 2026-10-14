"""Prepared P1/P2 experiment arms: one hypothesis, one declared change set.

Each YAML is self-contained (folds, labels, cache, weights, train). An arm
names its control and lists the sections/keys it changes; validation proves
that nothing else differs. `train` is passed to the job builder unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from rsna_knee.models.families import get_family, normalize_unfreeze, validate_img_size
from rsna_knee.models.study import POOLING_VERSIONS
from rsna_knee.paths import CONFIG_DIR

QUEUE_IDS = ("D0", "D1", "D2", "D3", "D4", "D5", "C1", "C2", "L1", "D1S2", "L1S2", "D3S2",
             "L1F1", "D3F1", "L1F2", "D3F2", "D3ALL", "D3E12", "D3ALLE12", "D3ALLE16", "D3ALLLR2", "D3LR2", "V5LR2", "V5ALLLR2",
             "LR2F1", "LR2F2", "LR2F3", "LR2F4", "LR2I1", "LR2I2", "LR2I3", "LR2I4", "PL0", "PLALL")
FOLDS = "state/folds-v2-allgold.csv"
# Whole-section identities: any difference inside counts as one change.
SECTIONS = ("folds", "labels", "cache", "weights", "outer_fold", "exclude_folds")  # outer_fold absent = "0"
TRAIN_KEYS = {
    "profile", "encoder", "init", "label_source", "pooling", "aggregator", "aggregator_layers",
    "aggregator_heads", "unfreeze_blocks", "unfreeze_after_epochs", "unfreeze_after_steps", "layer_decay",
    "train_centers_per_slot", "img_size", "resize_from_cache", "tta", "epochs", "effective_batch",
    "microbatch", "seed", "lr_head", "lr_backbone", "phase", "resume", "outer_fold",
}


def load_arms(directory: Path | None = None) -> dict[str, dict[str, Any]]:
    arms: dict[str, dict[str, Any]] = {}
    for path in sorted((directory or CONFIG_DIR / "experiments").glob("P[1-3]-*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        data["_path"] = str(path)
        if data.get("experiment_id") in arms:
            raise ValueError(f"duplicate experiment id {data.get('experiment_id')}")
        arms[str(data.get("experiment_id"))] = data
    return arms


def controlled_changes(arm: dict[str, Any], control: dict[str, Any]) -> set[str]:
    changed = {name for name in SECTIONS if arm.get(name) != control.get(name)}
    train, base = arm.get("train") or {}, control.get("train") or {}
    changed |= {f"train.{key}" for key in set(train) | set(base) if train.get(key) != base.get(key)}
    return changed


def validate_arm(arm: dict[str, Any], arms: dict[str, dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    train = dict(arm.get("train") or {})
    cache = arm.get("cache") or {}
    if arm.get("experiment_id") not in QUEUE_IDS:
        errors.append("unknown_queue_id")
    if arm.get("status") != "prepared":
        errors.append("status_not_prepared")
    if not str(arm.get("hypothesis") or "").strip():
        errors.append("hypothesis_missing")
    if arm.get("folds") != FOLDS:
        errors.append("folds_not_allgold")
    if arm.get("outer_fold") is not None and arm["outer_fold"] not in {"1", "2", "3", "4", "none"}:
        errors.append("invalid_outer_fold")  # fold 0 is the default and is never written; "none" trains on every fold
    excluded = arm.get("exclude_folds")
    if excluded is not None and (not isinstance(excluded, list) or not excluded
                                 or not set(excluded) <= {"0", "1", "2", "3", "4"}
                                 or arm.get("outer_fold", "0") in excluded):
        errors.append("invalid_exclude_folds")  # extra folds kept out of training entirely (nested teachers)
    if set(train) - TRAIN_KEYS:
        errors.append(f"unknown_train_keys:{sorted(set(train) - TRAIN_KEYS)}")
    if train.get("init") != "official_pretrained":
        errors.append("init_not_official_pretrained")
    if (arm.get("weights") or {}).get("encoder") != train.get("encoder"):
        errors.append("weights_encoder_mismatch")
    try:
        family = get_family(train.get("encoder")).name
        normalize_unfreeze(train.get("unfreeze_blocks", 1))
        size = validate_img_size(family, int(train.get("img_size", cache.get("img_size", 0))))
        if size != cache.get("img_size") and not train.get("resize_from_cache"):
            errors.append("img_size_differs_from_cache_without_resize")
    except (TypeError, ValueError) as exc:
        errors.append(f"invalid_model:{exc}")
    pooling = train.get("aggregator") or train.get("pooling", "mean")
    if pooling not in POOLING_VERSIONS:
        errors.append("invalid_pooling")
    slots = len(cache.get("slots") or [])
    if pooling == "plane_concat" and slots != 3:
        errors.append("plane_concat_requires_three_slots")
    windows = train.get("train_centers_per_slot")
    if windows is not None and not 1 <= int(windows) <= int(cache.get("centers") or 0):
        errors.append("train_centers_per_slot_out_of_range")
    control = arm.get("control")
    if control is not None:
        if control not in arms:
            errors.append("control_missing")
        else:
            declared = set(arm.get("changes") or [])
            actual = controlled_changes(arm, arms[control])
            if declared != actual:
                errors.append(f"undeclared_or_missing_changes:{sorted(declared ^ actual)}")
    elif arm.get("changes"):
        errors.append("changes_without_control")
    return errors


def job_train_spec(arm: dict[str, Any]) -> dict[str, Any]:
    """The exact `train` mapping for build_job. Call validate_arm first."""
    return dict(arm["train"])
