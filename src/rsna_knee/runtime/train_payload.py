"""Load folds, labels, and cache shards named by a job bundle."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rsna_knee.data.folds import assert_train_uids, image_training_uids, load_folds
from rsna_knee.errors import LeakageError
from rsna_knee.hashing import sha256_file, sha256_json
from rsna_knee.labels.canonical import apply_gold, load_gold_table, normalize_record
from rsna_knee.ontology import TARGET_COLUMNS
from rsna_knee.training.loop import NUMERICAL_POLICY_VERSION


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


def load_label_table(path: Path, gold_csv: Path | None = None) -> dict[str, dict[str, Any]]:
    """Read canonical rows or raw extractor envelopes. Gold metadata overrides both."""
    gold = load_gold_table(gold_csv)
    table: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record, error = normalize_record(json.loads(line))
            if record is None or error:
                continue
            apply_gold(record, gold.get(record["StudyInstanceUID"]))
            table[record["StudyInstanceUID"]] = {
                "y": [0.0 if record["targets"][name] is None else float(record["targets"][name]) for name in TARGET_COLUMNS],
                "y_mask": [float(record["y_mask"][name]) for name in TARGET_COLUMNS],
                "y_weight": [float(record["y_weight"][name]) for name in TARGET_COLUMNS],
                "source": record.get("source") or "weak",
            }
    return table


def assert_split_disjoint(
    train_uids: list[str],
    weak_uids: list[str],
    gold_uids: list[str],
    folds: list[dict[str, str]],
) -> None:
    """Reject a study or duplicate group that sits in more than one of train, weak, and gold."""
    sets = {"train": set(train_uids), "weak": set(weak_uids), "gold": set(gold_uids)}
    if sets["train"] & sets["weak"]:
        raise LeakageError("train_weak_overlap")
    if sets["train"] & sets["gold"]:
        raise LeakageError("train_gold_overlap")
    if sets["weak"] & sets["gold"]:
        raise LeakageError("weak_gold_overlap")
    roles: dict[str, set[str]] = {}
    for row in folds:
        uid = row["StudyInstanceUID"]
        group = row.get("group_id") or uid
        bucket = roles.setdefault(str(group), set())
        for name, members in sets.items():
            if uid in members:
                bucket.add(name)
    crossed = [group for group, names in roles.items() if len(names) > 1]
    if crossed:
        raise LeakageError(f"group_crosses_splits:{crossed[0]}")


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


class LazyStudyBatch:
    """One study. Pixel values are memmapped only when a tensor key is read."""

    def __init__(self, uid: str, meta: dict[str, Any], labels: dict[str, dict[str, Any]], root: Path):
        self.uid = uid
        self.meta = meta
        self.labels = labels
        self.root = root
        self.n_studies = 1
        shape = list(meta.get("shape") or [])
        if len(shape) != 5:
            raise ValueError(f"{uid} shard shape {shape} is not [slots,centers,3,H,W]")
        self._shape = tuple(int(v) for v in shape)

    def _images(self) -> torch.Tensor:
        shard = self.root / f"{self.uid}.npy"
        volume = np.load(shard, mmap_mode="r")
        if tuple(volume.shape) != self._shape:
            raise ValueError(f"{self.uid} shard shape changed")
        array = np.array(volume, dtype=np.float32, copy=True)
        if volume.dtype == np.uint8:
            array /= 255.0
        return torch.from_numpy(array).unsqueeze(0)

    def __getitem__(self, key: str) -> Any:
        slots, centers = self._shape[0], self._shape[1]
        if key == "images":
            return self._images()
        if key == "slot_mask":
            slot_mask = self.meta.get("slot_mask") or [1.0] * slots
            return torch.tensor(slot_mask, dtype=torch.float32).view(1, slots)
        if key == "center_mask":
            return torch.ones((1, slots, centers), dtype=torch.float32)
        lab = self.labels.get(self.uid) or {"y": [0.0] * 12, "y_mask": [0.0] * 12, "y_weight": [0.0] * 12}
        if key in {"y", "y_mask", "y_weight"}:
            return torch.tensor([lab[key]], dtype=torch.float32)
        if key == "uid":
            return self.uid
        raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        return key in {"images", "slot_mask", "center_mask", "y", "y_mask", "y_weight", "uid"}

    def items(self):
        for key in ("images", "slot_mask", "center_mask", "y", "y_mask", "y_weight", "uid"):
            yield key, self[key]


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
    manifest = json.loads(paths["cache"].read_text(encoding="utf-8"))
    if manifest.get("scope") not in {"pilot", "full"}:
        return None, "cache_scope_missing"
    studies = manifest.get("studies") or {}
    cache_root = paths["cache"].parent
    missing = [uid for uid, meta in studies.items() if not (cache_root / f"{uid}.npy").is_file() or sha256_file(cache_root / f"{uid}.npy") != meta.get("sha256")]
    if missing:
        return None, "shard_missing"
    gold_spec = resources.get("train_csv") or resources.get("metadata")
    gold_path = resolve_resource(gold_spec, job_dir) if isinstance(gold_spec, dict) else None
    if gold_path is not None:
        mismatch = _check_hash(gold_path, gold_spec.get("sha256") if isinstance(gold_spec, dict) else None)
        if mismatch:
            return None, f"train_csv_{mismatch}"
    labels = load_label_table(paths["labels"], gold_csv=gold_path)
    requested = job.get("train_uids")
    train_uids, weak_uids, gold_uids = pilot_splits(folds)
    if requested is not None:
        train_uids = [str(uid) for uid in requested]
    try:
        assert_split_disjoint(train_uids, weak_uids, gold_uids, folds)
        assert_train_uids(set(train_uids), folds)
    except LeakageError as exc:
        return None, f"leakage:{exc}"
    present = [uid for uid in train_uids if uid in studies]
    if not present:
        return None, "no_items"
    batches: list[LazyStudyBatch] = []
    for uid in present:
        try:
            batches.append(LazyStudyBatch(uid, studies[uid], labels, cache_root))
        except (OSError, ValueError):
            return None, "shard_unreadable"
    img_size = int(batches[0]._shape[-1])
    train_cfg = job.get("train") or {}
    identity = {
        key: train_cfg.get(key)
        for key in ("epochs", "effective_batch", "microbatch", "seed", "unfreeze_after_steps", "lr_head", "lr_backbone", "profile", "img_size")
        if key in train_cfg
    }
    identity["numerical_policy"] = NUMERICAL_POLICY_VERSION
    identity["pooling"] = train_cfg.get("pooling", "mean")
    identity["pooling_version"] = (
        "masked-mean.v1" if identity["pooling"] == "mean" else "plane-global-denominator.v1"
    )
    config_sha = (resources.get("config") or {}).get("sha256")
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
        "img_size": img_size,
        "slots": int(batches[0]._shape[0]),
        "centers": int(batches[0]._shape[1]),
        "scope": manifest.get("scope"),
        "input_id": sha256_file(paths["cache"]) + sha256_file(paths["folds"]) + sha256_file(paths["labels"]) + sha256_file(paths["weights"]),
        "config_id": sha256_json({"train": identity, "config_sha": config_sha}),
    }, None


def heldout_batches(payload: dict[str, Any], uids: list[str]) -> list[Any]:
    batches = []
    for uid in uids:
        if uid not in payload["studies"]:
            continue
        batches.append(LazyStudyBatch(uid, payload["studies"][uid], payload["labels"], payload["cache_root"]))
    return batches
