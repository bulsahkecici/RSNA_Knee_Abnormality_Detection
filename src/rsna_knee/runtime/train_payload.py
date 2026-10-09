"""Load folds, labels, and cache shards named by a job bundle."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rsna_knee.data.folds import assert_train_uids, image_training_uids, load_folds
from rsna_knee.errors import LeakageError
from rsna_knee.hashing import sha256_file, sha256_json
from rsna_knee.labels.canonical import apply_gold, load_gold_table, normalize_record
from rsna_knee.models.dinov2 import ENCODER_NAME
from rsna_knee.models.families import get_family, normalize_unfreeze, validate_img_size
from rsna_knee.models.study import POOLING_VERSIONS, RESIZE_VERSION
from rsna_knee.ontology import TARGET_COLUMNS
from rsna_knee.training.loop import IDENTITY_VERSION, NUMERICAL_POLICY_VERSION, WINDOW_SAMPLING_VERSION

INIT_SOURCES = {"official_pretrained", "supervised_checkpoint"}
HARD_LABEL_SOURCE = "hard_masked"
LEGACY_IDENTITY_KEYS = ("epochs", "effective_batch", "microbatch", "seed", "unfreeze_after_steps", "lr_head",
                        "lr_backbone", "profile", "img_size")


def shard_path(uid: str, meta: dict[str, Any], root: Path) -> Path:
    if not uid or uid in {".", ".."} or any(c not in "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ._-" for c in uid):
        raise ValueError("unsafe study uid")
    storage = meta.get("storage_format", "npy")
    if storage not in {"npy", "npz"}:
        raise ValueError("unsupported cache storage format")
    expected = f"{uid}.{storage}"
    if meta.get("shard_filename", expected) != expected:
        raise ValueError("cache shard filename mismatch")
    return root / expected


def load_volume(uid: str, meta: dict[str, Any], root: Path) -> np.ndarray:
    path = shard_path(uid, meta, root)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if archive.files != ["images"]:
                raise ValueError("compressed cache must contain only images")
            return archive["images"]
    return np.load(path, mmap_mode="r", allow_pickle=False)


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


def load_soft_label_table(path: Path) -> dict[str, dict[str, Any]]:
    """Versioned soft-target source for training only.

    Rows: {"StudyInstanceUID", "soft_targets": {target: p in [0,1] | null},
    optional "target_weights": {target: w >= 0}}. null/absent targets are masked;
    a present probability has mask 1 and is a plain BCE probability target.
    """
    table: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            uid = str(row.get("StudyInstanceUID") or "")
            targets = row.get("soft_targets")
            weights = row.get("target_weights") or {}
            if not uid or uid in table or not isinstance(targets, dict) or not isinstance(weights, dict):
                raise ValueError("soft label row is malformed or duplicated")
            if set(targets) - set(TARGET_COLUMNS) or set(weights) - set(TARGET_COLUMNS):
                raise ValueError("soft label row names an unknown target")
            y, mask, weight = [], [], []
            for name in TARGET_COLUMNS:
                value = targets.get(name)
                w = float(weights.get(name, 1.0))
                if value is not None and (not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0):
                    raise ValueError("soft targets must be probabilities in [0, 1]")
                if not math.isfinite(w) or w < 0:
                    raise ValueError("soft target weights must be finite and nonnegative")
                y.append(0.0 if value is None else float(value))
                mask.append(0.0 if value is None else 1.0)
                weight.append(w)
            table[uid] = {"y": y, "y_mask": mask, "y_weight": weight, "source": "soft"}
    return table


def resolve_label_source(job: dict[str, Any], job_dir: Path, hard_sha: str | None
                         ) -> tuple[dict[str, Any] | None, Path | None, str | None]:
    """(label identity, soft label path, error). The hard masked table is the default."""
    name = (job.get("train") or {}).get("label_source", HARD_LABEL_SOURCE)
    if name == HARD_LABEL_SOURCE:
        return {"name": name, "sha256": hard_sha}, None, None
    spec = (job.get("resources") or {}).get("soft_labels")
    if not isinstance(spec, dict):
        return None, None, "soft_labels_missing"
    path = resolve_resource(spec, job_dir)
    if path is None:
        return None, None, "soft_labels_missing"
    mismatch = _check_hash(path, spec.get("sha256"))
    if mismatch:
        return None, None, f"soft_labels_{mismatch}"
    return {"name": str(name), "sha256": spec["sha256"]}, path, None


def resolve_pooling(train_cfg: dict[str, Any]) -> str:
    """`aggregator` is an alias of `pooling`; both set must agree."""
    pooling, aggregator = train_cfg.get("pooling"), train_cfg.get("aggregator")
    if aggregator is not None and pooling not in {None, aggregator}:
        raise ValueError("pooling and aggregator disagree")
    return str(aggregator or pooling or "mean")


def slot_planes(studies: dict[str, Any]) -> list[str] | None:
    """Plane per slot from the cache policy; None when the manifest does not name slots."""
    names = {json.dumps(meta.get("slots")) for meta in studies.values()}
    if len(names) != 1:
        return None
    slots = json.loads(names.pop())
    if not isinstance(slots, list) or not slots:
        return None
    return [str(slot[0] if isinstance(slot, list) else str(slot).split(":")[0]) for slot in slots]


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
        volume = load_volume(self.uid, self.meta, self.root)
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


def resolve_init(job: dict[str, Any], job_dir: Path) -> tuple[Path | None, str | None, str | None]:
    """(init checkpoint, init_source, error). Official pretrained weights are the default."""
    train_cfg = job.get("train") or {}
    mode = train_cfg.get("init", "official_pretrained")
    if mode not in INIT_SOURCES:
        return None, None, "invalid_init"
    spec = (job.get("resources") or {}).get("init_checkpoint")
    if mode == "official_pretrained":
        return (None, None, "init_conflict") if spec else (None, "official_pretrained", None)
    if not isinstance(spec, dict):
        return None, None, "init_checkpoint_missing"
    path = resolve_resource(spec, job_dir)
    if path is None:
        return None, None, "init_checkpoint_missing"
    mismatch = _check_hash(path, spec.get("sha256"))
    if mismatch:
        return None, None, f"init_checkpoint_{mismatch}"
    return path, f"supervised_checkpoint:{spec['sha256']}", None


def legacy_config_identity(train_cfg: dict[str, Any], config_sha: str | None) -> dict[str, Any]:
    """Exact pre-v2 identity, kept so completed legacy checkpoints still verify."""
    identity = {key: train_cfg.get(key) for key in LEGACY_IDENTITY_KEYS if key in train_cfg}
    identity["numerical_policy"] = NUMERICAL_POLICY_VERSION
    identity["pooling"] = train_cfg.get("pooling", "mean")
    identity["pooling_version"] = (
        "masked-mean.v1" if identity["pooling"] == "mean" else "plane-global-denominator.v1"
    )
    return {"train": identity, "config_sha": config_sha}


def experiment_identity(train_cfg: dict[str, Any], *, config_sha: str | None, init_source: str | None,
                        preprocess_hash: str | None, slots: int, centers: int, img_size: int,
                        label_source: dict[str, Any] | None = None, model_img_size: int | None = None
                        ) -> dict[str, Any]:
    """Everything that defines a comparable experiment arm, with defaults made explicit.

    Evaluation-only options (tta) are recorded in metrics, not here, so a completed
    checkpoint can be re-evaluated with or without TTA.
    """
    pooling = resolve_pooling(train_cfg)
    model_img_size = img_size if model_img_size is None else model_img_size
    windows = train_cfg.get("train_centers_per_slot")
    layer_decay = train_cfg.get("layer_decay")
    identity = {key: train_cfg.get(key) for key in LEGACY_IDENTITY_KEYS if key in train_cfg}
    identity.update({
        "identity_version": IDENTITY_VERSION,
        "numerical_policy": NUMERICAL_POLICY_VERSION,
        "encoder": train_cfg.get("encoder", ENCODER_NAME),
        "pooling": pooling,
        "pooling_version": POOLING_VERSIONS.get(pooling),
        "unfreeze_blocks": normalize_unfreeze(train_cfg.get("unfreeze_blocks", 1)),
        "unfreeze_after_epochs": train_cfg.get("unfreeze_after_epochs"),
        "layer_decay": None if layer_decay is None else float(layer_decay),
        "train_centers_per_slot": None if windows is None else int(windows),
        "window_sampling": None if windows is None else WINDOW_SAMPLING_VERSION,
        "init_source": init_source,
        "preprocess_hash": preprocess_hash,
        "slots": slots,
        "centers": centers,
        "cache_img_size": img_size,
        "model_img_size": model_img_size,
        "input_resize": None if model_img_size == img_size else RESIZE_VERSION,
        "aggregator_layers": train_cfg.get("aggregator_layers", 2) if pooling == "window_transformer" else None,
        "aggregator_heads": train_cfg.get("aggregator_heads", 4) if pooling == "window_transformer" else None,
        "label_source": label_source,
    })
    return {"train": identity, "config_sha": config_sha}


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
    try:
        missing = [uid for uid, meta in studies.items()
                   if not shard_path(uid, meta, cache_root).is_file()
                   or sha256_file(shard_path(uid, meta, cache_root)) != meta.get("sha256")]
    except ValueError:
        return None, "invalid_cache_storage"
    if missing:
        return None, "shard_missing"
    gold_spec = resources.get("train_csv") or resources.get("metadata")
    gold_path = resolve_resource(gold_spec, job_dir) if isinstance(gold_spec, dict) else None
    if gold_path is not None:
        mismatch = _check_hash(gold_path, gold_spec.get("sha256") if isinstance(gold_spec, dict) else None)
        if mismatch:
            return None, f"train_csv_{mismatch}"
    labels = load_label_table(paths["labels"], gold_csv=gold_path)
    label_source, soft_path, label_error = resolve_label_source(job, job_dir, sha256_file(paths["labels"]))
    if label_error:
        return None, label_error
    train_labels = labels
    if soft_path is not None:
        try:
            train_labels = load_soft_label_table(soft_path)
        except (ValueError, TypeError, json.JSONDecodeError):
            return None, "soft_labels_invalid"
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
            batches.append(LazyStudyBatch(uid, studies[uid], train_labels, cache_root))
        except (OSError, ValueError):
            return None, "shard_unreadable"
    img_size = int(batches[0]._shape[-1])
    shapes = {batch._shape for batch in batches}
    if len(shapes) != 1:
        return None, "cache_shape_mixed"
    train_cfg = job.get("train") or {}
    try:
        family = get_family(train_cfg.get("encoder", ENCODER_NAME)).name
        normalize_unfreeze(train_cfg.get("unfreeze_blocks", 1))
        pooling = resolve_pooling(train_cfg)
        if pooling not in POOLING_VERSIONS:
            return None, "invalid_train_config"
        model_img_size = int(train_cfg.get("img_size") or img_size)
        if model_img_size != img_size and not train_cfg.get("resize_from_cache"):
            return None, "img_size_mismatch"
        validate_img_size(family, model_img_size)
        windows = train_cfg.get("train_centers_per_slot")
        if windows is not None and (isinstance(windows, bool) or not 1 <= int(windows) <= batches[0]._shape[1]):
            return None, "invalid_train_centers_per_slot"
    except (TypeError, ValueError):
        return None, "invalid_train_config"
    if pooling == "plane_concat" and batches[0]._shape[0] != 3:
        return None, "plane_concat_requires_three_slots"
    init_path, init_source, init_error = resolve_init(job, job_dir)
    if init_error:
        return None, init_error
    config_sha = (resources.get("config") or {}).get("sha256")
    slots, centers = int(batches[0]._shape[0]), int(batches[0]._shape[1])
    preprocess = manifest.get("preprocess_hash") or next(iter(studies.values()), {}).get("preprocess_hash")
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
        "slots": slots,
        "centers": centers,
        "preprocess_hash": preprocess,
        "scope": manifest.get("scope"),
        "model_img_size": model_img_size,
        "pooling": pooling,
        "slot_planes": slot_planes(studies),
        "label_source": label_source,
        "soft_label_coverage": None if soft_path is None else sum(uid in train_labels for uid in present),
        "init_checkpoint": init_path,
        "init_source": init_source,
        "input_id": sha256_file(paths["cache"]) + sha256_file(paths["folds"]) + sha256_file(paths["labels"]) + sha256_file(paths["weights"])
        + ("" if soft_path is None else label_source["sha256"]),
        "config_id": sha256_json(experiment_identity(train_cfg, config_sha=config_sha, init_source=init_source,
                                                     preprocess_hash=preprocess, slots=slots, centers=centers,
                                                     img_size=img_size, label_source=label_source,
                                                     model_img_size=model_img_size)),
        "legacy_config_id": sha256_json(legacy_config_identity(train_cfg, config_sha)),
    }, None


def heldout_batches(payload: dict[str, Any], uids: list[str]) -> list[Any]:
    """Evaluation always reads the hard masked gold/weak table, never a soft training source."""
    batches = []
    for uid in uids:
        if uid not in payload["studies"]:
            continue
        batches.append(LazyStudyBatch(uid, payload["studies"][uid], payload["labels"], payload["cache_root"]))
    return batches
