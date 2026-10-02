"""Uint8/float shard cache with memmap. Failures are explicit, not all-zero."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from rsna_knee.data.preprocess import PREPROCESS_VERSION, encode_uint8, preprocess_hash
from rsna_knee.hashing import sha256_file
from rsna_knee.paths import CACHE_DIR, ensure_runtime_dirs


class CacheIndex:
    def __init__(self, root: Path | None = None):
        ensure_runtime_dirs()
        self.root = root or CACHE_DIR
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "index.json"
        self.data: dict[str, Any] = {"version": PREPROCESS_VERSION, "studies": {}, "failures": {}}
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))

    def save(self) -> None:
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def record_ok(self, uid: str, shard: Path, meta: dict[str, Any]) -> None:
        self.data["studies"][uid] = {"shard": str(shard), "meta": meta, "sha256": sha256_file(shard)}
        self.data["failures"].pop(uid, None)
        self.save()

    def record_fail(self, uid: str, reason: str, series: str | None = None) -> None:
        self.data["failures"][uid] = {"reason": reason, "series": series}
        self.save()


def write_shard(uid: str, array: np.ndarray, meta: dict[str, Any], root: Path | None = None, dtype: str = "uint8") -> Path:
    root = root or CACHE_DIR
    root.mkdir(parents=True, exist_ok=True)
    if dtype == "uint8":
        stored = encode_uint8(array) if array.dtype != np.uint8 else array
    else:
        stored = array.astype(np.float32)
    path = root / f"{uid}.npy"
    tmp = root / f".{uid}.tmp.npy"
    np.save(tmp, stored)
    tmp.replace(path)
    meta_path = root / f"{uid}.json"
    payload = dict(meta)
    payload["dtype"] = dtype
    payload["shape"] = list(stored.shape)
    payload["preprocess_version"] = PREPROCESS_VERSION
    payload["preprocess_hash"] = preprocess_hash(meta)
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_shard(uid: str, root: Path | None = None) -> np.ndarray:
    root = root or CACHE_DIR
    path = root / f"{uid}.npy"
    if not path.exists():
        raise FileNotFoundError(uid)
    return np.load(path, mmap_mode="r")


def synthetic_study_cache(uid: str, n_slots: int = 3, n_slices: int = 3, size: int = 32, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    arr = rng.random((n_slots, n_slices, size, size)).astype(np.float32)
    return write_shard(uid, arr, {"synthetic": True, "size": size, "uid": uid}, dtype="uint8")
