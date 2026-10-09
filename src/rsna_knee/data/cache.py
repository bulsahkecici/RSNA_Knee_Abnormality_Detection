"""Study shard cache: legacy memmapped NPY (v3) or compressed NPZ (v4). Failures are explicit, not all-zero."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np

from rsna_knee.data.preprocess import PREPROCESS_VERSION, encode_uint8, preprocess_hash
from rsna_knee.hashing import sha256_file
from rsna_knee.paths import CACHE_DIR, ensure_runtime_dirs

STORAGE_FORMATS = ("npy", "npz")
DEFAULT_MAX_CACHE_BYTES = 18 * 1024**3


class CacheBudgetExceeded(RuntimeError):
    """Compressed payload would exceed the explicitly configured cache budget."""


def compressed_byte_total(root: Path) -> int:
    return sum(path.stat().st_size for path in root.glob("*.npz") if path.is_file())


class ByteBudget:
    """Running NPZ payload total, so the builder does not rescan the directory per shard."""

    def __init__(self, root: Path, limit: int):
        if limit < 1:
            raise ValueError("max_cache_bytes must be positive")
        self.limit = limit
        self.total = compressed_byte_total(root) if root.is_dir() else 0

    def admit(self, path: Path, size: int) -> None:
        prior = path.stat().st_size if path.is_file() else 0
        projected = self.total - prior + size
        if projected > self.limit:
            raise CacheBudgetExceeded(f"cache_budget_exceeded:projected={projected}:limit={self.limit}")
        self.total = projected


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


def write_shard(uid: str, array: np.ndarray, meta: dict[str, Any], root: Path | None = None, dtype: str = "uint8",
                storage_format: str = "npy", budget: ByteBudget | None = None) -> Path:
    """Write one study shard atomically.

    NPZ shards hold a single ``images`` array and are compressed in memory first,
    so a shard rejected by the byte budget leaves no partial output on disk.
    """
    root = root or CACHE_DIR
    root.mkdir(parents=True, exist_ok=True)
    if storage_format not in STORAGE_FORMATS:
        raise ValueError("unsupported cache storage format")
    if dtype == "uint8":
        stored = encode_uint8(array) if array.dtype != np.uint8 else array
    elif storage_format == "npz":
        raise ValueError("compressed cache requires uint8")
    else:
        stored = array.astype(np.float32)
    path = root / f"{uid}.{storage_format}"
    tmp = root / f".{uid}.tmp.{storage_format}"
    if storage_format == "npz":
        buffer = io.BytesIO()
        np.savez_compressed(buffer, images=stored)
        data = buffer.getvalue()
        if budget is not None:
            budget.admit(path, len(data))
        tmp.write_bytes(data)
    else:
        np.save(tmp, stored)
    tmp.replace(path)
    meta_path = root / f"{uid}.json"
    payload = dict(meta)
    payload["dtype"] = dtype
    payload["shape"] = list(stored.shape)
    payload["preprocess_version"] = PREPROCESS_VERSION
    payload["preprocess_hash"] = meta.get("preprocess_hash") or preprocess_hash(meta)
    if storage_format == "npz":
        payload["storage_format"] = "npz"
        payload["shard_filename"] = path.name
        payload["compressed_bytes"] = path.stat().st_size
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def read_shard(path: Path) -> np.ndarray:
    """Read either layout. NPZ archives must contain exactly one ``images`` array."""
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if archive.files != ["images"]:
                raise ValueError("compressed shard must contain only images")
            return archive["images"]
    return np.load(path, mmap_mode="r", allow_pickle=False)


def load_shard(uid: str, root: Path | None = None) -> np.ndarray:
    root = root or CACHE_DIR
    for suffix in ("npz", "npy"):
        path = root / f"{uid}.{suffix}"
        if path.exists():
            return read_shard(path)
    raise FileNotFoundError(uid)


def synthetic_study_cache(
    uid: str,
    n_slots: int = 3,
    n_slices: int = 3,
    size: int = 32,
    seed: int = 0,
    root: Path | None = None,
) -> Path:
    rng = np.random.default_rng(seed)
    arr = rng.random((n_slots, n_slices, size, size)).astype(np.float32)
    return write_shard(uid, arr, {"synthetic": True, "size": size, "uid": uid}, root=root, dtype="uint8")
