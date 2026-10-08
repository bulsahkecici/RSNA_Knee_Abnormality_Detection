"""Intensity normalize, resize, adjacent triplets. Same code for train cache and inference."""

from __future__ import annotations

from typing import Any

import numpy as np

from rsna_knee.data.dicom import apply_slope_intercept, invert_monochrome1, mm_crop_box
from rsna_knee.hashing import sha256_json

PREPROCESS_VERSION = "pp-v1-224-pct-monochrome1"


def percentile_clip(arr: np.ndarray, low: float = 0.5, high: float = 99.5) -> np.ndarray:
    lo, hi = np.percentile(arr, [low, high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(arr)), float(np.max(arr) + 1e-6)
    return np.clip(arr, lo, hi)


def to_unit(arr: np.ndarray) -> np.ndarray:
    mn, mx = float(arr.min()), float(arr.max())
    if mx <= mn:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - mn) / (mx - mn)).astype(np.float32)


def resize_2d(arr: np.ndarray, size: int = 224) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError:
        # Nearest-neighbor fallback without PIL.
        h, w = arr.shape
        ys = (np.linspace(0, h - 1, size)).astype(int)
        xs = (np.linspace(0, w - 1, size)).astype(int)
        return arr[ys][:, xs]
    img = Image.fromarray(arr)
    return np.asarray(img.resize((size, size), resample=Image.Resampling.BILINEAR), dtype=np.float32)


def encode_uint8(arr01: np.ndarray) -> np.ndarray:
    return np.clip(arr01 * 255.0, 0, 255).astype(np.uint8)


def decode_uint8(arr: np.ndarray) -> np.ndarray:
    return arr.astype(np.float32) / 255.0


def process_slice(
    pixels: np.ndarray,
    *,
    slope: float = 1.0,
    intercept: float = 0.0,
    photometric: str | None = None,
    spacing: tuple[float, float] | None = None,
    size: int = 224,
    crop_mm: float = 160.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    arr = apply_slope_intercept(np.asarray(pixels), slope, intercept)
    arr = invert_monochrome1(arr, photometric)
    rsl, csl, crop_mode = mm_crop_box(arr.shape[0], arr.shape[1], spacing, crop_mm=crop_mm)
    arr = arr[rsl, csl]
    arr = percentile_clip(arr)
    arr = to_unit(arr)
    arr = resize_2d(arr, size=size)
    meta = {"crop_mode": crop_mode, "size": size, "photometric": photometric, "version": PREPROCESS_VERSION}
    return arr, meta


def adjacent_triplet(stack: np.ndarray, center: int) -> np.ndarray:
    """stack: [S,H,W] -> [3,H,W] with edge clamp. Adjacent semantics preserved."""
    n = stack.shape[0]
    idx = [max(0, min(n - 1, center + d)) for d in (-1, 0, 1)]
    return stack[idx]


def select_centers(n: int, k: int = 1) -> list[int]:
    if n <= 0:
        return []
    if k == 1:
        return [n // 2]
    return list(np.linspace(0, n - 1, k).astype(int))


def preprocess_hash(config: dict[str, Any]) -> str:
    payload = dict(config)
    payload["preprocess_version"] = PREPROCESS_VERSION
    return sha256_json(payload)


def laterality_flip_needed(image_laterality: str | None, requested: str | None) -> bool:
    if not image_laterality or not requested:
        return False
    a, b = image_laterality.lower()[:1], requested.lower()[:1]
    return a in {"l", "r"} and b in {"l", "r"} and a != b
