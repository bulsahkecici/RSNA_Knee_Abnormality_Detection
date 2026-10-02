"""DICOM geometry: IOP × IPP ordering, not lexicographic SOPInstanceUID."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass
class SliceGeom:
    path: str
    iop: tuple[float, ...] | None
    ipp: tuple[float, ...] | None
    instance_number: int | None
    sop: str | None
    pixel_spacing: tuple[float, float] | None
    slope: float = 1.0
    intercept: float = 0.0
    photometric: str | None = None
    rows: int | None = None
    cols: int | None = None


def parse_ds_numbers(value: Any) -> tuple[float, ...] | None:
    if value is None:
        return None
    try:
        seq = [float(x) for x in list(value)]
        return tuple(seq)
    except Exception:
        try:
            return (float(value),)
        except Exception:
            return None


def slice_normal(iop: Sequence[float] | None) -> np.ndarray | None:
    if iop is None or len(iop) < 6:
        return None
    row = np.asarray(iop[:3], dtype=float)
    col = np.asarray(iop[3:6], dtype=float)
    n = np.cross(row, col)
    norm = np.linalg.norm(n)
    if norm < 1e-8:
        return None
    return n / norm


def projection_key(iop: Sequence[float] | None, ipp: Sequence[float] | None) -> float | None:
    normal = slice_normal(iop)
    if normal is None or ipp is None or len(ipp) < 3:
        return None
    return float(np.dot(normal, np.asarray(ipp[:3], dtype=float)))


def orientation_coherent(iops: list[Sequence[float] | None]) -> bool:
    normals = [slice_normal(iop) for iop in iops]
    valid = [n for n in normals if n is not None]
    if len(valid) < 2:
        return True
    ref = valid[0]
    return all(abs(float(np.dot(ref, n))) > 0.95 for n in valid[1:])


def order_slices(slices: list[SliceGeom]) -> tuple[list[SliceGeom], str]:
    keys = [projection_key(s.iop, s.ipp) for s in slices]
    if all(k is not None for k in keys) and orientation_coherent([s.iop for s in slices]):
        ranked = sorted(zip(keys, range(len(slices)), slices, strict=True), key=lambda t: (t[0], t[1]))
        return [t[2] for t in ranked], "iop_ipp"
    numbered = [s for s in slices if s.instance_number is not None]
    if len(numbered) == len(slices) and slices:
        return sorted(slices, key=lambda s: (s.instance_number or 0, s.path)), "instance_number_fallback"
    return sorted(slices, key=lambda s: s.path), "path_fallback_not_sop_lex"


def apply_slope_intercept(arr: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    out = arr.astype(np.float32) * float(slope) + float(intercept)
    return out


def invert_monochrome1(arr: np.ndarray, photometric: str | None) -> np.ndarray:
    if photometric and photometric.upper() == "MONOCHROME1":
        return arr.max() - arr
    return arr


def mm_crop_box(
    rows: int,
    cols: int,
    spacing: tuple[float, float] | None,
    crop_mm: float = 160.0,
) -> tuple[slice, slice, str]:
    if spacing is None or min(spacing) <= 0:
        side = min(rows, cols)
        r0 = max(0, (rows - side) // 2)
        c0 = max(0, (cols - side) // 2)
        return slice(r0, r0 + side), slice(c0, c0 + side), "missing_spacing_square_fallback"
    row_mm, col_mm = float(spacing[0]), float(spacing[1])
    h = min(rows, max(1, int(round(crop_mm / row_mm))))
    w = min(cols, max(1, int(round(crop_mm / col_mm))))
    r0 = max(0, (rows - h) // 2)
    c0 = max(0, (cols - w) // 2)
    return slice(r0, r0 + h), slice(c0, c0 + w), "pixel_spacing_mm"
