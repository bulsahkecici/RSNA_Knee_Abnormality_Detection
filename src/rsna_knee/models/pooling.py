"""Masked mean pooling over slots and slices."""

from __future__ import annotations

import numpy as np


def masked_mean(features: np.ndarray, mask: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """features [B, N, D], mask [B, N] -> [B, D]."""
    w = mask.astype(np.float32)[..., None]
    num = (features * w).sum(axis=1)
    den = w.sum(axis=1).clip(min=eps)
    return num / den
