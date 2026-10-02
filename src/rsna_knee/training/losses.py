"""Masked weighted BCE with logits. -1/NaN never enters the loss."""

from __future__ import annotations

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -60, 60)
    return 1.0 / (1.0 + np.exp(-x))


def masked_bce_with_logits(
    logits: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
    weight: np.ndarray | None = None,
    eps: float = 1e-8,
) -> float:
    """All arrays [B, 12]. Mask 0 drops the term. Sentinel -1 must already be masked."""
    t = np.where(np.isnan(targets), 0.0, targets)
    if np.any(t == -1):
        raise ValueError("sentinel -1 passed as BCE target; mask it first")
    p = sigmoid(logits)
    w = mask.astype(np.float32)
    if weight is not None:
        w = w * weight.astype(np.float32)
    loss = -(t * np.log(p + eps) + (1 - t) * np.log(1 - p + eps))
    den = w.sum()
    if den <= 0:
        return 0.0
    return float((loss * w).sum() / den)


def torch_masked_bce(logits, targets, mask, weight=None):
    import torch
    import torch.nn.functional as F

    t = torch.nan_to_num(targets, nan=0.0)
    if torch.any(t == -1):
        raise ValueError("sentinel -1 passed as BCE target; mask it first")
    loss = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
    w = mask
    if weight is not None:
        w = w * weight
    den = w.sum().clamp_min(1e-8)
    return (loss * w).sum() / den
