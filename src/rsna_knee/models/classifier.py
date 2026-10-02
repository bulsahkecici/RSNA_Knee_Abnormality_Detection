"""12-logit study classifier on pooled features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rsna_knee.models.pooling import masked_mean
from rsna_knee.ontology import TARGET_COLUMNS


@dataclass
class LinearHead:
    weight: np.ndarray  # [12, D]
    bias: np.ndarray  # [12]

    @classmethod
    def random(cls, dim: int, n_targets: int = 12, seed: int = 0) -> LinearHead:
        rng = np.random.default_rng(seed)
        return cls(weight=rng.normal(0, 0.02, size=(n_targets, dim)).astype(np.float32), bias=np.zeros(n_targets, dtype=np.float32))

    def logits(self, pooled: np.ndarray) -> np.ndarray:
        return pooled @ self.weight.T + self.bias


class StudyClassifier:
    def __init__(self, encoder, head: LinearHead | None = None):
        self.encoder = encoder
        dim = getattr(encoder, "embed_dim", 16)
        self.head = head or LinearHead.random(dim)

    def forward_numpy(self, images: np.ndarray, slot_mask: np.ndarray, slice_mask: np.ndarray) -> np.ndarray:
        """images [B,S,T,C,H,W] -> logits [B,12]."""
        b, s, t, c, h, w = images.shape
        flat = images.reshape(b * s * t, c, h, w)
        if c == 1:
            flat = np.repeat(flat, 3, axis=1) if flat.shape[1] == 1 else flat
            if flat.shape[1] == 1:
                flat = np.concatenate([flat, flat, flat], axis=1)
        feats = self.encoder(flat)
        feats = np.asarray(feats, dtype=np.float32).reshape(b, s * t, -1)
        mask = (slot_mask[:, :, None] * slice_mask).reshape(b, s * t)
        pooled = masked_mean(feats, mask)
        return self.head.logits(pooled)

    @property
    def n_targets(self) -> int:
        return len(TARGET_COLUMNS)
