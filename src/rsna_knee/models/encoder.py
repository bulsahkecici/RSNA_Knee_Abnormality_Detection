"""Encoders. Tiny conv is for tests; it is not a DINOv2 result."""

from __future__ import annotations

from typing import Any

import numpy as np


class TinyEncoder:
    """Deterministic tiny encoder used in synthetic/unit tests."""

    name = "tiny_test_encoder"
    embed_dim = 16

    def __init__(self, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0, 0.05, size=(self.embed_dim, 1)).astype(np.float32)

    def __call__(self, images: np.ndarray) -> np.ndarray:
        # images [B, C, H, W] or [N, C, H, W]
        pooled = images.mean(axis=(2, 3), keepdims=False)  # [N, C]
        if pooled.ndim == 1:
            pooled = pooled[:, None]
        feat = pooled.mean(axis=1, keepdims=True)  # [N, 1]
        return feat @ self.w.T


def load_encoder(kind: str = "tiny", checkpoint: str | None = None, **kwargs: Any):
    if kind in {"tiny", "test"}:
        return TinyEncoder(seed=int(kwargs.get("seed", 0)))
    if kind in {"dinov2", "dinov2_small", "dinov2_vits14"}:
        if not checkpoint:
            raise FileNotFoundError(
                "DINOv2 ViT-S/14 requires the pretrained checkpoint. A conv stub is not a production encoder."
            )
        from rsna_knee.models.dinov2 import load_dinov2_vits14

        return load_dinov2_vits14(checkpoint, img_size=int(kwargs.get("img_size", 224)))
    raise ValueError(kind)


class TorchDinoStub:
    """Removed production path. Constructing it is an error, including strict=False loads."""

    def __init__(self, checkpoint: str | None = None, **kwargs: Any):
        raise RuntimeError(
            "TorchDinoStub is not a DINOv2 encoder. Use load_dinov2_vits14 with the pretrained checkpoint."
        )
