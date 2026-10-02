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
    if kind in {"dinov2", "dinov2_small"}:
        try:
            import torch.nn as nn  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("torch required for DINOv2 encoder") from exc
        return TorchDinoStub(checkpoint=checkpoint, **kwargs)
    raise ValueError(kind)


class TorchDinoStub:
    """Loads a local checkpoint if present; otherwise a tiny conv stand-in.

    This stand-in is never reported as a DINOv2 competition result.
    """

    name = "dinov2_small_or_stub"
    embed_dim = 384

    def __init__(self, checkpoint: str | None = None, **kwargs: Any):
        import torch
        import torch.nn as nn

        self.torch = torch
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, self.embed_dim),
        )
        self.is_stub = True
        if checkpoint:
            try:
                state = torch.load(checkpoint, map_location="cpu")
                self.net.load_state_dict(state, strict=False)
                self.is_stub = False
            except Exception:
                self.is_stub = True

    def parameters(self):
        return self.net.parameters()

    def __call__(self, images):
        return self.net(images)
