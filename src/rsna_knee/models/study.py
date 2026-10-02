"""Study classifier: frozen DINOv2, masked pool, 12 logits. Input [B,slots,centers,3,H,W]."""

from __future__ import annotations

import torch
import torch.nn as nn

from rsna_knee.models.dinov2 import ENCODER_NAME, Dinov2ViTS14


class StudyModel(nn.Module):
    encoder_name = ENCODER_NAME

    def __init__(self, encoder: Dinov2ViTS14, n_targets: int = 12, freeze_encoder: bool = True):
        super().__init__()
        if not isinstance(encoder, Dinov2ViTS14):
            raise TypeError("production StudyModel requires Dinov2ViTS14, not a stub")
        self.encoder = encoder
        self.head = nn.Linear(encoder.embed_dim, n_targets)
        self.img_size = encoder.img_size
        if freeze_encoder:
            self.freeze_encoder()

    def freeze_encoder(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_last_block(self) -> None:
        for param in self.encoder.blocks[-1].parameters():
            param.requires_grad = True

    def forward(self, images: torch.Tensor, slot_mask: torch.Tensor, center_mask: torch.Tensor) -> torch.Tensor:
        if images.ndim != 6:
            raise ValueError(f"expected [B,slots,centers,3,H,W], got {tuple(images.shape)}")
        b, slots, centers, ch, h, w = images.shape
        if ch != 3:
            raise ValueError("channel axis must be 3 (adjacent triplet)")
        flat = images.reshape(b * slots * centers, ch, h, w)
        feats = self.encoder(flat).reshape(b, slots * centers, -1)
        mask = (slot_mask[:, :, None] * center_mask).reshape(b, slots * centers)
        weight = mask.unsqueeze(-1)
        denom = weight.sum(dim=1).clamp_min(1e-6)
        pooled = (feats * weight).sum(dim=1) / denom
        return self.head(pooled)
