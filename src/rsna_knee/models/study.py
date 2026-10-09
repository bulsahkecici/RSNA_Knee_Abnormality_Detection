"""Study classifier: frozen DINOv2, masked pool, 12 logits. Input [B,slots,centers,3,H,W]."""

from __future__ import annotations

import torch
import torch.nn as nn

from rsna_knee.models.dinov2 import ENCODER_NAME, Dinov2ViTS14


class StudyModel(nn.Module):
    encoder_name = ENCODER_NAME

    def __init__(self, encoder: Dinov2ViTS14, n_targets: int = 12, freeze_encoder: bool = True,
                 pooling: str = "mean"):
        super().__init__()
        if not isinstance(encoder, Dinov2ViTS14):
            raise TypeError("production StudyModel requires Dinov2ViTS14, not a stub")
        self.encoder = encoder
        if pooling not in {"mean", "plane_concat", "label_attention"}:
            raise ValueError(f"unsupported pooling: {pooling}")
        self.pooling = pooling
        self.pooling_version = {"mean": "masked-mean.v1", "plane_concat": "plane-global-denominator.v1",
                                "label_attention": "masked-label-attention.v1"}[pooling]
        self.head = nn.Linear(encoder.embed_dim * (3 if pooling == "plane_concat" else 1), n_targets)
        if pooling == "label_attention":
            self.attention = nn.Linear(encoder.embed_dim, n_targets, bias=False)
            nn.init.zeros_(self.attention.weight)
        self.img_size = encoder.img_size
        if freeze_encoder:
            self.freeze_encoder()

    def freeze_encoder(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_last_block(self, n_blocks: int = 1) -> None:
        if not isinstance(n_blocks, int) or isinstance(n_blocks, bool) or not 1 <= n_blocks <= len(self.encoder.blocks):
            raise ValueError("invalid number of encoder blocks to unfreeze")
        for block in self.encoder.blocks[-n_blocks:]:
            for param in block.parameters():
                param.requires_grad = True

    def initialize_from_mean_checkpoint(self, blob: dict) -> None:
        """Warm-start plane weights with exactly the mean model's function.

        Uses a model-only initialization: optimizer/schedule are never reused.
        Each plane's contribution is divided by the global valid-center count,
        so repeating the original head preserves missing-plane behavior too.
        """
        if self.pooling not in {"plane_concat", "label_attention"}:
            raise ValueError("mean-checkpoint conversion requires plane_concat or label_attention")
        if blob.get("pooling", "mean") != "mean" or blob.get("encoder_name") != ENCODER_NAME:
            raise ValueError("initialization requires a mean DINOv2 checkpoint")
        if blob.get("pooling_version", "masked-mean.v1") != "masked-mean.v1":
            raise ValueError("initialization pooling version is unsupported")
        if int(blob.get("img_size", self.img_size)) != self.img_size:
            raise ValueError("initialization image size mismatch")
        state = dict(blob["model"])
        head = state["head.weight"]
        if tuple(head.shape) != (self.head.out_features, self.encoder.embed_dim):
            raise ValueError("initialization head shape mismatch")
        if self.pooling == "plane_concat":
            state["head.weight"] = head.repeat(1, 3)
        else:
            state["attention.weight"] = torch.zeros_like(self.attention.weight)
        self.load_state_dict(state, strict=True)

    def forward(self, images: torch.Tensor, slot_mask: torch.Tensor, center_mask: torch.Tensor) -> torch.Tensor:
        if images.ndim != 6:
            raise ValueError(f"expected [B,slots,centers,3,H,W], got {tuple(images.shape)}")
        b, slots, centers, ch, h, w = images.shape
        if ch != 3:
            raise ValueError("channel axis must be 3 (adjacent triplet)")
        flat = images.reshape(b * slots * centers, ch, h, w)
        feats = self.encoder(flat).reshape(b, slots * centers, -1)
        mask = (slot_mask[:, :, None] * center_mask).reshape(b, slots * centers)
        if self.pooling == "label_attention":
            scores = self.attention(feats).float()
            valid = mask.unsqueeze(-1) > 0
            scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
            weights = scores.softmax(dim=1) * valid
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
            pooled = torch.einsum("bnt,bnd->btd", weights.to(feats.dtype), feats)
            return (pooled * self.head.weight.unsqueeze(0)).sum(dim=-1) + self.head.bias
        weight = mask.unsqueeze(-1)
        denom = weight.sum(dim=1).clamp_min(1e-6)
        weighted = feats * weight
        if self.pooling == "plane_concat":
            if slots != 3:
                raise ValueError("plane_concat requires exactly three ordered planes")
            pooled = (weighted.reshape(b, slots, centers, -1).sum(dim=2)
                      / denom.unsqueeze(1)).reshape(b, -1)
        else:
            pooled = weighted.sum(dim=1) / denom
        return self.head(pooled)
