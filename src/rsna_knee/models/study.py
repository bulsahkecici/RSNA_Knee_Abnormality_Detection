"""Study classifier: registered encoder, masked pool, 12 logits. Input [B,slots,centers,3,H,W].

Mean, label-attention and window-transformer pooling accept any slot/center
count (3- or 5-slot shards, sampled training windows). plane_concat is defined
for three ordered planes only.

window_transformer: each window feature gets a learned slot (series) embedding
and a learned window-position embedding indexed by its original cache center,
then a small pre-norm TransformerEncoder with a key-padding mask for absent
slots/windows, then the per-finding label attention head.

If the cache resolution differs from the encoder input size, `input_size_from`
resizes on the fly (bilinear, antialiased) inside forward, so training,
evaluation and inference share one code path; it is recorded in checkpoints.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from rsna_knee.models import families
from rsna_knee.models.dinov2 import ENCODER_NAME

POOLING_VERSIONS = {"mean": "masked-mean.v1", "plane_concat": "plane-global-denominator.v1",
                    "label_attention": "masked-label-attention.v1",
                    "window_transformer": "window-transformer-label-attention.v1"}
ATTENTION_POOLINGS = {"label_attention", "window_transformer"}
RESIZE_VERSION = "bilinear-antialias.v1"


class StudyModel(nn.Module):
    encoder_name = ENCODER_NAME

    def __init__(self, encoder: nn.Module, n_targets: int = 12, freeze_encoder: bool = True,
                 pooling: str = "mean", *, allow_test_encoder: bool = False,
                 input_size_from: int | None = None, aggregator_layers: int = 2, aggregator_heads: int = 4,
                 max_slots: int = 8, max_centers: int = 128):
        super().__init__()
        self.family = families.validate_encoder(encoder, allow_test_encoder=allow_test_encoder)
        self.encoder_name = self.family.name
        self.encoder = encoder
        if pooling not in POOLING_VERSIONS:
            raise ValueError(f"unsupported pooling: {pooling}")
        self.pooling = pooling
        self.pooling_version = POOLING_VERSIONS[pooling]
        self.head = nn.Linear(encoder.embed_dim * (3 if pooling == "plane_concat" else 1), n_targets)
        if pooling in ATTENTION_POOLINGS:
            self.attention = nn.Linear(encoder.embed_dim, n_targets, bias=False)
            nn.init.zeros_(self.attention.weight)
        self.aggregator_config = None
        if pooling == "window_transformer":
            dim = encoder.embed_dim
            if aggregator_layers < 1 or aggregator_heads < 1 or dim % aggregator_heads:
                raise ValueError(f"window_transformer needs layers>=1 and heads dividing {dim}")
            self.aggregator_config = {"layers": int(aggregator_layers), "heads": int(aggregator_heads),
                                      "max_slots": int(max_slots), "max_centers": int(max_centers)}
            self.slot_embedding = nn.Embedding(max_slots, dim)
            self.window_embedding = nn.Embedding(max_centers, dim)
            nn.init.normal_(self.slot_embedding.weight, std=0.02)
            nn.init.normal_(self.window_embedding.weight, std=0.02)
            layer = nn.TransformerEncoderLayer(dim, aggregator_heads, dim_feedforward=2 * dim, dropout=0.0,
                                               batch_first=True, norm_first=True)
            self.aggregator = nn.TransformerEncoder(layer, aggregator_layers, norm=nn.LayerNorm(dim),
                                                    enable_nested_tensor=False)
        self.img_size = encoder.img_size
        self.input_resize = None
        if input_size_from is not None and int(input_size_from) != self.img_size:
            self.input_resize = {"from": int(input_size_from), "to": int(self.img_size), "mode": RESIZE_VERSION}
        if freeze_encoder:
            self.freeze_encoder()
        self.train(True)

    @property
    def encoder_config(self) -> dict:
        return families.encoder_config(self.encoder)

    def train(self, mode: bool = True) -> StudyModel:
        """BatchNorm running statistics stay fixed (CoAtNet stem/MBConv).

        Studies arrive one at a time with a variable number of windows, so
        batch statistics would depend on microbatch and window sampling.
        """
        super().train(mode)
        for module in self.encoder.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def freeze_encoder(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self, depth: int | str = 1) -> None:
        """Train the last `depth` family units, or the whole encoder for "all"."""
        families.unfreeze(self.encoder, depth)

    def unfreeze_last_block(self, n_blocks: int = 1) -> None:
        if not isinstance(n_blocks, int) or isinstance(n_blocks, bool):
            raise ValueError("invalid number of encoder blocks to unfreeze")
        self.unfreeze_encoder(n_blocks)

    def backbone_param_groups(self, lr: float, layer_decay: float | None = None) -> list[dict]:
        return families.layerwise_groups(self.encoder, lr, layer_decay)

    def initialize_from_mean_checkpoint(self, blob: dict) -> None:
        """Warm-start plane weights with exactly the mean model's function.

        Uses a model-only initialization: optimizer/schedule are never reused.
        Each plane's contribution is divided by the global valid-center count,
        so repeating the original head preserves missing-plane behavior too.
        """
        if self.pooling not in {"plane_concat", "label_attention"}:
            raise ValueError("mean-checkpoint conversion requires plane_concat or label_attention")
        if blob.get("input_resize") != self.input_resize:
            raise ValueError("initialization input resize mismatch")
        if blob.get("pooling", "mean") != "mean" or blob.get("encoder_name") != self.encoder_name:
            raise ValueError(f"initialization requires a mean {self.family.display} checkpoint")
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

    def initialize_from_checkpoint(self, blob: dict) -> None:
        """Model-only supervised warm start. Optimizer, schedule and RNG are not reused."""
        if blob.get("encoder_name") != self.encoder_name:
            raise ValueError("initialization checkpoint is from another encoder family")
        if int(blob.get("img_size", self.img_size)) != self.img_size:
            raise ValueError("initialization image size mismatch")
        if int(blob.get("step") or 0) < 1:
            raise ValueError("initialization checkpoint has no optimizer update")
        if blob.get("pooling", "mean") != self.pooling:
            self.initialize_from_mean_checkpoint(blob)
            return
        if blob.get("pooling_version", self.pooling_version) != self.pooling_version:
            raise ValueError("initialization pooling version is unsupported")
        if blob.get("aggregator_config") != self.aggregator_config or blob.get("input_resize") != self.input_resize:
            raise ValueError("initialization aggregator or input resize mismatch")
        self.load_state_dict(blob["model"], strict=True)

    def _label_attention(self, feats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.attention(feats).float()
        valid = mask.unsqueeze(-1) > 0
        scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        weights = scores.softmax(dim=1) * valid
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        pooled = torch.einsum("bnt,bnd->btd", weights.to(feats.dtype), feats)
        return (pooled * self.head.weight.unsqueeze(0)).sum(dim=-1) + self.head.bias

    def _window_tokens(self, feats: torch.Tensor, mask: torch.Tensor, slots: int, centers: int,
                       center_index: torch.Tensor | None) -> torch.Tensor:
        b = feats.shape[0]
        cfg = self.aggregator_config
        if center_index is None:
            center_index = torch.arange(centers, device=feats.device).view(1, 1, centers).expand(b, slots, centers)
        center_index = center_index.to(feats.device).long()
        if slots > cfg["max_slots"] or int(center_index.max()) >= cfg["max_centers"]:
            raise ValueError("window_transformer slot/center index exceeds its embedding table")
        slot_index = torch.arange(slots, device=feats.device).view(1, slots, 1).expand(b, slots, centers)
        tokens = (feats + self.slot_embedding(slot_index.reshape(b, -1)).to(feats.dtype)
                  + self.window_embedding(center_index.reshape(b, -1)).to(feats.dtype))
        padding = mask <= 0
        # A study with no valid window would make attention NaN; its label pooling is masked anyway.
        padding = padding & ~padding.all(dim=1, keepdim=True)
        return self.aggregator(tokens, src_key_padding_mask=padding)

    def forward(self, images: torch.Tensor, slot_mask: torch.Tensor, center_mask: torch.Tensor,
                center_index: torch.Tensor | None = None) -> torch.Tensor:
        """center_index [B,slots,centers] gives each window's original cache center (sampled training)."""
        if images.ndim != 6:
            raise ValueError(f"expected [B,slots,centers,3,H,W], got {tuple(images.shape)}")
        b, slots, centers, ch, h, w = images.shape
        if ch != 3:
            raise ValueError("channel axis must be 3 (adjacent triplet)")
        if tuple(slot_mask.shape) != (b, slots) or tuple(center_mask.shape) != (b, slots, centers):
            raise ValueError("slot/center masks do not match the image grid")
        if self.pooling == "plane_concat" and slots != 3:
            raise ValueError(f"plane_concat requires exactly three ordered planes, got {slots} slots")
        flat = images.reshape(b * slots * centers, ch, h, w)
        if (h, w) != (self.img_size, self.img_size):
            if self.input_resize is None or (h, w) != (self.input_resize["from"],) * 2:
                raise ValueError(f"input {h}x{w} does not match model {self.img_size} and no resize is recorded")
            flat = F.interpolate(flat.float(), size=(self.img_size, self.img_size), mode="bilinear",
                                 align_corners=False, antialias=True)
        feats = self.encoder(flat).reshape(b, slots * centers, -1)
        mask = (slot_mask[:, :, None] * center_mask).reshape(b, slots * centers)
        if self.pooling == "window_transformer":
            return self._label_attention(self._window_tokens(feats, mask, slots, centers, center_index), mask)
        if self.pooling == "label_attention":
            return self._label_attention(feats, mask)
        weight = mask.unsqueeze(-1)
        denom = weight.sum(dim=1).clamp_min(1e-6)
        weighted = feats * weight
        if self.pooling == "plane_concat":
            pooled = (weighted.reshape(b, slots, centers, -1).sum(dim=2)
                      / denom.unsqueeze(1)).reshape(b, -1)
        else:
            pooled = weighted.sum(dim=1) / denom
        return self.head(pooled)
