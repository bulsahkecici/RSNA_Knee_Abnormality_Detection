"""DINOv2 ViT-S/14.

State-dict keys follow the facebookresearch/dinov2 ViT-S/14 checkpoint
(`patch_embed.proj`, `blocks.*.attn.qkv`, `blocks.*.ls1.gamma`, `mask_token`).
A mismatched key set is an error. `is_stub` is not a success signal.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from rsna_knee.models.weights import weight_trust

ENCODER_NAME = "dinov2_vits14"
EMBED_DIM = 384
DEPTH = 12
HEADS = 6
PATCH = 14


class PatchEmbed(nn.Module):
    def __init__(self, img_size: int, patch: int, dim: int):
        super().__init__()
        if img_size % patch != 0:
            raise ValueError(f"img_size {img_size} is not divisible by patch {patch}")
        self.img_size = img_size
        self.patch_size = patch
        self.grid = img_size // patch
        self.proj = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.num_heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scale = self.head_dim**-0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj(out)


class Mlp(nn.Module):
    def __init__(self, dim: int, ratio: int = 4):
        super().__init__()
        hidden = dim * ratio
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class LayerScale(nn.Module):
    def __init__(self, dim: int, init: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim) * init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class Block(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.ls1 = LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim)
        self.ls2 = LayerScale(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class Dinov2ViTS14(nn.Module):
    """Real ViT-S/14 module. Not a conv stub."""

    name = ENCODER_NAME
    embed_dim = EMBED_DIM

    def __init__(self, img_size: int = 224):
        super().__init__()
        self.img_size = img_size
        self.patch_embed = PatchEmbed(img_size, PATCH, EMBED_DIM)
        n_patches = self.patch_embed.grid**2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, EMBED_DIM))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + n_patches, EMBED_DIM))
        self.mask_token = nn.Parameter(torch.zeros(1, EMBED_DIM))
        self.blocks = nn.ModuleList([Block(EMBED_DIM, HEADS) for _ in range(DEPTH)])
        self.norm = nn.LayerNorm(EMBED_DIM)
        self.provenance: dict[str, Any] = {"encoder": ENCODER_NAME, "pretrained": False, "img_size": img_size}

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.shape[-1] != self.img_size or images.shape[-2] != self.img_size:
            raise ValueError(f"expected {self.img_size}px, got {tuple(images.shape)}")
        if images.shape[1] != 3:
            raise ValueError("DINOv2 expects 3 channels")
        images = images.float()
        if float(images.detach().max()) > 1.5:
            images = images / 255.0
        mean = images.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = images.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        images = (images - mean) / std
        x = self.patch_embed(images)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x[:, 0]


def interpolate_pos_embed(pos_embed: torch.Tensor, grid: int) -> torch.Tensor:
    cls, grid_tokens = pos_embed[:, :1], pos_embed[:, 1:]
    old = int(grid_tokens.shape[1] ** 0.5)
    if old * old != grid_tokens.shape[1]:
        raise ValueError("pos_embed is not a square grid")
    if old == grid:
        return pos_embed
    feat = grid_tokens.reshape(1, old, old, -1).permute(0, 3, 1, 2)
    feat = F.interpolate(feat, size=(grid, grid), mode="bicubic", align_corners=False)
    feat = feat.permute(0, 2, 3, 1).reshape(1, grid * grid, -1)
    return torch.cat([cls, feat], dim=1)


def _clean_state(blob: Any) -> dict[str, torch.Tensor]:
    if isinstance(blob, dict) and "model" in blob and isinstance(blob["model"], dict):
        blob = blob["model"]
    if isinstance(blob, dict) and "state_dict" in blob and isinstance(blob["state_dict"], dict):
        blob = blob["state_dict"]
    if not isinstance(blob, dict):
        raise ValueError("checkpoint is not a state dict")
    out: dict[str, torch.Tensor] = {}
    for key, value in blob.items():
        name = str(key)
        for prefix in ("module.", "backbone.", "student.backbone.", "teacher.backbone."):
            if name.startswith(prefix):
                name = name[len(prefix) :]
        if torch.is_tensor(value):
            out[name] = value
    return out


def load_dinov2_vits14(checkpoint: str | Path, img_size: int = 224) -> Dinov2ViTS14:
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"DINOv2 checkpoint missing: {path}")
    model = Dinov2ViTS14(img_size=img_size)
    raw = torch.load(path, map_location="cpu", weights_only=False)
    state = _clean_state(raw)
    notes: list[str] = []
    if "pos_embed" in state and tuple(state["pos_embed"].shape) != tuple(model.pos_embed.shape):
        grid = model.patch_embed.grid
        state["pos_embed"] = interpolate_pos_embed(state["pos_embed"], grid)
        notes.append(f"pos_embed_interpolated_to_{grid}")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "DINOv2 architecture mismatch. "
            f"missing={list(missing)[:12]} unexpected={list(unexpected)[:12]}"
        )
    trust = weight_trust(path)
    model.provenance = {
        "encoder": ENCODER_NAME,
        "pretrained": bool(trust["official"]),
        "allowlisted": bool(trust["allowlisted"]),
        "checkpoint": str(path),
        "sha256": trust["sha256"],
        "img_size": img_size,
        "notes": notes,
        "weights_reason": trust["reason"],
    }
    return model
