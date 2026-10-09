"""Encoder family registry for StudyModel.

Every family maps [N,3,H,W] images in [0,1] (or 0..255, auto-scaled as in
DINOv2) to [N, embed_dim] features and declares its own input normalization.

Adaptation units ("unfreeze the last N"), ordered input -> output:

* ``dinov2_vits14``: the 12 transformer blocks. Stem = patch_embed, cls_token,
  pos_embed, mask_token; top = final LayerNorm.
* ``convnext_tiny``: the 18 ConvNeXt blocks flattened over stages (3,3,9,3).
  A stage's downsample layer belongs to that stage's first block, so N=3
  trains the whole last stage including its downsample. Stem = patchify stem;
  top = norm_pre and the pooled-feature LayerNorm in ``head``.
* ``coatnet_0_rw_224``: the 4 stages (2 MBConv, 2 transformer). Stem = conv
  stem; top = final norm and ``head`` (pool only, fc removed).

``"all"`` trains stem, every unit and top. Last-N leaves stem and top frozen
for every family (the DINOv2 two-block recipe kept its final norm frozen).
BatchNorm statistics are never updated; see StudyModel.train.

Real families are built without network access: architecture via
``timm.create_model(..., pretrained=False)`` and weights from a local file
whose sha256 must match. Tiny test configs are marked and cannot pass the
production guard.
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml

from rsna_knee.hashing import sha256_file
from rsna_knee.models.dinov2 import ENCODER_NAME as DINOV2_NAME
from rsna_knee.models.dinov2 import Dinov2ViTS14, load_dinov2_vits14
from rsna_knee.paths import CONFIG_DIR

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class EncoderFamily:
    name: str
    display: str
    embed_dim: int
    n_units: int
    unit_kind: str
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    timm_name: str | None = None
    size_multiple: int = 32

    @property
    def normalization(self) -> dict[str, Any]:
        return {"input_range": "0-1", "mean": list(self.mean), "std": list(self.std)}


FAMILIES: dict[str, EncoderFamily] = {
    DINOV2_NAME: EncoderFamily(DINOV2_NAME, "DINOv2", 384, 12, "vit_block", IMAGENET_MEAN, IMAGENET_STD,
                               size_multiple=14),
    "convnext_tiny": EncoderFamily("convnext_tiny", "ConvNeXt-Tiny", 768, 18, "convnext_block",
                                   IMAGENET_MEAN, IMAGENET_STD, timm_name="convnext_tiny"),
    # timm's coatnet_0_rw_224.sw_in1k weights use 0.5/0.5 normalization, not ImageNet.
    "coatnet_0_rw_224": EncoderFamily("coatnet_0_rw_224", "CoAtNet-0", 768, 4, "coatnet_stage",
                                      (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), timm_name="coatnet_0_rw_224"),
}

TEST_ENCODER_NAMES = {"tiny", "test", "tiny_test_encoder", "dinov2_small_or_stub", "TorchDinoStub"}


def get_family(name: str | None) -> EncoderFamily:
    if name not in FAMILIES:
        raise ValueError(f"unregistered encoder family: {name!r}")
    return FAMILIES[name]


def validate_img_size(name: str, img_size: int) -> int:
    """Model input size per family: DINOv2 patches of 14 (224/308/392; pos_embed is
    interpolated on load), ConvNeXt/CoAtNet total stride 32 (224/256/320/384)."""
    spec = get_family(name)
    if isinstance(img_size, bool) or int(img_size) != img_size or img_size < spec.size_multiple:
        raise ValueError(f"invalid img_size {img_size!r} for {name}")
    if int(img_size) % spec.size_multiple:
        raise ValueError(f"{name} img_size must be a multiple of {spec.size_multiple}, got {img_size}")
    return int(img_size)


def normalize_unfreeze(value: Any) -> int | str:
    """Unfreeze depth is a positive int or "all"; legacy checkpoints stored ints."""
    if isinstance(value, str):
        if value == "all":
            return "all"
        if value.isdigit():
            value = int(value)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("invalid number of encoder blocks to unfreeze")
    return value


class TimmEncoder(nn.Module):
    """timm backbone with classifier removed; returns pooled pre-logit features."""

    def __init__(self, family: str, model: nn.Module, img_size: int, *, test_config: dict | None = None):
        super().__init__()
        spec = get_family(family)
        if spec.timm_name is None:
            raise ValueError(f"{family} is not a timm family")
        self.name = family
        self.model = model
        self.img_size = img_size
        self.embed_dim = int(getattr(model, "head_hidden_size", None) or model.num_features)
        self.is_test_encoder = test_config is not None
        self.test_config = test_config
        self.register_buffer("mean", torch.tensor(spec.mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(spec.std).view(1, 3, 1, 1), persistent=False)
        self.provenance: dict[str, Any] = {"encoder": family, "pretrained": False, "img_size": img_size,
                                           "test_config": self.is_test_encoder}

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.shape[-1] != self.img_size or images.shape[-2] != self.img_size:
            raise ValueError(f"expected {self.img_size}px, got {tuple(images.shape)}")
        if images.shape[1] != 3:
            raise ValueError(f"{self.name} expects 3 channels")
        images = images.float()
        if float(images.detach().max()) > 1.5:
            images = images / 255.0
        return self.model((images - self.mean) / self.std)


def _timm():
    try:
        import timm
    except ImportError as exc:  # pragma: no cover - depends on the runtime image
        raise RuntimeError("timm is required for ConvNeXt/CoAtNet encoders") from exc
    return timm


def _tiny_timm_model(family: str, img_size: int) -> nn.Module:
    """Random tiny architecture of the same family. Tests only; never a result."""
    if family == "convnext_tiny":
        from timm.models.convnext import ConvNeXt

        return ConvNeXt(depths=(1, 1, 1, 1), dims=(8, 16, 32, 32), num_classes=0)
    if family == "coatnet_0_rw_224":
        from timm.models.maxxvit import MaxxVit, model_cfgs

        base = model_cfgs["coatnet_0_rw"]
        cfg = dataclasses.replace(base, embed_dim=(8, 16, 32, 32), depths=(1, 1, 1, 1), stem_width=(8, 8),
                                  transformer_cfg=dataclasses.replace(base.transformer_cfg, dim_head=16))
        return MaxxVit(cfg, img_size=img_size, num_classes=0)
    raise ValueError(f"no tiny test config for {family}")


def build_encoder(name: str, img_size: int = 224, *, test_config: bool = False) -> nn.Module:
    """Architecture only, random init, no download. Weights come from load_encoder."""
    spec = get_family(name)
    validate_img_size(name, img_size)
    if name == DINOV2_NAME:
        if test_config:
            raise ValueError("DINOv2 tests use the real ViT-S/14 geometry at a small image size")
        return Dinov2ViTS14(img_size=img_size)
    if test_config:
        return TimmEncoder(name, _tiny_timm_model(name, img_size), img_size,
                           test_config={"family": name, "variant": "tiny_random"})
    kwargs: dict[str, Any] = {"pretrained": False, "num_classes": 0}
    if name.startswith("coatnet"):
        kwargs["img_size"] = img_size
    return TimmEncoder(name, _timm().create_model(spec.timm_name, **kwargs), img_size)


def encoder_config(encoder: nn.Module) -> dict[str, Any]:
    """Enough to rebuild the skeleton for evaluation; stored in checkpoints."""
    return {"family": family_of(encoder).name, "img_size": int(encoder.img_size),
            "test_config": bool(getattr(encoder, "is_test_encoder", False))}


def build_from_config(config: dict[str, Any] | None, encoder_name: str, img_size: int,
                      *, allow_test_encoder: bool = False) -> nn.Module:
    config = dict(config or {"family": encoder_name, "img_size": img_size, "test_config": False})
    if config.get("family") != encoder_name:
        raise RuntimeError("checkpoint encoder config disagrees with encoder_name")
    if config.get("test_config") and not allow_test_encoder:
        raise RuntimeError("checkpoint was trained with a tiny test encoder; not a production model")
    return build_encoder(encoder_name, img_size, test_config=bool(config.get("test_config")))


def family_of(encoder: nn.Module) -> EncoderFamily:
    if isinstance(encoder, Dinov2ViTS14):
        return FAMILIES[DINOV2_NAME]
    if isinstance(encoder, TimmEncoder):
        return get_family(encoder.name)
    raise TypeError("production StudyModel requires a registered encoder family (Dinov2ViTS14, "
                    f"convnext_tiny, coatnet_0_rw_224), not a stub: {type(encoder).__name__}")


def validate_encoder(encoder: nn.Module, *, allow_test_encoder: bool = False) -> EncoderFamily:
    """Production guard: registered family, real geometry, no tiny test config."""
    spec = family_of(encoder)
    if getattr(encoder, "is_test_encoder", False) and not allow_test_encoder:
        raise TypeError(f"{spec.name} tiny test encoder is not a production encoder")
    if getattr(encoder, "is_test_encoder", False):
        return spec
    if int(encoder.embed_dim) != spec.embed_dim or len(encoder_layers(encoder)[1]) != spec.n_units:
        raise TypeError(f"{spec.name} encoder geometry does not match the registered architecture")
    return spec


def _params(*items: Any) -> list[nn.Parameter]:
    out: list[nn.Parameter] = []
    for item in items:
        if isinstance(item, nn.Parameter):
            out.append(item)
        elif isinstance(item, nn.Module):
            out.extend(item.parameters())
    return out


def encoder_layers(encoder: nn.Module) -> tuple[list[nn.Parameter], list[list[nn.Parameter]], list[nn.Parameter]]:
    """(stem, units input->output, top). Every encoder parameter appears exactly once."""
    if isinstance(encoder, Dinov2ViTS14):
        stem = _params(encoder.patch_embed, encoder.cls_token, encoder.pos_embed, encoder.mask_token)
        return stem, [_params(block) for block in encoder.blocks], _params(encoder.norm)
    if not isinstance(encoder, TimmEncoder):
        raise TypeError(f"no adaptation units for {type(encoder).__name__}")
    model = encoder.model
    if encoder.name == "convnext_tiny":
        units: list[list[nn.Parameter]] = []
        for stage in model.stages:
            for index, block in enumerate(stage.blocks):
                units.append(_params(stage.downsample, block) if index == 0 else _params(block))
        return _params(model.stem), units, _params(model.norm_pre, model.head)
    if encoder.name == "coatnet_0_rw_224":
        return _params(model.stem), [_params(stage) for stage in model.stages], _params(model.norm, model.head)
    raise TypeError(f"no adaptation units for {encoder.name}")


def unfreeze(encoder: nn.Module, depth: int | str) -> None:
    depth = normalize_unfreeze(depth)
    stem, units, top = encoder_layers(encoder)
    if depth == "all":
        for param in encoder.parameters():
            param.requires_grad = True
        return
    if depth > len(units):
        raise ValueError("invalid number of encoder blocks to unfreeze")
    for unit in units[-depth:]:
        for param in unit:
            param.requires_grad = True


def layerwise_groups(encoder: nn.Module, lr: float, decay: float | None) -> list[dict[str, Any]]:
    """Optimizer groups for trainable encoder parameters.

    Without decay: one group at ``lr`` (the legacy layout). With decay d: the
    last unit and the top get ``lr``; a unit k steps below the last gets
    ``lr * d**k``; the stem gets ``lr * d**n_units``. Frozen params are skipped.
    """
    if decay is None:
        params = [p for p in encoder.parameters() if p.requires_grad]
        return [{"params": params, "lr": lr}] if params else []
    if not 0 < decay <= 1:
        raise ValueError("layer_decay must be in (0, 1]")
    stem, units, top = encoder_layers(encoder)
    n = len(units)
    layers = [(n, stem), *[(n - 1 - i, unit) for i, unit in enumerate(units)], (0, top)]
    groups = []
    for depth_from_top, params in layers:
        trainable = [p for p in params if p.requires_grad]
        if trainable:
            groups.append({"params": trainable, "lr": lr * decay**depth_from_top})
    return groups


def _allowlist(name: str) -> set[str]:
    """Locally hashed weight files: env JSON override, else configs/models/<name>.yaml.

    No published hash is invented here; add one only after hashing the file.
    """
    env = os.environ.get(f"RSNA_{name.upper()}_ALLOWLIST")
    path = Path(env) if env else CONFIG_DIR / "models" / f"{name}.yaml"
    if not path.is_file():
        return set()
    try:
        text = path.read_text(encoding="utf-8")
        data = json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
    except (OSError, json.JSONDecodeError, yaml.YAMLError):
        return set()
    values = data.get("weights_sha256_allowlist", data.get("sha256")) if isinstance(data, dict) else data
    return {str(v) for v in values if v} if isinstance(values, list) else set()


def weight_trust_for(name: str, path: Path) -> dict[str, Any]:
    """DINOv2 keeps its own trust policy. timm families need a locally hashed allowlist entry."""
    if name == DINOV2_NAME:
        from rsna_knee.models.weights import weight_trust

        return weight_trust(path)
    get_family(name)
    digest = sha256_file(path)
    allowlisted = digest in _allowlist(name)
    return {"sha256": digest, "allowlisted": allowlisted, "official": False, "pretrained": False,
            "reason": "allowlisted" if allowlisted else "not_allowlisted"}


def _read_state(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        raw: Any = load_file(str(path), device="cpu")
    else:
        raw = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("state_dict", "model"):
        if isinstance(raw, dict) and isinstance(raw.get(key), dict):
            raw = raw[key]
    if not isinstance(raw, dict):
        raise ValueError("weights file is not a state dict")
    return {str(k).removeprefix("module."): v for k, v in raw.items() if torch.is_tensor(v)}


def load_encoder(name: str, weights: str | Path, img_size: int = 224, *, expected_sha256: str | None = None,
                 test_config: bool = False) -> nn.Module:
    """Pretrained encoder from a local file. sha256 must match; key sets must match exactly."""
    path = Path(weights)
    if not path.is_file():
        raise FileNotFoundError(f"{name} weights missing: {path}")
    validate_img_size(name, img_size)
    if name == DINOV2_NAME:
        encoder = load_dinov2_vits14(path, img_size=img_size)
        if expected_sha256 and encoder.provenance["sha256"] != expected_sha256:
            raise RuntimeError("DINOv2 weights sha256 mismatch")
        return encoder
    if not expected_sha256:
        raise RuntimeError(f"{name} weights need an expected sha256")
    digest = sha256_file(path)
    if digest != expected_sha256:
        raise RuntimeError(f"{name} weights sha256 mismatch")
    encoder = build_encoder(name, img_size, test_config=test_config)
    state = {k: v for k, v in _read_state(path).items() if not k.startswith(("head.fc.", "fc."))}
    missing, unexpected = encoder.model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"{name} architecture mismatch. missing={list(missing)[:12]} "
                           f"unexpected={list(unexpected)[:12]}")
    encoder.provenance = {"encoder": name, "pretrained": True, "checkpoint": str(path), "sha256": digest,
                          "img_size": img_size, "test_config": encoder.is_test_encoder,
                          "weights_reason": weight_trust_for(name, path)["reason"]}
    return encoder
