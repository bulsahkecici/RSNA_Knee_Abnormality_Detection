"""Torch trainer for StudyModel. Effective batch is preserved when microbatch drops."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from rsna_knee.models.dinov2 import ENCODER_NAME
from rsna_knee.models.study import StudyModel
from rsna_knee.training.checkpoint import torch_load, torch_save
from rsna_knee.training.losses import torch_masked_bce


def _capture_rng() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])


def _amp_enabled(device: torch.device) -> bool:
    return device.type == "cuda"


def accumulation_for(effective_batch: int, microbatch: int) -> int:
    microbatch = max(1, microbatch)
    return max(1, int(np.ceil(effective_batch / microbatch)))


def train_study_model(
    model: StudyModel,
    batches: list[dict[str, torch.Tensor]],
    *,
    epochs: int = 1,
    lr_head: float = 1e-3,
    lr_backbone: float = 1e-5,
    effective_batch: int = 4,
    microbatch: int = 1,
    seed: int = 0,
    ckpt_path: Path | None = None,
    resume: bool = False,
    interrupt_after_steps: int | None = None,
    unfreeze_after_steps: int | None = None,
    device: str | None = None,
    oob_once: bool = False,
) -> dict[str, Any]:
    """Train with masked BCE. `oob_once` simulates a single CUDA OOM to test microbatch reduction."""
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(dev)
    head_params = [p for p in model.head.parameters() if p.requires_grad]
    backbone_params = [p for p in model.encoder.parameters() if p.requires_grad]
    groups = [{"params": head_params, "lr": lr_head}]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": lr_backbone})
    opt = torch.optim.AdamW(groups, weight_decay=0.05)
    steps_per_epoch = max(1, int(np.ceil(len(batches) / max(1, microbatch))))
    total_steps = max(1, epochs * max(1, steps_per_epoch // accumulation_for(effective_batch, microbatch)))
    sched = CosineAnnealingLR(opt, T_max=total_steps)
    scaler = torch.amp.GradScaler("cuda") if _amp_enabled(dev) else None
    epoch = 0
    step = 0
    cursor = 0
    order: list[int] | None = None
    img_size = model.img_size
    if resume and ckpt_path and ckpt_path.exists():
        blob = torch_load(ckpt_path, map_location="cpu")
        if blob.get("encoder_name") != ENCODER_NAME:
            raise RuntimeError("refusing to resume a non-DINOv2 checkpoint")
        if int(blob.get("img_size", img_size)) != img_size:
            raise RuntimeError("image size changed; start a new experiment id")
        model.load_state_dict(blob["model"])
        opt.load_state_dict(blob["optimizer"])
        sched.load_state_dict(blob["scheduler"])
        if scaler is not None and blob.get("scaler"):
            scaler.load_state_dict(blob["scaler"])
        epoch = int(blob["epoch"])
        step = int(blob["step"])
        cursor = int(blob.get("cursor", 0))
        order = list(blob.get("order") or [])
        microbatch = int(blob.get("microbatch", microbatch))
        _restore_rng(blob["rng"])
    else:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    accum = accumulation_for(effective_batch, microbatch)
    interrupted = False
    unfrozen = any(p.requires_grad for p in model.encoder.parameters())
    oob_fired = False
    opt.zero_grad(set_to_none=True)
    micro_in_accum = 0

    while epoch < epochs:
        if order is None:
            order = np.random.permutation(len(batches)).tolist()
        assert order is not None
        while cursor < len(order):
            if oob_once and not oob_fired and microbatch > 1:
                oob_fired = True
                microbatch = max(1, microbatch // 2)
                accum = accumulation_for(effective_batch, microbatch)
                # Resolution and architecture stay put.
                if model.img_size != img_size:
                    raise RuntimeError("OOM handler must not change image size")
            idx = order[cursor]
            batch = batches[idx]
            images = batch["images"].to(dev)
            y = batch["y"].to(dev)
            mask = batch["y_mask"].to(dev)
            weight = batch["y_weight"].to(dev)
            slot = batch["slot_mask"].to(dev)
            center = batch["center_mask"].to(dev)
            if _amp_enabled(dev):
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(images, slot, center)
                    loss = torch_masked_bce(logits, y, mask, weight)
                assert scaler is not None
                scaler.scale(loss / accum).backward()
            else:
                logits = model(images, slot, center)
                loss = torch_masked_bce(logits, y, mask, weight)
                (loss / accum).backward()
            micro_in_accum += 1
            cursor += 1
            if micro_in_accum >= accum:
                if scaler is not None:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
                micro_in_accum = 0
                if unfreeze_after_steps is not None and step >= unfreeze_after_steps and not unfrozen:
                    model.unfreeze_last_block()
                    params = [p for p in model.encoder.blocks[-1].parameters() if p.requires_grad]
                    if params:
                        opt.add_param_group({"params": params, "lr": lr_backbone})
                    unfrozen = True
                if ckpt_path:
                    _save(ckpt_path, model, opt, sched, scaler, epoch, step, cursor, order, microbatch, img_size)
                if interrupt_after_steps is not None and step >= interrupt_after_steps:
                    interrupted = True
                    break
        if interrupted:
            break
        cursor = 0
        order = None
        epoch += 1
    if ckpt_path:
        _save(ckpt_path, model, opt, sched, scaler, epoch, step, cursor, order or [], microbatch, img_size)
    return {
        "step": step,
        "interrupted": interrupted,
        "microbatch": microbatch,
        "effective_batch": effective_batch,
        "img_size": model.img_size,
        "encoder_name": ENCODER_NAME,
        "amp": _amp_enabled(dev),
    }


def _save(path, model, opt, sched, scaler, epoch, step, cursor, order, microbatch, img_size) -> None:
    torch_save(
        path,
        {
            "encoder_name": ENCODER_NAME,
            "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "scaler": None if scaler is None else scaler.state_dict(),
            "epoch": epoch,
            "step": step,
            "cursor": cursor,
            "order": list(order),
            "microbatch": microbatch,
            "img_size": img_size,
            "rng": _capture_rng(),
        },
    )
