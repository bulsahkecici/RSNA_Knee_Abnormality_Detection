"""Torch trainer for StudyModel.

Microbatch slices the study axis. Loss is the sample-weighted mean of the
optimizer group, so a smaller microbatch matches a larger one. The last
short group still applies an optimizer update.
"""

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
from rsna_knee.training.losses import torch_masked_bce_parts


def _capture_rng() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _amp_enabled(device: torch.device) -> bool:
    return device.type == "cuda"


def _is_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text


def accumulation_for(effective_batch: int, microbatch: int) -> int:
    microbatch = max(1, microbatch)
    return max(1, int(np.ceil(effective_batch / microbatch)))


def planned_updates(n_studies: int, epochs: int, effective_batch: int) -> int:
    if n_studies <= 0:
        return 1
    per_epoch = int(np.ceil(n_studies / max(1, effective_batch)))
    return max(1, epochs * max(1, per_epoch))


def _study_count(batches: list[Any]) -> int:
    total = 0
    for batch in batches:
        known = getattr(batch, "n_studies", None)
        if known is not None:
            total += int(known)
        else:
            total += int(batch["images"].shape[0])
    return total


def _slice_batch(batch: dict[str, torch.Tensor], start: int, end: int) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value[start:end]
        else:
            out[key] = value
    return out


def _optimizer(model: StudyModel, lr_head: float, lr_backbone: float, unfrozen: bool) -> torch.optim.AdamW:
    head_params = [p for p in model.head.parameters() if p.requires_grad]
    groups: list[dict[str, Any]] = [{"params": head_params, "lr": lr_head}]
    if unfrozen:
        block = [p for p in model.encoder.blocks[-1].parameters() if p.requires_grad]
        if block:
            groups.append({"params": block, "lr": lr_backbone})
    return torch.optim.AdamW(groups, weight_decay=0.05)


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
    input_id: str | None = None,
    config_id: str | None = None,
) -> dict[str, Any]:
    """Train with masked BCE. `oob_once` raises one CUDA-OOM-shaped error so retry can shrink the slice."""
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(dev)
    n_studies = _study_count(batches)
    total_steps = planned_updates(n_studies, epochs, effective_batch)
    epoch = 0
    step = 0
    cursor = 0
    batch_offset = 0
    samples_in_group = 0
    order: list[int] | None = None
    img_size = model.img_size
    unfrozen = False
    if resume and ckpt_path and ckpt_path.exists():
        blob = torch_load(ckpt_path, map_location="cpu")
        if blob.get("encoder_name") != ENCODER_NAME:
            raise RuntimeError("refusing to resume a non-DINOv2 checkpoint")
        if int(blob.get("img_size", img_size)) != img_size:
            raise RuntimeError("image size changed; start a new experiment id")
        if int(blob.get("effective_batch", effective_batch)) != int(effective_batch):
            raise RuntimeError("effective batch changed; start a new experiment id")
        if input_id is not None and blob.get("input_id") not in {None, input_id}:
            raise RuntimeError("input identity changed; refusing resume")
        if config_id is not None and blob.get("config_id") not in {None, config_id}:
            raise RuntimeError("config identity changed; refusing resume")
        saved_samples = int(blob.get("samples_in_group", 0))
        if saved_samples > 0:
            microbatch = int(blob.get("microbatch", microbatch))
        model.load_state_dict(blob["model"])
        unfrozen = bool(blob.get("unfrozen"))
        if unfrozen:
            model.unfreeze_last_block()
        else:
            model.freeze_encoder()
        model = model.to(dev)
        opt = _optimizer(model, lr_head, lr_backbone, unfrozen)
        if blob.get("scheduler_t_max"):
            total_steps = int(blob["scheduler_t_max"])
        sched = CosineAnnealingLR(opt, T_max=total_steps)
        opt.load_state_dict(blob["optimizer"])
        sched.load_state_dict(blob["scheduler"])
        scaler = torch.amp.GradScaler("cuda") if _amp_enabled(dev) else None
        if scaler is not None and blob.get("scaler"):
            scaler.load_state_dict(blob["scaler"])
        epoch = int(blob["epoch"])
        step = int(blob["step"])
        cursor = int(blob.get("cursor", 0))
        batch_offset = int(blob.get("batch_offset", 0))
        samples_in_group = saved_samples
        order = list(blob.get("order") or [])
        _restore_rng(blob["rng"])
        input_id = blob.get("input_id", input_id)
        config_id = blob.get("config_id", config_id)
    else:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        model.freeze_encoder()
        opt = _optimizer(model, lr_head, lr_backbone, False)
        sched = CosineAnnealingLR(opt, T_max=total_steps)
        scaler = torch.amp.GradScaler("cuda") if _amp_enabled(dev) else None

    interrupted = False
    oob_fired = False
    weight_acc = 0.0
    supervised_weight = 0.0
    opt.zero_grad(set_to_none=True)
    group_cursor = cursor
    group_offset = batch_offset

    def persist() -> None:
        if not ckpt_path:
            return
        _save(
            ckpt_path,
            model,
            opt,
            sched,
            scaler,
            epoch,
            step,
            cursor,
            batch_offset,
            order or [],
            microbatch,
            img_size,
            unfrozen=unfrozen,
            samples_in_group=samples_in_group,
            effective_batch=effective_batch,
            total_steps=total_steps,
            input_id=input_id,
            config_id=config_id,
        )

    def mark_group() -> None:
        nonlocal group_cursor, group_offset
        group_cursor = cursor
        group_offset = batch_offset

    def rollback_group() -> None:
        nonlocal cursor, batch_offset, samples_in_group, weight_acc
        cursor = group_cursor
        batch_offset = group_offset
        samples_in_group = 0
        weight_acc = 0.0
        opt.zero_grad(set_to_none=True)
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    def flush() -> None:
        nonlocal step, samples_in_group, weight_acc, supervised_weight, unfrozen, interrupted
        if samples_in_group <= 0:
            return
        if weight_acc <= 0:
            opt.zero_grad(set_to_none=True)
            samples_in_group = 0
            weight_acc = 0.0
            mark_group()
            return
        supervised_weight += weight_acc
        denom = weight_acc
        if scaler is not None:
            scaler.unscale_(opt)
        for group in opt.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.grad.div_(denom)
        if scaler is not None:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        opt.zero_grad(set_to_none=True)
        sched.step()
        step += 1
        samples_in_group = 0
        weight_acc = 0.0
        if unfreeze_after_steps is not None and step >= unfreeze_after_steps and not unfrozen:
            model.unfreeze_last_block()
            params = [p for p in model.encoder.blocks[-1].parameters() if p.requires_grad]
            if params:
                opt.add_param_group({"params": params, "lr": lr_backbone})
                sched.base_lrs.append(lr_backbone)
            unfrozen = True
        mark_group()
        persist()
        if interrupt_after_steps is not None and step >= interrupt_after_steps:
            interrupted = True

    mark_group()
    while epoch < epochs and not interrupted:
        if order is None:
            order = np.random.permutation(len(batches)).tolist() if batches else []
        while cursor < len(order) and not interrupted:
            batch = batches[order[cursor]]
            n = int(batch["images"].shape[0])
            while batch_offset < n and not interrupted:
                room = effective_batch - samples_in_group
                if room <= 0:
                    flush()
                    continue
                take = min(max(1, microbatch), n - batch_offset, room)
                try:
                    if oob_once and not oob_fired and take > 1:
                        oob_fired = True
                        raise RuntimeError("CUDA out of memory")
                    sl = _slice_batch(batch, batch_offset, batch_offset + take)
                    images = sl["images"].to(dev)
                    y = sl["y"].to(dev)
                    mask = sl["y_mask"].to(dev)
                    weight = sl["y_weight"].to(dev)
                    slot = sl["slot_mask"].to(dev)
                    center = sl["center_mask"].to(dev)
                    if _amp_enabled(dev):
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            logits = model(images, slot, center)
                            loss_sum, wsum = torch_masked_bce_parts(logits, y, mask, weight)
                        assert scaler is not None
                        scaler.scale(loss_sum).backward()
                    else:
                        logits = model(images, slot, center)
                        loss_sum, wsum = torch_masked_bce_parts(logits, y, mask, weight)
                        loss_sum.backward()
                    weight_acc += float(wsum.detach().cpu())
                    samples_in_group += take
                    batch_offset += take
                except RuntimeError as exc:
                    if not _is_oom(exc):
                        raise
                    if microbatch <= 1:
                        raise
                    rollback_group()
                    microbatch = max(1, microbatch // 2)
                    continue
                if samples_in_group >= effective_batch:
                    flush()
            if interrupted:
                break
            cursor += 1
            batch_offset = 0
        if interrupted:
            break
        if samples_in_group > 0:
            flush()
        if interrupted:
            break
        cursor = 0
        batch_offset = 0
        order = None
        epoch += 1
    if ckpt_path:
        persist()
    return {
        "step": step,
        "interrupted": interrupted,
        "microbatch": microbatch,
        "effective_batch": effective_batch,
        "img_size": model.img_size,
        "encoder_name": ENCODER_NAME,
        "amp": _amp_enabled(dev),
        "unfrozen": unfrozen,
        "updates_planned": total_steps,
        "supervised_weight": supervised_weight,
    }


def _save(
    path,
    model,
    opt,
    sched,
    scaler,
    epoch,
    step,
    cursor,
    batch_offset,
    order,
    microbatch,
    img_size,
    *,
    unfrozen: bool,
    samples_in_group: int,
    effective_batch: int,
    total_steps: int,
    input_id: str | None,
    config_id: str | None,
) -> None:
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
            "batch_offset": batch_offset,
            "order": list(order),
            "microbatch": microbatch,
            "img_size": img_size,
            "unfrozen": unfrozen,
            "samples_in_group": samples_in_group,
            "effective_batch": effective_batch,
            "scheduler_t_max": total_steps,
            "input_id": input_id,
            "config_id": config_id,
            "rng": _capture_rng(),
        },
    )
