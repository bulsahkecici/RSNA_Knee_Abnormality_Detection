"""Torch trainer for StudyModel.

Microbatch slices the study axis. Loss is the sample-weighted mean of the
optimizer group, so a smaller microbatch matches a larger one. The last
short group still applies an optimizer update.

`train_centers_per_slot` encodes a random subset of center windows per slot
for each training study. The subset is a pure function of (seed, epoch,
batch index, row), so it is invariant to microbatch/OOM retries and resume.
Evaluation never samples: it sees every center.
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from rsna_knee.models.families import normalize_unfreeze
from rsna_knee.models.study import StudyModel
from rsna_knee.training.checkpoint import torch_load, torch_save
from rsna_knee.training.losses import torch_masked_bce_parts

NUMERICAL_POLICY_VERSION = "amp-successful-update.v1"
WINDOW_SAMPLING_VERSION = "per-study-seeded-subset.v1"
# Checkpoints from identity version 2 compare against the full experiment
# identity; older checkpoints compare against the legacy config id.
IDENTITY_VERSION = 2


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


def _optimizer(model: StudyModel, lr_head: float, lr_backbone: float, unfrozen: bool,
               layer_decay: float | None = None) -> torch.optim.AdamW:
    head_params = [p for name, p in model.named_parameters()
                   if not name.startswith("encoder.") and p.requires_grad]
    groups: list[dict[str, Any]] = [{"params": head_params, "lr": lr_head}]
    if unfrozen:
        groups.extend(model.backbone_param_groups(lr_backbone, layer_decay))
    return torch.optim.AdamW(groups, weight_decay=0.05)


def sample_windows(batch: dict[str, Any], k: int, *, seed: int, epoch: int, batch_index: int,
                   row_offset: int) -> dict[str, Any]:
    """Select k center windows per slot for each study row, keeping physical order.

    Valid centers (center_mask > 0) are preferred; a slot with fewer valid
    centers keeps all of them and pads with masked ones, so absent slots and
    short series stay masked rather than duplicated into the loss.
    """
    images, center = batch["images"], batch["center_mask"]
    b, slots, centers = images.shape[:3]
    if k > centers:
        raise ValueError(f"train_centers_per_slot={k} exceeds {centers} cached centers")
    if k == centers:
        return batch
    index = torch.empty((b, slots, k), dtype=torch.long)
    valid_all = (center > 0).cpu().numpy()
    for row in range(b):
        rng = np.random.default_rng([seed, epoch, batch_index, row_offset + row])
        for slot in range(slots):
            valid = np.flatnonzero(valid_all[row, slot])
            invalid = np.flatnonzero(~valid_all[row, slot])
            if len(valid) >= k:
                chosen = rng.choice(valid, size=k, replace=False)
            else:
                chosen = np.concatenate([valid, invalid[: k - len(valid)]])
            index[row, slot] = torch.from_numpy(np.sort(chosen))
    rows = torch.arange(b).view(b, 1, 1)
    planes = torch.arange(slots).view(1, slots, 1)
    out = dict(batch)
    out["images"] = images[rows, planes, index]
    out["center_mask"] = center[rows, planes, index]
    previous = batch.get("center_index") if isinstance(batch, dict) else None
    out["center_index"] = index if previous is None else previous[rows, planes, index]
    return out


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
    unfreeze_blocks: int | str = 1,
    layer_decay: float | None = None,
    train_centers_per_slot: int | None = None,
    init_source: str | None = None,
    device: str | None = None,
    oob_once: bool = False,
    input_id: str | None = None,
    config_id: str | None = None,
    legacy_config_id: str | None = None,
) -> dict[str, Any]:
    """Train with masked BCE. `oob_once` raises one CUDA-OOM-shaped error so retry can shrink the slice.

    `unfreeze_blocks` is the number of last encoder units (family-defined) or
    "all"; `layer_decay` scales unfrozen units' learning rates by depth.
    `init_source` names where model weights came from (official_pretrained or
    supervised_checkpoint:<sha256>); it is recorded and checked on resume.
    """
    unfreeze_blocks = normalize_unfreeze(unfreeze_blocks)
    layer_decay = None if layer_decay is None else float(layer_decay)
    if train_centers_per_slot is not None:
        if isinstance(train_centers_per_slot, bool) or int(train_centers_per_slot) < 1:
            raise ValueError("train_centers_per_slot must be a positive integer")
        train_centers_per_slot = int(train_centers_per_slot)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(dev)
    window_seed = int(seed)
    identity_version = IDENTITY_VERSION
    n_studies = _study_count(batches)
    total_steps = planned_updates(n_studies, epochs, effective_batch)
    epoch = 0
    step = 0
    attempted_updates = 0
    skipped_updates = 0
    cursor = 0
    batch_offset = 0
    samples_in_group = 0
    order: list[int] | None = None
    img_size = model.img_size
    unfrozen = False
    validated_completed_resume = False
    resume_validation: dict[str, bool] = {}
    previous_supervised_weight = 0.0
    if resume and ckpt_path and ckpt_path.exists():
        blob = torch_load(ckpt_path, map_location="cpu")
        if blob.get("encoder_name") != model.encoder_name:
            raise RuntimeError(f"refusing cross-family resume: checkpoint encoder is {blob.get('encoder_name')}, "
                               f"model is {model.encoder_name}")
        if blob.get("encoder_config", model.encoder_config) != model.encoder_config:
            raise RuntimeError("encoder configuration changed; start a new experiment id")
        if blob.get("aggregator_config") != model.aggregator_config:
            raise RuntimeError("aggregator configuration changed; start a new experiment id")
        if blob.get("input_resize") != model.input_resize:
            raise RuntimeError("input resize changed; start a new experiment id")
        if int(blob.get("img_size", img_size)) != img_size:
            raise RuntimeError("image size changed; start a new experiment id")
        if blob.get("pooling", "mean") != model.pooling:
            raise RuntimeError("pooling changed; start a new experiment id")
        if blob.get("pooling_version", model.pooling_version) != model.pooling_version:
            raise RuntimeError("pooling version changed; start a new experiment id")
        if int(blob.get("effective_batch", effective_batch)) != int(effective_batch):
            raise RuntimeError("effective batch changed; start a new experiment id")
        if input_id is not None and blob.get("input_id") not in {None, input_id}:
            raise RuntimeError("input identity changed; refusing resume")
        expected_config = config_id
        # A resumed legacy checkpoint keeps its legacy identity for every later resume.
        identity_version = int(blob.get("identity_version", 1))
        if identity_version < IDENTITY_VERSION and legacy_config_id is not None:
            expected_config = legacy_config_id
        if expected_config is not None and blob.get("config_id") not in {None, expected_config}:
            raise RuntimeError("config identity changed; refusing resume")
        if normalize_unfreeze(blob.get("unfreeze_blocks", 1)) != unfreeze_blocks:
            raise RuntimeError("unfrozen block count changed; refusing resume")
        if blob.get("layer_decay") != layer_decay:
            raise RuntimeError("layer decay changed; refusing resume")
        if blob.get("train_centers_per_slot") != train_centers_per_slot:
            raise RuntimeError("training window sampling changed; refusing resume")
        if "init_source" in blob:
            if init_source is not None and blob["init_source"] != init_source:
                raise RuntimeError("initialization source changed; refusing resume")
            init_source = blob["init_source"]
        else:
            # Legacy checkpoints never recorded how they were initialized.
            init_source = "unrecorded_legacy"
        window_seed = int(blob.get("window_seed", window_seed))
        saved_samples = int(blob.get("samples_in_group", 0))
        if saved_samples > 0:
            microbatch = int(blob.get("microbatch", microbatch))
        model.load_state_dict(blob["model"])
        unfrozen = bool(blob.get("unfrozen"))
        model.freeze_encoder()
        if unfrozen:
            model.unfreeze_encoder(unfreeze_blocks)
        model = model.to(dev)
        opt = _optimizer(model, lr_head, lr_backbone, unfrozen, layer_decay)
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
        attempted_updates = int(blob.get("attempted_updates", step))
        skipped_updates = int(blob.get("skipped_updates", 0))
        cursor = int(blob.get("cursor", 0))
        batch_offset = int(blob.get("batch_offset", 0))
        samples_in_group = saved_samples
        order = list(blob.get("order") or [])
        _restore_rng(blob["rng"])
        previous_supervised_weight = float(blob.get("supervised_weight") or 0)
        # Completion is defined by consumed epochs, not successful updates:
        # AMP can skip optimizer updates and masked-only groups do not step.
        resume_validation = {
            "epochs_complete": epoch >= epochs,
            "positive_updates": step > 0,
            "update_counters_consistent": skipped_updates >= 0 and attempted_updates == step + skipped_updates,
            "optimizer_state_present": bool(blob.get("optimizer", {}).get("state")),
            "input_identity_matches": input_id is not None and blob.get("input_id") == input_id,
            "config_identity_matches": expected_config is not None and blob.get("config_id") == expected_config,
            "no_partial_group": saved_samples == 0,
        }
        validated_completed_resume = all(resume_validation.values())
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
    resumed_mid_epoch = bool(cursor or batch_offset or samples_in_group)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    train_started = epoch_started = time.perf_counter()
    epoch_seconds: list[float] = []
    studies_seen = 0
    weight_acc = 0.0
    supervised_weight = previous_supervised_weight
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
            attempted_updates=attempted_updates,
            skipped_updates=skipped_updates,
            unfreeze_blocks=unfreeze_blocks,
            supervised_weight=supervised_weight,
            layer_decay=layer_decay,
            train_centers_per_slot=train_centers_per_slot,
            window_seed=window_seed,
            init_source=init_source,
            identity_version=identity_version,
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
        nonlocal attempted_updates, skipped_updates, studies_seen
        if samples_in_group <= 0:
            return
        studies_seen += samples_in_group
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
        attempted_updates += 1
        updated = True
        if scaler is not None:
            scale_before = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            # GradScaler lowers its scale when nonfinite gradients skip the
            # optimizer. An unchanged or increased scale means it stepped.
            updated = scaler.get_scale() >= scale_before
        else:
            opt.step()
        opt.zero_grad(set_to_none=True)
        if updated:
            sched.step()
            step += 1
        else:
            skipped_updates += 1
        samples_in_group = 0
        weight_acc = 0.0
        if updated and unfreeze_after_steps is not None and step >= unfreeze_after_steps and not unfrozen:
            model.unfreeze_encoder(unfreeze_blocks)
            for group in model.backbone_param_groups(lr_backbone, layer_decay):
                opt.add_param_group(group)
                sched.base_lrs.append(group["lr"])
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
                    if train_centers_per_slot is not None:
                        sl = sample_windows(sl, train_centers_per_slot, seed=window_seed, epoch=epoch,
                                            batch_index=int(order[cursor]), row_offset=batch_offset)
                    images = sl["images"].to(dev)
                    y = sl["y"].to(dev)
                    mask = sl["y_mask"].to(dev)
                    weight = sl["y_weight"].to(dev)
                    slot = sl["slot_mask"].to(dev)
                    center = sl["center_mask"].to(dev)
                    index = sl.get("center_index") if isinstance(sl, dict) else None
                    extra = () if index is None else (index.to(dev),)
                    if _amp_enabled(dev):
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            logits = model(images, slot, center, *extra)
                            loss_sum, wsum = torch_masked_bce_parts(logits, y, mask, weight)
                        assert scaler is not None
                        scaler.scale(loss_sum).backward()
                    else:
                        logits = model(images, slot, center, *extra)
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
        now = time.perf_counter()
        epoch_seconds.append(round(now - epoch_started, 3))
        epoch_started = now
    train_seconds = time.perf_counter() - train_started
    peak_memory = int(torch.cuda.max_memory_allocated(dev)) if dev.type == "cuda" else None
    if ckpt_path:
        persist()
    return {
        "step": step,
        "attempted_updates": attempted_updates,
        "skipped_updates": skipped_updates,
        "numerical_policy": NUMERICAL_POLICY_VERSION,
        "interrupted": interrupted,
        "microbatch": microbatch,
        "effective_batch": effective_batch,
        "img_size": model.img_size,
        "encoder_name": model.encoder_name,
        "unfreeze_blocks": unfreeze_blocks,
        "layer_decay": layer_decay,
        "train_centers_per_slot": train_centers_per_slot,
        "window_sampling": None if train_centers_per_slot is None else WINDOW_SAMPLING_VERSION,
        "init_source": init_source,
        "amp": _amp_enabled(dev),
        "unfrozen": unfrozen,
        "updates_planned": total_steps,
        "supervised_weight": supervised_weight,
        "validated_completed_resume": validated_completed_resume,
        "resume_validation": resume_validation,
        "epochs_completed": epoch,
        # Timing covers this process only; a resumed run's first entry is a partial epoch.
        "epoch_wall_seconds": epoch_seconds,
        "resumed_mid_epoch": resumed_mid_epoch,
        "train_wall_seconds": round(train_seconds, 3),
        "train_studies_seen": studies_seen,
        "seconds_per_train_study": round(train_seconds / studies_seen, 6) if studies_seen else None,
        "peak_cuda_memory_bytes": peak_memory,
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
    attempted_updates: int = 0,
    skipped_updates: int = 0,
    unfreeze_blocks: int | str = 1,
    supervised_weight: float = 0.0,
    layer_decay: float | None = None,
    train_centers_per_slot: int | None = None,
    window_seed: int = 0,
    init_source: str | None = None,
    identity_version: int = IDENTITY_VERSION,
) -> None:
    torch_save(
        path,
        {
            "encoder_name": model.encoder_name,
            "encoder_config": model.encoder_config,
            "aggregator_config": model.aggregator_config,
            "input_resize": model.input_resize,
            "identity_version": identity_version,
            "layer_decay": layer_decay,
            "train_centers_per_slot": train_centers_per_slot,
            "window_sampling": None if train_centers_per_slot is None else WINDOW_SAMPLING_VERSION,
            "window_seed": window_seed,
            "init_source": init_source,
            "pooling": model.pooling,
            "pooling_version": model.pooling_version,
            "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "optimizer": opt.state_dict(),
            "unfreeze_blocks": unfreeze_blocks,
            "supervised_weight": supervised_weight,
            "scheduler": sched.state_dict(),
            "scaler": None if scaler is None else scaler.state_dict(),
            "epoch": epoch,
            "step": step,
            "attempted_updates": attempted_updates,
            "skipped_updates": skipped_updates,
            "numerical_policy": NUMERICAL_POLICY_VERSION,
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
