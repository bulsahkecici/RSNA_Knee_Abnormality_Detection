"""Training loop with effective-batch preservation and interruptible resume."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from rsna_knee.models.classifier import LinearHead, StudyClassifier
from rsna_knee.models.encoder import TinyEncoder
from rsna_knee.training.checkpoint import load_numpy_checkpoint, save_numpy_checkpoint
from rsna_knee.training.losses import masked_bce_with_logits, sigmoid


def _grad_head(logits: np.ndarray, y: np.ndarray, mask: np.ndarray, weight: np.ndarray, pooled: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = sigmoid(logits)
    t = np.where(np.isnan(y), 0.0, y)
    w = mask * weight
    den = max(float(w.sum()), 1e-8)
    delta = ((p - t) * w) / den  # [B,12]
    dW = delta.T @ pooled
    db = delta.sum(axis=0)
    return dW.astype(np.float32), db.astype(np.float32)


def train_tiny(
    dataset_items: list[dict[str, Any]],
    *,
    epochs: int = 2,
    lr: float = 0.05,
    seed: int = 0,
    ckpt_dir: Path | None = None,
    resume: bool = True,
    interrupt_after_steps: int | None = None,
    start_step: int = 0,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    encoder = TinyEncoder(seed=seed)
    head = LinearHead.random(encoder.embed_dim, seed=seed)
    clf = StudyClassifier(encoder, head)
    step = start_step
    history: list[float] = []
    if ckpt_dir and resume:
        last = ckpt_dir / "last.npz"
        if last.exists():
            saved = load_numpy_checkpoint(last)
            head.weight = saved["weight"]
            head.bias = saved["bias"]
            step = int(np.asarray(saved.get("step", [step])).reshape(-1)[0])
    interrupted = False
    for epoch in range(epochs):
        order = rng.permutation(len(dataset_items))
        for idx in order:
            item = dataset_items[int(idx)]
            images = item["images"][None]
            logits = clf.forward_numpy(images, item["slot_mask"][None], item["slice_mask"][None])
            loss = masked_bce_with_logits(logits, item["y"][None], item["y_mask"][None], item["y_weight"][None])
            # pooled features for head grad
            b, s, t, c, h, w = images.shape
            flat = images.reshape(b * s * t, c, h, w)
            if flat.shape[1] == 1:
                flat = np.concatenate([flat, flat, flat], axis=1)
            feats = encoder(flat).reshape(b, s * t, -1)
            mask = (item["slot_mask"][None, :, None] * item["slice_mask"][None]).reshape(b, s * t)
            from rsna_knee.models.pooling import masked_mean

            pooled = masked_mean(feats, mask)
            dW, db = _grad_head(logits, item["y"][None], item["y_mask"][None], item["y_weight"][None], pooled)
            head.weight -= lr * dW
            head.bias -= lr * db
            history.append(loss)
            step += 1
            if ckpt_dir:
                save_numpy_checkpoint(
                    ckpt_dir / "last.npz",
                    {
                        "weight": head.weight,
                        "bias": head.bias,
                        "step": np.array([step]),
                        "meta": {"epoch": epoch, "step": step, "seed": seed},
                    },
                )
            if interrupt_after_steps is not None and step >= interrupt_after_steps:
                interrupted = True
                break
        if interrupted:
            break
    if ckpt_dir:
        save_numpy_checkpoint(
            ckpt_dir / "best.npz",
            {"weight": head.weight, "bias": head.bias, "step": np.array([step]), "meta": {"epoch": epochs, "step": step}},
        )
    return {
        "step": step,
        "history": history,
        "interrupted": interrupted,
        "head": head,
        "encoder_name": encoder.name,
        "kind": "tiny_numpy",
        "metrics_kind": "synthetic" if all(i.get("synthetic") for i in dataset_items) else "real",
    }


def microbatch_for_gpu(name: str) -> int:
    table = {"A100": 4, "L4": 2, "T4": 1}
    return table.get(name, 1)


def accumulation_steps(effective_batch: int, microbatch: int) -> int:
    microbatch = max(1, microbatch)
    return max(1, int(np.ceil(effective_batch / microbatch)))
