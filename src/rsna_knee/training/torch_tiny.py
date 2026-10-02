"""Optional torch tiny step used by resume tests when torch is installed."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def torch_tiny_step(images: np.ndarray, y: np.ndarray, mask: np.ndarray, ckpt: Path | None = None, resume: bool = False) -> dict[str, Any]:
    import torch
    import torch.nn as nn

    from rsna_knee.training.checkpoint import torch_load, torch_save
    from rsna_knee.training.losses import torch_masked_bce

    model = nn.Sequential(nn.Flatten(), nn.Linear(int(np.prod(images.shape[1:])), 12))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    step = 0
    if resume and ckpt and ckpt.exists():
        blob = torch_load(ckpt, map_location="cpu")
        model.load_state_dict(blob["model"])
        opt.load_state_dict(blob["opt"])
        step = int(blob["step"])
    x = torch.tensor(images, dtype=torch.float32)
    yt = torch.tensor(np.nan_to_num(y, nan=0.0), dtype=torch.float32)
    m = torch.tensor(mask, dtype=torch.float32)
    opt.zero_grad()
    logits = model(x)
    loss = torch_masked_bce(logits, yt, m)
    loss.backward()
    opt.step()
    step += 1
    if ckpt:
        torch_save(ckpt, {"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "scaler": None})
    return {"loss": float(loss.detach()), "step": step}
