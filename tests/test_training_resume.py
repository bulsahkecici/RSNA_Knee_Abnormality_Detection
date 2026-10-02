from __future__ import annotations

import numpy as np
import pytest

from rsna_knee.models.classifier import StudyClassifier
from rsna_knee.models.encoder import TinyEncoder
from rsna_knee.training.checkpoint import load_numpy_checkpoint
from rsna_knee.training.losses import masked_bce_with_logits
from rsna_knee.training.train import train_tiny


def _item(seed=0):
    rng = np.random.default_rng(seed)
    return {
        "images": rng.random((3, 3, 1, 8, 8)).astype(np.float32),
        "slot_mask": np.ones(3, dtype=np.float32),
        "slice_mask": np.ones((3, 3), dtype=np.float32),
        "y": rng.integers(0, 2, size=12).astype(np.float32),
        "y_mask": np.ones(12, dtype=np.float32),
        "y_weight": np.ones(12, dtype=np.float32),
        "synthetic": True,
    }


def test_forward_backward_and_resume(tmp_path):
    items = [_item(i) for i in range(4)]
    clf = StudyClassifier(TinyEncoder(0))
    logits = clf.forward_numpy(items[0]["images"][None], items[0]["slot_mask"][None], items[0]["slice_mask"][None])
    loss = masked_bce_with_logits(logits, items[0]["y"][None], items[0]["y_mask"][None], items[0]["y_weight"][None])
    assert np.isfinite(loss)
    out1 = train_tiny(items, epochs=2, ckpt_dir=tmp_path, interrupt_after_steps=2, resume=False)
    assert out1["interrupted"] is True
    step1 = out1["step"]
    out2 = train_tiny(items, epochs=2, ckpt_dir=tmp_path, resume=True)
    assert out2["step"] >= step1
    ckpt = load_numpy_checkpoint(tmp_path / "last.npz")
    assert "weight" in ckpt
    # optional torch
    pytest.importorskip("torch")
    from rsna_knee.training.torch_tiny import torch_tiny_step

    img = items[0]["images"][None]
    y = items[0]["y"][None]
    m = items[0]["y_mask"][None]
    ck = tmp_path / "t.pt"
    a = torch_tiny_step(img, y, m, ckpt=ck, resume=False)
    b = torch_tiny_step(img, y, m, ckpt=ck, resume=True)
    assert b["step"] == a["step"] + 1
