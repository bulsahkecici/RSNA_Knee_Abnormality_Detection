from __future__ import annotations

import torch

from rsna_knee.models.dinov2 import Dinov2ViTS14
from rsna_knee.models.study import StudyModel
from rsna_knee.training.loop import train_study_model


def _fresh() -> StudyModel:
    torch.manual_seed(0)
    return StudyModel(Dinov2ViTS14(img_size=28), freeze_encoder=True)


def _batches():
    batches = []
    for i in range(2):
        batches.append(
            {
                "images": torch.zeros(1, 1, 1, 3, 28, 28),
                "slot_mask": torch.ones(1, 1),
                "center_mask": torch.ones(1, 1, 1),
                "y": torch.tensor([[float(i % 2)] + [0.0] * 11]),
                "y_mask": torch.ones(1, 12),
                "y_weight": torch.ones(1, 12),
            }
        )
    return batches


def test_interrupted_resume_matches_uninterrupted(tmp_path):
    batches = _batches()
    full = _fresh()
    train_study_model(
        full,
        batches,
        epochs=1,
        effective_batch=1,
        microbatch=1,
        seed=0,
        ckpt_path=tmp_path / "full.pt",
        resume=False,
    )
    part = _fresh()
    train_study_model(
        part,
        batches,
        epochs=1,
        effective_batch=1,
        microbatch=1,
        seed=0,
        ckpt_path=tmp_path / "part.pt",
        resume=False,
        interrupt_after_steps=1,
    )
    resumed = _fresh()
    train_study_model(
        resumed,
        batches,
        epochs=1,
        effective_batch=1,
        microbatch=1,
        seed=0,
        ckpt_path=tmp_path / "part.pt",
        resume=True,
    )
    fresh = _fresh()
    assert not torch.allclose(full.head.weight.detach(), fresh.head.weight.detach())
    for key, value in full.state_dict().items():
        assert torch.allclose(value, resumed.state_dict()[key], atol=1e-6), key
