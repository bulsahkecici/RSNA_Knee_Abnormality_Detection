from __future__ import annotations

import pytest
import torch

from rsna_knee.models.dinov2 import Dinov2ViTS14, load_dinov2_vits14
from rsna_knee.models.study import StudyModel


def test_strict_roundtrip_and_mismatch(tmp_path):
    model = Dinov2ViTS14(img_size=28)
    path = tmp_path / "ok.pt"
    torch.save(model.state_dict(), path)
    loaded = load_dinov2_vits14(path, img_size=28)
    assert loaded.provenance["pretrained"] is True
    x = torch.zeros(2, 3, 28, 28)
    y = loaded(x)
    assert y.shape == (2, 384)

    bad = dict(model.state_dict())
    bad["blocks.0.extra"] = torch.zeros(1)
    bad_path = tmp_path / "bad.pt"
    torch.save(bad, bad_path)
    with pytest.raises(RuntimeError, match="architecture mismatch"):
        load_dinov2_vits14(bad_path, img_size=28)

    missing = dict(model.state_dict())
    missing.pop("cls_token")
    miss_path = tmp_path / "miss.pt"
    torch.save(missing, miss_path)
    with pytest.raises(RuntimeError, match="architecture mismatch"):
        load_dinov2_vits14(miss_path, img_size=28)


def test_stub_cannot_pose_as_dinov2():
    from rsna_knee.models.encoder import TorchDinoStub

    with pytest.raises(RuntimeError, match="TorchDinoStub"):
        TorchDinoStub(checkpoint=None)


def test_partial_unfreeze_keeps_resolution(tmp_path):
    from rsna_knee.training.loop import train_study_model

    model = StudyModel(Dinov2ViTS14(img_size=28), freeze_encoder=True)
    batch = {
        "images": torch.zeros(1, 1, 1, 3, 28, 28),
        "slot_mask": torch.ones(1, 1),
        "center_mask": torch.ones(1, 1, 1),
        "y": torch.zeros(1, 12),
        "y_mask": torch.ones(1, 12),
        "y_weight": torch.ones(1, 12),
    }
    out = train_study_model(
        model,
        [batch],
        epochs=1,
        effective_batch=1,
        microbatch=1,
        unfreeze_after_steps=1,
        ckpt_path=tmp_path / "u.pt",
    )
    assert out["img_size"] == 28
    assert any(p.requires_grad for p in model.encoder.blocks[-1].parameters())
    assert not any(p.requires_grad for p in model.encoder.blocks[0].parameters())


def test_study_contract():
    model = StudyModel(Dinov2ViTS14(img_size=28))
    images = torch.zeros(2, 3, 2, 3, 28, 28)
    slot = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.float32)
    center = torch.ones(2, 3, 2)
    logits = model(images, slot, center)
    assert logits.shape == (2, 12)
    with pytest.raises(ValueError):
        model(images[:, :, :, :1], slot, center)
