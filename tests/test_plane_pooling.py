"""Model compatibility tests; mocked features are never production results."""

import pytest
import torch

from rsna_knee.evaluation.heldout import load_trained_study
from rsna_knee.models.dinov2 import ENCODER_NAME, Dinov2ViTS14
from rsna_knee.models.study import StudyModel
from rsna_knee.submission.offline import _load_model
from rsna_knee.training.checkpoint import torch_load, torch_save
from rsna_knee.training.loop import train_study_model


def blob(model):
    return {"encoder_name": ENCODER_NAME, "img_size": 28, "step": 1,
            "model": model.state_dict(), "pooling": model.pooling,
            "pooling_version": model.pooling_version}


def feature_forward(self, images):
    # Distinct image and feature dimensions expose concatenation/order mistakes.
    signal = images.mean(dim=(1, 2, 3)).unsqueeze(-1)
    return signal * torch.arange(1, 385, device=images.device).unsqueeze(0) / 384


@pytest.mark.parametrize("slot, center", [
    ([1, 1, 1], [[1, 1], [1, 1], [1, 1]]),
    ([1, 0, 1], [[1, 0], [1, 1], [1, 1]]),
    ([1, 1, 1], [[0, 1], [0, 0], [1, 1]]),
    ([0, 0, 0], [[1, 1], [1, 1], [1, 1]]),
])
def test_mean_conversion_preserves_masked_predictions(monkeypatch, slot, center):
    monkeypatch.setattr(Dinov2ViTS14, "forward", feature_forward)
    mean = StudyModel(Dinov2ViTS14(img_size=28))
    plane = StudyModel(Dinov2ViTS14(img_size=28), pooling="plane_concat")
    plane.initialize_from_mean_checkpoint(blob(mean))
    images = torch.arange(6).view(1, 3, 2, 1, 1, 1).expand(1, 3, 2, 3, 28, 28).float()
    slot_mask, center_mask = torch.tensor([slot]), torch.tensor([center])
    torch.testing.assert_close(mean(images, slot_mask, center_mask),
                               plane(images, slot_mask, center_mask), atol=1e-6, rtol=1e-6)
    for key, value in mean.encoder.state_dict().items():
        assert torch.equal(value, plane.encoder.state_dict()[key])


@pytest.mark.parametrize("pooling", ["mean", "plane_concat"])
def test_eval_and_offline_load_checkpoint_pooling(tmp_path, pooling):
    original = StudyModel(Dinov2ViTS14(img_size=28), pooling=pooling)
    checkpoint = tmp_path / "model.pt"
    saved = blob(original)
    if pooling == "mean":
        saved.pop("pooling")  # Historical mean checkpoints remain supported.
        saved.pop("pooling_version")
    torch_save(checkpoint, saved)
    evaluated, _ = load_trained_study(checkpoint, 28, torch.device("cpu"))
    inferred = _load_model(checkpoint, 28)
    for loaded in [evaluated, inferred]:
        assert loaded.pooling == pooling
        assert loaded.head.in_features == (1152 if pooling == "plane_concat" else 384)
        for key, value in original.state_dict().items():
            assert torch.equal(value, loaded.state_dict()[key])


def test_pooling_persisted_and_resume_mismatch_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(Dinov2ViTS14, "forward", feature_forward)
    model = StudyModel(Dinov2ViTS14(img_size=28), pooling="plane_concat")
    checkpoint = tmp_path / "plane.pt"
    batches = [{"images": torch.ones(1, 3, 1, 3, 28, 28),
                "slot_mask": torch.ones(1, 3), "center_mask": torch.ones(1, 3, 1),
                "y": torch.zeros(1, 12), "y_mask": torch.ones(1, 12), "y_weight": torch.ones(1, 12)}]
    train_study_model(model, batches, device="cpu", ckpt_path=checkpoint)
    saved = torch_load(checkpoint, map_location="cpu")
    assert saved["pooling"] == "plane_concat"
    assert saved["pooling_version"] == model.pooling_version
    train_study_model(StudyModel(Dinov2ViTS14(img_size=28), pooling="plane_concat"),
                      batches, device="cpu", ckpt_path=checkpoint, resume=True)
    with pytest.raises(RuntimeError, match="pooling changed"):
        train_study_model(StudyModel(Dinov2ViTS14(img_size=28)), batches,
                          device="cpu", ckpt_path=checkpoint, resume=True)


def test_conversion_rejects_invalid_or_incomplete_source():
    mean = StudyModel(Dinov2ViTS14(img_size=28))
    plane = StudyModel(Dinov2ViTS14(img_size=28), pooling="plane_concat")
    bad = blob(mean)
    bad["pooling"] = "plane_concat"
    with pytest.raises(ValueError, match="mean DINOv2"):
        plane.initialize_from_mean_checkpoint(bad)
    incomplete = blob(mean)
    incomplete["model"] = dict(incomplete["model"])
    incomplete["model"].pop("encoder.cls_token")
    with pytest.raises(RuntimeError, match="Missing key"):
        plane.initialize_from_mean_checkpoint(incomplete)
