"""Arbitrary slot counts, window-transformer aggregation and recorded input resize."""
import pytest
import torch

from rsna_knee.evaluation.heldout import load_trained_study
from rsna_knee.models.dinov2 import Dinov2ViTS14
from rsna_knee.models.study import StudyModel
from rsna_knee.training.loop import train_study_model


def _model(pooling, **kwargs):
    torch.manual_seed(1)
    model = StudyModel(Dinov2ViTS14(img_size=kwargs.pop("img_size", 28)), pooling=pooling, **kwargs).eval()
    if hasattr(model, "attention"):
        with torch.no_grad():
            model.attention.weight.normal_(std=0.5)
    return model


@pytest.mark.parametrize("pooling", ["mean", "label_attention", "window_transformer"])
def test_five_slot_forward_ignores_a_missing_slot(pooling):
    model = _model(pooling)
    images = torch.rand(2, 5, 3, 3, 28, 28, generator=torch.Generator().manual_seed(0))
    slots = torch.tensor([[1., 1., 1., 0., 1.], [1., 0., 1., 1., 1.]])
    centers = torch.ones(2, 5, 3)
    with torch.no_grad():
        base = model(images, slots, centers)
        changed = images.clone()
        changed[0, 3] = torch.rand_like(changed[0, 3])
        changed[1, 1] = 0
        assert base.shape == (2, 12) and torch.isfinite(base).all()
        assert torch.allclose(base, model(changed, slots, centers), atol=1e-5)
        changed[0, 4] *= 0.1
        assert not torch.allclose(base[0], model(changed, slots, centers)[0], atol=1e-5)


def test_plane_concat_is_three_slot_only():
    model = _model("plane_concat")
    with pytest.raises(ValueError, match="exactly three ordered planes, got 5"):
        model(torch.rand(1, 5, 2, 3, 28, 28), torch.ones(1, 5), torch.ones(1, 5, 2))


def test_all_absent_study_stays_finite_for_attention_aggregators():
    for pooling in ("label_attention", "window_transformer"):
        model = _model(pooling)
        logits = model(torch.rand(1, 5, 2, 3, 28, 28), torch.zeros(1, 5), torch.ones(1, 5, 2))
        assert torch.isfinite(logits).all()
        assert torch.allclose(logits, model.head.bias.unsqueeze(0))


def test_window_transformer_sampled_windows_equal_masked_full_grid():
    """Sampled windows with their original center indices see exactly what the full grid sees with the rest padded."""
    model = _model("window_transformer")
    images = torch.rand(1, 3, 6, 3, 28, 28, generator=torch.Generator().manual_seed(2))
    slots = torch.ones(1, 3)
    index = torch.tensor([[[0, 4], [1, 5], [2, 3]]])
    rows, planes = torch.arange(1).view(1, 1, 1), torch.arange(3).view(1, 3, 1)
    full_mask = torch.zeros(1, 3, 6)
    full_mask[rows, planes, index] = 1
    with torch.no_grad():
        full = model(images, slots, full_mask)
        sampled = model(images[rows, planes, index], slots, torch.ones(1, 3, 2), index)
        wrong_positions = model(images[rows, planes, index], slots, torch.ones(1, 3, 2))
    assert torch.allclose(full, sampled, atol=1e-5)
    assert not torch.allclose(full, wrong_positions, atol=1e-5)  # positions are real information


def test_window_transformer_config_is_checkpointed_and_refused_when_changed(tmp_path):
    torch.manual_seed(0)
    model = StudyModel(Dinov2ViTS14(img_size=28), pooling="window_transformer", aggregator_layers=1, aggregator_heads=2)
    batch = {"images": torch.rand(1, 5, 2, 3, 28, 28), "slot_mask": torch.tensor([[1., 1., 0., 1., 1.]]),
             "center_mask": torch.ones(1, 5, 2), "y": torch.ones(1, 12), "y_mask": torch.ones(1, 12),
             "y_weight": torch.ones(1, 12)}
    train_study_model(model, [batch], epochs=1, effective_batch=1, ckpt_path=tmp_path / "wt.pt")
    loaded, blob = load_trained_study(tmp_path / "wt.pt", 28, torch.device("cpu"))
    assert blob["aggregator_config"] == {"layers": 1, "heads": 2, "max_slots": 8, "max_centers": 128}
    model.eval()
    with torch.no_grad():
        assert torch.allclose(model(batch["images"], batch["slot_mask"], batch["center_mask"]),
                              loaded(batch["images"], batch["slot_mask"], batch["center_mask"]), atol=1e-6)
    torch.manual_seed(0)
    other = StudyModel(Dinov2ViTS14(img_size=28), pooling="window_transformer")
    with pytest.raises(RuntimeError, match="aggregator configuration changed"):
        train_study_model(other, [batch], epochs=2, effective_batch=1, ckpt_path=tmp_path / "wt.pt", resume=True)
    with pytest.raises(ValueError, match="heads dividing"):
        StudyModel(Dinov2ViTS14(img_size=28), pooling="window_transformer", aggregator_heads=5)


def test_window_transformer_gradients_reach_embeddings_and_attention():
    torch.manual_seed(0)
    model = StudyModel(Dinov2ViTS14(img_size=28), pooling="window_transformer")
    with torch.no_grad():
        model.attention.weight.normal_(std=0.1)
    model(torch.rand(1, 3, 2, 3, 28, 28), torch.ones(1, 3), torch.ones(1, 3, 2)).sum().backward()
    for param in (model.slot_embedding.weight, model.window_embedding.weight, model.attention.weight,
                  model.aggregator.layers[0].self_attn.in_proj_weight):
        assert param.grad is not None and param.grad.abs().sum() > 0


def test_input_resize_is_recorded_and_required(tmp_path):
    torch.manual_seed(0)
    model = StudyModel(Dinov2ViTS14(img_size=42), input_size_from=28)
    assert model.input_resize == {"from": 28, "to": 42, "mode": "bilinear-antialias.v1"}
    batch = {"images": torch.rand(1, 3, 2, 3, 28, 28), "slot_mask": torch.ones(1, 3),
             "center_mask": torch.ones(1, 3, 2), "y": torch.ones(1, 12), "y_mask": torch.ones(1, 12),
             "y_weight": torch.ones(1, 12)}
    train_study_model(model, [batch], epochs=1, effective_batch=1, ckpt_path=tmp_path / "resize.pt")
    loaded, _ = load_trained_study(tmp_path / "resize.pt", 28, torch.device("cpu"))
    assert loaded.img_size == 42 and loaded.input_resize == model.input_resize
    model.eval()
    with torch.no_grad():
        assert torch.allclose(model(batch["images"], batch["slot_mask"], batch["center_mask"]),
                              loaded(batch["images"], batch["slot_mask"], batch["center_mask"]), atol=1e-6)
    with pytest.raises(RuntimeError, match="cache image size"):
        load_trained_study(tmp_path / "resize.pt", 56, torch.device("cpu"))
    plain = StudyModel(Dinov2ViTS14(img_size=42))
    with pytest.raises(ValueError, match="no resize is recorded"):
        plain(batch["images"], batch["slot_mask"], batch["center_mask"])
    with pytest.raises(RuntimeError, match="input resize changed"):
        train_study_model(plain, [batch], epochs=2, effective_batch=1, ckpt_path=tmp_path / "resize.pt", resume=True)
