"""Training-time window sampling: deterministic, masked-slot safe, resume-equivalent, eval sees all centers."""
import pytest
import torch

from rsna_knee.evaluation.heldout import evaluate_checkpoint
from rsna_knee.models.dinov2 import Dinov2ViTS14
from rsna_knee.models.study import StudyModel
from rsna_knee.training.loop import sample_windows, train_study_model


def _fresh(pooling="mean"):
    torch.manual_seed(0)
    return StudyModel(Dinov2ViTS14(img_size=28), pooling=pooling)


def _grid(n=1, slots=3, centers=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    images = torch.rand(n, slots, centers, 3, 28, 28, generator=g)
    images[:, :, :, :, 0, 0] = torch.arange(centers, dtype=torch.float32).view(1, 1, centers, 1)
    return images


def _batches(slots=3, centers=6, n=3):
    out = []
    for i in range(n):
        slot_mask = torch.ones(1, slots)
        slot_mask[0, 1] = float(i != 1)  # study 1 has an absent slot
        y = torch.zeros(1, 12)
        y[0, i % 12] = 1.0
        out.append({"images": _grid(1, slots, centers, seed=i), "slot_mask": slot_mask,
                    "center_mask": torch.ones(1, slots, centers), "y": y,
                    "y_mask": torch.ones(1, 12), "y_weight": torch.ones(1, 12), "uid": f"s{i}"})
    return out


def _chosen(sampled):
    return sampled["images"][:, :, :, 0, 0, 0].round().long()


def test_sampling_is_deterministic_ordered_and_row_keyed():
    batch = {"images": _grid(2), "center_mask": torch.ones(2, 3, 6), "slot_mask": torch.ones(2, 3)}
    a = sample_windows(batch, 2, seed=5, epoch=1, batch_index=3, row_offset=0)
    b = sample_windows(batch, 2, seed=5, epoch=1, batch_index=3, row_offset=0)
    assert torch.equal(a["images"], b["images"]) and a["images"].shape == (2, 3, 2, 3, 28, 28)
    assert torch.equal(_chosen(a), a["center_index"])
    assert (a["center_index"][..., 0] < a["center_index"][..., 1]).all()  # physical order kept
    # Microbatch 1 on row 1 picks the same windows as row 1 of a two-study slice.
    single = {k: v[1:2] for k, v in batch.items()}
    row1 = sample_windows(single, 2, seed=5, epoch=1, batch_index=3, row_offset=1)
    assert torch.equal(row1["images"][0], a["images"][1])
    other_epoch = sample_windows(batch, 2, seed=5, epoch=2, batch_index=3, row_offset=0)
    assert not torch.equal(other_epoch["center_index"], a["center_index"])


def test_sampling_prefers_valid_centers_and_keeps_masks():
    center = torch.ones(1, 3, 6)
    center[0, 0, 2:] = 0  # slot 0 has only two valid centers
    center[0, 2, :] = 0  # slot 2 is entirely absent
    batch = {"images": _grid(), "center_mask": center, "slot_mask": torch.tensor([[1., 1., 0.]])}
    for seed in range(5):
        out = sample_windows(batch, 3, seed=seed, epoch=0, batch_index=0, row_offset=0)
        assert out["center_index"][0, 0, :2].tolist() == [0, 1]
        assert out["center_mask"][0, 0].tolist() == [1., 1., 0.]
        assert out["center_mask"][0, 2].sum() == 0
        assert out["center_mask"][0, 1].sum() == 3
    assert sample_windows(batch, 6, seed=0, epoch=0, batch_index=0, row_offset=0) is batch
    with pytest.raises(ValueError, match="exceeds"):
        sample_windows(batch, 7, seed=0, epoch=0, batch_index=0, row_offset=0)


def test_sampled_training_encodes_only_k_windows_per_slot():
    model = _fresh()
    seen = []
    forward = model.forward

    def spy(images, slot, center, *extra):
        seen.append(tuple(images.shape))
        return forward(images, slot, center, *extra)

    model.forward = spy
    train_study_model(model, _batches(), epochs=1, effective_batch=1, seed=0, train_centers_per_slot=2)
    assert seen and all(shape[1:3] == (3, 2) for shape in seen)


@pytest.mark.parametrize("pooling", ["mean", "label_attention", "window_transformer"])
def test_window_sampling_resume_equivalence(tmp_path, pooling):
    args = dict(epochs=2, effective_batch=1, microbatch=1, seed=3, train_centers_per_slot=2,
                unfreeze_after_steps=1, unfreeze_blocks=1)
    full, partial, resumed = _fresh(pooling), _fresh(pooling), _fresh(pooling)
    out = train_study_model(full, _batches(), ckpt_path=tmp_path / "full.pt", **args)
    train_study_model(partial, _batches(), ckpt_path=tmp_path / "part.pt", interrupt_after_steps=4, **args)
    train_study_model(resumed, _batches(), ckpt_path=tmp_path / "part.pt", resume=True, **args)
    assert out["train_centers_per_slot"] == 2 and out["window_sampling"]
    for key, value in full.state_dict().items():
        assert torch.allclose(value, resumed.state_dict()[key], atol=1e-6), key
    if pooling != "mean":
        assert full.attention.weight.abs().sum() > 0
    with pytest.raises(RuntimeError, match="window sampling changed"):
        train_study_model(_fresh(pooling), _batches(), ckpt_path=tmp_path / "part.pt", resume=True,
                          **{**args, "train_centers_per_slot": 3})
    with pytest.raises(RuntimeError, match="window sampling changed"):
        train_study_model(_fresh(pooling), _batches(), ckpt_path=tmp_path / "part.pt", resume=True,
                          **{**args, "train_centers_per_slot": None})


def test_sampling_changes_the_trajectory_and_evaluation_uses_all_centers(tmp_path):
    sampled, dense = _fresh("label_attention"), _fresh("label_attention")
    train_study_model(sampled, _batches(), epochs=1, effective_batch=1, seed=0, train_centers_per_slot=2,
                      ckpt_path=tmp_path / "sampled.pt")
    train_study_model(dense, _batches(), epochs=1, effective_batch=1, seed=0)
    assert not torch.allclose(sampled.head.weight, dense.head.weight)
    batches = _batches()
    metrics = evaluate_checkpoint(tmp_path / "sampled.pt", train_batches=None, weak_batches=batches,
                                  gold_batches=[], dest=tmp_path / "metrics.json", img_size=28, device="cpu")
    assert metrics["evaluation_windows"] == "all_centers" and metrics["evaluation_seconds"] >= 0
    import json
    rows = [json.loads(line) for line in (tmp_path / "predictions.jsonl").read_text().splitlines()]
    sampled.eval()
    with torch.no_grad():
        expected = torch.sigmoid(sampled(batches[0]["images"], batches[0]["slot_mask"], batches[0]["center_mask"]))
    from rsna_knee.ontology import TARGET_COLUMNS
    got = torch.tensor([rows[0][name] for name in TARGET_COLUMNS])
    assert torch.allclose(got, expected[0], atol=1e-5)
