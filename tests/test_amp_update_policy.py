"""Exercise overflow bookkeeping on CPU; no synthetic production metrics."""

from contextlib import nullcontext

import pytest
import torch

from rsna_knee.models.dinov2 import Dinov2ViTS14
from rsna_knee.models.study import StudyModel
from rsna_knee.training import loop
from rsna_knee.training.checkpoint import torch_load, torch_save


class SkippingScaler:
    def __init__(self, *args):
        self.value = 8.0
        self.calls = 0

    def get_scale(self):
        return self.value

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        pass

    def step(self, optimizer):
        self.calls += 1
        if self.calls > 1:
            optimizer.step()

    def update(self):
        if self.calls == 1:
            self.value /= 2

    def state_dict(self):
        return {"value": self.value, "calls": self.calls}

    def load_state_dict(self, state):
        self.value, self.calls = state["value"], state["calls"]


def model():
    torch.manual_seed(11)
    return StudyModel(Dinov2ViTS14(img_size=28))


def batch():
    return {
        "images": torch.zeros(1, 1, 1, 3, 28, 28),
        "slot_mask": torch.ones(1, 1),
        "center_mask": torch.ones(1, 1, 1),
        "y": torch.zeros(1, 12),
        "y_mask": torch.ones(1, 12),
        "y_weight": torch.ones(1, 12),
    }


@pytest.fixture
def simulated_amp(monkeypatch):
    monkeypatch.setattr(loop, "_amp_enabled", lambda device: True)
    monkeypatch.setattr(torch.amp, "GradScaler", SkippingScaler)
    monkeypatch.setattr(torch, "autocast", lambda **kwargs: nullcontext())


def test_skipped_update_keeps_weights_scheduler_and_encoder_frozen(tmp_path, simulated_amp):
    m = model()
    original = m.head.weight.detach().clone()
    path = tmp_path / "skip.pt"
    result = loop.train_study_model(m, [batch()], device="cpu", epochs=1,
                                    effective_batch=1, unfreeze_after_steps=0, ckpt_path=path)
    saved = torch_load(path, map_location="cpu")
    assert result["step"] == 0
    assert result["attempted_updates"] == result["skipped_updates"] == 1
    assert saved["scheduler"]["last_epoch"] == 0
    assert not saved["unfrozen"]
    assert torch.equal(original, m.head.weight)


def test_skipped_update_counters_and_scaler_resume_exactly(tmp_path, simulated_amp):
    args = dict(device="cpu", epochs=2, effective_batch=1, seed=9)
    full = model()
    expected = loop.train_study_model(full, [batch(), batch()], **args)
    checkpoint = tmp_path / "resume.pt"
    first = loop.train_study_model(model(), [batch(), batch()], **args,
                                   ckpt_path=checkpoint, interrupt_after_steps=1)
    assert first["attempted_updates"] == 2 and first["skipped_updates"] == 1
    resumed = model()
    actual = loop.train_study_model(resumed, [batch(), batch()], **args,
                                    ckpt_path=checkpoint, resume=True)
    assert actual["step"] == expected["step"] == 3
    assert actual["attempted_updates"] == expected["attempted_updates"] == 4
    assert actual["skipped_updates"] == expected["skipped_updates"] == 1
    for key, value in full.state_dict().items():
        assert torch.equal(value, resumed.state_dict()[key]), key


def test_legacy_checkpoint_without_update_counters_resumes(tmp_path):
    checkpoint = tmp_path / "legacy.pt"
    loop.train_study_model(model(), [batch(), batch()], device="cpu", epochs=1,
                           effective_batch=1, ckpt_path=checkpoint, interrupt_after_steps=1)
    saved = torch_load(checkpoint, map_location="cpu")
    for key in ("attempted_updates", "skipped_updates", "numerical_policy"):
        saved.pop(key)
    torch_save(checkpoint, saved)
    result = loop.train_study_model(model(), [batch(), batch()], device="cpu", epochs=1,
                                    effective_batch=1, ckpt_path=checkpoint, resume=True)
    assert result["step"] == result["attempted_updates"] == 2
    assert result["skipped_updates"] == 0


@pytest.mark.parametrize("configured, expected", [({}, (1e-3, 1e-5)),
                                               ({"lr_head": 3e-4, "lr_backbone": 2e-5}, (3e-4, 2e-5))])
def test_worker_forwards_effective_learning_rates(monkeypatch, tmp_path, configured, expected):
    from rsna_knee.runtime import worker

    payload = {"weights": tmp_path / "weights", "train_uids": ["test"],
               "img_size": 28, "batches": [batch()], "input_id": "input", "config_id": "config"}
    monkeypatch.setattr(worker, "prepare_training", lambda *args: (payload, None))
    monkeypatch.setattr(worker, "weight_trust", lambda path: {"allowlisted": True, "official": True})
    monkeypatch.setattr(worker, "execution_allowed", lambda **kwargs: (True, None))
    encoder = Dinov2ViTS14(img_size=28)
    monkeypatch.setattr(worker, "load_dinov2_vits14", lambda *args, **kwargs: encoder)
    monkeypatch.setattr(worker, "resolve_checkpoint_out", lambda *args: tmp_path / "new.pt")

    class StopAfterCapture(Exception):
        pass

    def capture(*args, **kwargs):
        assert (kwargs["lr_head"], kwargs["lr_backbone"]) == expected
        raise StopAfterCapture

    monkeypatch.setattr(worker, "train_study_model", capture)
    with pytest.raises(StopAfterCapture):
        worker._finish_training({"train": configured}, tmp_path)
