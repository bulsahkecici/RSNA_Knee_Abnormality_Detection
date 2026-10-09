"""Two-block optimizer coverage and restart equivalence on small real DINO geometry."""
import pytest
import torch
from rsna_knee.models.dinov2 import Dinov2ViTS14
from rsna_knee.models.study import StudyModel
from rsna_knee.training.loop import _optimizer, train_study_model
from tests.test_resume_equivalence import _fresh, _batches


def test_two_blocks_have_optimizer_coverage():
    model = StudyModel(Dinov2ViTS14(img_size=28))
    model.unfreeze_last_block(2)
    optimizer = _optimizer(model, 1e-4, 1e-5, True)
    covered = {id(p) for group in optimizer.param_groups for p in group['params']}
    assert covered == {id(p) for p in model.parameters() if p.requires_grad}
    assert all(p.requires_grad for block in model.encoder.blocks[-2:] for p in block.parameters())
    assert all(not p.requires_grad for block in model.encoder.blocks[:-2] for p in block.parameters())


def test_two_block_resume_equivalence(tmp_path):
    args = dict(epochs=2, effective_batch=1, microbatch=1, seed=0,
                unfreeze_after_steps=0, unfreeze_blocks=2)
    full, partial, resumed = _fresh(), _fresh(), _fresh()
    before = full.encoder.blocks[-2].mlp.fc1.weight.detach().clone()
    train_study_model(full, _batches(), ckpt_path=tmp_path/'full.pt', **args)
    train_study_model(partial, _batches(), ckpt_path=tmp_path/'part.pt', interrupt_after_steps=2, **args)
    train_study_model(resumed, _batches(), ckpt_path=tmp_path/'part.pt', resume=True, **args)
    assert not torch.equal(before, full.encoder.blocks[-2].mlp.fc1.weight)
    for key, value in full.state_dict().items():
        assert torch.allclose(value, resumed.state_dict()[key], atol=1e-6), key
    with pytest.raises(RuntimeError, match='block count changed'):
        train_study_model(_fresh(), _batches(), ckpt_path=tmp_path/'part.pt', resume=True,
                          **{**args, 'unfreeze_blocks':1})
