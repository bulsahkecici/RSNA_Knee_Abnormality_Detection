import torch
from rsna_knee.models.dinov2 import Dinov2ViTS14
from rsna_knee.models.study import StudyModel
from rsna_knee.training.loop import _optimizer


def test_attention_initialization_preserves_mean_and_masks():
    torch.manual_seed(3)
    mean = StudyModel(Dinov2ViTS14(img_size=28)).eval()
    attention = StudyModel(Dinov2ViTS14(img_size=28), pooling='label_attention').eval()
    attention.initialize_from_mean_checkpoint({'encoder_name':'dinov2_vits14', 'img_size':28,
                                               'model':mean.state_dict()})
    images = torch.randn(2,3,2,3,28,28)
    slots = torch.tensor([[1.,0.,1.],[0.,0.,0.]])
    centers = torch.tensor([[[1.,0.],[1.,1.],[1.,1.]],[[1.,1.],[1.,1.],[1.,1.]]])
    assert torch.allclose(mean(images,slots,centers),attention(images,slots,centers),atol=1e-6)
    changed = images.clone();changed[0,1] *= 100;changed[0,0,1] *= 100
    assert torch.allclose(attention(images,slots,centers),attention(changed,slots,centers),atol=1e-6)
    assert torch.isfinite(attention(images,slots,centers)).all()
    attention(images,slots,centers).sum().backward()
    assert attention.attention.weight.grad.abs().sum() > 0
    opt = _optimizer(attention,1e-4,1e-5,False)
    assert id(attention.attention.weight) in {id(p) for g in opt.param_groups for p in g['params']}


def test_attention_checkpoint_evaluator_preserves_pooling(tmp_path):
    from rsna_knee.evaluation.heldout import load_trained_study
    model = StudyModel(Dinov2ViTS14(img_size=28), pooling='label_attention').eval()
    with torch.no_grad():
        model.attention.weight.normal_(std=.01)
    checkpoint = tmp_path/'attention.pt'
    torch.save({'encoder_name':'dinov2_vits14','img_size':28,'step':1,
                'pooling':model.pooling,'pooling_version':model.pooling_version,
                'model':model.state_dict()}, checkpoint)
    loaded, _ = load_trained_study(checkpoint,28,torch.device('cpu'))
    assert loaded.pooling == 'label_attention'
    images = torch.randn(1,3,2,3,28,28)
    slots, centers = torch.ones(1,3), torch.ones(1,3,2)
    with torch.no_grad():
        assert torch.allclose(model(images,slots,centers),loaded(images,slots,centers),atol=1e-6)
