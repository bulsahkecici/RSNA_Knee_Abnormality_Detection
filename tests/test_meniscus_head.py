import pytest
import torch
from rsna_knee.models.meniscus_head import MeniscusHead

@pytest.mark.parametrize('ordered',[False,True])
def test_missing_series_has_no_effect(ordered):
    torch.manual_seed(11)
    model=MeniscusHead(12,ordered)
    x=torch.randn(2,4,16,12);mask=torch.ones(2,4,16,dtype=torch.bool);mask[:,3]=False
    changed=x.clone();changed[:,3]=10000
    assert torch.allclose(model(x,mask),model(changed,mask),atol=1e-6)
    model(x,mask).sum().backward()
    assert model.project.weight.grad.abs().sum()>0

def test_control_is_order_invariant_but_sequence_head_is_not():
    torch.manual_seed(4)
    x=torch.randn(2,4,16,12);mask=torch.ones(2,4,16,dtype=torch.bool)
    control=MeniscusHead(12,False);ordered=MeniscusHead(12,True)
    assert torch.allclose(control(x,mask),control(x.flip(2),mask.flip(2)),atol=1e-6)
    assert (ordered(x,mask)-ordered(x.flip(2),mask.flip(2))).abs().max()>1e-7

def test_all_missing_is_refused():
    with pytest.raises(ValueError,match='all sequence slots missing'):
        MeniscusHead(12)(torch.zeros(1,4,16,12),torch.zeros(1,4,16,dtype=torch.bool))
