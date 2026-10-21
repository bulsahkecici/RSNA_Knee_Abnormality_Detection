import numpy as np
import pytest
import torch
from rsna_knee.models.conditional_sampling import conditional_weights
from rsna_knee.models.meniscus_head import MeniscusHead


def test_inverse_sqrt_training_groups_and_unknown_companion():
    y=np.array([[0,0]]*9+[[1,1]]+[[0,1]])
    mask=np.ones_like(y,dtype=bool);mask[-1,0]=False
    w,c=conditional_weights(y,mask)
    assert c==[9,0,0,1]
    assert w[9]/w[0]==pytest.approx(3)
    assert w[-1]==1 and w[:10].mean()==pytest.approx(1)


def test_cap_and_masked_uncertain_labels():
    y=np.array([[0,0]]*100+[[1,1],[-1,1]])
    mask=np.ones_like(y,dtype=bool);mask[-1,0]=False
    w,_=conditional_weights(y,mask)
    assert w.max()<=3 and w[-1]==1
    with pytest.raises(ValueError):conditional_weights(y,np.ones_like(mask))
    with pytest.raises(ValueError):conditional_weights([[0,0]],[[False,False]])


def test_six_slot_head_mask_and_identical_init():
    torch.manual_seed(2026);a=MeniscusHead(8,False,n_slots=6)
    torch.manual_seed(2026);b=MeniscusHead(8,False,n_slots=6)
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    x=torch.randn(2,6,16,8);mask=torch.ones(2,6,16,dtype=torch.bool);mask[:,2]=False
    changed=x.clone();changed[:,2]=10000
    assert torch.allclose(a(x,mask),a(changed,mask),atol=1e-6)
    a(x,mask).sum().backward();assert a.project.weight.grad.abs().sum()>0
    with pytest.raises(ValueError):a(x[:,:4],mask[:,:4])
