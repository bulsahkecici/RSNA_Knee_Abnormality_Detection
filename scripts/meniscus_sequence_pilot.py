"""Real GPU feasibility pilot: frozen encoder + ordered sagittal/coronal head.
No anatomical ROI localization, validation AUC, or submission claim.
Executed inside the prepared Kaggle notebook with verified EXPERIMENT payload.
"""
import time
import random
import urllib.request
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import timm
from safetensors.torch import load_file
from rsna_knee.hashing import sha256_file
from rsna_knee.data.soft_pool import soft_table
from rsna_knee.data.volume_cache import _payload_path

started = time.monotonic()
random.seed(2026); np.random.seed(2026); torch.manual_seed(2026)
names = ['Medial Meniscus', 'Lateral Meniscus']
targets = soft_table(SOFT.read_bytes())
entries = {}
for mp in part_manifests:
    manifest = json.loads(mp.read_text())
    for uid, meta in manifest['studies'].items():
        assert uid not in entries
        entries[uid] = (mp.parent, meta)
weights = WORK / EXPERIMENT['weights_file']
urllib.request.urlretrieve(EXPERIMENT['weights_url'], weights)
assert sha256_file(weights) == EXPERIMENT['weights_sha256']
encoder = timm.create_model('convnext_tiny', pretrained=False, num_classes=0)
missing, extra = encoder.load_state_dict(load_file(str(weights)), strict=False)
assert not missing and set(extra) <= {'head.fc.weight', 'head.fc.bias'}, (missing, extra)
encoder.cuda().eval().requires_grad_(False)
features, labels, masks, used = [], [], [], []
slots = [0, 1, 3, 4]  # sagittal/coronal, FS and non-FS; no axial
mean = torch.tensor([.485,.456,.406], device='cuda')[None,:,None,None]
std = torch.tensor([.229,.224,.225], device='cuda')[None,:,None,None]
for uid in EXPERIMENT['train_uids']:
    assert time.monotonic()-started < 2700, 'pilot runtime limit'
    root, meta = entries[uid]
    path = _payload_path(root, meta['file'])
    assert sha256_file(path) == meta['sha256']
    f = torch.zeros(4,16,encoder.num_features)
    valid = torch.zeros(4,16,dtype=torch.bool)
    with np.load(path, allow_pickle=False) as arrays:
        for j,s in enumerate(slots):
            candidates = [c for c in meta['candidates'][s] if c['anatomical_order_verified']]
            if not candidates: continue
            c = candidates[0]; v = arrays[c['key']]
            assert v.dtype == np.uint8 and v.shape == (c['n'],384,384) and v.max()>v.min()
            # Adjacent triplets, ordered uniformly over the retained series.
            # Cache stores median spacing, not every slice coordinate: no exact-mm claim.
            centres = np.linspace(1,len(v)-2,16).round().astype(int)
            x = torch.from_numpy(v[centres[:,None]+np.array([-1,0,1])].copy())
            x = x[:,:,32:352,32:352]  # 320px native centre view, NOT a localized meniscus ROI
            for start in range(0,16,8):
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
                    z = encoder((x[start:start+8].cuda().float()/255-mean)/std)
                f[j,start:start+8] = z.float().cpu()
            valid[j] = True
    assert valid.any(), 'no geometry-verified sagittal/coronal series'
    y = [targets[uid].get(k) for k in names]
    # Conservative numeric-confidence proxy, not independently verified report labels.
    m = [q is not None and (q<=.1 or q>=.9) for q in y]
    assert any(m)
    features.append(f);masks.append(valid);labels.append(([0 if q is None else q for q in y],m));used.append(uid)
    print('encoded',len(used),'/',len(EXPERIMENT['train_uids']),flush=True)
x = torch.stack(features).cuda();mask = torch.stack(masks).cuda()
y = torch.tensor([a for a,b in labels],device='cuda'); ym = torch.tensor([b for a,b in labels],device='cuda')
assert torch.isfinite(x).all()
del encoder
class SequenceHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.project=nn.Linear(dim,128)
        self.pos=nn.Parameter(torch.randn(1,1,16,128)*.01)
        self.slot=nn.Parameter(torch.randn(1,4,1,128)*.01)
        self.temporal=nn.Conv1d(128,128,3,padding=1)
        self.attn=nn.Linear(128,2)
        self.readout=nn.Parameter(torch.randn(2,128)*.01)
        self.bias=nn.Parameter(torch.zeros(2))
    def forward(self,x,mask):
        b=x.shape[0]
        h=self.project(x)+self.pos+self.slot
        h=h+F.gelu(self.temporal(h.reshape(b*4,16,128).transpose(1,2)).transpose(1,2)).reshape(b,4,16,128)
        h=h.reshape(b,64,128);m=mask.reshape(b,64)
        a=self.attn(h).masked_fill(~m[:,:,None],-1e4).softmax(dim=1)
        pooled=torch.einsum('bnc,bnd->bcd',a,h)
        return (pooled*self.readout).sum(-1)+self.bias
model=SequenceHead(x.shape[-1]).cuda()
opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.02)
initial={k:v.detach().clone() for k,v in model.state_dict().items()}
losses=[];steps=0
for epoch in range(3):
    for ids in torch.randperm(len(x),device='cuda').split(8):
        opt.zero_grad();pred=model(x[ids],mask[ids])
        loss=(F.binary_cross_entropy_with_logits(pred,y[ids],reduction='none')*ym[ids]).sum()/ym[ids].sum().clamp_min(1)
        assert torch.isfinite(loss)
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),3);opt.step()
        losses.append(float(loss.detach()));steps+=1
assert steps>0 and any(not torch.equal(initial[k],v) for k,v in model.state_dict().items())
model.eval()
with torch.no_grad():
    original=model(x[:8],mask[:8]);reverse=model(x[:8].flip(2),mask[:8].flip(2))
    assert torch.isfinite(original).all()
    order_delta=float((original-reverse).abs().max())
    assert order_delta>1e-7, 'head did not respond to slice reversal'
ck=WORK/'meniscus-pilot-head.pt'
torch.save({'model':model.state_dict(),'experiment':EXPERIMENT,'scope':'train_only_feasibility_pilot'},ck)
result={'status':'PASS','kind':'real','scope':'train_only_feasibility_pilot','n_studies':len(used),'applied_updates':steps,
        'loss_first':losses[0],'loss_last':losses[-1],'order_reversal_max_logit_delta':order_delta,
        'seconds':time.monotonic()-started,'peak_vram_bytes':torch.cuda.max_memory_allocated(),
        'checkpoint_sha256':sha256_file(ck),'gold_in_training':0,'outer_fold':'0','auc':None,
        'roi_localized':False,'exact_physical_slice_coordinates':False,'submission_eligible':False}
(WORK/'pilot-result.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
