"""Executed after fold-specific frozen feature extraction in the OA notebook."""
from rsna_knee.models.meniscus_head import MeniscusHead
from rsna_knee.models.conditional_sampling import conditional_weights
from rsna_knee.data.folds import assert_group_integrity, assert_train_uids, load_folds
from sklearn.metrics import roc_auc_score
import math
fold_rows=load_folds(FOLDS);assert_group_integrity(fold_rows)
roles={r['StudyInstanceUID']:r for r in fold_rows}
ix={u:i for i,u in enumerate(used)}
x=torch.stack(features).cuda();mask=torch.stack(masks).cuda()
y=torch.tensor([a for a,b in labels],device='cuda');ym=torch.tensor([b for a,b in labels],device='cuda')
assert torch.isfinite(x).all()
del encoder,features,masks
fold=ACTIVE_FOLD
train=EXPERIMENT['fold_train_uids'][fold]
weak=EXPERIMENT['fold_eval_uids'][fold];gold=EXPERIMENT['gold_uids']
assert_train_uids(set(train),fold_rows)
assert all(str(roles[u]['fold'])!=fold for u in train)
assert {roles[u]['group_id'] for u in train}.isdisjoint({roles[u]['group_id'] for u in weak+gold})
tr=torch.tensor([ix[u] for u in train],device='cuda')
w,counts=conditional_weights(y[tr].cpu().numpy(),ym[tr].cpu().numpy())
assert counts==EXPERIMENT['sampling'][fold]['joint_counts']
assert hashlib.sha256(w.tobytes()).hexdigest()==EXPERIMENT['sampling'][fold]['weights_sha256']
results=[];predictions=[];checkpoints=[]
for kind in ('natural','conditional'):
    seed=2026+int(fold);torch.manual_seed(seed);np.random.seed(seed)
    model=MeniscusHead(x.shape[-1],False,n_slots=6).cuda()
    initial_sha=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.02)
    generator=torch.Generator().manual_seed(seed)
    weights=torch.tensor(w if kind=='conditional' else np.ones(len(w)),dtype=torch.double)
    step=0;loss_sum=0.
    for epoch in range(EXPERIMENT['head_epochs']):
        model.train()
        # Same draw budget, replacement policy and RNG seed; only probabilities differ.
        draws=torch.multinomial(weights,len(train),replacement=True,generator=generator)
        for sample in draws.split(32):
            ids=tr[sample.to(tr.device)]
            opt.zero_grad(set_to_none=True)
            logits=model(x[ids],mask[ids])
            loss=(F.binary_cross_entropy_with_logits(logits,y[ids],reduction='none')*ym[ids]).sum()/ym[ids].sum()
            assert torch.isfinite(loss)
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),3);opt.step()
            step+=1;loss_sum+=float(loss.detach())
        print('OA_EPOCH',fold,kind,epoch+1,loss_sum/step,flush=True)
    assert step==EXPERIMENT['head_epochs']*math.ceil(len(train)/32)
    model.eval()
    ck=WORK/f'oa-{kind}-f{fold}.pt'
    torch.save({'model':model.state_dict(),'dim':x.shape[-1],'n_slots':6,'ordered':False,
                'train_uids':train,'experiment':EXPERIMENT,'applied_updates':step},ck)
    checkpoints.append({'file':ck.name,'sha256':sha256_file(ck),'initial_sha256':initial_sha})
    for split,uids in [('weak',weak),('gold',gold)]:
        pp=[]
        with torch.no_grad():
            for st in range(0,len(uids),32):
                ids=[ix[u] for u in uids[st:st+32]]
                pp.extend(model(x[ids],mask[ids]).sigmoid().cpu().tolist())
        assert np.isfinite(pp).all()
        per_class={}
        for j,name in enumerate(names):
            pairs=[(eval_rows[u]['targets'][name],p[j]) for u,p in zip(uids,pp)
                   if eval_rows.get(u,{}).get('y_mask',{}).get(name) and eval_rows[u]['targets'].get(name) in (0,1)]
            yy=[a for a,b in pairs]; ss=[b for a,b in pairs]
            per_class[name]={'auc':float(roc_auc_score(yy,ss)) if len(set(yy))==2 else None,'n':len(yy),'n_pos':sum(yy),'n_neg':len(yy)-sum(yy)}
        results.append({'fold':fold,'model':kind,'split':split,'per_class':per_class,'applied_updates':step,'initial_sha256':initial_sha})
        predictions.extend({'StudyInstanceUID':u,'fold':fold,'model':kind,'split':split,**dict(zip(names,p))} for u,p in zip(uids,pp))
    del model,opt
assert len({c['initial_sha256'] for c in checkpoints})==1,'paired initial weights differ'
