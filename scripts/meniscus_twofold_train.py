"""Fixed-schedule two-fold evaluation; frozen features extracted by the preceding cell."""
from rsna_knee.models.meniscus_head import MeniscusHead
from rsna_knee.data.folds import assert_group_integrity, assert_train_uids, load_folds
from sklearn.metrics import roc_auc_score
fold_rows=load_folds(FOLDS);assert_group_integrity(fold_rows)
roles={r['StudyInstanceUID']:r for r in fold_rows}
eval_rows={r['StudyInstanceUID']:r for r in map(json.loads,(WORK/'evaluation-targets.jsonl').read_text().splitlines())}
ix={u:i for i,u in enumerate(used)}
x=torch.stack(features).cuda();mask=torch.stack(masks).cuda()
y=torch.tensor([a for a,b in labels],device='cuda');ym=torch.tensor([b for a,b in labels],device='cuda')
assert torch.isfinite(x).all()
del encoder,features,masks
results=[];predictions=[];checkpoints=[]

def score(uids, probs):
    out={}
    for j,name in enumerate(names):
        yy=[];pp=[]
        for u,p in zip(uids,probs):
            r=eval_rows.get(u,{})
            t=r.get('targets',{}).get(name)
            if r.get('y_mask',{}).get(name) and t in (0,1):yy.append(t);pp.append(float(p[j]))
        out[name]={'auc':float(roc_auc_score(yy,pp)) if len(set(yy))==2 else None,
                   'n':len(yy),'n_pos':sum(yy),'n_neg':len(yy)-sum(yy)}
    valid=[r['auc'] for r in out.values() if r['auc'] is not None]
    return {'per_class':out,'two_target_macro_auc':float(np.mean(valid)) if len(valid)==2 else None}

for fold in ('0','1'):
    train=EXPERIMENT['fold_train_uids'][fold]
    assert_train_uids(set(train),fold_rows)
    assert all(roles[u]['fold']!=fold for u in train)
    assert set(train).isdisjoint(EXPERIMENT['gold_uids'])
    weak=EXPERIMENT['fold_eval_uids'][fold];gold=EXPERIMENT['gold_uids']
    assert set(train).isdisjoint(weak)
    assert {roles[u]['group_id'] for u in train}.isdisjoint({roles[u]['group_id'] for u in weak+gold})
    tr=torch.tensor([ix[u] for u in train],device='cuda')
    for ordered in (False,True):
        seed=2026+int(fold);torch.manual_seed(seed);np.random.seed(seed)
        model=MeniscusHead(x.shape[-1],ordered).cuda()
        opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.02)
        step=0;loss_sum=0.
        for epoch in range(EXPERIMENT['head_epochs']):
            model.train()
            for ids in tr[torch.randperm(len(tr),device='cuda')].split(32):
                opt.zero_grad(set_to_none=True)
                logits=model(x[ids],mask[ids])
                loss=(F.binary_cross_entropy_with_logits(logits,y[ids],reduction='none')*ym[ids]).sum()/ym[ids].sum().clamp_min(1)
                assert torch.isfinite(loss)
                loss.backward();nn.utils.clip_grad_norm_(model.parameters(),3);opt.step()
                step+=1;loss_sum+=float(loss.detach())
            print('HEAD_EPOCH',fold,ordered,epoch+1,loss_sum/step,flush=True)
        assert step>0
        model.eval();kind='ordered' if ordered else 'invariant'
        ck=WORK/f'meniscus-{kind}-f{fold}.pt'
        torch.save({'model':model.state_dict(),'ordered':ordered,'dim':x.shape[-1],
                    'train_uids':train,'experiment':EXPERIMENT,'applied_updates':step},ck)
        checkpoints.append({'file':ck.name,'sha256':sha256_file(ck)})
        for split,uids in [('weak',weak),('gold',gold)]:
            pp=[]
            with torch.no_grad():
                for st in range(0,len(uids),32):
                    ids=[ix[u] for u in uids[st:st+32]]
                    pp.extend(model(x[ids],mask[ids]).sigmoid().cpu().tolist())
            assert np.isfinite(pp).all()
            results.append({'fold':fold,'model':kind,'split':split,**score(uids,pp),'applied_updates':step})
            predictions.extend({'StudyInstanceUID':u,'fold':fold,'model':kind,'split':split,
                                **dict(zip(names,p))} for u,p in zip(uids,pp))
        del model,opt
report={'kind':'real','status':'COMPLETE','scope':'two_outer_fold_two_target_evaluation','results':results,
        'checkpoints':checkpoints,'n_feature_studies':len(used),'seconds':time.monotonic()-started,
        'peak_vram_bytes':torch.cuda.max_memory_allocated(),'gold_in_training':0,
        'gold_is_independent':False,'weak_labels_are_image_truth':False,'competition_macro_auc':None,
        'submission_eligible':False,'needs_review_and_reference_complementarity':True}
(WORK/'meniscus-metrics.json').write_text(json.dumps(report,indent=2))
(WORK/'meniscus-predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in predictions))
print(json.dumps(report),flush=True)
