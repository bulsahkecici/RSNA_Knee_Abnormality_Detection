"""Review real two-fold head ablation; no leaderboard or independent gold claim."""
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from rsna_knee.data.folds import load_folds
from rsna_knee.hashing import sha256_file
ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'artifacts/kaggle/rsna-meniscus-twofold-20261020'
NAMES=['Medial Meniscus','Lateral Meniscus']

def review():
    exp=json.loads((FOLDER/'experiment.json').read_text())
    out=FOLDER/'output'
    assert json.loads((out/'experiment.json').read_text())==exp
    metrics=json.loads((out/'meniscus-metrics.json').read_text())
    assert metrics['kind']=='real' and metrics['status']=='COMPLETE' and metrics['gold_in_training']==0
    assert len(metrics['results'])==8 and all(r['applied_updates']>0 for r in metrics['results'])
    labels_path=ROOT/'state/campaigns/campaign-20261003-v2/numeric-training.jsonl'
    assert sha256_file(labels_path)==exp['labels_sha256']
    assert sha256_file(ROOT/exp['folds_file'])==exp['folds_sha256']
    labels={r['StudyInstanceUID']:r for r in map(json.loads,labels_path.read_text().splitlines())}
    groups={r['StudyInstanceUID']:r['group_id'] for r in load_folds(ROOT/exp['folds_file'])}
    preds={}
    for r in map(json.loads,(out/'meniscus-predictions.jsonl').read_text().splitlines()):
        key=(r['fold'],r['model'],r['split'],r['StudyInstanceUID'])
        assert key not in preds
        preds[key]=r
    fold_delta={};class_delta={};records=[]
    for fold in ('0','1'):
        deltas=[]
        for name in NAMES:
            uids=[u for u in exp['fold_eval_uids'][fold] if labels.get(u,{}).get('y_mask',{}).get(name) and labels[u]['targets'][name] in (0,1)]
            y=np.array([labels[u]['targets'][name] for u in uids]);assert len(set(y))==2
            a=np.array([preds[(fold,'ordered','weak',u)][name] for u in uids])
            b=np.array([preds[(fold,'invariant','weak',u)][name] for u in uids])
            assert np.isfinite(a).all() and np.isfinite(b).all()
            delta=float(roc_auc_score(y,a)-roc_auc_score(y,b));deltas.append(delta)
            class_delta[fold+':'+name]=delta
            records.append((fold,name,uids,y,a,b))
        fold_delta[fold]=float(np.mean(deltas))
    # Resample duplicate groups within fold. Never pool cross-fold raw scores:
    # AUC is calculated within each fold/class then averaged.
    rng=np.random.default_rng(2026);boots=[]
    indexed={}
    for f in ('0','1'):
        group_ids=sorted({groups[u] for u in exp['fold_eval_uids'][f]})
        indexed[f]=(group_ids,{g:i for i,g in enumerate(group_ids)})
    for _ in range(2000):
        counts={f:np.bincount(rng.integers(0,len(v[0]),len(v[0])),minlength=len(v[0])) for f,v in indexed.items()}
        ds=[]
        for f,name,uids,y,a,b in records:
            w=np.array([counts[f][indexed[f][1][groups[u]]] for u in uids])
            if (w[y==0].sum()==0 or w[y==1].sum()==0):break
            ds.append(roc_auc_score(y,a,sample_weight=w)-roc_auc_score(y,b,sample_weight=w))
        if len(ds)==4:boots.append(float(np.mean(ds)))
    assert len(boots)>=1900
    ci=np.quantile(boots,[.025,.975]).tolist()
    passed=all(d>0 for d in fold_delta.values()) and ci[0]>0
    result={'kind':'real','decision':'NEEDS_REFERENCE_COMPLEMENTARITY' if passed else 'STOP_NO_SUPPORTED_SEQUENCE_GAIN',
            'fold_macro_deltas':fold_delta,'per_class_deltas':class_delta,'paired_group_bootstrap_ci95':ci,
            'bootstrap_draws':len(boots),'gold_is_independent':False,'competition_auc':None,
            'submission_eligible':False,'results':metrics['results'],
            'note':'Passing head ablation alone does not establish gain over R6/CoAtNet reference.'}
    (FOLDER/'review.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
    return result
if __name__=='__main__':review()
