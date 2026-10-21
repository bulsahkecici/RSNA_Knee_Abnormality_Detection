"""Numeric-only, descriptive error audit. Never fits weights or reads reports/images."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from rsna_knee.data.folds import load_folds

ROOT = Path(__file__).resolve().parents[1]
SCREEN = ROOT/'artifacts/kaggle/rsna-meniscus-complementarity-20261020'
OUT = ROOT/'artifacts/research/oa-synovitis-diagnostic-20261020'
TARGETS = ('Lateral OA', 'Synovitis')


def pair_errors(y, score):
    """Positive-negative misorder cost: 1 reversal, .5 tie, 0 correct."""
    y=np.asarray(y);score=np.asarray(score,dtype=float)
    assert y.shape==score.shape and y.ndim==1 and np.isfinite(score).all()
    assert set(np.unique(y))=={0,1}
    p=np.flatnonzero(y==1);n=np.flatnonzero(y==0)
    d=score[p,None]-score[n]
    return p,n,(d<0).astype(float)+.5*(d==0)


def aligned_csv(path, uids):
    frame=pd.read_csv(path,dtype={'StudyInstanceUID':str})
    assert not frame.StudyInstanceUID.duplicated().any()
    assert set(frame.StudyInstanceUID)==set(uids)
    return frame.set_index('StudyInstanceUID').loc[uids]


def weak_confirmation(evidence):
    """Check descriptive hypotheses on each own model's held-out weak fold."""
    lp=ROOT/'state/campaigns/campaign-20261003-v2/numeric-training.jsonl'
    label_hash=hashlib.sha256(lp.read_bytes()).hexdigest()
    evidence[str(lp.relative_to(ROOT))]=label_hash
    labels={r['StudyInstanceUID']:r for r in map(json.loads,lp.read_text().splitlines())}
    results={}
    for tag in ('K1','K1F1'):
        folder=ROOT/f'artifacts/colab/p1-{tag}/output'
        exp=json.loads((folder/'experiment.json').read_text())
        metrics=json.loads((folder/'job/metrics.json').read_text())
        assert metrics['kind']=='real' and metrics['applied_updates']>0
        assert exp['labels_sha256']==label_hash
        fp=ROOT/exp['folds_file']
        assert hashlib.sha256(fp.read_bytes()).hexdigest()==exp['folds_sha256']
        roles={r['StudyInstanceUID']:r for r in load_folds(fp)}
        path=folder/'job/predictions.jsonl'
        for p in (path,folder/'experiment.json',folder/'job/metrics.json'):
            evidence[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
        pp=[r for r in map(json.loads,path.read_text().splitlines()) if r['split']=='weak']
        us=[r['StudyInstanceUID'] for r in pp]
        assert len(us)==len(set(us)) and set(us)==set(exp['weak_uids'])
        assert all(str(roles[u]['fold'])==str(exp['outer_fold']) for u in us)
        train=set(metrics['train_uids']);assert train==set(exp['train_uids'])
        assert {roles[u]['group_id'] for u in train}.isdisjoint({roles[u]['group_id'] for u in us+exp['gold_uids']})
        result={}
        for target in TARGETS:
            companion='Medial OA' if target=='Lateral OA' else 'Effusion'
            slices={}
            for value in (None,0,1):
                pairs=[]
                for r in pp:
                    lab=labels.get(r['StudyInstanceUID'],{});mask=lab.get('y_mask',{});ys=lab.get('targets',{})
                    if not mask.get(target) or ys.get(target) not in (0,1):continue
                    if value is not None and (not mask.get(companion) or ys.get(companion)!=value):continue
                    pairs.append((ys[target],r[target]))
                y=np.array([a for a,b in pairs]);s=np.array([b for a,b in pairs]);assert np.isfinite(s).all()
                slices['all' if value is None else f'{companion}={value}']={'n':len(y),'positives':int(y.sum()),
                    'negatives':int((y==0).sum()),'auc':float(roc_auc_score(y,s)) if len(set(y))==2 else None}
            result[target]=slices
        results[tag]={'outer_fold':exp['outer_fold'],'targets':result}
    return {'scope':'Held-out weak report labels, not image truth; descriptive subgroup confirmation only','models':results}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    evidence={}
    def read_json(path):
        evidence[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())
    plan=read_json(SCREEN/'plan.json');proof=read_json(SCREEN/'output/screen-audit.json')
    assert proof['status']=='PASS' and proof['kind']=='real'
    assert hashlib.sha256((SCREEN/'output/reference-gold.csv').read_bytes()).hexdigest()==proof['predictions_sha256']
    rows=read_json(SCREEN/'gold-targets.json'); labels={r['StudyInstanceUID']:r for r in rows}
    assert len(rows)==len(labels)==58
    uids=sorted(labels)
    paths={'reference':'reference-gold.csv','coatnet384':'raptor_sub.csv','coatnet224':'raptor224_sub.csv','r6':'ours_sub.csv','reader':'_own.csv'}
    frames={}
    for name,file in paths.items():
        p=SCREEN/'output'/file
        evidence[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
        frames[name]=aligned_csv(p,uids)
    exp=read_json(ROOT/'artifacts/kaggle/rsna-meniscus-mri-twofold-20261020/experiment.json')
    metas={}
    for part in exp['cache_parts']:
        p=ROOT/'artifacts/kaggle'/part['kernel'].split('/')[-1]/'output/cache/manifest.json'
        assert hashlib.sha256(p.read_bytes()).hexdigest()==part['manifest_sha256']
        metas.update(read_json(p)['studies'])
    assert set(uids)<=set(metas)
    fp=ROOT/plan['folds_file'];assert hashlib.sha256(fp.read_bytes()).hexdigest()==plan['folds_sha256']
    group={r['StudyInstanceUID']:r['group_id'] for r in load_folds(fp)}
    quality={u:{'sagittal_missing':not any(metas[u]['slot_mask'][s] for s in (0,3)),
                'coronal_missing':not any(metas[u]['slot_mask'][s] for s in (1,4)),
                'axial_missing':not any(metas[u]['slot_mask'][s] for s in (2,5)),
                'any_series_failure':bool(metas[u]['series_failures'])} for u in uids}
    result={'kind':'real','independent_holdout':False,'submission_eligible':False,'targets':{},
            'limitation':'58 reused development studies; CoAtNet training overlap. Descriptive associations cannot establish image pathology or error causes. Cache metadata describes our cache, not the public models decoding.',
            'method':'Positive-negative ranking errors; no classification threshold, blend search, label edits or report reads.'}
    cases=[]
    for target in TARGETS:
        us=[u for u in uids if labels[u]['y_mask'].get(target) and labels[u]['targets'].get(target) in (0,1)]
        y=np.array([labels[u]['targets'][target] for u in us]);scores={k:f.loc[us,target].to_numpy(float) for k,f in frames.items()}
        p,n,err=pair_errors(y,scores['reference']);auc=roc_auc_score(y,scores['reference'])
        assert np.isclose(auc,1-err.mean())
        entry={'n':len(us),'positives':len(p),'negatives':len(n),'reference_auc':float(auc),
               'n_pairs':int(err.size),'misordered_pair_equivalents':float(err.sum()),'components':{},'strata':{}}
        for name,score in scores.items():
            _,_,e=pair_errors(y,score)
            entry['components'][name]={'auc':float(roc_auc_score(y,score)),
                'reference_error_pairs_corrected':int(((err>0)&(e==0)).sum()),
                'reference_correct_pairs_broken':int(((err==0)&(e>0)).sum()),
                'spearman_with_reference':float(pd.Series(score).corr(pd.Series(scores['reference']),method='spearman'))}
        errors=[pair_errors(y,scores[k])[2] for k in paths if k!='reference']
        entry['pairs_wrong_in_all_four_components']=int(np.logical_and.reduce([e>0 for e in errors]).sum())
        rng=np.random.default_rng(2026);groups=sorted({group[u] for u in us});gi=np.array([groups.index(group[u]) for u in us]);boots=[]
        for _ in range(2000):
            counts=np.bincount(rng.integers(len(groups),size=len(groups)),minlength=len(groups));w=counts[gi]
            if w[y==0].sum() and w[y==1].sum():boots.append(roc_auc_score(y,scores['reference'],sample_weight=w))
        entry['descriptive_group_bootstrap_auc_ci95']=np.quantile(boots,[.025,.975]).tolist()
        entry['bootstrap_draws']=len(boots)
        companion='Medial OA' if target=='Lateral OA' else 'Effusion'
        for attr,values in [('companion_'+companion,[labels[u]['targets'].get(companion) if labels[u]['y_mask'].get(companion) else None for u in us])]+[(key,[quality[u][key] for u in us]) for key in quality[us[0]]]:
            strata={}
            for value in sorted(set(values),key=str):
                ids=np.array([i for i,v in enumerate(values) if v==value]);yy=y[ids]
                strata[str(value)]={'n':len(ids),'positives':int(yy.sum()),'negatives':int((yy==0).sum()),
                    'auc':float(roc_auc_score(yy,scores['reference'][ids])) if len(set(yy))==2 else None}
            entry['strata'][attr]=strata
        for i,u in enumerate(us):
            burden=float(err[np.flatnonzero(p==i)[0]].mean()) if y[i] else float(err[:,np.flatnonzero(n==i)[0]].mean())
            cases.append({'StudyInstanceUID':u,'target':target,'label':int(y[i]),'pair_error_fraction':burden,
                          'reference_rank':float(pd.Series(scores['reference']).rank(pct=True).iloc[i]),
                          'companion_target':companion,'companion_label':labels[u]['targets'].get(companion),
                          **quality[u],**{name+'_score':float(s[i]) for name,s in scores.items()}})
        result['targets'][target]=entry
    pd.DataFrame(cases).sort_values(['target','pair_error_fraction'],ascending=[True,False]).to_csv(OUT/'case-review-local.csv',index=False)
    result['own_weak_confirmation']=weak_confirmation(evidence)
    result['evidence_sha256']=evidence
    (OUT/'diagnostic.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='evidence_sha256'},indent=2))

if __name__=='__main__': main()
