"""Review OA paired sampling on held-out weak folds, then require separate ensemble evidence."""
import argparse
import fcntl
import hashlib
import json
import math
import time
import numpy as np
from sklearn.metrics import roc_auc_score
from rsna_knee.hashing import sha256_file
from rsna_knee.data.folds import load_folds
from prepare_oa_conditional import ROOT,FOLDER,NAMES
from check_public_r6 import cli


def review():
    exp=json.loads((FOLDER/'experiment.json').read_text());out=FOLDER/'output'
    assert json.loads((out/'experiment.json').read_text())==exp
    proof=json.loads((FOLDER/'local-audit.json').read_text())
    assert sha256_file(FOLDER/'train.ipynb')==proof['notebook_sha256']
    metrics=json.loads((out/'oa-metrics.json').read_text())
    assert metrics['kind']=='real' and metrics['status']=='COMPLETE' and metrics['gold_in_training']==0
    assert len(metrics['results'])==8 and metrics['sampling']==exp['sampling']
    expected_results={(f,m,s) for f in ('0','1') for m in ('natural','conditional') for s in ('weak','gold')}
    assert {(str(r['fold']),r['model'],r['split']) for r in metrics['results']}==expected_results
    for f in ('0','1'):
        rs=[r for r in metrics['results'] if str(r['fold'])==f]
        assert len({r['initial_sha256'] for r in rs})==1
        assert all(r['applied_updates']==15*math.ceil(len(exp['fold_train_uids'][f])/32) for r in rs)
    lp=ROOT/'state/campaigns/campaign-20261003-v2/numeric-training.jsonl'
    assert sha256_file(lp)==exp['labels_sha256']
    assert sha256_file(ROOT/exp['folds_file'])==exp['folds_sha256']
    labels={r['StudyInstanceUID']:r for r in map(json.loads,lp.read_text().splitlines())}
    roles={r['StudyInstanceUID']:r for r in load_folds(ROOT/exp['folds_file'])}
    preds={}
    for r in map(json.loads,(out/'oa-predictions.jsonl').read_text().splitlines()):
        key=(str(r['fold']),r['model'],r['split'],r['StudyInstanceUID']);assert key not in preds
        assert all(np.isfinite(r[k]) and 0<=r[k]<=1 for k in NAMES)
        preds[key]=r
    expected={(f,m,s,u) for f,m,s in expected_results for u in (exp['gold_uids'] if s=='gold' else exp['fold_eval_uids'][f])}
    assert set(preds)==expected
    records=[];results={};secondary=[];primary=[]
    for f in ('0','1'):
        train=exp['fold_train_uids'][f];eval_uids=exp['fold_eval_uids'][f]
        assert {roles[u]['group_id'] for u in train}.isdisjoint({roles[u]['group_id'] for u in eval_uids+exp['gold_uids']})
        assert all(str(roles[u]['fold'])==f for u in eval_uids)
        result={}
        for name,subgroup in [('Lateral OA',False),('Lateral OA',True),('Medial OA',False)]:
            us=[u for u in eval_uids if labels.get(u,{}).get('y_mask',{}).get(name) and labels[u]['targets'].get(name) in (0,1)]
            if subgroup:us=[u for u in us if labels[u]['y_mask'].get('Medial OA') and labels[u]['targets'].get('Medial OA')==1]
            y=np.array([labels[u]['targets'][name] for u in us]);assert len(set(y))==2
            a=np.array([preds[(f,'conditional','weak',u)][name] for u in us])
            b=np.array([preds[(f,'natural','weak',u)][name] for u in us])
            aa=float(roc_auc_score(y,a));bb=float(roc_auc_score(y,b));delta=aa-bb
            key=name+(' | Medial OA positive' if subgroup else '')
            result[key]={'n':len(us),'n_pos':int(y.sum()),'natural_auc':bb,'conditional_auc':aa,'delta':delta}
            if subgroup:secondary.append(delta)
            elif name=='Lateral OA':records.append((f,us,y,a,b));primary.append(delta)
        results[f]=result
    rng=np.random.default_rng(2026);boots=[]
    ids={f:sorted({roles[u]['group_id'] for u in exp['fold_eval_uids'][f]}) for f in ('0','1')}
    indices={f:{g:i for i,g in enumerate(gs)} for f,gs in ids.items()}
    for _ in range(2000):
        counts={f:np.bincount(rng.integers(len(gs),size=len(gs)),minlength=len(gs)) for f,gs in ids.items()}
        ds=[]
        for f,us,y,a,b in records:
            w=np.array([counts[f][indices[f][roles[u]['group_id']]] for u in us])
            if not w[y==0].sum() or not w[y==1].sum():break
            ds.append(roc_auc_score(y,a,sample_weight=w)-roc_auc_score(y,b,sample_weight=w))
        if len(ds)==2:boots.append(float(np.mean(ds)))
    assert len(boots)>=1900
    ci=np.quantile(boots,[.025,.975]).tolist()
    passed=all(v>0 for v in primary) and all(v>=0 for v in secondary) and ci[0]>0
    result={'kind':'real','decision':'NEEDS_REFERENCE_COMPLEMENTARITY' if passed else 'REJECT_CONDITIONAL_SAMPLING',
            'weak_fold_results':results,'paired_group_bootstrap_primary_delta_ci95':ci,'bootstrap_draws':len(boots),
            'all_remote_results':metrics['results'],'seconds':metrics['seconds'],
            'submission_eligible':False,'gold_is_independent':False,'competition_auc':None,
            'note':'Weak report-label agreement only. Passing requires separate reference complementarity assessment.'}
    (FOLDER/'review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    if not args.watch:review();return
    lock=(FOLDER/'watcher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    kernel=json.loads((FOLDER/'kernel-metadata.json').read_text())['id']
    deadline=time.monotonic()+7*3600
    while time.monotonic()<deadline:
        try:status=cli('kernels','status',kernel+'/1')
        except Exception as exc:
            (FOLDER/'last-query-error.txt').write_text(repr(exc));time.sleep(45);continue
        (FOLDER/'status.txt').write_text(status);print(status.strip(),flush=True)
        if 'COMPLETE' in status:
            cli('kernels','output',kernel+'/1','-p',str(FOLDER/'output'),'--force','--quiet','--file-pattern',r'(^|/)(experiment.json|oa-metrics.json|oa-predictions.jsonl|handshake.json|.*\.log)$')
            result=review();(FOLDER/'campaign-state.json').write_text(json.dumps({'status':'REVIEWED','decision':result['decision'],'submission_made':False},indent=2));return
        if any(v in status for v in ('ERROR','CANCEL','FAILED')):
            cli('kernels','output',kernel+'/1','-p',str(FOLDER/'output'),'--force','--quiet','--file-pattern',r'.*\.log$')
            (FOLDER/'campaign-state.json').write_text(json.dumps({'status':'REMOTE_FAILED','submission_made':False},indent=2));raise RuntimeError(status)
        time.sleep(45)
    raise TimeoutError('OA run did not finish in monitoring window')
if __name__=='__main__':main()
