"""Build OA-COND-01 using existing MRI checkpoints and read-only volume caches."""
import base64
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import make_colab_notebook as mcn
import prepare_k1_kaggle_training as prep
from prepare_meniscus_mri import decode
from rsna_knee.hashing import sha256_file
from rsna_knee.data.folds import load_folds, image_training_uids, assert_train_uids, assert_group_integrity
from rsna_knee.models.conditional_sampling import conditional_weights
ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'artifacts/kaggle/rsna-oa-conditional-twofold-20261020'
NAMES=['Medial OA','Lateral OA']


def main():
    FOLDER.mkdir(parents=True,exist_ok=True)
    assert not (FOLDER/'push-attempt.json').exists(),'already attempted launch'
    base=ROOT/'artifacts/kaggle/rsna-meniscus-mri-twofold-20261020'
    e=json.loads((base/'experiment.json').read_text());n=json.loads((base/'train.ipynb').read_text())
    rows=load_folds(ROOT/e['folds_file']);assert_group_integrity(rows)
    assert sha256_file(ROOT/e['folds_file'])==e['folds_sha256']
    roles={r['StudyInstanceUID']:r for r in rows}
    lp=ROOT/'state/campaigns/campaign-20261003-v2/numeric-training.jsonl'
    assert sha256_file(lp)==e['labels_sha256']
    labs={r['StudyInstanceUID']:r for r in map(json.loads,lp.read_text().splitlines())}
    metas={}
    for part in e['cache_parts']:
        p=ROOT/'artifacts/kaggle'/part['kernel'].split('/')[-1]/'output/cache/manifest.json'
        assert sha256_file(p)==part['manifest_sha256']
        metas.update(json.loads(p.read_text())['studies'])
    eligible={u for u,m in metas.items() if any(c['anatomical_order_verified'] and c['n']>=3 for cs in m['candidates'] for c in cs)}
    allowed=set(image_training_uids(rows))&eligible
    def target(u):
        r=labs.get(u,{});m=[bool(r.get('y_mask',{}).get(k)) and r.get('targets',{}).get(k) in (0,1) for k in NAMES]
        return [float(r['targets'][k]) if ok else 0. for k,ok in zip(NAMES,m)],m
    observed={u for u in allowed if any(target(u)[1])}
    train={f:sorted(u for u in observed if str(roles[u]['fold'])!=f) for f in ('0','1')}
    val={f:sorted(u for u in allowed if str(roles[u]['fold'])==f) for f in ('0','1')}
    gold=e['gold_uids'];assert set(gold)<=eligible and len(gold)==58
    sampling={}
    for f,tag in [('0','K1'),('1','K1F1')]:
        assert_train_uids(set(train[f]),rows)
        held=val[f]+gold
        assert {roles[u]['group_id'] for u in train[f]}.isdisjoint({roles[u]['group_id'] for u in held})
        orig=ROOT/f'artifacts/colab/p1-{tag}/output'
        cfg=json.loads((orig/'experiment.json').read_text());metrics=json.loads((orig/'job/metrics.json').read_text())
        assert metrics['kind']=='real' and metrics['applied_updates']>0
        assert cfg['folds_sha256']==e['folds_sha256'] and str(cfg['outer_fold'])==f
        assert sha256_file(orig/'job/checkpoint.pt')==e['mri_encoders'][f]['checkpoint_sha256']==metrics['oof_provenance']['checkpoint_sha256']
        enc_train=set(metrics['train_uids']);assert enc_train==set(cfg['train_uids']);assert_train_uids(enc_train,rows)
        assert all(str(roles[u]['fold'])!=f for u in enc_train)
        assert {roles[u]['group_id'] for u in enc_train}.isdisjoint({roles[u]['group_id'] for u in held})
        yy,mm=zip(*(target(u) for u in train[f]));w,counts=conditional_weights(yy,mm)
        assert min(counts)>0 and len(train[f])>500
        sampling[f]={'joint_counts':counts,'weights_sha256':hashlib.sha256(w.tobytes()).hexdigest(),
                     'n_train':len(train[f]),'weight_min':float(w.min()),'weight_max':float(w.max()),'unknown_companion_weight':1.}
    archive,files=mcn.source_archive('WORKTREE',extra=('configs','pyproject.toml'))
    assert 'src/rsna_knee/models/conditional_sampling.py' in files
    payload=decode(n['cells'][4]);lines=payload.splitlines();assert lines[0].startswith('SOURCE_B64 = ')
    lines[0]='SOURCE_B64 = '+repr(base64.b64encode(archive).decode());payload='\n'.join(lines)+'\n'
    prefix=(ROOT/'scripts/meniscus_sequence_pilot.py').read_text().split('x = torch.stack(features).cuda();mask = torch.stack(masks).cuda()')[0]
    prefix=prefix.replace("names = ['Medial Meniscus', 'Lateral Meniscus']",'names = '+repr(NAMES))
    prefix=prefix.replace('targets = soft_table(SOFT.read_bytes())',"eval_rows={r['StudyInstanceUID']:r for r in map(json.loads,(WORK/'evaluation-targets.jsonl').read_text().splitlines())}")
    start=prefix.index('weights = WORK / EXPERIMENT');end=prefix.index('encoder.cuda().eval().requires_grad_(False)')
    prefix=prefix[:start]+'''expected_ck=EXPERIMENT['mri_encoders'][ACTIVE_FOLD]['checkpoint_sha256']
paths=[p for p in Path('/kaggle/input').rglob('checkpoint.pt') if sha256_file(p)==expected_ck]
assert paths, 'pinned encoder checkpoint missing'
ck=torch.load(paths[0],map_location='cpu',weights_only=False)
assert {roles[u]['group_id'] for u in ck['train_uids']}.isdisjoint({roles[u]['group_id'] for u in EXPERIMENT['fold_eval_uids'][ACTIVE_FOLD]+EXPERIMENT['gold_uids']}) if 'train_uids' in ck else True
encoder=timm.create_model('convnext_tiny',pretrained=False,num_classes=0)
encoder.load_state_dict({k.removeprefix('enc.'):v for k,v in ck['model'].items() if k.startswith('enc.')},strict=True)
del ck
'''+prefix[end:]
    prefix=prefix.replace('slots = [0, 1, 3, 4]  # sagittal/coronal, FS and non-FS; no axial','slots = list(range(6))')
    prefix=prefix.replace('torch.zeros(4,16,','torch.zeros(6,16,').replace(' < 2700',' < 18000')
    prefix=prefix.replace("EXPERIMENT['train_uids']","EXPERIMENT['fold_extract_uids'][ACTIVE_FOLD]")
    prefix=prefix.replace("if c['anatomical_order_verified']]","if c['anatomical_order_verified'] and c['n']>=3]")
    a=prefix.index('    y = [targets[uid]');b=prefix.index('    features.append',a)
    prefix=prefix[:a]+'''    row=eval_rows.get(uid,{})
    m=[bool(row.get('y_mask',{}).get(k)) and row.get('targets',{}).get(k) in (0,1) for k in names]
    y=[float(row['targets'][k]) if ok else 0. for k,ok in zip(names,m)]
'''+prefix[b:]
    prefix=prefix.replace("    print('encoded'","    if len(used)%50==0: print('encoded'")
    code="import time, hashlib\nfrom rsna_knee.data.folds import load_folds\nroles={r['StudyInstanceUID']:r for r in load_folds(FOLDS)}\n_all_started=time.monotonic()\n_all_results=[]\n_all_preds=[]\n_all_ck=[]\n"
    code+="for ACTIVE_FOLD in ('0','1'):\n"+'\n'.join('    '+line for line in (prefix+'\n'+(ROOT/'scripts/oa_conditional_train.py').read_text()).splitlines())+'\n'
    code+="    _all_results.extend(results);_all_preds.extend(predictions);_all_ck.extend(checkpoints)\n    del x, mask, y, ym\n    torch.cuda.empty_cache()\n"
    code+="report={'kind':'real','status':'COMPLETE','results':_all_results,'checkpoints':_all_ck,'seconds':time.monotonic()-_all_started,'gold_in_training':0,'submission_eligible':False,'gold_is_independent':False,'sampling':EXPERIMENT['sampling']}\n(WORK/'oa-metrics.json').write_text(json.dumps(report,indent=2))\n(WORK/'oa-predictions.jsonl').write_text(''.join(json.dumps(r)+'\\n' for r in _all_preds))\nprint(json.dumps(report),flush=True)\n"
    compile(code,'oa-training','exec')
    e.update(id='OA_COND_01',targets=NAMES,train_uids=sorted(set(train['0']+train['1'])),fold_train_uids=train,fold_eval_uids=val,
             fold_extract_uids={f:sorted(set(train[f]+val[f]+gold)) for f in train},weak_uids=sorted(set(val['0']+val['1'])),
             sampling=sampling,head_epochs=15,source_sha256=hashlib.sha256(archive).hexdigest(),
             training_cell_sha256=hashlib.sha256(code.encode()).hexdigest(),
             scope='Six-slot frozen MRI encoder; masked binary OA targets; paired uniform vs conditional replacement sampling',
             decision_rule={'primary':'Lateral OA positive delta both folds and paired group bootstrap mean delta CI lower > 0',
                            'secondary':'Medial OA positive subgroup lateral OA delta nonnegative both folds','n_boot':2000,'no_automatic_promotion':True})
    for key in ['extract_uids','control_kernel','config_sha256']:e.pop(key,None)
    e['train_uids_sha256']=hashlib.sha256('\n'.join(e['train_uids']).encode()).hexdigest()
    e['config_sha256']=hashlib.sha256(json.dumps(e,sort_keys=True).encode()).hexdigest()
    n['cells'][0]=mcn._md('# OA-COND-01\nPaired two-fold conditional sampling experiment. Existing MRI encoders, six slots, masked hard labels. No gold fitting or submission.')
    n['cells'][1]=prep.packed_code('import json\nEXPERIMENT=json.loads('+repr(json.dumps(e))+')','oa-config')
    n['cells'][4]=prep.packed_code(payload,'oa-payload');n['cells'][5]=prep.packed_code(code,'oa-training')
    n['cells'][6]=mcn._code("print((WORK/'oa-metrics.json').read_text())")
    for c in n['cells']:
        if c['cell_type']=='code':compile(''.join(c['source']),'cell','exec');c['outputs']=[];c['execution_count']=None
    (FOLDER/'train.ipynb').write_text(json.dumps(n,indent=1))
    meta=json.loads((base/'kernel-metadata.json').read_text());meta.update(id='bulsahkecici/'+FOLDER.name,title=FOLDER.name)
    (FOLDER/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    (FOLDER/'experiment.json').write_text(json.dumps(e,indent=2))
    audit={'status':'PASS','sampling':sampling,'n_eval':{f:len(v) for f,v in val.items()},'n_gold':len(gold),'notebook_sha256':sha256_file(FOLDER/'train.ipynb'),'gold_group_overlap':0,'submission':False}
    (FOLDER/'local-audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
