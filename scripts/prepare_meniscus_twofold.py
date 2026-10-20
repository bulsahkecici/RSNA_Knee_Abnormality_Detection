"""Build a frozen-feature, paired two-fold meniscus experiment without reading reports."""
import ast,base64,copy,hashlib,json,lzma
from pathlib import Path
import make_colab_notebook as mcn
import prepare_k1_kaggle_training as prep
from rsna_knee.data.folds import load_folds,image_training_uids,assert_train_uids,assert_group_integrity
from rsna_knee.data.soft_pool import soft_table
from rsna_knee.hashing import sha256_file
ROOT=Path(__file__).resolve().parents[1]

def main():
    parts,_,cached=prep.cache_receipts()
    old=ROOT/'artifacts/kaggle/rsna-k1-kneenet-fold0-20261015'
    n=json.loads((old/'K1.ipynb').read_text());ns={};exec(''.join(n['cells'][1]['source']),ns)
    e=copy.deepcopy(ns['EXPERIMENT'])
    table=soft_table((ROOT/'state/soft-labels/public-mean4-v0.jsonl').read_bytes())
    rows=load_folds(ROOT/e['folds_file']);assert_group_integrity(rows);role={r['StudyInstanceUID']:r for r in rows}
    assert sha256_file(ROOT/e['folds_file'])==e['folds_sha256']
    assert sha256_file(ROOT/'state/soft-labels/public-mean4-v0.jsonl')==e['soft_labels_sha256']
    metas={}
    for part in parts:
        p=ROOT/'artifacts/kaggle'/part['kernel'].split('/')[-1]/'output/cache/manifest.json'
        metas.update(json.loads(p.read_text())['studies'])
    eligible={u for u in cached if any(c['anatomical_order_verified'] for s in (0,1,3,4) for c in metas[u]['candidates'][s])}
    allowed=set(image_training_uids(rows))&eligible
    confident={u for u in allowed if any(table.get(u,{}).get(k) is not None and (table[u][k]<=.1 or table[u][k]>=.9) for k in ['Medial Meniscus','Lateral Meniscus'])}
    train={f:sorted(u for u in confident if role[u]['fold']!=f) for f in ('0','1')}
    val={f:sorted(u for u in allowed if role[u]['fold']==f) for f in ('0','1')}
    gold=sorted(set(e['gold_uids'])&eligible)
    assert len(gold)==58,'gold coverage changed'
    for f in train:
        assert_train_uids(set(train[f]),rows)
        assert {role[u]['group_id'] for u in train[f]}.isdisjoint({role[u]['group_id'] for u in val[f]+gold})
        assert len(train[f])>500 and len(val[f])>100
    extract=sorted(set(gold).union(*map(set,train.values()),*map(set,val.values())))
    archive,names=mcn.source_archive('WORKTREE',extra=('configs','pyproject.toml'))
    assert 'src/rsna_knee/models/meniscus_head.py' in names
    # Decode the already audited numeric/source payload; replace only the source archive.
    packed=''.join(n['cells'][4]['source'])
    const=max((x.value for x in ast.walk(ast.parse(packed)) if isinstance(x,ast.Constant) and isinstance(x.value,str)),key=len)
    payload=lzma.decompress(base64.b64decode(const)).decode()
    lines=payload.splitlines();assert lines[0].startswith('SOURCE_B64 = ')
    lines[0]='SOURCE_B64 = '+repr(base64.b64encode(archive).decode())
    payload='\n'.join(lines)+'\n'
    prefix=(ROOT/'scripts/meniscus_sequence_pilot.py').read_text().split('x = torch.stack(features).cuda();mask = torch.stack(masks).cuda()')[0]
    prefix=prefix.replace("EXPERIMENT['train_uids']","EXPERIMENT['extract_uids']").replace(' < 2700',' < 18000')
    prefix=prefix.replace("targets[uid].get(k)","targets.get(uid,{}).get(k)").replace('    assert any(m)\n','')
    prefix=prefix.replace("    print('encoded',len(used),'/',len(EXPERIMENT['extract_uids']),flush=True)","    if len(used)%50==0: print('encoded',len(used),'/',len(EXPERIMENT['extract_uids']),flush=True)")
    code=prefix+'\n'+(ROOT/'scripts/meniscus_twofold_train.py').read_text()
    compile(code,'meniscus-full','exec')
    e.update(id='MENISCUS_TWO_FOLD',train_uids=sorted(set(train['0'])|set(train['1'])),
             gold_uids=gold,weak_uids=sorted(set(val['0'])|set(val['1'])),fold_train_uids=train,fold_eval_uids=val,
             extract_uids=extract,source_sha256=hashlib.sha256(archive).hexdigest(),cache_parts=parts,
             head_epochs=15,scope='two-fold matched frozen-feature ablation; no model selection on gold',
             training_cell_sha256=hashlib.sha256(code.encode()).hexdigest(),submission=False,
             decision_rule={'paired_weak_delta_positive_both_folds':True,'bootstrap_pooled_ci_lower_gt':0,
                            'n_boot':2000,'no_gold_weight_fit':True,'no_automatic_promotion':True})
    e['train_uids_sha256']=hashlib.sha256('\n'.join(e['train_uids']).encode()).hexdigest()
    e['config_sha256']=hashlib.sha256(json.dumps(e,sort_keys=True).encode()).hexdigest()
    n['cells'][0]=mcn._md('# Meniscus two-fold matched-head experiment\nFrozen ImageNet features; ordered vs invariant heads; 15 fixed epochs. Gold excluded from training. Weak AUC is report-label agreement; gold is reused development data. No submission in this notebook.')
    n['cells'][1]=prep.packed_code('import json\nEXPERIMENT=json.loads('+repr(json.dumps(e))+')','twofold-config')
    n['cells'][4]=prep.packed_code(payload,'twofold-payload')
    n['cells'][5]=prep.packed_code(code,'twofold-training')
    n['cells'][6]=mcn._code("print((WORK/'meniscus-metrics.json').read_text())")
    for c in n['cells']:
        if c['cell_type']=='code':compile(''.join(c['source']),'cell','exec');c['outputs']=[];c['execution_count']=None
    p=ROOT/'artifacts/kaggle/rsna-meniscus-twofold-20261020';p.mkdir(parents=True,exist_ok=True)
    assert not (p/'push-receipt.json').exists(),'already launched'
    (p/'train.ipynb').write_text(json.dumps(n,indent=1))
    meta=json.loads((old/'kernel-metadata.json').read_text());meta.update(id='bulsahkecici/'+p.name,title=p.name,code_file='train.ipynb')
    (p/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    (p/'experiment.json').write_text(json.dumps(e,indent=2))
    audit={'status':'PASS','n_train':{f:len(v) for f,v in train.items()},'n_weak':{f:len(v) for f,v in val.items()},'n_gold':len(gold),'n_features':len(extract),'gold_train_overlap':0,'heldout_group_overlap':0,'notebook_sha256':sha256_file(p/'train.ipynb'),'submission':False}
    (p/'local-audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit))
if __name__=='__main__':main()
