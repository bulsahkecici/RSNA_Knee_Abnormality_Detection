"""Prepare a train-only sequence-head feasibility pilot on existing remote caches."""
import copy,json,hashlib,sys
from pathlib import Path
import prepare_k1_kaggle_training as prep
from rsna_knee.data.soft_pool import soft_table
from rsna_knee.data.folds import load_folds,assert_train_uids,assert_group_integrity
from rsna_knee.hashing import sha256_file
ROOT=Path(__file__).resolve().parents[1]
def main():
    parts,notes,cached=prep.cache_receipts()
    source=ROOT/'artifacts/kaggle/rsna-k1-kneenet-fold0-20261015'
    notebook=json.loads((source/'K1.ipynb').read_text());ns={}
    exec(''.join(notebook['cells'][1]['source']),ns)
    exp=copy.deepcopy(ns['EXPERIMENT'])
    soft=ROOT/'state/soft-labels/public-mean4-v0.jsonl'
    assert sha256_file(soft)==exp['soft_labels_sha256']
    table=soft_table(soft.read_bytes());metas={}
    for part in parts:
        p=ROOT/'artifacts/kaggle'/part['kernel'].split('/')[-1]/'output/cache/manifest.json'
        metas.update(json.loads(p.read_text())['studies'])
    eligible=[]
    for u in exp['train_uids']:
        if not any(table[u].get(k) is not None and (table[u][k]<=.1 or table[u][k]>=.9) for k in ['Medial Meniscus','Lateral Meniscus']):continue
        # Require sagittal and coronal geometry-verified candidates; no label-based class balancing.
        if all(any(c['anatomical_order_verified'] for s in group for c in metas[u]['candidates'][s]) for group in [(0,3),(1,4)]):eligible.append(u)
    selected=sorted(eligible,key=lambda u:hashlib.sha256(u.encode()).hexdigest())[:96]
    assert len(selected)==96
    folds=load_folds(ROOT/exp['folds_file']);assert_group_integrity(folds);assert_train_uids(set(selected),folds)
    role={r['StudyInstanceUID']:r for r in folds}
    assert all(role[u]['fold']!='0' for u in selected)
    assert set(selected).isdisjoint(exp['gold_uids']+exp['weak_uids'])
    exp.update(id='MENISCUS_SEQUENCE_PILOT',train_uids=selected,gold_uids=[],weak_uids=[],
               train_uids_sha256=hashlib.sha256('\n'.join(selected).encode()).hexdigest(),
               scope='96 train-only studies, three head epochs, frozen ImageNet encoder; no validation or ROI claim',
               cache_parts=parts,submission=False)
    train=(ROOT/'scripts/meniscus_sequence_pilot.py').read_text()
    exp['training_cell_sha256']=hashlib.sha256(train.encode()).hexdigest()
    exp['config_sha256']=hashlib.sha256(json.dumps(exp,sort_keys=True).encode()).hexdigest()
    notebook['cells'][0]={'cell_type':'markdown','metadata':{},'source':['# Meniscus ordered-sequence feasibility pilot\n96 train-only studies. Frozen encoder; two-target head. No gold or outer-fold training. No AUC, anatomical ROI, or submission claim.']}
    notebook['cells'][1]=prep.packed_code('import json\nEXPERIMENT=json.loads('+repr(json.dumps(exp))+')','meniscus-pilot-config')
    notebook['cells'][5]=prep.packed_code(train,'meniscus-sequence-pilot')
    notebook['cells'][6]={'cell_type':'code','metadata':{},'source':["print((WORK/'pilot-result.json').read_text())"],'outputs':[],'execution_count':None}
    for c in notebook['cells']:
        if c['cell_type']=='code':compile(''.join(c['source']),'pilot','exec');c['outputs']=[];c['execution_count']=None
    p=ROOT/'artifacts/kaggle/rsna-meniscus-sequence-pilot-20261020';p.mkdir(parents=True,exist_ok=True)
    assert not (p/'push-receipt.json').exists(),'already launched'
    (p/'pilot.ipynb').write_text(json.dumps(notebook,indent=1))
    meta=json.loads((source/'kernel-metadata.json').read_text());meta.update(id='bulsahkecici/'+p.name,title=p.name,code_file='pilot.ipynb')
    (p/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    (p/'experiment.json').write_text(json.dumps(exp,indent=2))
    (p/'local-audit.json').write_text(json.dumps({'status':'PASS','n_train':96,'gold_overlap':0,'holdout_fold_overlap':0,'group_integrity':True,'notebook_sha256':sha256_file(p/'pilot.ipynb'),'training_cell_sha256':exp['training_cell_sha256'],'submission':False},indent=2))
    print(p)
if __name__=='__main__':main()
