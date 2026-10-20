"""Fold-matched MRI encoder ablation: immutable weights, no images/reports upload."""
import ast,base64,copy,hashlib,json,lzma
from pathlib import Path
import torch
from safetensors.torch import save_file
import prepare_k1_kaggle_training as prep
from rsna_knee.hashing import sha256_file
from rsna_knee.data.folds import load_folds,assert_train_uids,assert_group_integrity
ROOT=Path(__file__).resolve().parents[1]
SLUG='rsna-meniscus-mri-twofold-20261020'
DATASET='rsna-meniscus-fold-encoders-20261020'

def decode(cell):
    s=''.join(cell['source'])
    val=max((x.value for x in ast.walk(ast.parse(s)) if isinstance(x,ast.Constant) and isinstance(x.value,str)),key=len)
    return lzma.decompress(base64.b64decode(val)).decode()

def main():
    base=ROOT/'artifacts/kaggle/rsna-meniscus-twofold-20261020'
    e=json.loads((base/'experiment.json').read_text());n=json.loads((base/'train.ipynb').read_text())
    folder=ROOT/'artifacts/kaggle'/SLUG;folder.mkdir(parents=True,exist_ok=True)
    assert not (folder/'push-receipt.json').exists(),'already pushed'
    upload=ROOT/'artifacts/kaggle'/DATASET;upload.mkdir(parents=True,exist_ok=True)
    rows=load_folds(ROOT/e['folds_file']);assert_group_integrity(rows)
    roles={r['StudyInstanceUID']:r for r in rows};encoders={}
    for f,tag in [('0','K1'),('1','K1F1')]:
        p=ROOT/f'artifacts/colab/p1-{tag}/output'
        cfg=json.loads((p/'experiment.json').read_text());metrics=json.loads((p/'job/metrics.json').read_text())
        ck=p/'job/checkpoint.pt'
        assert sha256_file(ck)==metrics['oof_provenance']['checkpoint_sha256']
        assert cfg['folds_sha256']==e['folds_sha256']==sha256_file(ROOT/e['folds_file'])
        assert str(cfg['outer_fold'])==f and metrics['kind']=='real' and metrics['applied_updates']>0
        train=set(metrics['train_uids']);assert train==set(cfg['train_uids']);assert_train_uids(train,rows)
        assert all(roles[u]['fold']!=f for u in train)
        hold=set(e['fold_eval_uids'][f]+e['gold_uids'])
        assert {roles[u]['group_id'] for u in train}.isdisjoint({roles[u]['group_id'] for u in hold})
        state=torch.load(ck,map_location='cpu',weights_only=False)['model']
        encoder={k.removeprefix('enc.'):v.contiguous() for k,v in state.items() if k.startswith('enc.')}
        assert encoder
        dest=upload/f'encoder-f{f}.safetensors';save_file(encoder,str(dest))
        encoders[f]={'file':dest.name,'sha256':sha256_file(dest),'checkpoint_sha256':sha256_file(ck),
                     'outer_fold':f,'n_train':len(train),'training_uids_sha256':hashlib.sha256('\n'.join(sorted(train)).encode()).hexdigest(),
                     'heldout_group_overlap':0,'gold_overlap':0}
    (upload/'encoder-manifest.json').write_text(json.dumps(encoders,indent=2))
    (upload/'dataset-metadata.json').write_text(json.dumps({'id':'bulsahkecici/'+DATASET,'title':DATASET,
        'licenses':[{'name':'other'}],'description':'Private fold-matched ConvNeXt encoders from our K1/K1F1 checkpoints. No MRI, reports or labels. Original model and upstream pretrained component terms retained; no new license grant.'},indent=2))
    old=decode(n['cells'][5]);prefix,tail=old.split('from rsna_knee.models.meniscus_head import MeniscusHead',1)
    tail='from rsna_knee.models.meniscus_head import MeniscusHead'+tail
    start=prefix.index('weights = WORK / EXPERIMENT');end=prefix.index('encoder.cuda().eval().requires_grad_(False)')
    load="""expected_ck=EXPERIMENT['mri_encoders'][ACTIVE_FOLD]['checkpoint_sha256']
checkpoints_found=[p for p in Path('/kaggle/input').rglob('checkpoint.pt') if sha256_file(p)==expected_ck]
assert checkpoints_found, 'pinned prior Kaggle training checkpoint not mounted'
checkpoint=torch.load(checkpoints_found[0],map_location='cpu',weights_only=False)
encoder=timm.create_model('convnext_tiny',pretrained=False,num_classes=0)
encoder.load_state_dict({k.removeprefix('enc.'):v for k,v in checkpoint['model'].items() if k.startswith('enc.')},strict=True)
del checkpoint
"""
    prefix=prefix[:start]+load+prefix[end:]
    prefix=prefix.replace("EXPERIMENT['extract_uids']","EXPERIMENT['fold_extract_uids'][ACTIVE_FOLD]")
    tail=tail.replace("for fold in ('0','1'):","for fold in (ACTIVE_FOLD,):")
    e.update(id='MENISCUS_MRI_TWO_FOLD',mri_encoders=encoders,
        fold_extract_uids={f:sorted(set(e['fold_train_uids'][f]+e['fold_eval_uids'][f]+e['gold_uids'])) for f in ('0','1')},
        control_kernel='bulsahkecici/rsna-meniscus-twofold-20261020',
        scope='Fold-matched MRI encoder features; matched input/head/15-epoch recipe; no ROI or exact-mm claim')
    code="import time\n_all_started=time.monotonic()\n_all_results=[]\n_all_preds=[]\n_all_ck=[]\n"
    code+="for ACTIVE_FOLD in ('0','1'):\n"
    code+='\n'.join('    '+line for line in (prefix+'\n'+tail).splitlines())+'\n'
    code+="    _all_results.extend(results);_all_preds.extend(predictions);_all_ck.extend(checkpoints)\n    del x, mask, y, ym\n    torch.cuda.empty_cache()\n"
    code+="report.update(results=_all_results,checkpoints=_all_ck,seconds=time.monotonic()-_all_started,n_feature_studies=sum(len(v) for v in EXPERIMENT['fold_extract_uids'].values()),encoder_scope='fold-matched MRI-finetuned')\n(WORK/'meniscus-metrics.json').write_text(json.dumps(report,indent=2))\n(WORK/'meniscus-predictions.jsonl').write_text(''.join(json.dumps(r)+'\\n' for r in _all_preds))\nprint(json.dumps(report),flush=True)\n"
    compile(code,'mri-training','exec')
    e['training_cell_sha256']=hashlib.sha256(code.encode()).hexdigest()
    e['config_sha256']=hashlib.sha256(json.dumps(e,sort_keys=True).encode()).hexdigest()
    n['cells'][0]={'cell_type':'markdown','metadata':{},'source':['# Fold-matched MRI encoder ablation\nSame heads, inputs, masks and 15-epoch schedule as ImageNet experiment. Each encoder excludes its outer validation fold and all gold groups. No automatic submission.']}
    n['cells'][1]=prep.packed_code('import json\nEXPERIMENT=json.loads('+repr(json.dumps(e))+')','mri-config')
    n['cells'][5]=prep.packed_code(code,'mri-twofold')
    for c in n['cells']:
        if c['cell_type']=='code':compile(''.join(c['source']),'cell','exec');c['outputs']=[];c['execution_count']=None
    (folder/'train.ipynb').write_text(json.dumps(n,indent=1))
    meta=json.loads((base/'kernel-metadata.json').read_text());meta.update(id='bulsahkecici/'+SLUG,title=SLUG,dataset_sources=[],kernel_sources=meta['kernel_sources']+['bulsahkecici/rsna-k1-kneenet-fold0-20261015','bulsahkecici/rsna-k1f1-kneenet-fold1-20261015'])
    (folder/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    (folder/'experiment.json').write_text(json.dumps(e,indent=2))
    (folder/'local-audit.json').write_text(json.dumps({'status':'PASS','encoders':encoders,'notebook_sha256':sha256_file(folder/'train.ipynb'),'no_reports_or_images_uploaded':True,'submission':False},indent=2))
    print(folder);print(upload)
if __name__=='__main__':main()
