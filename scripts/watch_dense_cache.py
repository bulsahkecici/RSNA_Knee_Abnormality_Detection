"""Monitor one CPU cache, verify remotely, and keep competition submissions disabled."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
from filelock import FileLock
from autonomous_campaign import call,write,notify,ROOT

VERIFY='''import hashlib,json,shutil
from pathlib import Path
import numpy as np
roots=[p.parent for p in Path('/kaggle/input').rglob('campaign-result.json')]
assert len(roots)==1
root=roots[0];result=json.loads((root/'campaign-result.json').read_text())
assert result['status']=='ok' and result['requested_complete'] and result['n_quarantine']==0
path=root/'cache/manifest.json';assert hashlib.sha256(path.read_bytes()).hexdigest()==result['manifest_sha256']
m=json.loads(path.read_text());policy=m['preprocess_config']
assert policy['storage_format']=='npz' and policy['n_centers']==12
assert len(m['studies'])==4219 and set(m['studies'])==set(m['requested_uids'])
shape=(len(policy['slots']),12,3,224,224)
for uid,meta in m['studies'].items():
 assert meta['synthetic'] is False and meta['storage_format']=='npz' and meta['shard_filename']==uid+'.npz'
 assert meta['preprocess_hash']==m['preprocess_hash']
 assert len(meta['slot_mask'])==len(policy['slots']) and any(meta['slot_mask']) and all(v in (0,1) for v in meta['slot_mask'])
 p=root/'cache'/(uid+'.npz');h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 assert h.hexdigest()==meta['sha256']
 with np.load(p,allow_pickle=False) as archive:
  assert archive.files==['images'];a=archive['images']
  assert a.shape==shape and a.dtype==np.uint8
  for slot,present in enumerate(meta['slot_mask']):
   assert (present and a[slot].max()>a[slot].min()) or (not present and not np.any(a[slot]))
shutil.copyfile(path,'/kaggle/working/manifest.json')
shutil.copyfile(root/'campaign-result.json','/kaggle/working/campaign-result.json')
shutil.copyfile(root/'pilot-measurement.json','/kaggle/working/pilot-measurement.json')
Path('/kaggle/working/verification.json').write_text(json.dumps({'pass':True,'n_studies':4219,'shape':shape,'all_shard_hashes_checked':True,'storage_format':'npz'}))
'''


def watch(tag):
    folder=ROOT/'state/campaigns'/('cache-'+tag+'-20261009');folder.mkdir(parents=True,exist_ok=True)
    path=folder/'status.json'
    with FileLock(str(folder/'watch.lock'),timeout=0):
        state=json.loads(path.read_text()) if path.exists() else {'kernel':'bulsahkecici/rsna-'+tag+'-cache-20261009'}
        def save(phase):state.update(phase=phase,checked_ts=time.time());write(path,state)
        try:
            for phase,kernel in [('CACHE_RUNNING',state['kernel']),('VERIFYING','bulsahkecici/rsna-'+tag+'-verify-20261009')]:
                if phase=='VERIFYING':
                    asset=ROOT/'artifacts/kaggle'/(tag+'-verify');asset.mkdir(parents=True,exist_ok=True)
                    (asset/'entry.py').write_text(VERIFY)
                    (asset/'kernel-metadata.json').write_text(json.dumps({'id':kernel,'title':tag+' verification','code_file':'entry.py',
                        'language':'python','kernel_type':'script','is_private':True,'enable_gpu':False,'enable_internet':False,
                        'competition_sources':[],'dataset_sources':[],'kernel_sources':[state['kernel']]},indent=2))
                    if state.get('verify_push')=='UNKNOWN':raise RuntimeError('Reconcile uncertain verification push before retry')
                    if not state.get('verify_receipt'):
                        state['verify_push']='UNKNOWN';save('VERIFY_PREPARING')
                        state['verify_receipt']=call(['kernels','push','-p',str(asset)]);state['verify_push']='ACKNOWLEDGED';save('VERIFYING')
                started=time.time()
                while True:
                    status=call(['kernels','status',kernel]);state['observed']=status.strip();save(phase)
                    if 'COMPLETE' in status:break
                    if 'ERROR' in status:raise RuntimeError('CPU cache/verification kernel failed: '+kernel)
                    if time.time()-started>10*3600:raise RuntimeError('CPU cache monitoring exceeded planning timeout')
                    time.sleep(60)
                if phase=='CACHE_RUNNING':
                    dest=ROOT/'artifacts/kaggle'/tag/'output';dest.mkdir(exist_ok=True)
                    call(['kernels','output',kernel,'-p',str(dest),'--file-pattern','campaign-result.json$|pilot-measurement.json$|cache/manifest.json$'])
                    result=json.loads((dest/'campaign-result.json').read_text());assert result['status']=='ok'
                    import hashlib
                    prep=json.loads((ROOT/'artifacts/kaggle'/tag/'preparation.json').read_text())
                    assert result['source_archive_sha256']==prep['source_archive_sha256']
                    assert hashlib.sha256((dest/'cache/manifest.json').read_bytes()).hexdigest()==result['manifest_sha256']
                    state['cache_result']=result
                else:
                    dest=asset/'output';dest.mkdir(exist_ok=True)
                    call(['kernels','output',kernel,'-p',str(dest),'--file-pattern','verification.json$|campaign-result.json$|pilot-measurement.json$|manifest.json$'])
                    proof=json.loads((dest/'verification.json').read_text());assert proof['pass']
                    state['verification']=proof;save('VERIFIED');notify('Yoğun 12 merkezli önbellek doğrulandı; Colab deneyine hazır. Kaggle gönderimi kapalı.')
        except BaseException as error:
            state['error']=str(error)[:1500];save('NEEDS_ATTENTION');notify('Yoğun önbellek kontrolü inceleme gerektiriyor.');raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--tag',default='v4-dense12');watch(parser.parse_args().tag)
