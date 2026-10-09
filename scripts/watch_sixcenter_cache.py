"""Track the auxiliary CPU cache and validate shards in a small-output bridge."""
import json,subprocess,time
from pathlib import Path
from filelock import FileLock
from autonomous_campaign import call,write,notify,ROOT,FOLDER
CACHE='bulsahkecici/rsna-v3-sixcenters-cache-20261009'
BRIDGE='bulsahkecici/rsna-v3-sixcenters-verify-20261009'
VERIFY='''import hashlib,json,shutil
from pathlib import Path
import numpy as np
roots=[p.parent for p in Path('/kaggle/input').rglob('campaign-result.json')]
assert len(roots)==1
root=roots[0];result=json.loads((root/'campaign-result.json').read_text());path=root/'cache/manifest.json'
assert result['status']=='ok' and result['n_quarantine']==0 and result['requested_complete']
assert hashlib.sha256(path.read_bytes()).hexdigest()==result['manifest_sha256']
m=json.loads(path.read_text());assert m['preprocess_config']['n_centers']==6
assert len(m['studies'])==4219 and set(m['studies'])==set(m['requested_uids'])
for uid,meta in m['studies'].items():
 assert meta['synthetic'] is False and meta['preprocess_hash']==m['preprocess_hash']
 assert len(meta['slot_mask'])==3 and any(meta['slot_mask']) and all(x in (0,1) for x in meta['slot_mask'])
 p=root/'cache'/f'{uid}.npy';h=hashlib.sha256()
 with p.open('rb') as handle:
  for block in iter(lambda:handle.read(1048576),b''):h.update(block)
 assert h.hexdigest()==meta['sha256']
 a=np.load(p,mmap_mode='r',allow_pickle=False);assert a.shape==(3,6,3,224,224) and a.dtype==np.uint8
 for slot,present in enumerate(meta['slot_mask']):
  assert (present and a[slot].max()>a[slot].min()) or (not present and not np.any(a[slot]))
shutil.copyfile(path,'/kaggle/working/manifest.json')
shutil.copyfile(root/'campaign-result.json','/kaggle/working/campaign-result.json')
Path('/kaggle/working/verification.json').write_text(json.dumps({'pass':True,'n_studies':4219,'n_centers':6,'all_shard_hashes_checked':True}))
'''
if __name__=='__main__':
 with FileLock(str(FOLDER/'sixcenter-watch.lock'),timeout=0):
  p=FOLDER/'sixcenter-cache.json';s=json.loads(p.read_text())
  try:
   while True:
    status=call(['kernels','status',CACHE]);s.update(phase='CACHE_RUNNING',observed=status.strip(),checked_ts=time.time());write(p,s)
    if 'COMPLETE' in status:break
    if 'ERROR' in status:raise RuntimeError('Six-center cache failed')
    time.sleep(60)
   dest=ROOT/'artifacts/kaggle/v3-sixcenters-verify';dest.mkdir(exist_ok=True);(dest/'entry.py').write_text(VERIFY)
   metadata={'id':BRIDGE,'title':BRIDGE.split('/')[1],'code_file':'entry.py','language':'python','kernel_type':'script','is_private':True,'enable_gpu':False,'enable_internet':False,'competition_sources':[],'dataset_sources':[],'kernel_sources':[CACHE]};(dest/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
   if s.get('verify_push')=='UNKNOWN':raise RuntimeError('Reconcile bridge push before retry')
   if 'verify_receipt' not in s:
    s['verify_push']='UNKNOWN';write(p,s);s['verify_receipt']=call(['kernels','push','-p',str(dest)]);s['verify_push']='ACKNOWLEDGED';write(p,s)
   while True:
    status=call(['kernels','status',BRIDGE]);s.update(phase='VERIFYING',verify_status=status.strip(),checked_ts=time.time());write(p,s)
    if 'COMPLETE' in status:break
    if 'ERROR' in status:raise RuntimeError('Six-center verification failed')
    time.sleep(60)
   call(['kernels','output',BRIDGE,'-p',str(dest/'output'),'--file-pattern','verification.json$|campaign-result.json$|manifest.json$'])
   verification=json.loads((dest/'output/verification.json').read_text());assert verification['pass']
   s.update(phase='VERIFIED',verification=verification);write(p,s);notify('Altı kesitli cache doğrulandı; sonraki eğitim deneyine hazır')
  except Exception as e:
   s.update(phase='NEEDS_ATTENTION',error=str(e)[:1500]);write(p,s);notify('Altı kesitli cache kontrolü inceleme gerektiriyor');raise
