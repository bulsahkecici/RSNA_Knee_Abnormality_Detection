"""Package a reviewed candidate, prove offline execution, then submit once."""
import csv
import io
import hashlib
import json
import math
import shutil
import tarfile
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def track_submission(state,statepath,call,write,notify):
 """Read-only reconciliation by the unique description, never resubmit."""
 description=state.get('submission_description')
 if not description:raise RuntimeError('Reconcile submission without recorded description')
 while True:
  rows=list(csv.DictReader(io.StringIO(call(['competitions','submissions','-c','rsna-knee-abnormality-detection','--csv']))))
  matches=[row for row in rows if row['description']==description]
  if len(matches)>1:raise RuntimeError('Ambiguous submission description; reconcile receipts')
  if matches:
   row=matches[0]
   if state.get('submission_ref') and str(state['submission_ref'])!=row['ref']:
    raise RuntimeError('Submission receipt changed')
   state.update(submission_ref=row['ref'],submission_status=row['status'],updated_ts=time.time())
   terminal=row['status'].split('.')[-1].upper()
   if terminal in {'COMPLETE','ERROR'}:
    score=row.get('publicScore','')
    if score:
     value=float(score)
     if not math.isfinite(value) or not 0<=value<=1:raise RuntimeError('Invalid Kaggle public score')
     state['public_score']=value
    state['phase']='SCORED' if terminal=='COMPLETE' and score else 'SUBMISSION_FAILED'
    write(statepath,state);notify('Kaggle sonucu: '+str(state.get('public_score',row['status'])))
    return state
   write(statepath,state)
  time.sleep(60)

def publish(item,folder,call,write,notify):
 statepath=folder/'publication.json'
 if statepath.exists():
  state=json.loads(statepath.read_text())
  if state['phase'] in ['SCORED','SUBMISSION_FAILED']:return state
  if state['phase']=='SUBMITTED':return track_submission(state,statepath,call,write,notify)
  if state['phase'] in ['SUBMIT_UNKNOWN','ASSET_PUSH_UNKNOWN','KERNEL_PUSH_UNKNOWN']:
   raise RuntimeError('Reconcile ambiguous/publication state before retry')
 else:state={'candidate':item['kernel'],'phase':'PREPARING'}
 def save(phase):state.update(phase=phase,updated_ts=time.time());write(statepath,state)
 out=ROOT/item['directory']/'output';checkpoint=out/'job/checkpoint.pt';metrics=json.loads((out/'job/metrics.json').read_text())
 assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==metrics['oof_provenance']['checkpoint_sha256']
 tag=item['id'];slug='rsna-auto-20261009-'+tag+'-asset';kernel='bulsahkecici/rsna-auto-20261009-'+tag+'-offline'
 asset=folder/(tag+'-asset');asset.mkdir(exist_ok=True);(asset/'assets').mkdir(exist_ok=True)
 shutil.copytree(ROOT/'src',asset/'src',dirs_exist_ok=True)
 for name in ['cache_build.py','cache.py','preprocess.py','dicom.py']:
  shutil.copyfile(ROOT/'scripts/cache_v3_source/rsna_knee/data'/name,asset/'src/rsna_knee/data'/name)
 source=asset/'src/rsna_knee/submission/offline.py';code=source.read_text().replace('strict_series=False,','strict_series=True,\n        center_strategy="interior", center_lower=0.15, center_upper=0.85, min_present_slots=1,').replace('meta["series_errors"] for uid, meta','meta.get("series_errors", []) for uid, meta');compile(code,str(source),'exec');source.write_text(code)
 cache_metadata=ROOT/item['preprocess_config'] if item.get('preprocess_config') else ROOT/'artifacts/kaggle/inspect-v3-cache/output/preprocess-config.json'
 shutil.copyfile(cache_metadata,asset/'assets/preprocess-config.json')
 shutil.copyfile(checkpoint,asset/'assets/checkpoint.pt');shutil.copytree(ROOT/'artifacts/kaggle/v3-offline-asset/wheels',asset/'wheels',dirs_exist_ok=True)
 files={str(p.relative_to(asset)):hashlib.sha256(p.read_bytes()).hexdigest() for p in asset.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='MANIFEST.json'}
 (asset/'MANIFEST.json').write_text(json.dumps({'internet_required':False,'files':files},indent=2))
 upload=folder/(tag+'-upload');upload.mkdir(exist_ok=True)
 with tarfile.open(upload/'offline-bundle.tar.gz','w:gz') as tar:
  for name in [*files,'MANIFEST.json']:tar.add(asset/name,arcname=name)
 (upload/'dataset-metadata.json').write_text(json.dumps({'title':slug,'id':'bulsahkecici/'+slug,'licenses':[{'name':'CC0-1.0'}]},indent=2))
 if 'asset_receipt' not in state:
  save('ASSET_PUSH_UNKNOWN');state['asset_receipt']=call(['datasets','create','-p',str(upload)]);save('ASSET_CREATED')
 while 'ready' not in call(['datasets','status','bulsahkecici/'+slug]).lower():time.sleep(60)
 dest=folder/(tag+'-offline');dest.mkdir(exist_ok=True)
 entry=(ROOT/'artifacts/kaggle/v3-offline-check/entry.py').read_text().replace("checkpoint=asset/'assets/checkpoint.pt',","checkpoint=asset/'assets/checkpoint.pt',preprocess_config=asset/'assets/preprocess-config.json',")
 (dest/'entry.py').write_text(entry)
 meta=json.loads((ROOT/'artifacts/kaggle/v3-offline-check/kernel-metadata.json').read_text());meta.update(id=kernel,title=kernel.split('/')[1],dataset_sources=['bulsahkecici/'+slug]);(dest/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
 if 'kernel_receipt' not in state:
  save('KERNEL_PUSH_UNKNOWN');state['kernel_receipt']=call(['kernels','push','-p',str(dest)]);save('OFFLINE_RUNNING')
 while True:
  observed=call(['kernels','status',kernel]);state['observed']=observed.strip();save('OFFLINE_RUNNING')
  if 'COMPLETE' in observed:break
  if 'ERROR' in observed:raise RuntimeError('Candidate offline validation failed')
  time.sleep(60)
 call(['kernels','output',kernel,'-p',str(dest/'output'),'--file-pattern','offline-benchmark.json$|csv-validation.json$|submission.csv$'])
 benchmark=json.loads((dest/'output/offline-benchmark.json').read_text());runtime=benchmark['runtime'];assert runtime['n_measured']>=30 and runtime['includes_decode_and_load'];assert runtime['projected_seconds']<9*3600
 validation=json.loads((dest/'output/csv-validation.json').read_text())['submission.csv'];assert validation['finite_probabilities'] and validation['unique_uids'] and validation['columns_match']
 with (dest/'output/submission.csv').open() as handle:
  reader=csv.DictReader(handle);rows=list(reader)
 assert len(rows)>0 and all(math.isfinite(float(r[k])) and 0<=float(r[k])<=1 for r in rows for k in reader.fieldnames if k!='StudyInstanceUID')
 save('OFFLINE_PASSED')
 limits=call(['competitions','submission-limits','-c','rsna-knee-abnormality-detection']);state['limits']=limits
 if 'Remaining today: 0' in limits:save('DAILY_LIMIT_WAIT');notify('Aday doğrulandı; günlük gönderim limiti bekleniyor');return
 state['submission_description']='Reviewed autonomous v3 candidate '+tag+' ['+str(time.time_ns())+']'
 save('SUBMIT_UNKNOWN')
 state['receipt']=call(['competitions','submit','rsna-knee-abnormality-detection','-k',kernel,'-v','1','-f','submission.csv','-m',state['submission_description']]);save('SUBMITTED');notify('Yeni aday Kaggle yarışmasına gönderildi: '+tag)
 return track_submission(state,statepath,call,write,notify)
