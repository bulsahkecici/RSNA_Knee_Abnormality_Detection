import subprocess,json,time,csv,math
from pathlib import Path
root=Path.cwd();folder=root/'state/campaigns/rsna-ab-20261008';statepath=folder/'final-submission.json';kernel='bulsahkecici/rsna-v3-final-inference-20261009'
def save(s):
 p=statepath.with_suffix('.tmp');p.write_text(json.dumps(s,indent=2));p.replace(statepath)
def call(args):
 r=subprocess.run(['kaggle',*args],capture_output=True,text=True,timeout=300)
 if r.returncode:raise RuntimeError(r.stdout+r.stderr)
 return r.stdout
s={'phase':'WAITING_FOR_KERNEL','kernel':kernel,'version':1,'submitted':False}
if statepath.exists():
 old=json.loads(statepath.read_text())
 if old.get('phase') in ['SUBMIT_REQUEST_UNKNOWN','SUBMITTED']:raise RuntimeError('Reconcile prior request; refusing duplicate')
save(s)
try:
 while True:
  observed=call(['kernels','status',kernel]);print(observed,flush=True)
  if 'COMPLETE' in observed:break
  if 'ERROR' in observed:raise RuntimeError('Final notebook failed')
  time.sleep(30)
 dest=root/'artifacts/kaggle/v3-final-inference/output';call(['kernels','output',kernel,'-p',str(dest),'--file-pattern','submission.csv$|csv-validation.json$|offline-test-result.json$'])
 validation=json.loads((dest/'csv-validation.json').read_text())['submission.csv'];assert all(validation[k] for k in ['finite_probabilities','unique_uids','columns_match'])
 with (dest/'submission.csv').open() as h:
  reader=csv.DictReader(h);rows=list(reader);cols=reader.fieldnames
 assert len(rows)==validation['rows'] and len(rows)>0
 assert all(math.isfinite(float(r[k])) and 0<=float(r[k])<=1 for r in rows for k in cols if k!='StudyInstanceUID')
 s.update(phase='SUBMIT_REQUEST_UNKNOWN',validation=validation);save(s)
 receipt=call(['competitions','submit','rsna-knee-abnormality-detection','-k',kernel,'-v','1','-f','submission.csv','-m','Local Qwen labels, v3 DINOv2 finetuned, verified offline inference'])
 s.update(phase='SUBMITTED',submitted=True,receipt=receipt,submitted_ts=time.time());save(s);print(receipt,flush=True)
except Exception as e:
 s.update(error=str(e)[:1500]);save(s);raise
