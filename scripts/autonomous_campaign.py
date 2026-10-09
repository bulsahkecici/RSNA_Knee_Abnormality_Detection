"""Bounded durable Kaggle experiment queue; receipts precede success states."""
import json
import re
import subprocess
import time
from pathlib import Path
from filelock import FileLock
ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'state/campaigns/autonomous-20261009'

def write(path,value):
 temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)

def call(args):
 for attempt in range(4):
  r=subprocess.run(['kaggle',*args],capture_output=True,text=True,timeout=600)
  if r.returncode==0:return r.stdout
  if '429' in r.stdout+r.stderr and args[:2] in [['kernels','status'],['kernels','output']] and attempt<3:
   time.sleep(60*(attempt+1));continue
  raise RuntimeError(r.stdout+r.stderr)

def quota():
 output=call(['quota']);match=re.search(r'GPU\s+[\d.]+h\s+([\d.]+)h',output)
 if not match:raise RuntimeError('GPU quota cannot be parsed; refusing launch')
 return float(match.group(1))

def notify(message):
 script='on run argv\ndisplay notification (item 1 of argv) with title "RSNA deney döngüsü"\nend run'
 subprocess.run(['/usr/bin/osascript','-e',script,message],capture_output=True,timeout=30)

def run():
 FOLDER.mkdir(parents=True,exist_ok=True)
 with FileLock(str(FOLDER/'controller.lock'),timeout=0):
  plan=json.loads((FOLDER/'plan.json').read_text());path=FOLDER/'status.json'
  state=json.loads(path.read_text()) if path.exists() else {'started_ts':time.time(),'jobs':{},'gpu_hours_observed':0}
  def save(phase):state.update(phase=phase,updated_ts=time.time());write(path,state)
  try:
   for item in plan['jobs'][:plan['max_jobs']]:
    key=item['id'];rec=state['jobs'].get(key)
    if rec and rec.get('phase')=='COMPLETE':continue
    if time.time()-state['started_ts']>plan['max_elapsed_hours']*3600:save('ELAPSED_BUDGET_REACHED');return
    if rec is None:
     remaining=quota()
     if remaining<item['reserve_gpu_hours'] or state['gpu_hours_observed']+item['reserve_gpu_hours']>plan['max_gpu_hours']:
      save('GPU_BUDGET_REACHED');notify('GPU bütçesi nedeniyle yeni iş başlatılmadı');return
     rec={'kernel':item['kernel'],'phase':'PUSH_UNKNOWN','quota_before':remaining};state['jobs'][key]=rec;save('PUSHING')
     rec['receipt']=call(['kernels','push','-p',str(ROOT/item['directory'])]);rec['phase']='RUNNING';save('TRAINING')
    elif rec['phase']=='PUSH_UNKNOWN':raise RuntimeError('Ambiguous push: reconcile before retry')
    while True:
     if time.time()-state['started_ts']>plan['max_elapsed_hours']*3600:
      save('ELAPSED_BUDGET_REACHED_REMOTE_JOB_TRACKED');return
     observed=call(['kernels','status',rec['kernel']]);rec['observed']=observed.strip();save('TRAINING')
     if 'COMPLETE' in observed:break
     if 'ERROR' in observed:raise RuntimeError('Training failed; no blind retry')
     time.sleep(60)
    out=ROOT/item['directory']/'output'
    call(['kernels','output',rec['kernel'],'-p',str(out),'--file-pattern','campaign-result.json$|metrics.json$|predictions.jsonl$|checkpoint.pt$|initialization.json$'])
    from campaign_review import review
    decision=review(out,ROOT/plan['baseline_output'])
    rec.update(phase='COMPLETE',review=decision)
    after=quota()
    if after>rec['quota_before']:
     save('QUOTA_REFRESH_REQUIRES_ACCOUNTING');notify('GPU kota dönemi değişti; bütçe hesabı incelenecek');return
    state['gpu_hours_observed']+=max(0,rec['quota_before']-after);save('EVALUATED');notify(key+' tamamlandı: '+json.dumps(decision['summary']))
   candidates=[item for item in plan['jobs'] if state['jobs'].get(item['id'],{}).get('review',{}).get('candidate_for_offline_check')]
   if candidates and plan.get('publish_reviewed_candidate',False):
    best=max(candidates,key=lambda item:state['jobs'][item['id']]['review']['summary']['weak_auc'])
    if quota()<9 or state['gpu_hours_observed']+9>plan['max_gpu_hours']:
     save('OFFLINE_BUDGET_WAIT');return
    save('PACKAGING_REVIEWED_CANDIDATE')
    from candidate_publish import publish
    publish(best,FOLDER,call,write,notify)
   save('ROUND_COMPLETE')
   notify('Deney turu tamamlandı; değerlendirme kararları kaydedildi')
  except Exception as exc:
   state['error']=str(exc)[:2000];save('NEEDS_ATTENTION');notify('Deney döngüsü hata nedeniyle durdu; kayıtları kontrol edin');raise
if __name__=='__main__':run()
