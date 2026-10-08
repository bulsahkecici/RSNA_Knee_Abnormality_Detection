"""Poll registered Kaggle jobs, collect real outputs and compare two seeds locally."""
import csv
import json
import subprocess
import time
from pathlib import Path
import numpy as np
from filelock import FileLock
from rsna_knee.evaluation.metrics import per_class_auc
from rsna_knee.ontology import TARGET_COLUMNS
ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'state/campaigns/rsna-ab-20261008'
KERNEL='bulsahkecici/rsna-v3-seed2047-20261009'
OUT=ROOT/'artifacts/kaggle/rsna-v3-seed2047-20261009/output'
def command(args):
 r=subprocess.run(['kaggle',*args],capture_output=True,text=True,timeout=300)
 if r.returncode:raise RuntimeError('Kaggle read failed')
 return r.stdout

def compare():
 old=ROOT/'artifacts/kaggle/rsna-v3-finetune-20261008-finetune/output/job'
 new=OUT/'job'
 a=json.loads((old/'metrics.json').read_text());b=json.loads((new/'metrics.json').read_text())
 assert a['kind']==b['kind']=='real' and set(a['train_uids'])==set(b['train_uids'])
 assert json.loads((OUT/'campaign-result.json').read_text())['exit_reason']=='ok'
 tables=[]
 for path in [old,new]:
  rows=[json.loads(x) for x in (path/'predictions.jsonl').read_text().splitlines()]
  table={(r['split'],r['StudyInstanceUID']):r for r in rows};assert len(table)==len(rows);tables.append(table)
 assert set(tables[0])==set(tables[1])
 labels={r['StudyInstanceUID']:r for r in map(json.loads,(ROOT/'state/campaigns/campaign-20261003-v2/numeric-training.jsonl').read_text().splitlines())}
 report={'kind':'real','averaging':'equal_probability_average','splits':{}}
 for split in ['gold','weak']:
  keys=sorted(k for k in tables[0] if k[0]==split)
  y=np.array([[labels[k[1]]['targets'][t] if labels[k[1]]['y_mask'][t] else np.nan for t in TARGET_COLUMNS] for k in keys],dtype=float)
  probs=[np.array([[table[k][t] for t in TARGET_COLUMNS] for k in keys]) for table in tables]
  assert all(np.isfinite(p).all() and (p>=0).all() and (p<=1).all() for p in probs)
  report['splits'][split]={'n':len(keys),'seed2026':per_class_auc(y,probs[0])['macro_auc'],'seed2047':per_class_auc(y,probs[1])['macro_auc'],'ensemble':per_class_auc(y,(probs[0]+probs[1])/2)['macro_auc']}
 (FOLDER/'ensemble-comparison.json').write_text(json.dumps(report,indent=2))
 return report

def notify(message):
 script='on run argv\ndisplay notification (item 1 of argv) with title "RSNA otomatik takip"\nend run'
 subprocess.run(['/usr/bin/osascript','-e',script,message],capture_output=True,timeout=30)

if __name__=='__main__':
 with FileLock(str(FOLDER/'training-watch.lock'),timeout=0):
  previous=None;last_notify=0
  while True:
   try:
    status=command(['kernels','status',KERNEL]).strip()
    report=None
    if 'COMPLETE' in status and not (FOLDER/'ensemble-comparison.json').exists():
     command(['kernels','output',KERNEL,'-p',str(OUT),'--file-pattern','metrics.json$|predictions.jsonl$|campaign-result.json$|checkpoint.pt$'])
     report=compare()
    record={'checked_ts':time.time(),'kernel_status':status,'ensemble_comparison_ready':(FOLDER/'ensemble-comparison.json').exists()}
    (FOLDER/'training-watch-status.json').write_text(json.dumps(record,indent=2))
    if status!=previous or time.time()-last_notify>=3600:
     message=status
     if record['ensemble_comparison_ready']:
      result=json.loads((FOLDER/'ensemble-comparison.json').read_text());message+='; gold AUC: '+str(result['splits']['gold'])
     notify(message);last_notify=time.time()
    previous=status
    print(json.dumps(record),flush=True)
   except Exception as exc:
    print(json.dumps({'error_type':type(exc).__name__}),flush=True)
   time.sleep(60)
