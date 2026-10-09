"""Independent local gates for candidate promotion; no leaderboard-based fitting."""
import json
from pathlib import Path
import numpy as np
from rsna_knee.data.folds import load_folds,assert_train_uids
from rsna_knee.evaluation.metrics import per_class_auc
from rsna_knee.ontology import TARGET_COLUMNS
ROOT=Path(__file__).resolve().parents[1]

def validate_prediction_table(rows):
 table={(r['split'],r['StudyInstanceUID']):r for r in rows}
 if len(table)!=len(rows):raise ValueError('duplicate_prediction_uid')
 if len({r['StudyInstanceUID'] for r in rows})!=len(rows):raise ValueError('prediction_split_overlap')
 if not rows or any(r['split'] not in {'gold','weak'} for r in rows):raise ValueError('invalid_prediction_split')
 values=np.array([[r[t] for t in TARGET_COLUMNS] for r in rows],dtype=float)
 if not np.isfinite(values).all() or not ((values>=0)&(values<=1)).all():raise ValueError('invalid_prediction_probability')
 return table

def review(candidate,baseline):
 m=json.loads((candidate/'job/metrics.json').read_text());a=json.loads((baseline/'job/metrics.json').read_text())
 assert m['kind']==a['kind']=='real'
 assert json.loads((candidate/'campaign-result.json').read_text())['exit_reason']=='ok'
 assert set(m['train_uids'])==set(a['train_uids']);assert_train_uids(set(m['train_uids']),load_folds())
 tables=[]
 for folder in [candidate,baseline]:
  rows=[json.loads(line) for line in (folder/'job/predictions.jsonl').read_text().splitlines()]
  tables.append(validate_prediction_table(rows))
 assert set(tables[0])==set(tables[1])
 labels={r['StudyInstanceUID']:r for r in map(json.loads,(ROOT/'state/campaigns/campaign-20261003-v2/numeric-training.jsonl').read_text().splitlines())}
 recomputed=[]
 for table,metrics in zip(tables,[m,a],strict=True):
  measured={}
  for split in ['gold','weak']:
   split_keys=sorted(k for k in table if k[0]==split)
   if not split_keys:raise ValueError('empty_validation_split')
   yy=np.array([[labels[k[1]]['targets'][t] if labels[k[1]]['y_mask'][t] else np.nan for t in TARGET_COLUMNS] for k in split_keys],dtype=float)
   pp=np.array([[table[k][t] for t in TARGET_COLUMNS] for k in split_keys])
   result=per_class_auc(yy,pp)
   if result['n_defined']!=len(TARGET_COLUMNS):raise ValueError('undefined_validation_target')
   measured[split]=result['macro_auc']
   if abs(measured[split]-metrics[split+'_metrics']['macro_auc'])>1e-8:raise ValueError('reported_metrics_mismatch')
  recomputed.append(measured)
 keys=sorted(k for k in tables[0] if k[0]=='gold')
 y=np.array([[labels[k[1]]['targets'][t] for t in TARGET_COLUMNS] for k in keys]);p=[np.array([[tab[k][t] for t in TARGET_COLUMNS] for k in keys]) for tab in tables]
 assert all(np.isfinite(x).all() and (x>=0).all() and (x<=1).all() for x in p)
 rng=np.random.default_rng(2026);d=[]
 for _ in range(500):
  idx=rng.integers(0,len(y),len(y));d.append(per_class_auc(y[idx],p[0][idx])['macro_auc']-per_class_auc(y[idx],p[1][idx])['macro_auc'])
 gold=recomputed[0]['gold']-recomputed[1]['gold'];weak=recomputed[0]['weak']-recomputed[1]['weak'];low,high=np.percentile(d,[2.5,97.5])
 result={'summary':{'gold_auc':m['macro_auc'],'weak_auc':m['weak_metrics']['macro_auc'],'gold_delta':gold,'weak_delta':weak},'paired_gold_delta_ci95':[float(low),float(high)],'candidate_for_offline_check':gold>=0 and weak>=0.002 and low>=-0.03,'automatic_submission_allowed':False,'note':'Offline runtime/parity proof required; 29-study gold uncertainty is wide.'}
 (candidate/'review.json').write_text(json.dumps(result,indent=2));return result
