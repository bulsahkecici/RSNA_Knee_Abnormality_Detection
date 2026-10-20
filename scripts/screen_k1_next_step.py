"""Fixed reused-gold development screen; no weight search, independent validation or promotion."""
import csv,json,sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / 'scripts'))
import review_experiment as r
from rsna_knee.ontology import TARGET_COLUMNS
from rsna_knee.hashing import sha256_file
root=ROOT;out=root/'artifacts/research/k1-next-step-screen.json'
plan={'kind':'fixed_development_screen_plan_not_independent_validation','primary_candidate':'rank50_reader_and_equal_probability_mean_K1_K1F1','weights':'fixed 0.5 reader rank + 0.5 rank(mean(K1,K1F1)); no weight or class routing search','sensitivity':'single K1 rank50 only descriptive; no weight selection','gold':'58 reused development studies; external reader overlap unknown; never independent OOF','promotion':False}
(root/'artifacts/research/k1-next-step-screen-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
labels=r.labels_table();a=r.load_predictions(r.out_dir('K1'));b=r.load_predictions(r.out_dir('K1F1'))
path=root/'artifacts/kaggle/rsna-reader-gold-eval-20261014/output/reader-gold.csv'
with path.open(newline='') as f: reader={x['StudyInstanceUID']:np.array([float(x[t]) for t in TARGET_COLUMNS]) for x in csv.DictReader(f)}
uids=sorted(reader);assert len(uids)==58 and all(u in a and u in b and a[u][0]==b[u][0]=='gold' for u in uids)
y=np.stack([labels[u][0] for u in uids]);pr=np.stack([reader[u] for u in uids]);p0=np.stack([a[u][1] for u in uids]);p1=np.stack([b[u][1] for u in uids]);pe=(p0+p1)/2
rank=lambda p:rankdata(p,axis=0,method='average')/len(p)
ps={'reader':pr,'K1':p0,'K1F1':p1,'K1_twofold_mean':pe,'primary_rank50':(rank(pr)+rank(pe))/2,'single_K1_rank50_descriptive':(rank(pr)+rank(p0))/2}
records={}
for name,p in ps.items():
 score,detail=r.macro(y,p);records[name]={'macro_auc':score,'per_class':{t:v['auc'] for t,v in detail['per_class'].items()}}
report={'kind':'reused_gold_fixed_screen_not_independent_validation','plan':plan,'n':58,'reader_file_sha256':sha256_file(path),'scores':records,'reader_difference':{n:v['macro_auc']-records['reader']['macro_auc'] for n,v in records.items()}}
report['decision']={'primary_candidate_pass':False,'reason':'Negative point delta versus reader; no CI can satisfy the positive-point-delta gate.' if report['reader_difference']['primary_rank50']<=0 else 'Requires paired CI before any promotion','reader_only_preserved':True,'next_hypothesis':'K2R320 fixed resolution experiment'}
out.write_text(json.dumps(report,indent=2)+'\n')

print(json.dumps({n:{'auc':v['macro_auc'],'delta_reader':report['reader_difference'][n]} for n,v in records.items()},indent=2))
