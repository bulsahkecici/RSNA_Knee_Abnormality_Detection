"""One fixed complementarity screen on reused gold, never a weight search."""
import csv
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
import review_experiment as r
from rsna_knee.ontology import TARGET_COLUMNS
from rsna_knee.hashing import sha256_file

ROOT=Path(__file__).resolve().parents[1]

def main():
    plan={'kind':'fixed_reused_gold_screen_not_independent_validation',
          'candidate':'0.5 rank(reader) + 0.5 rank(K3EV2)',
          'no_weight_search':True,'gold_reused':True,'external_reader_training_overlap':'unknown',
          'automatic_submission':False,'reason':'Test complementary model family once at the existing fixed 50/50 recipe.'}
    folder=ROOT/'artifacts/research';(folder/'k3-reader-screen-plan.json').write_text(json.dumps(plan,indent=2))
    preds=r.load_predictions(r.out_dir('K3EV2'));labels=r.labels_table()
    file=ROOT/'artifacts/kaggle/rsna-reader-gold-eval-20261014/output/reader-gold.csv'
    with file.open(newline='') as handle:
        reader={row['StudyInstanceUID']:np.array([float(row[t]) for t in TARGET_COLUMNS]) for row in csv.DictReader(handle)}
    uids=sorted(reader);assert len(uids)==58 and all(u in preds and preds[u][0]=='gold' for u in uids)
    y=np.stack([labels[u][0] for u in uids]);pr=np.stack([reader[u] for u in uids]);pk=np.stack([preds[u][1] for u in uids])
    blend=(rankdata(pr,axis=0,method='average')+rankdata(pk,axis=0,method='average'))/(2*len(uids))
    scores={name:r.macro(y,p)[0] for name,p in [('reader',pr),('K3EV2',pk),('fixed_rank50',blend)]}
    delta=scores['fixed_rank50']-scores['reader']
    report={'plan':plan,'scores':scores,'delta_reader':delta,'n_gold':58,
            'reader_file_sha256':sha256_file(file),
            'candidate_checkpoint_sha256':json.loads((r.out_dir('K3EV2')/'verification.json').read_text())['checkpoint_sha256'],
            'decision':'reject_negative_point_delta' if delta<=0 else 'requires_independent_confirmation',
            'leaderboard_score':None,'no_submission':True}
    (folder/'k3-reader-screen.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
