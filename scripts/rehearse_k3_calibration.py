"""Execute the actual pilot stop path with synthetic DICOM volumes on CPU.

This is engineering verification only, never GPU resource or validation evidence.
Run after rehearse_kneenet_notebook.py has created its synthetic fixture.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from make_submission_kernel import notebook_cells
from rehearse_kneenet_notebook import CPU

ROOT = Path(__file__).resolve().parents[1]
CHILD = r'''
import json, sys, os, shutil, hashlib
from pathlib import Path
import torch
torch.set_num_threads(4)
args = json.loads(Path(sys.argv[1]).read_text())
src = args['cells']; ns = {'__name__': '__main__'}
exec(src[0], ns)
work = Path(args['work']);work.mkdir(parents=True, exist_ok=True)
ns.update(WORK=work, sys=sys, os=os, json=json, Path=Path, shutil=shutil, torch=torch, INPUT_ROOT=work / 'mounted')
exec(src[3], ns)
exp = ns['EXPERIMENT'];exp['train_uids'] = args['train_uids'];exp['gold_uids'] = exp['weak_uids'] = []
exp['train'].update(epochs=14, batch=2, res=64, k=2)
exp['runtime_guard'].update(warmup_micro_batches=2, measure_micro_batches=2, max_projected_training_seconds=1e9)
ns['SOFT'].write_text(Path(args['soft']).read_text())
ns['part_manifests'] = [Path(p) for p in args['parts']]
exp['cache_parts'] = [{'kernel':f'synthetic/{i}', 'manifest_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
                      for i,p in enumerate(ns['part_manifests'])]
shutil.copy(args['weights'],work / exp['weights_file'])
exec(src[4], ns)
assert not (work / 'job/metrics.json').exists()
assert not (work / 'job/predictions.jsonl').exists()
assert not (work / 'job/checkpoint.pt').exists()
p = work / 'calibration-result.json';result=json.loads(p.read_text())
assert result['kind'] == 'real_resource_calibration_not_validation' and result['applied_optimizer_steps'] > 0
assert result['n_gold'] == result['n_weak'] == 0
result['kind'] = 'synthetic_engineering_calibration_stop_test';p.write_text(json.dumps(result))
p = work / 'resource-gate.json';gate=json.loads(p.read_text());gate['kind']='synthetic_engineering_resource_gate';p.write_text(json.dumps(gate))
print(json.dumps({'kind':'synthetic_engineering_test', 'pass':True, 'no_validation_artifacts':True,
                  'applied_updates':result['applied_optimizer_steps']}))
'''


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--fixture',type=Path,required=True);args=parser.parse_args()
    spec=json.loads((args.fixture / 'run-K3EV2/spec-straight.json').read_text())
    # Explicit code-cell selection preserves the exact decoded training wrapper.
    raw=json.loads((ROOT / 'artifacts/kaggle/rsna-k3ev2-resource-pilot-20261016/K3EV2PILOT.ipynb').read_text())
    cells=[cell for cell,meta in zip(notebook_cells('K3EV2PILOT',ROOT / 'artifacts/kaggle/rsna-k3ev2-resource-pilot-20261016'),raw['cells']) if meta['cell_type']=='code']
    for old,new in CPU:cells[4]=cells[4].replace(old,new)
    work=args.fixture / 'pilot-stop-test';shutil.rmtree(work,ignore_errors=True)
    spec.update(cells=cells,work=str(work))
    target=args.fixture / 'pilot-spec.json';target.write_text(json.dumps(spec))
    child=args.fixture / 'pilot-child.py';child.write_text(CHILD)
    p=subprocess.run([sys.executable,'-I',str(child),str(target)],capture_output=True,text=True)
    if p.returncode:raise RuntimeError(p.stdout[-1500:]+'\n'+p.stderr[-3000:])
    print(p.stdout.strip().splitlines()[-1])

if __name__=='__main__':main()
