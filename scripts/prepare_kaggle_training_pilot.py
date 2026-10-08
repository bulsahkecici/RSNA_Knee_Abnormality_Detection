"""Prepare a private GPU pilot using numeric labels and a verified cache output.

No report text, label-cache payloads, or credentials enter the upload package.
The remote script calls the existing hashed rsna worker protocol.
"""
from __future__ import annotations
import argparse
import base64
import io
import json
import tarfile
from pathlib import Path

from rsna_knee.data.cache_build import manifest_ready
from rsna_knee.data.folds import load_folds, assert_group_integrity, assert_train_uids
from rsna_knee.hashing import sha256_file
from rsna_knee.labels.canonical import normalize_record
from rsna_knee.runtime.train_payload import pilot_splits, assert_split_disjoint


def prepare(cache: Path, labels: Path, run_id: str, dest: Path) -> dict:
    ready, reason = manifest_ready(cache / 'manifest.json')
    if not ready:
        raise ValueError(reason)
    folds = load_folds()
    assert_group_integrity(folds)
    table = {}
    for line in labels.read_text().splitlines():
        row, error = normalize_record(json.loads(line))
        if error or row is None:
            raise ValueError(error)
        table[row['StudyInstanceUID']] = row
    studies = json.loads((cache / 'manifest.json').read_text())['studies']
    train, weak, gold = pilot_splits(folds)
    roles = {r['StudyInstanceUID']: r for r in folds}
    selected = [u for u in train if u in studies and u in table and any(table[u]['y_mask'].values())]
    selected.sort(key=lambda u: (roles[u]['is_gold'] != '1', u))
    selected = selected[:50]
    assert selected
    assert_train_uids(set(selected), folds)
    assert_split_disjoint(selected, weak, gold, folds)
    assert all(u in studies and u in table for u in gold), 'Missing gold eval coverage'
    # Re-normalizing strips every nonnumeric source envelope and evidence span.
    label_bytes = ''.join(json.dumps(r) + '\n' for r in table.values()).encode()
    fold_bytes = Path('state/folds.csv').read_bytes()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as tar:
        files = [Path('pyproject.toml')]
        files += sorted(Path('src/rsna_knee').rglob('*.py'))
        files += sorted(Path('configs').rglob('*.yaml'))
        files += [Path('configs/dinov2_vits14_allowlist.json')]
        files += sorted(Path('schemas').glob('*.json'))
        for p in files:
            tar.add(p, arcname=str(p), recursive=False)
    source = base64.b64encode(buf.getvalue()).decode()
    weights_sha = sha256_file(Path('artifacts/weights/dinov2_vits14_pretrain.pth'))
    cache_sha = sha256_file(cache / 'manifest.json')
    script = f'''import base64, hashlib, io, json, os, subprocess, sys, tarfile, urllib.request, zipfile
from pathlib import Path
WORK = Path('/kaggle/working')
SRC = Path('/tmp/rsna-pilot-source')
SRC.mkdir(parents=True, exist_ok=True)
source_bytes = base64.b64decode({source!r})
with tarfile.open(fileobj=io.BytesIO(source_bytes), mode='r:gz') as tar:
    for m in tar.getmembers():
        assert not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
    tar.extractall(SRC, filter='data')
os.chdir(SRC)
sys.path.insert(0, str(SRC/'src'))
subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', 'pydantic>=2.7', 'pyyaml', 'filelock', 'scikit-learn', 'jsonschema'], check=True)
import torch
assert torch.cuda.is_available(), 'GPU required; no CPU fallback'
torch.manual_seed(2026)
from rsna_knee.runtime.worker import handshake, run_job
from rsna_knee.runtime.jobs import build_job
from rsna_knee.runtime.transport import FileTransport
from rsna_knee.data.cache_build import manifest_ready
from rsna_knee.hashing import sha256_file
INPUT = Path('/tmp/rsna-pilot-input')
INPUT.mkdir(parents=True, exist_ok=True)
zips = list(Path('/kaggle/input').rglob('rsna_cache_pilot-20261003-021558.zip'))
assert len(zips) == 1, f'Expected one cache ZIP, found {{zips}}'
with zipfile.ZipFile(zips[0]) as archive:
    assert archive.testzip() is None
    for name in archive.namelist():
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
    archive.extractall(INPUT)
cache = INPUT/'cache/manifest.json'
assert sha256_file(cache) == {cache_sha!r}, 'Cache version changed'
assert manifest_ready(cache)[0]
(INPUT/'labels.jsonl').write_bytes(base64.b64decode({base64.b64encode(label_bytes).decode()!r}))
(INPUT/'folds.csv').write_bytes(base64.b64decode({base64.b64encode(fold_bytes).decode()!r}))
weights = INPUT/'dinov2_vits14_pretrain.pth'
urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth', weights)
assert sha256_file(weights) == {weights_sha!r}, 'Weights hash mismatch'
hs=handshake()
(WORK/'handshake.json').write_text(json.dumps(hs, indent=2))
print(json.dumps(hs), flush=True)
resources={{'config':str(SRC/'configs/default.yaml'), 'folds':str(INPUT/'folds.csv'), 'labels':str(INPUT/'labels.jsonl'), 'cache':str(cache), 'weights':str(weights)}}
job=build_job(run_id={run_id!r},stage='TRAINING',inputs={{k:sha256_file(Path(v)) for k,v in resources.items()}},dest=WORK/'job',resources=resources,train={{'profile':'pilot','epochs':2,'effective_batch':8,'microbatch':1,'seed':2026,'img_size':224}},train_uids={selected!r},input_root=INPUT,checkpoint_out='checkpoint.pt')
rec=run_job(job,FileTransport(WORK/'transport'),expected_token=job['fencing_token'])
(WORK/'result.json').write_text(rec.model_dump_json(indent=2))
print(rec.model_dump_json(indent=2), flush=True)
outputs={{str(p.relative_to(WORK)):sha256_file(p) for p in WORK.rglob('*') if p.is_file()}}
(WORK/'output_hashes.json').write_text(json.dumps(outputs,indent=2))
assert rec.exit_reason=='ok', rec.exit_reason
'''
    compile(script, 'train.py', 'exec')
    dest.mkdir(parents=True, exist_ok=True)
    (dest/'train.py').write_text(script)
    meta = {'id':'bulsahkecici/rsna-knee-training-pilot','title':'RSNA Knee Training Pilot','code_file':'train.py','language':'python','kernel_type':'script','is_private':True,'enable_gpu':True,'enable_internet':True,'competition_sources':[],'dataset_sources':[],'kernel_sources':['bulsahkecici/rsna-knee-expanded-pilot-cache']}
    (dest/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    summary = {'run_id':run_id,'train_studies':len(selected),'gold_eval_studies':len(gold),'weak_holdout_studies_in_cache':len(set(weak)&set(studies)),'label_rows':len(table),'cache_sha256':cache_sha,'weights_sha256':weights_sha,'source_sha256':hashlib_sha(buf.getvalue()),'raw_reports_uploaded':False,'leakage':'PASS'}
    (dest/'preparation.json').write_text(json.dumps(summary,indent=2))
    return summary


def hashlib_sha(data: bytes) -> str:
    import hashlib
    return hashlib.sha256(data).hexdigest()


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--run-id',required=True)
    p.add_argument('--dest',type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(prepare(a.cache,a.labels,a.run_id,a.dest),indent=2))
