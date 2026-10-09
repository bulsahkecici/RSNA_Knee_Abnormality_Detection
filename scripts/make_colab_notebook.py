"""Generate one self-contained Colab training notebook per experiment YAML.

Replaces hand-built notebooks with stacked base64 patches. Everything the run
needs is pinned by hash: the committed ``src/rsna_knee`` archive (git ref), the
numeric label table, the fold manifest, the Kaggle cache output (manifest
sha256) and the official DINOv2 weights (allowlisted sha256). The ``train:``
mapping of the experiment YAML is passed to the worker unchanged, so new
training options need no generator change.

The notebook embeds private numeric labels: outputs go under notebooks/generated/,
which is gitignored. It never submits to Kaggle and reads no raw reports.

Usage:
  python scripts/make_colab_notebook.py configs/experiments/D1.yaml \
      --cache-kernel bulsahkecici/rsna-v4-dense12-cache-20261009 --cache-manifest-sha256 <sha>
"""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from make_kaggle_kernel import git_identity, source_archive  # noqa: E402

from rsna_knee.data.folds import image_training_uids, load_folds  # noqa: E402
from rsna_knee.training.experiments import job_train_spec, load_arms, validate_arm  # noqa: E402

LABELS = ROOT / "state/campaigns/campaign-20261003-v2/numeric-training.jsonl"
WEIGHTS_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"
OUT_DIR = ROOT / "notebooks/generated"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def train_uids(folds_path: Path, labels: bytes, outer_fold: str) -> list[str]:
    """Image-trainable studies outside the outer holdout fold with at least one observed target."""
    folds = load_folds(folds_path)
    observed = {row["StudyInstanceUID"] for row in map(json.loads, labels.decode().splitlines())
                if row.get("source") != "gold" and any(row["y_mask"].values())}
    fold_of = {r["StudyInstanceUID"]: r["fold"] for r in folds}
    return sorted(u for u in image_training_uids(folds) if fold_of[u] != outer_fold and u in observed)


def _code(source: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": source.strip("\n").splitlines(keepends=True)}


def _md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip("\n").splitlines(keepends=True)}


SETUP = r'''
import sys, subprocess, shutil, json, os
from pathlib import Path
import torch
assert torch.cuda.is_available(), 'Select a Colab GPU runtime first.'
assert sys.version_info >= (3, 11), 'Python 3.11+ required.'
assert shutil.disk_usage('/content').free > 40 * 1024**3, 'At least 40 GB free disk required for the cache.'
print('GPU:', torch.cuda.get_device_name(0), '| PyTorch:', torch.__version__)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'kaggle', 'pydicom==3.0.1', 'pydantic>=2.7',
                'pyyaml', 'filelock', 'scikit-learn', 'jsonschema', 'timm'], check=True)
from google.colab import drive
drive.mount('/content/drive')
WORK = Path('/content/drive/MyDrive/RSNA/colab') / EXPERIMENT['id']
WORK.mkdir(parents=True, exist_ok=True)
print('Durable output (resume-safe):', WORK)
'''

AUTH_AND_CACHE = r'''
import getpass, re
if not (os.environ.get('KAGGLE_API_TOKEN') or os.environ.get('KAGGLE_KEY') or Path.home().joinpath('.kaggle/kaggle.json').is_file()):
    method = input('Kaggle access: 1 = API token, 2 = kaggle.json contents: ').strip()
    if method == '1':
        os.environ['KAGGLE_API_TOKEN'] = getpass.getpass('Kaggle API token (hidden): ').strip()
    elif method == '2':
        credentials = json.loads(getpass.getpass('Paste kaggle.json contents (hidden): '))
        os.environ['KAGGLE_USERNAME'], os.environ['KAGGLE_KEY'] = credentials['username'], credentials['key']
        del credentials
    else:
        raise ValueError('Choose 1 or 2.')
def redact(text):
    for name in ('KAGGLE_API_TOKEN', 'KAGGLE_KEY'):
        if os.environ.get(name): text = text.replace(os.environ[name], '[REDACTED]')
    return re.sub(r'KGAT_[A-Za-z0-9_-]+', '[REDACTED]', text)
CACHE_DIR = Path('/content/rsna-input') / EXPERIMENT['cache_kernel'].split('/')[-1]
manifest_path = CACHE_DIR / 'cache/manifest.json'
if not manifest_path.is_file():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    token, seen = None, set()
    while True:  # kernel output is paginated
        cmd = [shutil.which('kaggle'), 'kernels', 'output', EXPERIMENT['cache_kernel'], '-p', str(CACHE_DIR), '--page-size', '200']
        if token: cmd += ['--page-token', token]
        done = subprocess.run(cmd, capture_output=True, text=True)
        if done.returncode: raise RuntimeError('Kaggle download failed:\n' + redact(done.stderr + done.stdout)[-2000:])
        nxt = re.search(r'Next page token:\s*(\S+)', done.stdout)
        if not nxt: break
        token = nxt.group(1); assert token not in seen, 'pagination loop'; seen.add(token)
    found = list(CACHE_DIR.rglob('cache/manifest.json'))
    assert len(found) == 1, f'expected one cache manifest, found {len(found)}'
    manifest_path = found[0]
import hashlib
digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
assert digest == EXPERIMENT['cache_manifest_sha256'], f'cache manifest mismatch: {digest}'
print('Cache manifest verified:', manifest_path)
'''

PAYLOAD = r'''
import base64, gzip, io, tarfile, hashlib
def unpack(b64, sha, gz=False):
    data = base64.b64decode(b64)
    assert hashlib.sha256(data).hexdigest() == sha, 'embedded payload integrity failure'
    return gzip.decompress(data) if gz else data
SOURCE = Path('/tmp/rsna-src-' + EXPERIMENT['source_sha256'][:16])
with tarfile.open(fileobj=io.BytesIO(unpack(SOURCE_B64, EXPERIMENT['source_sha256'])), mode='r:gz') as archive:
    archive.extractall(SOURCE, filter='data')
sys.path.insert(0, str(SOURCE / 'src'))
LABELS = WORK / 'labels.jsonl'; LABELS.write_bytes(unpack(LABELS_B64, EXPERIMENT['labels_gz_sha256'], gz=True))
FOLDS = WORK / 'folds.csv'; FOLDS.write_bytes(unpack(FOLDS_B64, EXPERIMENT['folds_sha256']))
SOFT = None
if EXPERIMENT['soft_labels_sha256']:
    SOFT = WORK / 'soft-labels.jsonl'; SOFT.write_bytes(unpack(SOFT_B64, EXPERIMENT['soft_labels_sha256']))
from rsna_knee.hashing import sha256_file
assert sha256_file(LABELS) == EXPERIMENT['labels_sha256'] and sha256_file(FOLDS) == EXPERIMENT['folds_sha256']
print('Source, labels and folds verified.')
'''

TRAIN = r'''
import urllib.request, time
from rsna_knee.data.cache_build import manifest_ready
from rsna_knee.runtime.jobs import build_job
from rsna_knee.runtime.transport import FileTransport
from rsna_knee.runtime.worker import handshake, run_job
t0 = time.time()
ready, reason = manifest_ready(manifest_path)  # re-hashes every shard, checks masks and shapes
assert ready, reason
print(f'Cache verified in {time.time() - t0:.0f}s:', reason, flush=True)
weights = WORK / 'dinov2_vits14_pretrain.pth'
if not weights.is_file() or sha256_file(weights) != EXPERIMENT['weights_sha256']:
    urllib.request.urlretrieve(EXPERIMENT['weights_url'], weights)
assert sha256_file(weights) == EXPERIMENT['weights_sha256'], 'official weights hash mismatch'
(WORK / 'handshake.json').write_text(json.dumps(handshake(), indent=2))
resources = {'config': str(SOURCE / 'configs/default.yaml') if (SOURCE / 'configs/default.yaml').is_file() else None,
             'folds': str(FOLDS), 'labels': str(LABELS), 'cache': str(manifest_path), 'weights': str(weights)}
if SOFT is not None:
    resources['soft_labels'] = str(SOFT)
resources = {k: v for k, v in resources.items() if v}
job = build_job(run_id=EXPERIMENT['id'], stage='TRAINING', inputs={k: sha256_file(Path(v)) for k, v in resources.items()},
                dest=WORK / 'job', resources=resources, train=EXPERIMENT['train'], train_uids=EXPERIMENT['train_uids'],
                checkpoint_out='checkpoint.pt')
started = time.time()
torch.manual_seed(int(EXPERIMENT['train'].get('seed', 2026)))
result = run_job(job, FileTransport(WORK / 'transport'), expected_token=job['fencing_token'])
(WORK / 'campaign-result.json').write_text(result.model_dump_json(indent=2))
(WORK / 'timing.json').write_text(json.dumps({'run_job_wall_seconds': round(time.time() - started, 1),
    'gpu': torch.cuda.get_device_name(0), 'peak_cuda_bytes': torch.cuda.max_memory_allocated()}, indent=2))
print('Training status:', result.exit_reason, flush=True)
assert result.exit_reason == 'ok', result.exit_reason
'''

COLLECT = r'''
import zipfile
metrics = json.loads((WORK / 'job/metrics.json').read_text())
print({k: v for k, v in metrics.items() if 'auc' in k and not isinstance(v, (dict, list))})
archive_path = WORK / 'colab-results.zip'
with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as archive:
    for rel in ['campaign-result.json', 'handshake.json', 'timing.json', 'experiment.json', 'job/checkpoint.pt',
                'job/metrics.json', 'job/predictions.jsonl', 'job/job.json', 'job/input_manifest.json']:
        if (WORK / rel).is_file(): archive.write(WORK / rel, rel)
print('Result archive (download this one file):', archive_path)
'''


def generate(config_path: Path, *, cache_kernel: str, cache_manifest_sha256: str, ref: str, outer_fold: str) -> Path:
    spec = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    exp_id = spec.get("experiment_id") or spec["id"]
    errors = validate_arm(spec, load_arms())
    if errors:
        raise SystemExit(f"{exp_id} fails experiment validation: {errors}")
    train = job_train_spec(spec)
    soft = b""
    if train.get("label_source", "hard_masked") != "hard_masked":
        soft_path = ROOT / spec["labels"]["file"]
        soft = soft_path.read_bytes()
        if _sha(soft) != spec["labels"]["sha256"]:
            raise SystemExit("soft label table does not match the sha256 pinned in the experiment config")
    folds_path = ROOT / spec.get("folds", "state/folds-v2-allgold.csv")
    labels, folds = LABELS.read_bytes(), folds_path.read_bytes()
    labels_gz = gzip.compress(labels, mtime=0)
    archive, names = source_archive(ref)
    uids = train_uids(folds_path, labels, outer_fold)
    weights_sha = json.loads((ROOT / "configs/dinov2_vits14_allowlist.json").read_text())["sha256"][0]
    experiment = {
        "id": exp_id, "hypothesis": spec.get("hypothesis"), "config_file": str(config_path.relative_to(ROOT)),
        "config_sha256": _sha(config_path.read_bytes()), "train": train, "outer_fold": outer_fold,
        "train_uids": uids, "train_uids_sha256": _sha("\n".join(uids).encode()),
        "cache_kernel": cache_kernel, "cache_manifest_sha256": cache_manifest_sha256,
        "labels_sha256": _sha(labels), "labels_gz_sha256": _sha(labels_gz),
        "folds_file": str(folds_path.relative_to(ROOT)), "folds_sha256": _sha(folds),
        "source_sha256": _sha(archive), "n_source_files": len(names), **git_identity(ref),
        "weights_url": WEIGHTS_URL, "weights_sha256": weights_sha, "submission": False,
        "soft_labels_sha256": _sha(soft) if soft else None,
    }
    header = (f"# RSNA {exp_id}\n\n**Hypothesis:** {spec.get('hypothesis', '')}\n\n"
              f"Controls: {spec.get('controls', 'see config')}. Outer holdout fold {outer_fold}; all 58 gold "
              "studies excluded from training (folds v2). Results are local validation, never a Kaggle score. "
              "This notebook does not submit.\n\nSelect a Colab GPU runtime (VS Code: Select Kernel → Colab), then "
              "Run All. Outputs and checkpoints live on Drive, so re-running after a disconnect resumes.")
    cells = [
        _md(header),
        _code("EXPERIMENT = " + json.dumps(experiment, indent=1)),
        _code(SETUP),
        _code(AUTH_AND_CACHE),
        _code(f"SOURCE_B64 = {base64.b64encode(archive).decode()!r}\n"
              f"LABELS_B64 = {base64.b64encode(labels_gz).decode()!r}\n"
              f"FOLDS_B64 = {base64.b64encode(folds).decode()!r}\n"
              f"SOFT_B64 = {base64.b64encode(soft).decode()!r}\n" + PAYLOAD
              + "\n(WORK / 'experiment.json').write_text(json.dumps(EXPERIMENT, indent=1))\n"),
        _code(TRAIN),
        _code(COLLECT),
    ]
    notebook = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
                "metadata": {"accelerator": "GPU", "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{exp_id}.ipynb"
    out.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    (OUT_DIR / f"{exp_id}.build.json").write_text(json.dumps({k: v for k, v in experiment.items() if k != "train_uids"}
                                                             | {"n_train_uids": len(uids)}, indent=2), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--cache-kernel", required=True)
    parser.add_argument("--cache-manifest-sha256", required=True)
    parser.add_argument("--ref", default="HEAD")
    parser.add_argument("--outer-fold", default="0")
    args = parser.parse_args()
    out = generate(args.config.resolve(), cache_kernel=args.cache_kernel, cache_manifest_sha256=args.cache_manifest_sha256,
                   ref=args.ref, outer_fold=args.outer_fold)
    print(out)


if __name__ == "__main__":
    main()
