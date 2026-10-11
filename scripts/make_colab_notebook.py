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
  (repeat --cache-kernel/--cache-manifest-sha256 for a sharded cache; parts are merged in Colab)
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
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'kaggle>=2.2', 'pydicom==3.0.1', 'pydantic>=2.7',
                'pyyaml', 'filelock', 'scikit-learn', 'jsonschema', 'timm'], check=True)
from google.colab import drive  # pyright: ignore[reportMissingImports]  (exists only on Colab)
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
import hashlib, time
INPUT_ROOT = Path('/content/rsna-input')
def download_part(kernel, part_dir, log_path, attempts=5, stall_seconds=900):
    # The CLI fetches every page itself, one file at a time with no timeout: watch for stalls and retry.
    # A retry skips files already complete on disk, so interrupting this cell and re-running also resumes.
    for attempt in range(1, attempts + 1):
        cmd = [shutil.which('kaggle'), 'kernels', 'output', kernel, '-p', str(part_dir), '--page-size', '200']
        with open(log_path, 'w') as log:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
        # Progress is new bytes on disk or new CLI log lines: a resumed run spends a long time only skipping files.
        progress = lambda: sum(f.stat().st_size for f in part_dir.rglob('*') if f.is_file()) + log_path.stat().st_size
        last_progress, last_growth, started, stalled = progress(), time.time(), time.time(), False
        while proc.poll() is None:
            time.sleep(60)
            now = progress()
            if now > last_progress: last_progress, last_growth = now, time.time()
            log_text = log_path.read_text()
            on_disk = sum(f.stat().st_size for f in part_dir.rglob('*') if f.is_file())
            print(f'  {part_dir.name}: {on_disk / 2**30:.1f} GiB on disk, {log_text.count("Output file downloaded")} downloaded '
                  f'+ {log_text.count("Skipping")} already present this attempt, {(time.time() - started) / 60:.0f} min', flush=True)
            if time.time() - last_growth > stall_seconds:
                proc.kill(); proc.wait(); stalled = True
                print(f'  no progress for {stall_seconds // 60} min, restarting (resume skips finished files)', flush=True)
        text = redact(log_path.read_text())
        if proc.returncode == 0:
            assert 'Next page token' not in text, 'CLI stopped after one page; expected it to fetch every page'
            return
        error = 'stalled, killed' if stalled else '\n'.join(l for l in text.splitlines() if 'Skipping' not in l)[-1500:]
        print(f'  attempt {attempt}/{attempts} failed (exit {proc.returncode}):', error, flush=True)
        time.sleep(30 * attempt)
    raise RuntimeError(f'Kaggle download failed after {attempts} attempts:\n' + error)
import tarfile
DRIVE_CACHE = Path('/content/drive/MyDrive/RSNA/cache')  # one tar per verified cache part, reused by later runtimes
def manifest_digest(part_dir):
    found = list(part_dir.rglob('cache/manifest.json'))
    assert len(found) == 1, f'expected one cache manifest in {part_dir}, found {len(found)}'
    return found[0], hashlib.sha256(found[0].read_bytes()).hexdigest()
def restore_from_drive(archive, part_dir, expected):
    print('Restoring', archive.name, 'from Drive', flush=True)
    started = time.time()
    try:
        with tarfile.open(archive) as tar:
            tar.extractall(INPUT_ROOT, filter='data')
        if manifest_digest(part_dir)[1] == expected:
            print(f'  restored in {(time.time() - started) / 60:.1f} min', flush=True)
            return True
        print('  Drive copy has a different manifest', flush=True)
    except (OSError, tarfile.TarError, AssertionError, TypeError) as exc:
        print('  Drive copy unusable:', exc, flush=True)
    shutil.rmtree(part_dir, ignore_errors=True)
    archive.unlink(missing_ok=True)  # re-saved from the verified Kaggle download below
    print('  removed it; falling back to the Kaggle download', flush=True)
    return False
def save_to_drive(part_dir, archive):
    size = sum(f.stat().st_size for f in part_dir.rglob('*') if f.is_file())
    archive.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(archive.parent).free
    if free < size * 1.05:
        print(f'  not saved to Drive: needs {size / 2**30:.1f} GiB, {free / 2**30:.1f} GiB free', flush=True)
        return
    partial = archive.with_name(archive.name + '.partial')
    print(f'Saving {size / 2**30:.1f} GiB to Drive as {archive.name} so later runtimes skip Kaggle', flush=True)
    started = time.time()
    try:
        with tarfile.open(partial, 'w') as tar:  # .npz is already compressed
            tar.add(part_dir, arcname=part_dir.name)
        partial.rename(archive)
        print(f'  saved in {(time.time() - started) / 60:.1f} min', flush=True)
    except OSError as exc:
        partial.unlink(missing_ok=True)
        print('  not saved to Drive:', exc, flush=True)
part_manifests = []
for part in EXPERIMENT['cache_parts']:  # one or more Kaggle cache outputs built with one policy
    part_dir = INPUT_ROOT / part['kernel'].split('/')[-1]
    complete = INPUT_ROOT / (part_dir.name + '.download-complete')  # manifest.json can land before the shards
    archive = DRIVE_CACHE / f"{part_dir.name}-{part['manifest_sha256'][:12]}.tar"
    if not complete.is_file() and archive.is_file() and restore_from_drive(archive, part_dir, part['manifest_sha256']):
        complete.touch()
    if not complete.is_file():
        part_dir.mkdir(parents=True, exist_ok=True)
        print('Downloading', part['kernel'], flush=True)
        download_part(part['kernel'], part_dir, INPUT_ROOT / (part_dir.name + '.log'))
        complete.touch()
    manifest, digest = manifest_digest(part_dir)
    assert digest == part['manifest_sha256'], f'cache manifest mismatch for {part["kernel"]}: {digest}'
    if not archive.is_file():  # only a part whose manifest matched is saved; shards are re-hashed before training
        save_to_drive(part_dir, archive)
    part_manifests.append(manifest)
print('Cache part manifests verified:', [str(p) for p in part_manifests])
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
# rsna_knee finds the repository root (configs/, weight allowlist) from the working directory.
os.chdir(SOURCE)
os.environ['RSNA_DINOV2_ALLOWLIST'] = str(SOURCE / 'configs/dinov2_vits14_allowlist.json')
assert EXPERIMENT['weights_sha256'] in json.loads((SOURCE / 'configs/dinov2_vits14_allowlist.json').read_text())['sha256']
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
if len(part_manifests) == 1:
    manifest_path = part_manifests[0]
else:  # move shard parts into one directory; manifest_ready below re-verifies every shard
    from rsna_knee.data.cache_build import merge_cache_parts
    merged_dir = INPUT_ROOT / ('merged-' + EXPERIMENT['cache_manifest_sha256'][:12]) / 'cache'
    if not (merged_dir / 'manifest.json').is_file():
        merge_cache_parts([m.parent for m in part_manifests], merged_dir)
    manifest_path = merged_dir / 'manifest.json'
EXPERIMENT['merged_manifest_sha256'] = sha256_file(manifest_path)
(WORK / 'experiment.json').write_text(json.dumps(EXPERIMENT, indent=1))
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


def generate(config_path: Path, *, cache_kernels: list[str], cache_manifest_sha256s: list[str], ref: str,
             outer_fold: str) -> Path:
    if not cache_kernels or len(cache_kernels) != len(cache_manifest_sha256s):
        raise SystemExit("give one manifest sha256 per cache kernel")
    parts = [{"kernel": k, "manifest_sha256": s} for k, s in zip(cache_kernels, cache_manifest_sha256s, strict=True)]
    # One id for the whole cache: the part manifest itself, or a hash over the ordered part hashes.
    cache_id = parts[0]["manifest_sha256"] if len(parts) == 1 else _sha(",".join(cache_manifest_sha256s).encode())
    spec = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    exp_id = spec.get("experiment_id") or spec["id"]
    errors = validate_arm(spec, load_arms())
    if errors:
        raise SystemExit(f"{exp_id} fails experiment validation: {errors}")
    train = job_train_spec(spec)
    train["resume"] = True  # checkpoint lives on Drive, so Run All after a disconnect resumes; not run identity
    soft = b""
    if train.get("label_source", "hard_masked") != "hard_masked":
        soft_path = ROOT / spec["labels"]["file"]
        soft = soft_path.read_bytes()
        if _sha(soft) != spec["labels"]["sha256"]:
            raise SystemExit("soft label table does not match the sha256 pinned in the experiment config")
    folds_path = ROOT / spec.get("folds", "state/folds-v2-allgold.csv")
    labels, folds = LABELS.read_bytes(), folds_path.read_bytes()
    labels_gz = gzip.compress(labels, mtime=0)
    archive, names = source_archive(ref, extra=("configs", "pyproject.toml"))
    if "configs/dinov2_vits14_allowlist.json" not in names:
        raise SystemExit("source archive lacks the DINOv2 weight allowlist")
    if train.get("encoder") != "dinov2_vits14":  # the notebook downloads and pins only the official DINOv2 file
        raise SystemExit(f"encoder {train.get('encoder')!r}: only dinov2_vits14 weights are wired into the notebook")
    uids = train_uids(folds_path, labels, outer_fold)
    weights_sha = json.loads((ROOT / "configs/dinov2_vits14_allowlist.json").read_text())["sha256"][0]
    experiment = {
        "id": exp_id, "hypothesis": spec.get("hypothesis"), "config_file": str(config_path.relative_to(ROOT)),
        "config_sha256": _sha(config_path.read_bytes()), "train": train, "outer_fold": outer_fold,
        "train_uids": uids, "train_uids_sha256": _sha("\n".join(uids).encode()),
        "cache_parts": parts, "cache_manifest_sha256": cache_id,
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
        # JSON literals (null/true/false) are not Python: decode the pinned JSON at run time.
        _code("import json\nEXPERIMENT = json.loads(" + repr(json.dumps(experiment, indent=1)) + ")"),
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
    namespace: dict = {}
    exec("".join(cells[1]["source"]), namespace)  # the config cell must run, not only parse
    if namespace["EXPERIMENT"] != experiment:
        raise SystemExit("EXPERIMENT cell does not round-trip")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{exp_id}.ipynb"
    out.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    (OUT_DIR / f"{exp_id}.build.json").write_text(json.dumps({k: v for k, v in experiment.items() if k != "train_uids"}
                                                             | {"n_train_uids": len(uids)}, indent=2), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--cache-kernel", required=True, action="append", help="repeat for sharded caches")
    parser.add_argument("--cache-manifest-sha256", required=True, action="append", help="one per --cache-kernel, same order")
    parser.add_argument("--ref", default="HEAD")
    parser.add_argument("--outer-fold", default="0")
    args = parser.parse_args()
    out = generate(args.config.resolve(), cache_kernels=args.cache_kernel,
                   cache_manifest_sha256s=args.cache_manifest_sha256, ref=args.ref, outer_fold=args.outer_fold)
    print(out)


if __name__ == "__main__":
    main()
