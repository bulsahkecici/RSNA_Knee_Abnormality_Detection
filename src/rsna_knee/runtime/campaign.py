"""Private Kaggle campaign bundles with numeric labels and no report text."""

from __future__ import annotations

import base64
import gzip
import io
import json
import re
import tarfile
from pathlib import Path
from typing import Any

from rsna_knee.data.folds import assert_group_integrity, assert_train_uids, load_folds
from rsna_knee.hashing import sha256_file, sha256_text
from rsna_knee.labels.canonical import normalize_record
from rsna_knee.paths import ROOT
from rsna_knee.runtime.train_payload import assert_split_disjoint, pilot_splits


def prepare_campaign_kernel(
    labels: Path, *, run_id: str, owner: str, dest: Path,
    phase: str, cache_kernel: str | None = None, cache_sha256: str | None = None,
) -> dict[str, Any]:
    if not re.fullmatch(r"[a-z0-9-]+", run_id) or not re.fullmatch(r"[A-Za-z0-9_-]+", owner):
        raise ValueError("invalid_campaign_identity")
    if phase not in {"cache", "frozen", "finetune"}:
        raise ValueError("invalid_campaign_phase")
    folds = load_folds()
    assert_group_integrity(folds)
    table = {}
    for line in labels.read_text().splitlines():
        rec, error = normalize_record(json.loads(line))
        if error or rec is None:
            raise ValueError("invalid_numeric_labels")
        uid = rec["StudyInstanceUID"]
        if uid in table:
            raise ValueError("duplicate_label_uid")
        table[uid] = rec
    train, weak, gold = pilot_splits(folds)
    selected = [u for u in train if u in table and any(table[u]["y_mask"].values())]
    assert_train_uids(set(selected), folds)
    assert_split_disjoint(selected, weak, gold, folds)
    if len(selected) < 500 or not set(gold) <= set(table):
        raise ValueError("insufficient_training_or_gold_coverage")
    observed = [u for u in table if any(table[u]["y_mask"].values())]
    requested = sorted(set(observed) | set(gold))
    numeric = "".join(json.dumps(table[u]) + "\n" for u in sorted(table))
    # File allowlist: never package metadata/train.csv, prompts, SQLite, reports,
    # raw extractor envelopes, credentials, or an entire working directory.
    files = sorted((ROOT / "src/rsna_knee").rglob("*.py"))
    files += sorted((ROOT / "configs").rglob("*.yaml"))
    files += sorted((ROOT / "schemas").glob("*.json"))
    files += [ROOT / "configs/dinov2_vits14_allowlist.json", ROOT / "state/folds.csv"]
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path in files:
            archive.add(path, arcname=str(path.relative_to(ROOT)), recursive=False)
    encoded = base64.b64encode(buffer.getvalue()).decode()
    label_data = base64.b64encode(gzip.compress(numeric.encode())).decode()
    config = {
        "run_id": run_id, "phase": phase, "selected": selected, "requested": requested,
        "labels_sha256": sha256_text(numeric), "folds_sha256": sha256_file(ROOT / "state/folds.csv"),
        "cache_sha256": cache_sha256,
        "weights_sha256": sha256_file(ROOT / "artifacts/weights/dinov2_vits14_pretrain.pth"),
    }
    header = f'''import base64, gzip, io, json, os, subprocess, sys, tarfile
from pathlib import Path
SOURCE = Path('/tmp/rsna-campaign-source')
SOURCE.mkdir(parents=True, exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode({encoded!r})), mode='r:gz') as archive:
    for member in archive.getmembers():
        assert not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
    archive.extractall(SOURCE, filter='data')
os.chdir(SOURCE)
sys.path.insert(0, str(SOURCE/'src'))
CONFIG = json.loads({json.dumps(config)!r})
WORK = Path('/kaggle/working')
LABELS = SOURCE/'labels.jsonl'
LABELS.write_bytes(gzip.decompress(base64.b64decode({label_data!r})))
subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', 'numpy', 'Pillow', 'pydicom==3.0.1', 'pylibjpeg', 'pylibjpeg-libjpeg', 'pylibjpeg-openjpeg', 'pydantic>=2.7', 'pyyaml', 'filelock', 'scikit-learn', 'jsonschema'], check=True)
from rsna_knee.hashing import sha256_file
assert sha256_file(LABELS) == CONFIG['labels_sha256']
assert sha256_file(SOURCE/'state/folds.csv') == CONFIG['folds_sha256']
'''
    if phase == "cache":
        body = '''from rsna_knee.data.metadata import discover_input_root
from rsna_knee.data.cache_build import build_cache, manifest_ready
root = discover_input_root()
assert root is not None, 'Competition input not mounted'
dicom_root = root/'train_series'
assert dicom_root.is_dir(), 'Missing train_series'
manifest = build_cache(series_csv=root/'train_series.csv', dicom_root=dicom_root,
    dest=WORK/'cache', study_ids=CONFIG['requested'], size=224, n_centers=3, resume=True)
assert manifest_ready(WORK/'cache/manifest.json')[0]
assert not manifest['quarantine'], 'Strict DICOM quarantine requires review'
assert set(manifest['studies']) == set(CONFIG['requested'])
summary = {'kind':'real', 'phase':'cache', 'n_studies':manifest['n_studies'],
    'n_quarantine':manifest['n_quarantine'], 'manifest_sha256':sha256_file(WORK/'cache/manifest.json'),
    'raw_reports_uploaded':False}
(WORK/'campaign-result.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary), flush=True)
'''
    else:
        if not cache_kernel or not cache_sha256:
            raise ValueError("verified_cache_required")
        body = '''import torch, urllib.request
from rsna_knee.runtime.worker import handshake, run_job
from rsna_knee.runtime.jobs import build_job
from rsna_knee.runtime.transport import FileTransport
assert torch.cuda.is_available(), 'GPU required'
paths = [p for p in Path('/kaggle/input').rglob('cache/manifest.json')
    if sha256_file(p) == CONFIG['cache_sha256']]
assert len(paths) == 1, 'Expected one verified cache'
weights = SOURCE/'weights.pth'
urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth', weights)
assert sha256_file(weights) == CONFIG['weights_sha256']
hs = handshake()
(WORK/'handshake.json').write_text(json.dumps(hs, indent=2))
resources = {'config':str(SOURCE/'configs/default.yaml'), 'folds':str(SOURCE/'state/folds.csv'),
    'labels':str(LABELS), 'cache':str(paths[0]), 'weights':str(weights)}
train = {'profile':'campaign', 'epochs':8, 'effective_batch':8, 'microbatch':1,
    'seed':2026, 'img_size':224, 'phase':CONFIG['phase']}
if CONFIG['phase'] == 'finetune':
    train['unfreeze_after_steps'] = (len(CONFIG['selected']) + 7)//8
job = build_job(run_id=CONFIG['run_id']+'-'+CONFIG['phase'], stage='TRAINING',
    inputs={k:sha256_file(Path(v)) for k,v in resources.items()}, dest=WORK/'job',
    resources=resources, train=train, train_uids=CONFIG['selected'], checkpoint_out='checkpoint.pt')
torch.manual_seed(2026)
result = run_job(job, FileTransport(WORK/'transport'), expected_token=job['fencing_token'])
(WORK/'campaign-result.json').write_text(result.model_dump_json(indent=2))
print(result.model_dump_json(indent=2), flush=True)
assert result.exit_reason == 'ok', result.exit_reason
'''
    script = header + body
    compile(script, "campaign_entry.py", "exec")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "campaign_entry.py").write_text(script)
    metadata = {
        "id": f"{owner}/{run_id}-{phase}", "title": f"{run_id}-{phase}",
        "code_file": "campaign_entry.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": phase != "cache", "enable_internet": True,
        "competition_sources": ["rsna-knee-abnormality-detection"] if phase == "cache" else [],
        "dataset_sources": [], "kernel_sources": [cache_kernel] if cache_kernel else [],
    }
    (dest / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
    summary = {"kernel": metadata["id"], "phase": phase, "train_studies": len(selected),
        "cache_studies": len(requested), "gold_eval_studies": len(gold),
        "labels_sha256": config["labels_sha256"], "source_sha256": sha256_text(script),
        "raw_reports_uploaded": False, "leakage": "PASS"}
    (dest / "preparation.json").write_text(json.dumps(summary, indent=2))
    return summary
