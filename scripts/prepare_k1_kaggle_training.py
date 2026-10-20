"""Bind completed real volume-cache receipts and prepare one private K1 Kaggle GPU job.

Only numeric labels/code enter this upload. MRI volumes are mounted read-only
from the five authorized cache kernels, never copied to working output or Mac.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import lzma
from pathlib import Path
import yaml

import make_colab_notebook as mcn
import make_kneenet_notebook as kn

from rsna_knee.data.folds import assert_group_integrity, assert_train_uids, load_folds
from rsna_knee.data.volume_cache import VERSION
from rsna_knee.hashing import sha256_file, sha256_json

ROOT = Path(__file__).resolve().parents[1]
SLUG = 'rsna-k1-kneenet-fold0-20261015'

SETUP = r'''
import sys, subprocess, shutil, json, os
from pathlib import Path
import torch
assert torch.cuda.is_available(), 'GPU required; full K1 training cannot fall back to CPU.'
assert sys.version_info >= (3, 11)
WORK = Path('/kaggle/working')
WORK.mkdir(parents=True, exist_ok=True)
INPUT_ROOT = Path('/tmp/rsna-k1-mounted-view')
INPUT_ROOT.mkdir(parents=True, exist_ok=True)
assert shutil.disk_usage(WORK).free > 4 * 1024**3, 'At least 4 GiB working space required.'
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'timm', 'safetensors',
                'pydicom==3.0.1', 'pydantic>=2.7', 'pyyaml', 'filelock', 'scikit-learn', 'jsonschema'], check=True)
handshake = {'gpu': torch.cuda.get_device_name(0), 'vram_bytes': torch.cuda.get_device_properties(0).total_memory,
             'capability': list(torch.cuda.get_device_capability(0)), 'torch': torch.__version__,
             'cache_access': 'read_only_kernel_mounts_symlink_view_in_tmp'}
import importlib.metadata
handshake['packages'] = {n: importlib.metadata.version(n) for n in ('numpy', 'timm', 'safetensors', 'pydicom')}
(WORK / 'handshake.json').write_text(json.dumps(handshake, indent=2))
print('GPU handshake:', handshake, flush=True)
'''

MOUNTS = r'''
import hashlib
def file_sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()
expected = {p['manifest_sha256'] for p in EXPERIMENT['cache_parts']}
found = {}
for path in Path('/kaggle/input').rglob('manifest.json'):
    digest = file_sha256(path)
    if digest in expected:
        assert digest not in found, 'ambiguous mounted cache manifest'
        found[digest] = path
assert expected == set(found), 'one or more pinned cache outputs are not mounted'
part_manifests = [found[p['manifest_sha256']] for p in EXPERIMENT['cache_parts']]
(WORK / 'cache-bindings.json').write_text(json.dumps([
    {**p, 'mounted_manifest': str(m)} for p, m in zip(EXPERIMENT['cache_parts'], part_manifests, strict=True)
], indent=2))
print('Pinned mounted cache manifests:', [str(p) for p in part_manifests], flush=True)
'''

COLLECT = r'''
import zipfile
archive_path = WORK / 'colab-results.zip'  # existing review/ingest contract
files = ['campaign-result.json', 'timing.json', 'experiment.json', 'handshake.json',
         'throughput.json', 'resource-gate.json', 'cache-bindings.json', 'job/checkpoint.pt', 'job/metrics.json', 'job/predictions.jsonl']
with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as archive:
    for name in files:
        path = WORK / name
        if path.is_file():
            archive.write(path, name)
print('K1 training results:', archive_path, flush=True)
'''


def packed_code(source: str, label: str):
    raw = source.encode()
    packed = base64.b64encode(lzma.compress(raw, preset=9)).decode()
    code = ('import base64, lzma, hashlib\n'
            f'_payload = lzma.decompress(base64.b64decode({packed!r}))\n'
            f'assert hashlib.sha256(_payload).hexdigest() == {hashlib.sha256(raw).hexdigest()!r}\n'
            f'exec(compile(_payload, {label!r}, "exec"), globals())\n'
            'del _payload\n')
    return mcn._code(code)


def cache_receipts():
    plan = json.loads((ROOT / 'artifacts/research/k1-cache-execution-plan.json').read_text())
    all_uids, parts, notes = set(), [], []
    policy_hash = series_hash = None
    for job in plan['jobs']:
        out = ROOT / job['folder'] / 'output'
        status = json.loads((out / 'campaign-result.json').read_text())
        path = out / 'cache/manifest.json'
        digest = sha256_file(path)
        if status['status'] != 'ok' or status['scope'] != 'real_dicom_preprocessing_only' or status['pilot']:
            raise ValueError('cache_job_not_complete_real_full_scope')
        if status['build_info']['source_sha256'] != job['source_sha256'] or status['manifest_sha256'] != digest:
            raise ValueError('cache_receipt_identity_mismatch')
        if status['build_info']['defines']['SHARD'] != job['shard']:
            raise ValueError('cache_shard_identity_mismatch')
        m = json.loads(path.read_text())
        if m['version'] != VERSION or m['preprocess_hash'] != sha256_json(m['preprocess_config']):
            raise ValueError('cache_policy_hash_mismatch')
        if m['manifest_sha256'] != sha256_json({k: v for k, v in m.items() if k != 'manifest_sha256'}):
            raise ValueError('cache_manifest_content_hash_mismatch')
        if set(m['requested_uids']) != set(m['studies']) or m['quarantine'] or all_uids & set(m['studies']):
            raise ValueError('incomplete_or_overlapping_cache_parts')
        if policy_hash is not None and (m['preprocess_hash'] != policy_hash or m['series_csv_sha256'] != series_hash):
            raise ValueError('inconsistent_cache_parts')
        for name, code_hash in m['preprocess_config']['code_sha256'].items():
            if sha256_file(ROOT / 'src/rsna_knee/data' / name) != code_hash:
                raise ValueError('cache_preprocessing_source_drift')
        policy_hash, series_hash = m['preprocess_hash'], m['series_csv_sha256']
        all_uids.update(m['studies'])
        parts.append({'kernel': job['kernel'], 'manifest_sha256': digest})
        notes.append({'kernel': job['kernel'], 'n_studies': len(m['studies']),
                      'payload_bytes': sum(s['bytes'] for s in m['studies'].values())})
    folds = load_folds(ROOT / 'state/folds-v2-allgold.csv')
    if all_uids != {r['StudyInstanceUID'] for r in folds}:
        raise ValueError('cache_union_differs_from_complete_study_universe')
    return parts, notes, all_uids


def window_cache_receipts():
    """Prove the full v5 UID scope from real receipts; full payload QC runs on Kaggle."""
    parts = json.loads((mcn.OUT_DIR / 'K0.build.json').read_text())['cache_parts']
    universe = sorted(r['StudyInstanceUID'] for r in load_folds(ROOT / 'state/folds-v2-allgold.csv'))
    notes, cached = [], set()
    for i, part in enumerate(parts):
        folder = ROOT / 'artifacts/kaggle' / part['kernel'].split('/')[-1]
        receipt = json.loads((folder / 'output/campaign-result.json').read_text())
        uids = universe[i::len(parts)]
        if (receipt['status'] != 'ok' or not receipt['all_built_shards_verified']
                or not receipt['manifest_ready'][0] or not receipt['requested_complete']
                or receipt['n_quarantine'] or receipt['bad_shards']
                or receipt['manifest_sha256'] != part['manifest_sha256']
                or receipt['shard'] != {'index': i, 'count': len(parts)}
                or receipt['shard_uids_sha256'] != hashlib.sha256('\n'.join(uids).encode()).hexdigest()
                or receipt['n_studies'] != len(uids) or receipt['n_requested'] != len(uids)):
            raise ValueError('v5_cache_receipt_scope_or_identity_mismatch')
        cached.update(uids)
        notes.append({'kernel': part['kernel'], 'n_studies': len(uids),
                      'payload_bytes': receipt['compressed_byte_total'],
                      'coverage_proof': 'real_campaign_receipt_with_exact_UID_hash'})
    return parts, notes, cached


def prepare(ref='WORKTREE', *, config_path=None, slug=None):
    config_path = Path(config_path or ROOT / 'configs/experiments/P5-K1-kneenet-fold0.yaml').resolve()
    spec = yaml.safe_load(config_path.read_text())
    exp_id = spec['experiment_id']
    slug = slug or (SLUG if exp_id == 'K1' else f"rsna-{exp_id.lower()}-kneenet-fold{spec['outer_fold']}-20261015")
    is_volume = spec.get('cache_layout') == 'canonical_volumes'
    parts, notes, cached = cache_receipts() if is_volume else window_cache_receipts()
    out = kn.generate(config_path,
                      cache_kernels=[p['kernel'] for p in parts],
                      cache_manifest_sha256s=[p['manifest_sha256'] for p in parts], ref=ref)
    notebook = json.loads(out.read_text())
    ns = {}
    exec(''.join(notebook['cells'][1]['source']), ns)
    exp = ns['EXPERIMENT']
    folds = load_folds(ROOT / exp['folds_file'])
    assert_group_integrity(folds)
    assert_train_uids(set(exp['train_uids']), folds)
    roles = {r['StudyInstanceUID']: r for r in folds}
    if any(roles[u]['fold'] == str(exp['outer_fold']) for u in exp['train_uids']):
        raise ValueError('outer_holdout_entered_training')
    if not set(exp['train_uids'] + exp['gold_uids'] + exp['weak_uids']) <= cached:
        raise ValueError('K1 train/evaluation coverage missing')
    exp.update({'cache_access': 'mounted_symlinks', 'platform': 'kaggle',
                'platform_recipe_sha256': sha256_json({'setup': SETUP, 'mounts': MOUNTS, 'collect': COLLECT,
                                                       'packing': 'lzma_sha256_v1'}),
                'platform_generator_sha256': sha256_file(Path(__file__))})
    notebook['cells'][0] = mcn._md(f"# {exp_id} — KneeNet, outer fold {exp['outer_fold']}\n\n"
                                  f"Private Kaggle GPU training; pinned labels/seed/schedule; {exp['train']['res']} px forward "
                                  'resolution. Real cache outputs are pinned and mounted read-only. '
                                  'All payload hashes and volume QC are checked before training; gold and groups '
                                  'remain excluded. No competition submission. Results: colab-results.zip.')
    notebook['cells'][1] = packed_code('import json\nEXPERIMENT = json.loads(' + repr(json.dumps(exp, indent=1)) + ')',
                                      exp_id + '-verified-config')
    notebook['cells'][2], notebook['cells'][3], notebook['cells'][6] = (mcn._code(s) for s in (SETUP, MOUNTS, COLLECT))
    # Kaggle SaveKernel accepts <1 MB of source. Compress the numeric/code payload,
    # preserving its exact decoded bytes and the existing inner integrity checks.
    notebook['cells'][4] = packed_code(''.join(notebook['cells'][4]['source']), exp_id + '-verified-payload')
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), 'K1-Kaggle', 'exec')
    wire = json.loads(json.dumps(notebook))
    for cell in wire['cells']:
        cell['source'] = ''.join(cell['source'])
    if len(json.dumps(wire).encode()) >= 1_000_000:
        raise ValueError('Kaggle source must remain below its 1 MB API limit')
    out.write_text(json.dumps(notebook, indent=1))
    build = {k: v for k, v in exp.items() if k not in ('train_uids', 'gold_uids', 'weak_uids')}
    build.update({'n_train_uids': len(exp['train_uids']), 'n_weak_uids': len(exp['weak_uids'])})
    (mcn.OUT_DIR / f'{exp_id}.build.json').write_text(json.dumps(build, indent=2))
    folder = ROOT / 'artifacts/kaggle' / slug
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f'{exp_id}.ipynb').write_bytes(out.read_bytes())
    (folder / 'build.json').write_text(json.dumps(build | {'cache_audit': notes}, indent=2))
    metadata = {'id': 'bulsahkecici/' + slug, 'title': slug, 'code_file': f'{exp_id}.ipynb',
                'language': 'python', 'kernel_type': 'notebook', 'is_private': True,
                'enable_gpu': True, 'enable_internet': True, 'competition_sources': [], 'dataset_sources': [],
                'kernel_sources': [p['kernel'] for p in parts]}
    (folder / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2))
    audit = {'kind': 'real_cache_metadata_audit_payload_QC_repeated_remotely', 'n_cached_studies': len(cached),
             'n_train': len(exp['train_uids']), 'n_gold': len(exp['gold_uids']), 'n_weak': len(exp['weak_uids']),
             'train_gold_overlap': len(set(exp['train_uids']) & set(exp['gold_uids'])),
             'total_payload_gib': sum(n['payload_bytes'] for n in notes) / 2**30, 'parts': notes,
             'raw_mri_downloaded_to_mac': False, 'notebook': str(out), 'package': str(folder)}
    (ROOT / f'artifacts/research/{exp_id.lower()}-full-cache-audit.json').write_text(json.dumps(audit, indent=2))
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', default='WORKTREE')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--slug')
    args = parser.parse_args()
    print(json.dumps(prepare(args.ref, config_path=args.config, slug=args.slug), indent=2))
