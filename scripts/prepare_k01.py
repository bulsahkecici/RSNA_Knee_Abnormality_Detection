"""Prepare K0 handoff and a numeric-only coverage audit; never starts a remote job.

Default cache pins come from the existing K5F0 build record. Supply downloaded
--cache-manifest paths to prove payload accessibility; absent local files mean
unverified coverage, never a successful cache audit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import make_colab_notebook as mcn
import make_kneenet_notebook as kn
import yaml

from rsna_knee.data.cache_build import manifest_ready
from rsna_knee.data.folds import image_training_uids, load_folds
from rsna_knee.data.soft_pool import soft_training_uids
from rsna_knee.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[1]


def prepare(*, ref='WORKTREE', manifests=()):
    config = ROOT / 'configs/experiments/P5-K0-kneenet-fold0.yaml'
    spec = yaml.safe_load(config.read_text())
    folds_path = ROOT / spec['folds']
    folds = load_folds(folds_path)
    soft = (ROOT / spec['labels']['file']).read_bytes()
    if mcn._sha(soft) != spec['labels']['sha256']:
        raise ValueError('soft-label source hash mismatch')
    legacy = mcn.train_uids(folds_path, mcn.LABELS.read_bytes(), '0')
    uids = soft_training_uids(folds, soft, '0')
    all_soft = soft_training_uids(folds, soft, 'none')
    old_build = json.loads((mcn.OUT_DIR / 'K5F0.build.json').read_text())
    parts = old_build['cache_parts']
    checks, cached, supplied_pins = [], set(), set()
    for path in map(Path, manifests):
        ready, reason = manifest_ready(path)
        digest = sha256_file(path)
        if digest not in {p['manifest_sha256'] for p in parts}:
            raise ValueError('local cache manifest differs from the pinned K0 cache')
        supplied_pins.add(digest)
        if ready:
            cached.update(json.loads(path.read_text())['studies'])
        checks.append({'path': str(path), 'verified': ready, 'reason': reason, 'sha256': digest})
    gold = {r['StudyInstanceUID'] for r in folds if r.get('gold_set')}
    weak = {r['StudyInstanceUID'] for r in folds if r['fold'] == '0' and not r.get('gold_set')}
    required = set(uids) | gold | weak
    coverage_verified = (all(c['verified'] for c in checks) and supplied_pins == {p['manifest_sha256'] for p in parts}
                         and required <= cached)
    notebooks = []
    for exp in ('K0BASE', 'K0'):
        notebooks.append(str(kn.generate(ROOT / f'configs/experiments/P5-{exp}-kneenet-fold0.yaml',
                                         cache_kernels=[p['kernel'] for p in parts],
                                         cache_manifest_sha256s=[p['manifest_sha256'] for p in parts], ref=ref)))
    result = {'kind': 'preparation_not_training', 'eligible_non_gold': len(image_training_uids(folds)),
              'all_soft_training_pool': len(all_soft), 'legacy_fold0_train': len(legacy), 'soft_fold0_train': len(uids),
              'added_fold0': len(set(uids) - set(legacy)), 'train_gold_overlap': len(set(uids) & gold),
              'cache_coverage_verified': coverage_verified, 'cache_checks': checks,
              'missing_locally_verified_required_studies': len(required - cached),
              'K5F0_result_present': (ROOT / 'artifacts/colab/p1-K5F0/output/job/metrics.json').exists(),
              'notebooks': notebooks, 'K1': 'remote real-DICOM pilot required; no cache hash invented',
              'remote_job_started': False}
    out = ROOT / 'artifacts/research/k01-preparation.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', default='WORKTREE')
    parser.add_argument('--cache-manifest', type=Path, action='append', default=[])
    args = parser.parse_args()
    print(json.dumps(prepare(ref=args.ref, manifests=args.cache_manifest), indent=2))
