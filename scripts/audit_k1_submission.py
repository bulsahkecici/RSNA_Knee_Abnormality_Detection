"""Bind a completed real K1 offline check to the guarded rsna submit command.

This is operational submission readiness, not second-fold confirmation or promotion.
No Kaggle request is made here. Download the real kernel outputs/status first.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from rsna_knee.hashing import sha256_file
from rsna_knee.ontology import TARGET_COLUMNS
from rsna_knee.submission.package import _checkpoint_errors
from rsna_knee.submission.validate import validate_submission
from rsna_knee.workflow.registry import Registry
from rsna_knee.workflow.state import Stage

ROOT = Path(__file__).resolve().parents[1]


def audit(slug: str, version: str, register: bool = False) -> dict:
    folder = ROOT / 'artifacts/kaggle' / slug
    asset = folder.with_name(slug + '-asset')
    bundle = asset / 'bundle'
    package = asset / 'upload/offline-bundle.tar.gz'
    output = folder / 'output'
    proof = output / 'offline-result.json'
    receipt = folder / f'push-receipt-v{version}.json'
    completed = json.loads(receipt.read_text())
    assert completed['returncode'] == 0 and completed['observed_status'] == 'KernelWorkerStatus.COMPLETE'
    assert completed['version_number'] == int(version)
    assert sha256_file(folder / 'entry.py') == completed['entry_sha256']
    assert sha256_file(folder / 'kernel-metadata.json') == completed['metadata_sha256']
    metadata = json.loads((folder / 'kernel-metadata.json').read_text())
    assert metadata['id'] == 'bulsahkecici/' + slug and not metadata['enable_internet']
    manifest = json.loads((bundle / 'MANIFEST.json').read_text())
    assert manifest['internet_required'] is False
    for name, digest in manifest['files'].items():
        assert sha256_file(bundle / name) == digest, f'changed asset: {name}'
    assert sha256_file(package) == json.loads((asset / 'upload-receipt.json').read_text())['package_sha256']
    result = json.loads(proof.read_text())
    assert result['identity'] == manifest['identity'] and result['status'] == 'ok' and not result['rerun']
    identity = result['identity']
    assert identity['experiment'] == 'K1' and len(identity['members']) == 1 and identity['reader'] is None
    member = identity['members'][0]
    checkpoint = bundle / 'members/K1/checkpoint.pt'
    assert sha256_file(checkpoint) == member['checkpoint_sha256']
    assert not _checkpoint_errors(checkpoint)
    metrics_path = ROOT / 'artifacts/colab/p1-K1/output/job/metrics.json'
    metrics = json.loads(metrics_path.read_text())
    assert metrics['kind'] == 'real' and metrics['applied_updates'] > 0
    assert member['checkpoint_sha256'] == metrics['oof_provenance']['checkpoint_sha256']
    assert member['parity_reference'] == 'fp32' and member['eval_windows'] == 12
    assert result['parity']['n_studies'] == len(metrics['gold_fp32']) == 58
    assert result['parity']['max_abs'] <= .02 and result['parity']['mean_abs'] <= .002
    assert result['n_test'] > 0 and result['n_fallback'] == 0
    csv_check = validate_submission(output / 'submission.csv')
    assert csv_check['ok'] and csv_check['n_rows'] == result['n_test']
    with (output / 'submission.csv').open(newline='') as handle:
        preview_uids = [r['StudyInstanceUID'] for r in csv.DictReader(handle)]
    assert len(set(preview_uids)) == len(preview_uids)
    run_path = ROOT / 'artifacts/colab/p1-K1/output/experiment.json'
    run = json.loads(run_path.read_text())
    folds = ROOT / run['folds_file']
    config = ROOT / run['config_file']
    assert sha256_file(folds) == run['folds_sha256'] and sha256_file(config) == run['config_sha256']
    assert member['training_source_sha256'] == run['source_sha256']
    assert identity['cache_manifest_sha256'] == run['cache_manifest_sha256']
    assert sha256_file(bundle / f"model-source-{run['source_sha256'][:16]}.blob") == run['source_sha256']
    assert sha256_file(bundle / 'preprocess-source.blob') == identity['preprocess_source_sha256']
    train_uids, gold_uids = set(run['train_uids']), set(run['gold_uids'])
    assert train_uids.isdisjoint(gold_uids) and gold_uids == set(metrics['gold_fp32'])
    parity_rows = [json.loads(line) for line in (bundle / 'members/K1/parity.jsonl').read_text().splitlines()]
    assert len(parity_rows) == len(gold_uids) and {r['StudyInstanceUID'] for r in parity_rows} == gold_uids
    for row in parity_rows:
        assert [row[t] for t in TARGET_COLUMNS] == metrics['gold_fp32'][row['StudyInstanceUID']]
    with folds.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    gold_groups = {r['group_id'] for r in rows if r['gold_split'] == 'eval'}
    assert train_uids <= {r['StudyInstanceUID'] for r in rows}
    train_groups = {r['group_id'] for r in rows if r['StudyInstanceUID'] in train_uids}
    assert train_groups.isdisjoint(gold_groups)
    policy_path = ROOT / 'state/submission-policy.json'
    policy = json.loads(policy_path.read_text())
    assert policy['allow_submission'] and policy['scope'] == 'K1 reviewed checkpoint offline submission 2026-10-15'
    # Conservative estimate includes fresh DICOM decode and fp32 inference; 1300 is approximate.
    seconds_per_study = max(result['parity']['seconds'] / 58, result['seconds_per_test_study'])
    projected_seconds = 600 + 2 * 1300 * seconds_per_study
    assert projected_seconds < 32400, 'conservative runtime estimate exceeds nine hours'
    evidence = [receipt, proof, output / 'submission.csv', bundle / 'MANIFEST.json',
                metrics_path, run_path, folds, config, policy_path,
                ROOT / 'artifacts/research/k1-run-verification.json',
                ROOT / 'artifacts/colab/p1-K1/output/review.json',
                ROOT / 'src/rsna_knee/submission/package.py', Path(__file__).resolve()]
    record = {'overall': 'PASS', 'kind': 'real_offline_operational_submission_audit',
              'run_id': 'K1-submit-20261015', 'kernel': metadata['id'], 'version': str(version),
              'checkpoint_sha256': sha256_file(checkpoint), 'package_sha256': sha256_file(package),
              'evidence_files': {str(p): sha256_file(p) for p in evidence},
              'parity': result['parity'], 'csv': csv_check,
              'gold_group_exclusion': True, 'runtime_1300_studies_2x_plus_600_seconds': projected_seconds,
              'independent_holdout': False, 'promotion': False, 'second_fold_confirmation': 'not_run',
              'authorization': 'Explicit user-requested exploratory K1 submission after first-fold review',
              'created_at_utc': datetime.now(timezone.utc).isoformat()}
    audit_path = folder / 'submission-audit.json'
    audit_path.write_text(json.dumps(record, indent=2) + '\n')
    if register:
        registry = Registry()
        assert registry.submission_for_kernel(metadata['id'], version) is None, 'already submitted; reconcile'
        registry.upsert_run(record['run_id'], 'k1-offline-submission', Stage.READY_TO_SUBMIT,
                            {'experiment': 'K1', 'artifacts': {'checkpoint': str(checkpoint),
                             'package_tar': str(package), 'audit': str(audit_path)},
                             'promotion': False, 'source_run': str(run_path)}, synthetic=False)
        registry.close()
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--slug', default='rsna-k1-submit-20261015')
    parser.add_argument('--version', default='1')
    parser.add_argument('--register', action='store_true')
    args = parser.parse_args()
    print(json.dumps(audit(args.slug, args.version, args.register), indent=2))
