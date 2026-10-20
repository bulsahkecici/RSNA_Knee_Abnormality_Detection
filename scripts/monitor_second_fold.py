"""Monitor the two authorized Kaggle trainings, ingest exact outputs and review the pair.

Does not start/restart training, submit, blend or change final selections. Polls
only the two receipt-bound kernels. --once performs one pass; default watches up
to eight hours. Only numeric provenance, predictions and checkpoints are fetched.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import review_experiment as review
from make_submission_kernel import notebook_blob
from rsna_knee.hashing import sha256_file
from rsna_knee.ontology import TARGET_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'artifacts/research/second-fold-campaign.json'
KAGGLE = ROOT / '.venv/bin/kaggle'
FILES = r'^(campaign-result\.json|timing\.json|experiment\.json|handshake\.json|throughput\.json|cache-bindings\.json|job/(checkpoint\.pt|metrics\.json|predictions\.jsonl))$'


def now():
    return datetime.now(timezone.utc).isoformat()


def save(plan):
    plan['last_checked_at_utc'] = now()
    tmp = PLAN.with_suffix('.json.tmp');tmp.write_text(json.dumps(plan, indent=2) + '\n');tmp.replace(PLAN)


def cli(args, timeout=120):
    p = subprocess.run([str(KAGGLE), *args], capture_output=True, text=True, cwd=ROOT, timeout=timeout)
    if p.returncode:
        raise RuntimeError((p.stderr or p.stdout)[-2000:])
    return p.stdout


def verify_result(job, *, build_dir=None):
    exp_id = job['experiment']
    folder = ROOT / f'artifacts/colab/p1-{exp_id}/output'
    metrics, _ = review.verify(exp_id, folder, build_dir=build_dir)
    exp = json.loads((folder / 'experiment.json').read_text())
    assert sha256_file(ROOT / exp['folds_file']) == exp['folds_sha256']
    assert sha256_file(ROOT / exp['config_file']) == exp['config_sha256']
    assert sha256_file(review.LABELS) == exp['labels_sha256']
    checkpoint = folder / 'job/checkpoint.pt'
    if sha256_file(checkpoint) != metrics['oof_provenance']['checkpoint_sha256']:
        raise ValueError('evaluated_checkpoint_hash_mismatch')
    blob = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert blob['kneenet'] == {k: exp['train'][k] for k in ('backbone', 'dim', 'depth', 'res')}
    assert blob['reader_code_sha256'] == exp['reader_code_sha256']
    assert blob.get('input_normalization') == exp.get('input_normalization')
    weights = blob['model'];assert weights and all(torch.isfinite(w).all().item() for w in weights.values())
    assert blob['step'] == metrics['applied_updates'] > 0
    attempted = metrics['attempted_updates']
    micro = metrics['micro_batch'];accum = exp['train']['batch'] // micro
    assert attempted == exp['train']['epochs'] * (len(exp['train_uids']) // micro // accum)
    assert attempted == metrics['applied_updates'] + metrics['skipped_updates']
    assert metrics['eval_precision'] == 'fp32'
    rows = [json.loads(line) for line in (folder / 'job/predictions.jsonl').read_text().splitlines()]
    assert len(rows) == len({r['StudyInstanceUID'] for r in rows})
    for split in ('gold', 'weak'):
        assert {r['StudyInstanceUID'] for r in rows if r['split'] == split} == set(exp[split + '_uids'])
    assert all(r['split'] in ('gold', 'weak') and all(math.isfinite(r[t]) and 0 <= r[t] <= 1 for t in TARGET_COLUMNS) for r in rows)
    gold = {r['StudyInstanceUID']: [r[t] for t in TARGET_COLUMNS] for r in rows if r['split'] == 'gold'}
    assert gold == metrics['gold_fp32'], 'strict-fp32 reference differs from prediction file'
    # Supply the exact, pinned source for the shared strict checkpoint validator.
    source = notebook_blob(exp_id, 'KNEE_B64')
    assert __import__('hashlib').sha256(source).hexdigest() == exp['reader_code_sha256']
    (folder / 'job/knee.py').write_bytes(source)
    from rsna_knee.submission.package import _checkpoint_errors
    assert not _checkpoint_errors(checkpoint)
    return {'checkpoint_sha256': sha256_file(checkpoint), 'weights_finite': True,
            'attempted_updates': attempted, 'applied_updates': metrics['applied_updates'],
            'skipped_updates': metrics['skipped_updates'], 'n_train': len(exp['train_uids']),
            'n_gold': len(exp['gold_uids']), 'n_weak_predictions': len(exp['weak_uids']),
            'eval_precision': 'fp32', 'verified_at_utc': now()}


def compare(plan):
    for exp_id in ('K0F1', 'K1F1'):
        review.review(exp_id)
    newer = json.loads((review.out_dir('K1F1') / 'review.json').read_text())
    first = json.loads((review.out_dir('K1') / 'review.json').read_text())
    platforms = []
    for exp_id in ('K0F1', 'K1F1'):
        folder = review.out_dir(exp_id)
        hs = json.loads((folder / 'handshake.json').read_text())
        metrics = json.loads((folder / 'job/metrics.json').read_text())
        platforms.append({k: hs[k] for k in ('gpu', 'torch', 'packages')} |
                         {k: metrics[k] for k in ('amp_dtype', 'micro_batch', 'eval_precision')})
    assert platforms[0] == platforms[1], 'paired runs used different runtime/precision; review confounding first'
    d0 = first['vs_control']['weak_hard_labels']
    d1 = newer['vs_control']['weak_hard_labels']
    direction = d0['delta_unrounded'] > 0 and d1['delta_unrounded'] > 0
    # Descriptive two-fold OOF comparison. Do not count the reused gold studies twice.
    labels = review.labels_table();pooled = []
    for new, old in (('K1', 'K0'), ('K1F1', 'K0F1')):
        a, b = review.load_predictions(review.out_dir(new)), review.load_predictions(review.out_dir(old))
        uids = sorted(u for u in a if a[u][0] == 'weak' and u in labels)
        assert all(u in b and b[u][0] == 'weak' for u in uids)
        pooled.extend((u, labels[u][0], a[u][1], b[u][1]) for u in uids)
    assert len({r[0] for r in pooled}) == len(pooled), 'weak folds overlap'
    y, pn, pc = (np.stack([r[i] for r in pooled]) for i in (1, 2, 3))
    combined = review.paired_delta(y, pn, pc, seed=2026, n_boot=2000)
    record = {'kind': 'real_second_outer_fold_comparison_not_leaderboard', 'created_at_utc': now(),
              'fold0': d0, 'fold1': d1, 'same_positive_direction': bool(direction),
              'fold1_ci95_lower_positive': d1['ci95_unrounded'][0] > 0,
              'fold1_strict_threshold_pass': newer['decision']['status'] == 'passes_this_fold_only',
              'fixed_confirmation_rule_satisfied': first['decision']['status'] == 'passes_this_fold_only' and bool(direction),
              'descriptive_two_fold_weak_paired': combined | {'n_studies': len(pooled)},
              'runtime_pair': platforms[0], 'gold_reused_development': True, 'gold_pooled': False,
              'fold1_review': str(review.out_dir('K1F1') / 'review.json'),
              'promotion': False, 'reader_only_candidate_changed': False}
    report_path = ROOT / 'artifacts/research/second-fold-review.json'
    report_path.write_text(json.dumps(record, indent=2) + '\n')
    with (ROOT / 'docs/EXPERIMENTS.md').open('a') as f:
        f.write('\n\n## K0F1/K1F1 completed real comparison — ' + now() + '\n\n')
        f.write(f"Both real 14-epoch trainings and checkpoint/prediction identity checks passed. "
                f"Matched runtime: {platforms[0]}. Fold1 weak K1F1−K0F1 delta "
                f"{d1['delta_unrounded']:+.6f}, paired bootstrap 95% CI {d1['ci95_unrounded']}; "
                f"{d1['n_paired']} scored pairs. Same positive direction across folds: {direction}. "
                f"Fold1 strict threshold pass: {record['fold1_strict_threshold_pass']}. "
                f"Gold remains reused development data and was not pooled. "
                f"Class regressions and per-class AUCs are in the individual reviews. "
                f"Reader-only/final selections unchanged; no automatic promotion/submission. "
                f"Evidence: artifacts/research/second-fold-review.json and each p1-*/output/review.json.\n")
    return record


def tick():
    plan = json.loads(PLAN.read_text())
    if plan['status'] == 'COMPLETE_REVIEWED':
        return True
    for job in plan['jobs']:
        if job['status'] == 'VERIFIED':
            continue
        folder = ROOT / job['folder']
        receipt = json.loads((folder / 'push-receipt-v1.json').read_text())
        assert receipt['returncode'] == 0 and receipt['version_number'] == 1
        metadata = json.loads((folder / 'kernel-metadata.json').read_text())
        assert sha256_file(folder / metadata['code_file']) == job['notebook_sha256']
        assert sha256_file(folder / 'kernel-metadata.json') == job['metadata_sha256']
        assert sha256_file(ROOT / f"notebooks/generated/{job['experiment']}.build.json") == job['build_sha256']
        try:
            raw = cli(['kernels', 'status', job['kernel']])
            status = re.search(r'KernelWorkerStatus\.([A-Z_]+)', raw)
            if not status:
                raise RuntimeError('unrecognized Kaggle status: ' + raw)
            job['remote_status'] = status.group(1)
            print(now(), job['experiment'], job['remote_status'], flush=True)
            if job['remote_status'] in ('ERROR', 'CANCELLED'):
                job['status'] = 'REMOTE_FAILED'
                continue
            if job['remote_status'] != 'COMPLETE':
                job['status'] = 'REMOTE_' + job['remote_status']
                continue
            output = ROOT / f"artifacts/colab/p1-{job['experiment']}/output"
            job['download_attempts'] = job.get('download_attempts', 0) + 1
            cli(['kernels', 'output', job['kernel'], '-p', str(output), '--file-pattern', FILES,
                 '--page-size', '200', '-q', '-o'], timeout=1800)
            try:
                downloaded = json.loads((output / 'job/metrics.json').read_text())
                assert sha256_file(output / 'job/checkpoint.pt') == downloaded['oof_provenance']['checkpoint_sha256']
            except (AssertionError, OSError, json.JSONDecodeError) as error:
                if job['download_attempts'] >= 3:
                    raise ValueError('checkpoint download failed identity check three times') from error
                raise RuntimeError('incomplete/corrupt artifact download; retry with pinned checkpoint hash') from error
            job['verification'] = verify_result(job)
            job['status'] = 'VERIFIED';job.pop('last_error', None)
        except (RuntimeError, subprocess.TimeoutExpired, OSError) as error:
            job['last_error'] = str(error)[-2000:]
            print(now(), job['experiment'], 'retry later:', job['last_error'], flush=True)
        save(plan)
    if all(j['status'] == 'VERIFIED' for j in plan['jobs']):
        result = compare(plan)
        plan.update(status='COMPLETE_REVIEWED', finished_at_utc=now(),
                    same_positive_direction=result['same_positive_direction'],
                    review='artifacts/research/second-fold-review.json')
        save(plan);return True
    if any(j['status'] == 'REMOTE_FAILED' for j in plan['jobs']):
        plan['status'] = 'FAILED_OR_PARTIAL';save(plan);return True
    plan['status'] = 'TRAINING_OR_WAITING_OUTPUT';save(plan);return False


if __name__ == '__main__':
    import fcntl
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    lock = (ROOT / 'artifacts/research/second-fold-monitor.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock.write(str(os.getpid()));lock.flush()
    os.chdir(ROOT);torch.set_num_threads(4)
    deadline = time.monotonic() + 8 * 3600
    try:
        while True:
            done = tick()
            if done or args.once:
                break
            if time.monotonic() >= deadline:
                plan = json.loads(PLAN.read_text());plan['status'] = 'MONITOR_TIME_LIMIT';save(plan);break
            time.sleep(60)
    except Exception as error:
        plan = json.loads(PLAN.read_text());plan.update(status='REVIEW_OR_MONITOR_FAILED', last_error=str(error));save(plan)
        raise
