"""Monitor only the single bounded K2R320 job; collect and review, never submit/restart."""
import argparse
import fcntl
import json
import os
import re
import time
from pathlib import Path

import monitor_second_fold as shared
from rsna_knee.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'artifacts/research/k2-campaign.json'
FILES = shared.FILES.replace('campaign-result\\.json|', 'campaign-result\\.json|resource-gate\\.json|')


def save(plan):
    plan['last_checked_at_utc'] = shared.now()
    tmp = PLAN.with_suffix('.json.tmp');tmp.write_text(json.dumps(plan, indent=2) + '\n');tmp.replace(PLAN)


def tick():
    plan = json.loads(PLAN.read_text());job = plan['job']
    if plan['status'] in ('COMPLETE_REVIEWED', 'STOPPED_BY_RESOURCE_GATE', 'REMOTE_FAILED'):
        return True
    folder = ROOT / job['folder']
    receipt = json.loads((folder / 'push-receipt-v1.json').read_text())
    assert receipt['returncode'] == 0 and receipt['version_number'] == 1
    assert sha256_file(folder / 'K2R320.ipynb') == job['notebook_sha256']
    assert sha256_file(folder / 'kernel-metadata.json') == job['metadata_sha256']
    assert sha256_file(ROOT / 'notebooks/generated/K2R320.build.json') == job['build_sha256']
    try:
        raw = shared.cli(['kernels', 'status', job['kernel']])
        status = re.search(r'KernelWorkerStatus\.([A-Z_]+)', raw)
        if not status:
            raise RuntimeError('unrecognized Kaggle state: ' + raw)
        job['remote_status'] = status.group(1)
        print(shared.now(), 'K2R320', job['remote_status'], flush=True)
        if job['remote_status'] in ('ERROR', 'CANCELLED'):
            try:
                shared.cli(['kernels', 'output', job['kernel'], '-p', str(folder / 'stopped-output'),
                            '--file-pattern', r'^(resource-gate\.json|campaign-result\.json|handshake\.json|throughput\.json)$', '-q'])
            except RuntimeError as error:
                plan['output_fetch_error'] = str(error)
            gate = folder / 'stopped-output/resource-gate.json'
            plan['status'] = 'REMOTE_FAILED'
            if gate.is_file():
                plan['resource_gate'] = json.loads(gate.read_text())
                if plan['resource_gate']['pass'] is False:
                    plan['status'] = 'STOPPED_BY_RESOURCE_GATE'
            save(plan)
            with (ROOT / 'docs/EXPERIMENTS.md').open('a') as handle:
                handle.write('\n\nK2R320 stopped: ' + plan['status'] + '. No completed training/AUC claimed. '
                             'Evidence: artifacts/research/k2-campaign.json and the kernel stopped-output/log.\n')
            return True
        if job['remote_status'] != 'COMPLETE':
            plan['status'] = 'REMOTE_' + job['remote_status'];save(plan);return False
        output = ROOT / 'artifacts/colab/p1-K2R320/output'
        job['download_attempts'] = job.get('download_attempts', 0) + 1
        shared.cli(['kernels', 'output', job['kernel'], '-p', str(output), '--file-pattern', FILES,
                    '--page-size', '200', '-q', '-o'], timeout=1800)
        metrics = json.loads((output / 'job/metrics.json').read_text())
        if sha256_file(output / 'job/checkpoint.pt') != metrics['oof_provenance']['checkpoint_sha256']:
            if job['download_attempts'] >= 3:
                raise ValueError('checkpoint hash mismatch after three downloads')
            raise RuntimeError('checkpoint download incomplete; retrying pinned output')
        gate = json.loads((output / 'resource-gate.json').read_text())
        assert gate['kind'] == 'real_measured_training_resource_gate_not_AUC' and gate['pass']
        job['verification'] = shared.verify_result(job)
        report = shared.review.review('K2R320')
        plan.update(status='COMPLETE_REVIEWED', resource_gate=gate,
                    review=str(output / 'review.json'), decision=report['decision'], completed_at_utc=shared.now())
        save(plan)
        with (ROOT / 'docs/EXPERIMENTS.md').open('a') as handle:
            delta = report['vs_control']['weak_hard_labels']
            handle.write('\n\n## K2R320 completed — ' + shared.now() + '\n\n')
            handle.write(f"Real K2R320 versus K1: weak delta {delta['delta_unrounded']:+.6f}, "
                         f"95% paired bootstrap CI {delta['ci95_unrounded']}. Decision: {report['decision']['status']}. "
                         'Checkpoint/coverage/finite-output checks passed. Reused gold/class regressions in review.json. '
                         'No automatic submission/promotion; confirm fold1 only if fold0 passes.\n')
        return True
    except (RuntimeError, OSError, TimeoutError, shared.subprocess.TimeoutExpired) as error:
        plan['last_error'] = str(error)[-2000:];save(plan)
        print(shared.now(), 'Retry later:', plan['last_error'], flush=True);return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__);parser.add_argument('--once', action='store_true')
    args = parser.parse_args();os.chdir(ROOT)
    shared.torch.set_num_threads(4)
    lock = (ROOT / 'artifacts/research/k2-monitor.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB);lock.write(str(os.getpid()));lock.flush()
    deadline = time.monotonic() + 9 * 3600
    try:
        while True:
            if tick() or args.once:
                break
            if time.monotonic() > deadline:
                plan = json.loads(PLAN.read_text());plan['status'] = 'MONITOR_TIME_LIMIT';save(plan);break
            time.sleep(60)
    except Exception as error:
        plan = json.loads(PLAN.read_text());plan.update(status='REVIEW_OR_MONITOR_FAILED', last_error=str(error));save(plan)
        raise
