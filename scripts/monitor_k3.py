"""One bounded resource pilot, then at most one full job after verified quota refresh.

Never submits, searches weights, runs a second fold, or retries an uncertain push.
Only numeric provenance, predictions and checkpoints may be downloaded.
"""
import argparse
import fcntl
import json
import os
import re
import subprocess
import time
from pathlib import Path
import monitor_second_fold as shared
from rsna_knee.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'artifacts/research/k3-campaign.json'
TERMINAL = {'COMPLETE_REVIEWED', 'STOPPED_CALIBRATION', 'STOPPED_FULL', 'PUSH_UNKNOWN',
            'STOPPED_VERIFICATION', 'MONITOR_TIME_LIMIT', 'COLAB_HANDOFF'}
PILOT_FILES = r'^(calibration-result\.json|resource-gate\.json|experiment\.json|handshake\.json|throughput\.json|campaign-result\.json|cache-bindings\.json)$'


def save(plan):
    plan['last_checked_at_utc'] = shared.now()
    temp = PLAN.with_suffix('.tmp');temp.write_text(json.dumps(plan, indent=2) + '\n');temp.replace(PLAN)


def quota():
    raw = json.loads(shared.cli(['quota', '--format', 'json']))
    gpu = [r for r in raw if r['resource'] == 'GPU']
    assert len(gpu) == 1
    return float(gpu[0]['remaining'].removesuffix('h')), gpu[0]


def bindings(job):
    folder = ROOT / job['folder']
    assert sha256_file(folder / (job['id'] + '.ipynb')) == job['notebook_sha256']
    assert sha256_file(folder / 'kernel-metadata.json') == job['metadata_sha256']
    if job['id'] == 'K3EV2':
        assert sha256_file(ROOT / 'notebooks/generated/K3EV2.build.json') == job['build_sha256']
    return folder


def push(plan, key, timeout):
    job = plan[key];folder = bindings(job)
    attempt = folder / 'push-attempt.json'
    assert not attempt.exists(), 'Never automatically repeat a push'
    attempt.write_text(json.dumps({'kernel': job['kernel'], 'at_utc': shared.now(),
                                  'notebook_sha256': job['notebook_sha256'], 'timeout_seconds': timeout}, indent=2))
    plan['status'] = 'PUSH_UNKNOWN';save(plan)
    done = subprocess.run([str(shared.KAGGLE), 'kernels', 'push', '-p', str(folder), '-t', str(timeout),
                           '--accelerator', 'NvidiaTeslaT4'], capture_output=True, text=True, timeout=240, cwd=ROOT)
    version = re.search(r'Kernel version (\d+) successfully pushed', done.stdout)
    receipt = {'kind': 'real_training_push_not_competition_submission', 'kernel': job['kernel'],
               'version_number': int(version.group(1)) if version else None,
               'returncode': done.returncode, 'stdout': done.stdout, 'stderr': done.stderr,
               'accepted_at_utc': shared.now(), 'timeout_seconds': timeout,
               'notebook_sha256': job['notebook_sha256'], 'competition_submission': False}
    (folder / 'push-receipt-v1.json').write_text(json.dumps(receipt, indent=2))
    assert done.returncode == 0 and receipt['version_number'] == 1, 'Uncertain push; manual reconciliation required'
    job['version_number'] = 1
    plan['status'] = 'PILOT_RUNNING' if key == 'pilot' else 'FULL_RUNNING';save(plan)


def remote(job):
    bindings(job)
    receipt = json.loads((ROOT / job['folder'] / 'push-receipt-v1.json').read_text())
    assert receipt['returncode'] == 0 and receipt['version_number'] == 1
    raw = shared.cli(['kernels', 'status', job['kernel']])
    match = re.search(r'KernelWorkerStatus\.([A-Z_]+)', raw)
    assert match, raw
    job['remote_status'] = match.group(1)
    return job['remote_status']


def verify_pilot(plan, folder):
    exp = json.loads((folder / 'experiment.json').read_text())
    result = json.loads((folder / 'calibration-result.json').read_text())
    build = json.loads((ROOT / plan['pilot']['folder'] / 'build.json').read_text())
    for key in ('id', 'config_sha256', 'training_cell_sha256', 'weights_sha256', 'source_sha256',
                'train_uids', 'train_uids_sha256', 'input_normalization', 'runtime_guard'):
        assert exp[key] == build[key], key
    assert result['kind'] == 'real_resource_calibration_not_validation' and result['pass'] is True
    assert result['experiment_id'] == exp['id']
    for key in ('config_sha256', 'training_cell_sha256', 'weights_sha256'):
        assert result[key] == exp[key]
    assert result['applied_optimizer_steps'] > 0 and result['n_train'] == 192
    assert result['n_gold'] == result['n_weak'] == 0 and exp['gold_uids'] == exp['weak_uids'] == []
    assert result['fp32_eval_windows'] == 12
    full = json.loads((ROOT / 'notebooks/generated/K3EV2.build.json').read_text())
    assert exp['runtime_guard']['projection_training_studies'] == full['n_train_uids'] == 3479
    from rsna_knee.data.folds import assert_train_uids, load_folds
    folds = load_folds(ROOT / exp['folds_file']);assert_train_uids(set(exp['train_uids']), folds)
    roles = {r['StudyInstanceUID']: r for r in folds}
    assert all(roles[u]['fold'] != '0' for u in exp['train_uids'])
    gate = result['resource_gate']
    assert gate == json.loads((folder / 'resource-gate.json').read_text()) and gate['pass'] is True
    assert gate['projected_training_seconds'] <= 8 * 3600
    assert result['train_and_eval_peak_cuda_bytes'] <= .9 * gate['device_vram_bytes']
    hs = json.loads((folder / 'handshake.json').read_text())
    assert hs['gpu'] == 'Tesla T4', 'Full recipe pinned to calibrated T4'
    return result


def tick():
    plan = json.loads(PLAN.read_text())
    if plan['status'] in TERMINAL:return True
    if plan['status'] == 'PREPARED_NOT_RUN':
        checks = json.loads((ROOT / 'artifacts/research/k3-local-checks.json').read_text())
        assert checks['pass'] is True
        assert checks['full_notebook_sha256'] == plan['full']['notebook_sha256']
        assert checks['pilot_notebook_sha256'] == plan['pilot']['notebook_sha256']
        remaining, snapshot = quota();plan['quota_before_pilot'] = snapshot
        if remaining < 1.5:
            plan['status'] = 'WAITING_PILOT_QUOTA';save(plan);return False
        push(plan, 'pilot', 3600);return False
    if plan['status'] == 'WAITING_PILOT_QUOTA':
        remaining, plan['last_quota'] = quota()
        if remaining >= 1.5:plan['status'] = 'PREPARED_NOT_RUN'
        save(plan);return False
    if plan['status'] == 'PILOT_RUNNING':
        status = remote(plan['pilot']);save(plan)
        if status not in ('COMPLETE', 'ERROR', 'CANCELLED'):return False
        folder = ROOT / plan['pilot']['folder'] / 'output'
        shared.cli(['kernels', 'output', plan['pilot']['kernel'], '-p', str(folder),
                    '--file-pattern', PILOT_FILES, '--page-size', '200', '-q', '-o'], timeout=600)
        if status != 'COMPLETE':
            plan['status'] = 'STOPPED_CALIBRATION';save(plan);return True
        plan['calibration'] = verify_pilot(plan, folder)
        plan['status'] = 'WAITING_QUOTA';save(plan);return False
    if plan['status'] == 'WAITING_QUOTA':
        remaining, plan['last_quota'] = quota()
        bindings(plan['full'])
        if remaining < plan['minimum_remaining_gpu_hours']:
            save(plan);return False
        push(plan, 'full', 36000);return False
    if plan['status'] == 'FULL_RUNNING':
        job = plan['full'];status = remote(job);save(plan)
        if status not in ('COMPLETE', 'ERROR', 'CANCELLED'):return False
        folder = ROOT / 'artifacts/colab/p1-K3EV2/output'
        shared.cli(['kernels', 'output', job['kernel'], '-p', str(folder),
                    '--file-pattern', shared.FILES[:-2] + r'|resource-gate\.json)$',
                    '--page-size', '200', '-q', '-o'], timeout=1800)
        if status != 'COMPLETE':
            plan['status'] = 'STOPPED_FULL';save(plan);return True
        gate = json.loads((folder / 'resource-gate.json').read_text());assert gate['pass']
        hs = json.loads((folder / 'handshake.json').read_text());assert hs['gpu'] == 'Tesla T4'
        plan['verification'] = shared.verify_result({'experiment': 'K3EV2'})
        report = shared.review.review('K3EV2')
        plan.update(status='COMPLETE_REVIEWED', decision=report['decision'], completed_at_utc=shared.now())
        save(plan)
        with (ROOT / 'docs/EXPERIMENTS.md').open('a') as handle:
            handle.write('\n\nK3EV2 completed: ' + json.dumps(report['vs_control']['weak_hard_labels']) +
                         '. Decision: ' + report['decision']['status'] +
                         '. Strict checkpoint/coverage verification passed. No automatic submission or fold1.\n')
        return True
    raise ValueError('Unknown controller state: ' + plan['status'])


def main():
    parser = argparse.ArgumentParser();parser.add_argument('--once', action='store_true');args = parser.parse_args()
    os.chdir(ROOT);shared.torch.set_num_threads(4)
    lock = (ROOT / 'artifacts/research/k3-monitor.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB);lock.write(str(os.getpid()));lock.flush()
    deadline = time.monotonic() + 26 * 3600
    while True:
        try:
            if tick() or args.once:break
        except (AssertionError, ValueError, KeyError) as error:
            plan = json.loads(PLAN.read_text())
            if plan['status'] != 'PUSH_UNKNOWN':plan['status'] = 'STOPPED_VERIFICATION'
            plan['last_error'] = repr(error);save(plan);raise
        except (RuntimeError, OSError, TimeoutError, subprocess.TimeoutExpired) as error:
            plan = json.loads(PLAN.read_text());plan['last_error'] = str(error)[-2000:];save(plan)
            if plan['status'] == 'PUSH_UNKNOWN':raise
            print(shared.now(), 'Will retry read-only check:', error, flush=True)
        plan = json.loads(PLAN.read_text());print(shared.now(), plan['status'], flush=True)
        if time.monotonic() >= deadline:
            plan['status'] = 'MONITOR_TIME_LIMIT';save(plan);break
        time.sleep(60)

if __name__ == '__main__':main()
