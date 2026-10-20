"""Fetch and audit the public reference run. Never submits or downloads MRI/weights."""
import hashlib
import json
import subprocess
import time
from pathlib import Path

from rsna_knee.submission.validate import validate_submission

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'artifacts/kaggle/rsna-r6-coatnet-reference-20261018'
KERNEL = 'bulsahkecici/rsna-r6-coatnet-public-reference-20261018'


def cli(*args):
    # Only retry read operations; a submission must never be blindly repeated.
    retryable = args[:2] in (('kernels', 'status'), ('kernels', 'output'),
                            ('competitions', 'submissions'), ('competitions', 'submission-limits'))
    for attempt in range(4 if retryable else 1):
        try:
            result = subprocess.run([str(ROOT/'.venv/bin/kaggle'), *args], capture_output=True, text=True, check=True)
            return result.stdout
        except subprocess.CalledProcessError:
            if not retryable or attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))


def main():
    status = cli('kernels', 'status', KERNEL+'/1')
    (FOLDER/'status.txt').write_text(status)
    print(status.strip())
    if 'KernelWorkerStatus.COMPLETE' not in status:
        return
    output = FOLDER/'output'
    cli('kernels', 'output', KERNEL+'/1', '-p', str(output), '--force', '--quiet',
        '--file-pattern', r'(^|/)(reference-audit.json|submission.csv|submission_receipt.json|blend_receipt.json|.*\.log|worker[0-9]+_log.txt)$')
    provenance = json.loads((FOLDER/'provenance.json').read_text())
    assert hashlib.sha256((FOLDER/'reference.ipynb').read_bytes()).hexdigest() == provenance['notebook_sha256']
    proof = json.loads((output/'reference-audit.json').read_text())
    assert proof['status'] == 'PASS' and proof['source_notebook_sha256'] == provenance['source_sha256']
    assert proof['r6_manifest_sha256'] == '6508ce6921eda986c885b2226fa55dd4d2aa8e5fc570b4c007c9a60c05bfc648'
    assert hashlib.sha256((output/'submission.csv').read_bytes()).hexdigest() == proof['submission_sha256']
    csv = validate_submission(output/'submission.csv')
    assert csv['ok'] and csv['n_rows'] == proof['n_test']
    receipt = json.loads((output/'submission_receipt.json').read_text())
    assert receipt['status'] == 'PASS' and receipt['fallback_studies'] == 0
    assert receipt['scored'] == proof['n_test']
    assert receipt['worker_exit_codes'] and all(c == 0 for c in receipt['worker_exit_codes'].values())
    blend = json.loads((output/'blend_receipt.json').read_text())
    assert blend['raptor_failed'] == blend['ours_failed'] == blend['both_failed'] == 0
    if 'expected_weights' in provenance:
        assert blend['weights']['raptor_per_class'] == provenance['expected_weights'], 'candidate weights differ'
    # Sample runtime is diagnostic; it does not establish full-hidden-test runtime.
    evidence = [FOLDER/'reference.ipynb', FOLDER/'provenance.json', FOLDER/'status.txt',
                output/'reference-audit.json', output/'submission_receipt.json', output/'blend_receipt.json', output/'submission.csv']
    record = {'overall':'PASS', 'kind':'public_reference_operational_check', 'kernel':KERNEL,
              'version':1,'independent_holdout':False,'promotion':False,'csv':csv,
              'full_hidden_runtime_verified':False,'remote_proof':proof,
              'evidence_files':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in evidence}}
    (FOLDER/'operational-audit.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'overall':record['overall'],'n_test':proof['n_test'],
                      'seconds':proof['elapsed_seconds'],'r6_fallbacks':receipt['fallback_studies']}))


if __name__ == '__main__':
    main()
