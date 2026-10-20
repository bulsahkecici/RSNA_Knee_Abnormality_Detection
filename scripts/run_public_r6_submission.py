"""Watch the authorized public reference and submit once after operational checks.

This is an external public notebook reproduction, not promotion of a locally
trained rsna run. No synthetic metrics or local training registry are fabricated.
"""
import argparse
import fcntl
import json
import subprocess
import time
from datetime import datetime, timezone

from check_public_r6 import FOLDER, KERNEL, ROOT, cli, main as audit
import check_public_r6 as audit_module
from pathlib import Path

COMPETITION = 'rsna-knee-abnormality-detection'
MESSAGE = 'Public R6 CoAtNet reference 20261018 v1; original recipe; operational audit'


def save(name, data):
    path = FOLDER/name
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2)+'\n')
    temp.replace(path)


def history():
    raw = cli('competitions', 'submissions', COMPETITION, '--format', 'json', '--page-size', '100')
    # CLI can prefix the JSON array with a pagination token.
    rows = json.loads(raw[raw.index('['):])
    (FOLDER/'submission-history.json').write_text(json.dumps(rows,indent=2)+'\n')
    matches = [r for r in rows if r.get('description') == MESSAGE]
    assert len(matches) <= 1, 'multiple matching submissions: manual reconciliation required'
    return matches[0] if matches else None


def main():
    global FOLDER, KERNEL, MESSAGE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--run-folder', type=Path)
    args = parser.parse_args()
    if args.run_folder:
        FOLDER = args.run_folder.resolve()
        assert FOLDER.is_relative_to(ROOT/'artifacts/kaggle')
        metadata = json.loads((FOLDER/'kernel-metadata.json').read_text())
        provenance = json.loads((FOLDER/'provenance.json').read_text())
        KERNEL = metadata['id']
        assert KERNEL.startswith('bulsahkecici/') and metadata['enable_internet'] is False
        MESSAGE = provenance['submission_message']
        assert MESSAGE and len(MESSAGE) < 200
        audit_module.FOLDER, audit_module.KERNEL = FOLDER, KERNEL
    lock = (FOLDER/'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    deadline = time.monotonic()+12*3600
    while True:
        status = cli('kernels', 'status', KERNEL+'/1')
        print(status.strip(), flush=True)
        if 'KernelWorkerStatus.COMPLETE' in status:
            break
        if any(x in status for x in ('ERROR', 'CANCEL', 'FAILED')):
            raise RuntimeError(status)
        if time.monotonic() >= deadline:
            raise TimeoutError('Notebook not complete; no submission made')
        time.sleep(45)
    audit()
    output = FOLDER/'output'
    logs = list(output.glob('*.log'))
    assert logs, 'execution log missing; cannot check silent recipe fallback'
    log_text = '\n'.join(p.read_text(errors='replace') for p in logs)
    assert 'own leg: blended' in log_text, 'reader completion not established'
    assert 'own leg FAILED' not in log_text, 'reader silently fell back'
    assert '224' in log_text, '224 stage evidence missing'
    found = history()
    if not found:
        if (FOLDER/'submit-attempt.json').exists():
            raise RuntimeError('Prior attempt unresolved; reconcile history, never blindly resubmit')
        if not args.submit:
            print('READY, no submit requested', flush=True)
            return
        limits = cli('competitions', 'submission-limits', '-c', COMPETITION)
        (FOLDER/'submission-limits.txt').write_text(limits)
        import re
        match = re.search(r'Remaining today:\s*(\d+)', limits)
        assert match and int(match.group(1)) > 0, 'daily capacity not established'
        command = ['competitions','submit',COMPETITION,'-k',KERNEL,'-v','1',
                   '-f','submission.csv','-m',MESSAGE]
        save('submit-attempt.json', {'status':'ATTEMPT_RECORDED','kernel':KERNEL,'version':1,
             'message':MESSAGE,'created_at_utc':datetime.now(timezone.utc).isoformat()})
        result = subprocess.run([str(ROOT/'.venv/bin/kaggle'),*command],capture_output=True,text=True)
        save('submit-response.json', {'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
        print(result.stdout, flush=True)
        for _ in range(6):
            found = history()
            if found:
                break
            time.sleep(10)
        assert found, 'No receipt yet; reconcile, do not repeat submission'
    assert str(found.get('ref','')).isdigit(), 'missing server receipt'
    deadline = time.monotonic()+12*3600
    while True:
        save('submission-record.json', found)
        print(json.dumps(found), flush=True)
        status = str(found.get('status','')).upper()
        if any(x in status for x in ('COMPLETE','ERROR','CANCEL','FAILED')):
            return
        if time.monotonic() >= deadline:
            return
        time.sleep(45)
        found = history()
        assert found, 'receipt disappeared from history'


if __name__ == '__main__':
    main()
