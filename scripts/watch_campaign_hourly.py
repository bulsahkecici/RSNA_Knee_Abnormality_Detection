"""Read-only hourly Kaggle checks and local macOS notifications."""
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from filelock import FileLock

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'state/campaigns/campaign-20261003-v2'


def check():
    state = json.loads((FOLDER / 'status.json').read_text())
    observed = {}
    for phase, remote in state.get('remote', {}).items():
        try:
            result = subprocess.run(
                [shutil.which('kaggle') or 'kaggle', 'kernels', 'status', remote['kernel']],
                capture_output=True, text=True, timeout=90,
            )
            observed[phase] = result.stdout.strip() if result.returncode == 0 else 'Durum alınamadı; ağ/API kontrolü gerekli'
        except (OSError, subprocess.TimeoutExpired):
            observed[phase] = 'Durum alınamadı; ağ/API kontrolü gerekli'
    stamp = datetime.now(ZoneInfo('Europe/Istanbul')).isoformat(timespec='seconds')
    message = '; '.join(f'{phase}: {value}' for phase, value in observed.items())
    message += f". Yerel aşama: {state.get('phase', 'bilinmiyor')}"
    record = {'checked_at': stamp, 'phase': state.get('phase'), 'remote': observed}
    with (FOLDER / 'hourly-checks.jsonl').open('a') as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + '\n')
    script = 'on run argv\ndisplay notification (item 1 of argv) with title "RSNA saatlik durum"\nend run'
    result = subprocess.run(['/usr/bin/osascript', '-e', script, message], capture_output=True, timeout=30)
    record['notification_command_ok'] = result.returncode == 0
    temp = FOLDER / 'hourly-latest.tmp'
    temp.write_text(json.dumps(record, ensure_ascii=False, indent=2))
    temp.replace(FOLDER / 'hourly-latest.json')
    print(json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    with FileLock(str(FOLDER / 'hourly-watcher.lock'), timeout=0):
        while True:
            try:
                check()
            except Exception as exc:
                print(json.dumps({'check_error': type(exc).__name__}), flush=True)
            if '--once' in sys.argv:
                break
            time.sleep(3600)
