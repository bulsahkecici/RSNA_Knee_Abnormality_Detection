"""K1 remote header/decode pilot (default 64 studies), or bounded full-cache shard.

Run via make_kaggle_kernel; mounts competition data without downloading to Mac.
No training, reports, uploads or submissions. Full builds are a separate explicit job.
"""
from __future__ import annotations

import base64
import csv
import hashlib
import io
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

SOURCE_B64 = ""
SOURCE_SHA256 = ""
BUILD_INFO = {}
LIMIT = 64  # define
SHARD = {'index': 0, 'count': 1}  # define
MAX_CACHE_BYTES = 18 * 1024**3  # define
SIZE = 384  # define


def main():
    import tempfile

    work = Path(os.environ.get('RSNA_WORK_DIR', '/kaggle/working'))
    work.mkdir(parents=True, exist_ok=True)
    data = base64.b64decode(SOURCE_B64, validate=True)
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA256:
        raise ValueError('source_integrity_failure')
    source = Path(tempfile.gettempdir()) / ('rsna-k1-' + SOURCE_SHA256[:16])
    source.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data)) as t:
        t.extractall(source, filter='data')
    sys.path.insert(0, str(source / 'src'))
    smoke = os.environ.get('RSNA_LOCAL_SMOKE') == '1'
    if not smoke:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'pydicom==3.0.1',
                        'opencv-python-headless>=4.10,<5', 'pylibjpeg==2.1.0',
                        'pylibjpeg-libjpeg==2.4.0', 'pylibjpeg-openjpeg==2.6.0'], check=True)
    import numpy as np

    from rsna_knee.data.cache_build import _atomic_json
    from rsna_knee.data.volume_cache import build_volume_cache, volume_manifest_ready
    from rsna_knee.hashing import sha256_file
    explicit = os.environ.get('RSNA_INPUT_ROOT')
    roots = ([Path(explicit)] if explicit else []) + [Path('/kaggle/input/rsna-knee-abnormality-detection'),
                                                    Path('/kaggle/input/competitions/rsna-knee-abnormality-detection')]
    root = next((p for p in roots if (p / 'train_series.csv').is_file() and (p / 'train_series').is_dir()), None)
    if root is None:
        raise RuntimeError('competition data not mounted')
    if not 0 <= SHARD['index'] < SHARD['count']:
        raise ValueError('invalid shard')
    with (root / 'train_series.csv').open() as f:
        uids = sorted({r['StudyInstanceUID'] for r in csv.DictReader(f)})
    uids = uids[SHARD['index']::SHARD['count']]
    if LIMIT is not None:
        if LIMIT < 1:
            raise ValueError('invalid limit')
        uids = sorted(np.random.default_rng(2026).permutation(uids)[:LIMIT].tolist())
    result = {'status': 'running', 'scope': 'synthetic_dicom_preprocessing_only' if smoke else 'real_dicom_preprocessing_only',
              'build_info': BUILD_INFO,
              'n_requested': len(uids), 'pilot': LIMIT is not None}
    _atomic_json(work / 'campaign-result.json', result)
    started = time.perf_counter()
    try:
        m = build_volume_cache(series_csv=root / 'train_series.csv', dicom_root=root / 'train_series',
                               dest=work / 'cache', study_ids=uids, size=SIZE, max_cache_bytes=MAX_CACHE_BYTES)
        ready, reason = volume_manifest_ready(work / 'cache/manifest.json', required_uids=uids)
        if not ready:
            raise RuntimeError(reason)
        seconds = time.perf_counter() - started
        payload_bytes = sum(s['bytes'] for s in m['studies'].values())
        _atomic_json(work / 'campaign-result.json', result | {'status': 'ok', 'elapsed_seconds': seconds,
                     'seconds_per_study': seconds / len(uids), 'payload_bytes': payload_bytes,
                     'manifest_sha256': sha256_file(work / 'cache/manifest.json'),
                     'n_quarantine': len(m['quarantine']),
                     'note': 'Decode throughput only; measure real GPU training throughput separately.'})
    except Exception as exc:
        _atomic_json(work / 'campaign-result.json', result | {'status': 'failed', 'error': str(exc)[:500]})
        raise


if __name__ == '__main__':
    main()
