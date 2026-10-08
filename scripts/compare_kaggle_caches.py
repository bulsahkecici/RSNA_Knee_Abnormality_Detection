"""Verify mounted cache outputs without rebuilding or exporting MRI shards."""
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


reports = {}
for version in ('v2', 'v3'):
    start = time.monotonic()
    roots = [p.parent for p in Path('/kaggle/input').rglob('campaign-result.json')
             if f'campaign-20261003-{version}-cache' in str(p)]
    if len(roots) != 1:
        reports[version] = {'pass': False, 'error': 'missing_or_ambiguous_mount'}
        continue
    root = roots[0]
    result = json.loads((root / 'campaign-result.json').read_text())
    manifest_path = root / 'cache/manifest.json'
    if not manifest_path.is_file():
        reports[version] = {'pass': False, 'error': 'manifest_missing', 'result': result}
        continue
    manifest = json.loads(manifest_path.read_text())
    errors = Counter()
    if digest(manifest_path) != result.get('manifest_sha256'):
        errors['manifest_hash'] += 1
    studies = manifest['studies']
    if set(studies) != set(manifest['requested_uids']) or manifest['quarantine']:
        errors['coverage'] += 1
    total_bytes = 0
    present_planes = Counter()
    for number, (uid, meta) in enumerate(studies.items(), 1):
        path = root / 'cache' / f'{uid}.npy'
        try:
            total_bytes += path.stat().st_size
            if digest(path) != meta.get('sha256'):
                errors['shard_hash'] += 1
            array = np.load(path, mmap_mode='r', allow_pickle=False)
            if list(array.shape) != [3, 3, 3, 224, 224] or array.dtype != np.uint8:
                errors['shape_or_dtype'] += 1
            mask = meta.get('slot_mask', [])
            if len(mask) != 3 or any(x not in (0, 1) for x in mask) or not any(mask):
                errors['mask'] += 1
            else:
                present_planes[str(int(sum(mask)))] += 1
                for slot, present in enumerate(mask):
                    if (present and array[slot].max() == array[slot].min()) or (not present and np.any(array[slot])):
                        errors['mask_pixels'] += 1
            if meta.get('synthetic') is not False:
                errors['synthetic'] += 1
        except Exception as exc:
            errors[type(exc).__name__] += 1
        if number % 500 == 0:
            print(json.dumps({'version': version, 'checked': number}), flush=True)
    reports[version] = {'pass': not errors, 'errors': dict(errors), 'n_studies': len(studies),
                        'image_bytes': total_bytes, 'present_planes': dict(present_planes),
                        'verification_seconds': round(time.monotonic() - start, 2),
                        'result': result, 'preprocess_hash': manifest['preprocess_hash']}
    # Make small metadata available through a notebook with only a few output files.
    (Path('/kaggle/working') / f'{version}-campaign-result.json').write_text(json.dumps(result, indent=2))
    (Path('/kaggle/working') / f'{version}-manifest.json').write_bytes(manifest_path.read_bytes())
out = Path('/kaggle/working/cache-comparison.json')
out.write_text(json.dumps(reports, indent=2))
print(json.dumps(reports), flush=True)
if not all(reports[v]['pass'] for v in ('v2', 'v3')):
    raise RuntimeError('cache_integrity_failed_see_comparison')
