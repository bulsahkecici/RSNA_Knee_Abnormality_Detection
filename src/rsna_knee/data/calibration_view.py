"""A train-only, bounded derived view; preprocessing bytes remain unchanged."""
import json
from pathlib import Path
from rsna_knee.data.volume_mounts import mounted_volume_view
from rsna_knee.data.cache_build import _atomic_json
from rsna_knee.hashing import sha256_json


def calibration_view(parts, dest, *, expected_hashes, uids):
    if not uids or len(uids) != len(set(uids)):
        raise ValueError('invalid_calibration_scope')
    dest = Path(dest)
    full = dest.parent / (dest.name + '-bindings')
    m = mounted_volume_view(parts, full, expected_hashes=expected_hashes)
    if not set(uids) <= set(m['studies']):
        raise ValueError('calibration_scope_missing')
    derived = {**m, 'requested_uids': sorted(uids), 'studies': {u: m['studies'][u] for u in sorted(uids)},
               'derived_from_manifest_sha256s': expected_hashes,
               'purpose': 'train_only_resource_calibration_not_validation'}
    derived['manifest_sha256'] = sha256_json({k: v for k, v in derived.items() if k != 'manifest_sha256'})
    dest.mkdir(parents=True, exist_ok=True)
    previous = dest / 'manifest.json'
    if previous.exists() and json.loads(previous.read_text()) != derived:
        raise ValueError('calibration_binding_changed')
    for meta in derived['studies'].values():
        source = (full / meta['file']).resolve(strict=True)
        target = dest / meta['file']
        if target.is_symlink():
            if target.resolve(strict=True) != source:
                raise ValueError('calibration_link_changed')
        elif target.exists():
            raise ValueError('calibration_destination_occupied')
        else:
            target.symlink_to(source)
    _atomic_json(previous, derived)
    return previous
