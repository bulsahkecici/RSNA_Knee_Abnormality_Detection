"""Verified symlink view over read-only canonical cache parts; never copies MRI payloads."""
from __future__ import annotations

import json
from pathlib import Path

from rsna_knee.data.cache_build import _atomic_json
from rsna_knee.data.volume_cache import VERSION, _payload_path
from rsna_knee.hashing import sha256_file, sha256_json


def mounted_volume_view(parts: list[Path], dest: Path, *, expected_hashes: list[str]) -> dict:
    """Bind exactly pinned manifests into one flat view; full pixel QC follows in the trainer.

    Every source is read-only. Metadata/policy/scope and compressed-file sizes are
    checked before symlinking; volume_manifest_ready then hashes/decodes every
    linked payload once. Destination may be reused only for identical bindings.
    """
    if not parts or len(parts) != len(expected_hashes):
        raise ValueError('one expected hash required per cache part')
    dest = Path(dest)
    loaded, requested, filenames = [], set(), set()
    for part, digest in zip(parts, expected_hashes, strict=True):
        manifest_path = Path(part) / 'manifest.json'
        if sha256_file(manifest_path) != digest:
            raise ValueError('mounted_manifest_hash_mismatch')
        m = json.loads(manifest_path.read_text())
        if m['version'] != VERSION or m['preprocess_hash'] != sha256_json(m['preprocess_config']):
            raise ValueError('mounted_policy_hash_mismatch')
        if m['manifest_sha256'] != sha256_json({k: v for k, v in m.items() if k != 'manifest_sha256'}):
            raise ValueError('mounted_manifest_content_hash_mismatch')
        uids = m['requested_uids']
        if len(uids) != len(set(uids)) or set(uids) != set(m['studies']) or m['quarantine']:
            raise ValueError('mounted_cache_scope_incomplete')
        if requested & set(uids):
            raise ValueError('overlapping_mounted_cache_parts')
        if loaded and any(m[k] != loaded[0][1][k] for k in ('preprocess_hash', 'series_csv_sha256')):
            raise ValueError('incompatible_mounted_cache_parts')
        for meta in m['studies'].values():
            source = _payload_path(Path(part), meta['file'])
            if source.name in filenames:
                raise ValueError('duplicate_mounted_payload_filename')
            if source.stat().st_size != meta['bytes']:
                raise ValueError('mounted_payload_size_mismatch')
            filenames.add(source.name)
        requested.update(uids)
        loaded.append((Path(part), m))
    merged = {**loaded[0][1], 'studies': {}, 'quarantine': {}, 'requested_uids': sorted(requested)}
    for _, manifest in loaded:
        merged['studies'].update(manifest['studies'])
    merged['manifest_sha256'] = sha256_json({k: v for k, v in merged.items() if k != 'manifest_sha256'})
    dest.mkdir(parents=True, exist_ok=True)
    old = dest / 'manifest.json'
    if old.exists() and json.loads(old.read_text()) != merged:
        raise ValueError('mounted_view_binding_changed_use_new_destination')
    for part, m in loaded:
        for meta in m['studies'].values():
            target = dest / meta['file']
            source = _payload_path(part, meta['file']).resolve(strict=True)
            if target.is_symlink():
                if target.resolve(strict=True) != source:
                    raise ValueError('mounted_view_link_target_mismatch')
            elif target.exists():
                raise ValueError('mounted_view_would_overwrite_existing_file')
            else:
                target.symlink_to(source)
    _atomic_json(dest / 'manifest.json', merged)
    return merged
