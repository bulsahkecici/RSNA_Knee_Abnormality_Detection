"""Verified read-only symlink view of the v5 window cache, without copying MRI payloads."""
from __future__ import annotations

import json
from pathlib import Path

from rsna_knee.data.cache_build import _atomic_json, _coverage, _safe_uid
from rsna_knee.data.preprocess import preprocess_hash
from rsna_knee.hashing import sha256_file, sha256_json


def mounted_window_view(parts: list[Path], dest: Path, *, expected_hashes: list[str]) -> dict:
    """Bind pinned, complete NPZ parts; manifest_ready performs full pixel QC afterward."""
    if not parts or len(parts) != len(expected_hashes):
        raise ValueError('one expected hash required per cache part')
    loaded, studies, requested, links = [], {}, [], []
    for part, digest in zip(parts, expected_hashes, strict=True):
        part = Path(part)
        path = part / 'manifest.json'
        if sha256_file(path) != digest:
            raise ValueError('mounted_manifest_hash_mismatch')
        m = json.loads(path.read_text())
        policy = m['preprocess_config']
        if policy.get('storage_format') != 'npz' or preprocess_hash(policy) != m['preprocess_hash']:
            raise ValueError('mounted_window_policy_mismatch')
        if (m.get('scope') not in ('full', 'pilot') or m.get('budget_error') or m.get('quarantine')
                or set(m['requested_uids']) != set(m['studies'])
                or len(m['requested_uids']) != len(set(m['requested_uids']))
                or len(m['studies']) != m['n_studies'] or len(m['requested_uids']) != m['n_requested']):
            raise ValueError('mounted_cache_scope_incomplete')
        if loaded and (policy != loaded[0]['preprocess_config']
                       or m['builder_version'] != loaded[0]['builder_version']):
            raise ValueError('incompatible_mounted_cache_parts')
        if set(studies) & set(m['studies']):
            raise ValueError('overlapping_mounted_cache_parts')
        for uid, meta in m['studies'].items():
            if not _safe_uid(uid) or meta.get('uid') != uid:
                raise ValueError('invalid_mounted_study_uid')
            for suffix in ('npz', 'json'):
                source = part / f'{uid}.{suffix}'
                if not source.is_file():
                    raise ValueError('mounted_payload_missing')
                links.append(source)
        loaded.append(m)
        studies.update(m['studies'])
        requested.extend(m['requested_uids'])
    total = sum(p.stat().st_size for p in links if p.suffix == '.npz')
    if total != sum(m['compressed_byte_total'] for m in loaded):
        raise ValueError('mounted_payload_size_mismatch')
    merged = {**loaded[0], 'studies': studies, 'quarantine': {}, 'n_studies': len(studies),
              'n_quarantine': 0, 'requested_uids': requested, 'n_requested': len(requested),
              'compressed_byte_total': total, 'max_cache_bytes': sum(m['max_cache_bytes'] for m in loaded),
              'requested_complete': True,
              'merged_from': [{'manifest_sha256': sha256_json(m), 'n_studies': m['n_studies']} for m in loaded],
              'coverage': _coverage(studies, {}, [tuple(s) for s in loaded[0]['preprocess_config']['slots']])}
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    old = dest / 'manifest.json'
    if old.exists() and json.loads(old.read_text()) != merged:
        raise ValueError('mounted_view_binding_changed_use_new_destination')
    for source in links:
        target, resolved = dest / source.name, source.resolve(strict=True)
        if target.is_symlink():
            if target.resolve(strict=True) != resolved:
                raise ValueError('mounted_view_link_target_mismatch')
        elif target.exists():
            raise ValueError('mounted_view_would_overwrite_existing_file')
        else:
            target.symlink_to(resolved)
    _atomic_json(dest / 'manifest.json', merged)
    return merged
