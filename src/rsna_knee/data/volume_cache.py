"""K1 canonical full-series cache. No reports, labels, training or downloads.

Axes/FOV/window match the audited reader; malformed geometry is quarantined
instead of silently becoming a successful zero volume. Each study is a ragged
NPZ of every usable series; training draws candidates and windows dynamically.
"""
from __future__ import annotations

import importlib.metadata
import io
import json
import time
from pathlib import Path

import numpy as np

from rsna_knee.data.cache import ByteBudget
from rsna_knee.data.cache_build import (
    _atomic_json,
    _decode_pixels,
    _load_headers,
    _read_series_csv,
    _safe_uid,
)
from rsna_knee.data.dicom import apply_slope_intercept, order_slices, slice_normal
from rsna_knee.hashing import sha256_file, sha256_json

VERSION = 'k1-canonical-volumes-v1'
PLANES = ('Sagittal', 'Coronal', 'Axial')
# slice direction; column LPS axis/sign; row LPS axis/sign
CANON = ((1, (1, 1), (2, -1)), (1, (0, 1), (2, -1)), (-1, (0, 1), (1, 1)))


def policy(size=384, crop_mm=153.6, max_slices=64):
    if size < 4 or crop_mm <= 0 or max_slices < 3:
        raise ValueError('invalid volume policy')
    here = Path(__file__).parent
    return {'version': VERSION, 'size': size, 'crop_mm': crop_mm, 'max_slices': max_slices,
            'slots': [[p, fs] for fs in (1, 0) for p in PLANES], 'percentiles': [1, 99.5],
            'code_sha256': {n: sha256_file(here / n) for n in
                            ('volume_cache.py', 'cache_build.py', 'dicom.py', 'cache.py')},
            'runtime_versions': {n: importlib.metadata.version(n) for n in
                                 ('numpy', 'pydicom', 'opencv-python-headless')}}


def orient_slice(image, geom, plane):
    """Canonical signed permutation, including swapped anisotropic spacing."""
    if geom.iop is None or len(geom.iop) != 6:
        raise ValueError('missing_iop')
    spacing = geom.pixel_spacing
    if spacing is None or len(spacing) != 2 or min(spacing) <= 0:
        raise ValueError('missing_or_invalid_spacing')
    r, c = np.asarray(geom.iop[:3]), np.asarray(geom.iop[3:])
    _, (cax, csign), (rax, rsign) = CANON[plane]
    transpose = bool(abs(r[cax]) < abs(r[rax]))
    if transpose:
        image, r, c, spacing = image.T, c, r, spacing[::-1]
    # Reject off-plane/unusable axes; do not claim an oblique reslicing operation.
    if abs(r[cax]) < .7 or abs(c[rax]) < .7:
        raise ValueError('unsupported_in_plane_geometry')
    flip_col, flip_row = bool(r[cax] * csign < 0), bool(c[rax] * rsign < 0)
    if flip_col:
        image = image[:, ::-1]
    if flip_row:
        image = image[::-1]
    return image, spacing, {'transpose': transpose, 'flip_col': flip_col, 'flip_row': flip_row}


def physical_resample(image, spacing, size, crop_mm):
    import cv2
    mm_per_px = crop_mm / size
    h, w = image.shape
    ch, cw = min(h, max(1, round(crop_mm / spacing[0]))), min(w, max(1, round(crop_mm / spacing[1])))
    y, x = (h - ch) // 2, (w - cw) // 2
    nh, nw = max(1, min(size, round(ch * spacing[0] / mm_per_px))), max(1, min(size, round(cw * spacing[1] / mm_per_px)))
    crop = np.ascontiguousarray(image[y:y + ch, x:x + cw])
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA if nh < ch else cv2.INTER_LINEAR)
    out = np.zeros((size, size), np.float32)
    oy, ox = (size - nh) // 2, (size - nw) // 2
    out[oy:oy + nh, ox:ox + nw] = resized
    return out, {'resampled_shape': [nh, nw], 'padding_pixels': [size - nh, size - nw]}


def canonical_volume(headers, get_pixels, plane_name, *, size=384, crop_mm=153.6, max_slices=64):
    if len(headers) < 3:
        raise ValueError('fewer_than_three_slices')
    normals = [slice_normal(g.iop) for g in headers]
    if any(n is None for n in normals):
        raise ValueError('missing_iop')
    ref = normals[0]
    plane = int(np.abs(ref).argmax())
    if PLANES[plane] != plane_name:
        raise ValueError('plane_metadata_geometry_mismatch')
    # A shared normal is required; abs(dot) would accept reversed orientations.
    if any(float(np.dot(ref, n)) < .95 for n in normals):
        raise ValueError('incoherent_orientation')
    if any(not np.allclose(g.iop, headers[0].iop, atol=.02) for g in headers):
        raise ValueError('incoherent_in_plane_orientation')
    if len({g.sop for g in headers}) != len(headers):
        raise ValueError('duplicate_sop')
    ordered, order = order_slices(headers)
    # InstanceNumber is an explicitly recorded fallback when IPP is unavailable.
    if order not in {'iop_ipp', 'instance_number_fallback'}:
        raise ValueError('missing_slice_order')
    reversed_order = order == 'iop_ipp' and CANON[plane][0] * np.sign(ref[plane]) < 0
    if reversed_order:
        ordered = ordered[::-1]
    if order == 'iop_ipp':
        projected = np.array([np.dot(ref, g.ipp) for g in ordered])
        if np.any(np.abs(np.diff(projected)) < 1e-5):
            raise ValueError('duplicate_slice_position')
    else:
        projected = None
    keep = np.rint(np.linspace(0, len(ordered) - 1, min(len(ordered), max_slices))).astype(int)
    images, transforms, resamples = [], [], []
    for i in keep:
        g = ordered[i]
        image = apply_slope_intercept(get_pixels(g), g.slope, g.intercept)
        # Scale first, then invert: honors rescale semantics even with negative slope.
        if (g.photometric or '').upper() == 'MONOCHROME1':
            image = image.max() - image
        image, spacing, transform = orient_slice(image, g, plane)
        image, resample = physical_resample(image, spacing, size, crop_mm)
        images.append(image)
        transforms.append(transform)
        resamples.append(resample)
    volume = np.stack(images)
    if not np.isfinite(volume).all():
        raise ValueError('nonfinite_pixels')
    lo, hi = np.percentile(volume[:, ::4, ::4], [1, 99.5])
    if hi <= lo:
        raise ValueError('empty_intensity_window')
    volume = np.rint(np.clip((volume - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    if not volume.any() or volume.max() == volume.min():
        raise ValueError('constant_or_zero_volume')
    return volume, {'n_raw': len(headers), 'n': len(volume), 'retained_indices': keep.tolist(),
                    'order': order, 'anatomical_order_verified': order == 'iop_ipp',
                    'slice_order_reversed': bool(reversed_order), 'plane': PLANES[plane],
                    'slice_gap_mm': None if projected is None else float(np.median(np.abs(np.diff(projected)))),
                    'intensity_window': [float(lo), float(hi)], 'transforms': transforms, 'resampling': resamples}


def _payload_path(root, filename):
    if Path(filename).name != filename or not filename.endswith('.npz'):
        raise ValueError('unsafe volume shard filename')
    return root / filename


def _check_study(root, meta, size):
    path = _payload_path(root, meta['file'])
    if sha256_file(path) != meta['sha256'] or path.stat().st_size != meta['bytes']:
        raise ValueError('volume_payload_hash_mismatch')
    candidates = meta['candidates']
    if len(candidates) != 6 or not any(candidates):
        raise ValueError('invalid_volume_candidates')
    keys = [c['key'] for slot in candidates for c in slot]
    if len(keys) != len(set(keys)):
        raise ValueError('duplicate_candidate_key')
    with np.load(path, allow_pickle=False) as arrays:
        if set(arrays.files) != set(keys):
            raise ValueError('volume_candidate_payload_mismatch')
        for slot in candidates:
            for c in slot:
                a = arrays[c['key']]
                if a.dtype != np.uint8 or a.shape != (c['n'], size, size) or c['n'] < 3 or not a.any() or a.max() == a.min():
                    raise ValueError('invalid_volume_shape_or_pixels')


def volume_manifest_ready(path, required_uids=None):
    try:
        path = Path(path)
        m = json.loads(path.read_text())
        if m['version'] != VERSION or m['preprocess_hash'] != sha256_json(m['preprocess_config']):
            raise ValueError('volume_policy_hash_mismatch')
        if m['manifest_sha256'] != sha256_json({k: v for k, v in m.items() if k != 'manifest_sha256'}):
            raise ValueError('volume_manifest_hash_mismatch')
        if not m['studies']:
            raise ValueError('no_usable_studies')
        requested = m['requested_uids']
        if len(requested) != len(set(requested)) or set(requested) != set(m['studies']) | set(m['quarantine']):
            raise ValueError('volume_requested_scope_incomplete')
        if set(m['studies']) & set(m['quarantine']):
            raise ValueError('study_in_success_and_quarantine')
        if required_uids is not None and set(required_uids) - set(m['studies']):
            raise ValueError('required_studies_missing_from_volume_cache')
        for meta in m['studies'].values():
            _check_study(path.parent, meta, m['preprocess_config']['size'])
        return True, 'volume_cache_verified'
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return False, str(exc)


def build_volume_cache(*, series_csv, dicom_root, dest, study_ids=None, size=384, crop_mm=153.6,
                       max_slices=64, max_cache_bytes=18 * 1024**3, resume=True):
    started = time.perf_counter()
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    config = policy(size, crop_mm, max_slices)
    digest = sha256_json(config)
    grouped = _read_series_csv(Path(series_csv))
    requested = sorted(grouped if study_ids is None else set(study_ids))
    if not requested or any(not _safe_uid(u) for u in requested):
        raise ValueError('empty_or_unsafe_study_list')
    source_hash = sha256_file(Path(series_csv))
    path = dest / 'manifest.json'
    old = json.loads(path.read_text()) if resume and path.exists() else {}
    if old and (old['preprocess_hash'] != digest or old['series_csv_sha256'] != source_hash):
        raise ValueError('cache_policy_or_metadata_changed_use_new_destination')
    budget = ByteBudget(dest, max_cache_bytes)
    manifest = {'version': VERSION, 'preprocess_config': config, 'preprocess_hash': digest,
                'series_csv_sha256': source_hash, 'requested_uids': requested,
                'studies': {}, 'quarantine': {}, 'scope': 'preprocessing_only'}
    reused = 0
    for uid in requested:
        prior = old.get('studies', {}).get(uid)
        if prior:
            try:
                _check_study(dest, prior, size)
                manifest['studies'][uid] = prior
                reused += 1
                continue
            except (OSError, ValueError, KeyError):
                pass
        arrays, candidates, failures = {}, [[] for _ in range(6)], []
        seen = set()
        for row in grouped.get(uid, []):
            sid = row['SeriesInstanceUID']
            if sid in seen:
                raise ValueError('duplicate_series_metadata')
            seen.add(sid)
            try:
                plane, fs = row['Anatomical_Plane'], row['Fat_Suppression']
                if plane not in PLANES or fs not in {'0', '1'}:
                    raise ValueError('invalid_slot_metadata')
                headers, error = _load_headers(Path(dicom_root) / uid / sid)
                if error:
                    raise ValueError(error)
                volume, details = canonical_volume(headers, lambda g: _decode_pixels(g.path), plane,
                                                   size=size, crop_mm=crop_mm, max_slices=max_slices)
                key = f'v{len(arrays):04d}'
                arrays[key] = volume
                slot = PLANES.index(plane) + 3 * (1 - int(fs))
                candidates[slot].append({'series': sid, 'key': key, **details})
            except (OSError, ValueError, RuntimeError) as exc:
                failures.append({'series': sid, 'error': str(exc)[:200]})
        if not arrays:
            manifest['quarantine'][uid] = {'error': 'no_usable_series', 'series_failures': failures}
        else:
            for slot in candidates:
                slot.sort(key=lambda c: (-c['n'], c['series']))
            buffer = io.BytesIO()
            np.savez_compressed(buffer, **arrays)
            data = buffer.getvalue()
            shard = dest / f'{uid}.npz'
            budget.admit(shard, len(data))
            tmp = shard.with_suffix('.npz.tmp')
            tmp.write_bytes(data)
            tmp.replace(shard)
            manifest['studies'][uid] = {'file': shard.name, 'sha256': sha256_file(shard), 'bytes': len(data),
                                       'candidates': candidates, 'slot_mask': [bool(s) for s in candidates],
                                       'series_failures': failures}
        manifest['manifest_sha256'] = sha256_json({k: v for k, v in manifest.items() if k != 'manifest_sha256'})
        _atomic_json(path, manifest)
    manifest['manifest_sha256'] = sha256_json({k: v for k, v in manifest.items() if k != 'manifest_sha256'})
    _atomic_json(path, manifest)
    _atomic_json(dest / 'build-summary.json', {'n_requested': len(requested), 'n_built': len(manifest['studies']),
                 'n_quarantine': len(manifest['quarantine']), 'reused': reused,
                 'elapsed_seconds': time.perf_counter() - started, 'compressed_bytes': budget.total,
                 'kind': 'real_dicom_preprocessing_no_model_metrics'})
    return manifest


def merge_volume_parts(parts, dest):
    import shutil
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    merged = None
    for part in parts:
        part = Path(part)
        ready, reason = volume_manifest_ready(part / 'manifest.json')
        if not ready:
            raise ValueError(reason)
        m = json.loads((part / 'manifest.json').read_text())
        if merged is None:
            merged = {**m, 'studies': {}, 'quarantine': {}, 'requested_uids': []}
        if any(merged[k] != m[k] for k in ('preprocess_hash', 'series_csv_sha256')):
            raise ValueError('incompatible_volume_parts')
        if set(merged['requested_uids']) & set(m['requested_uids']):
            raise ValueError('overlapping_volume_parts')
        for meta in m['studies'].values():
            shutil.copy2(_payload_path(part, meta['file']), _payload_path(dest, meta['file']))
        merged['studies'].update(m['studies'])
        merged['quarantine'].update(m['quarantine'])
        merged['requested_uids'].extend(m['requested_uids'])
    if merged is None:
        raise ValueError('no_volume_parts')
    merged['requested_uids'].sort()
    merged['manifest_sha256'] = sha256_json({k: v for k, v in merged.items() if k != 'manifest_sha256'})
    _atomic_json(dest / 'manifest.json', merged)
    return merged
