"""K0 pool leakage and K1 anatomical/physical geometry, integrity and sampling."""
from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pydicom
import pytest
import yaml

from rsna_knee.data.cache import CacheBudgetExceeded
from rsna_knee.data.dicom import SliceGeom
from rsna_knee.data.soft_pool import soft_table, soft_training_uids
from rsna_knee.data.volume_cache import (
    build_volume_cache,
    canonical_volume,
    merge_volume_parts,
    orient_slice,
    physical_resample,
    volume_manifest_ready,
)
from rsna_knee.data.volume_dataset import VolumeStudy
from rsna_knee.ontology import TARGET_COLUMNS
from tests.test_cache_manifest import _write_slice

IOPS = [(0, 1, 0, 0, 0, -1), (1, 0, 0, 0, 0, -1), (1, 0, 0, 0, 1, 0)]
PLANES = ['Sagittal', 'Coronal', 'Axial']


def test_k0_k1_keep_labels_schedule_seed_and_forward_resolution_matched():
    root = Path(__file__).resolve().parents[1]
    a, b = (yaml.safe_load((root / f'configs/experiments/P5-{arm}-kneenet-fold0.yaml').read_text()) for arm in ('K0', 'K1'))
    for key in ('train', 'labels', 'weights', 'reader_code', 'folds', 'outer_fold', 'uid_selection', 'eval_precision'):
        assert a[key] == b[key], key
    assert a['train']['res'] == 256 and b['cache']['img_size'] == 384
    assert b['control'] == a['experiment_id'] and a['control'] == 'K0BASE'


def test_paired_bootstrap_handles_zero_auc_and_uses_requested_draw_count():
    from scripts.review_experiment import paired_delta
    y = np.tile([0., 1.], 20)[:, None]
    result = paired_delta(y, y, 1 - y, n_boot=25)
    assert result['delta_unrounded'] == 1 and result['n_boot'] == 25
    assert result['ci95_unrounded'] == [1., 1.]


def targets(uid, value=.5):
    return {'StudyInstanceUID': uid, 'soft_targets': dict.fromkeys(TARGET_COLUMNS, value)}


def test_pool_uses_soft_coverage_and_excludes_gold_groups_and_outer_fold():
    folds = [{'StudyInstanceUID': uid, 'fold': fold, 'group_id': group, 'gold_split': split}
             for uid, fold, group, split in [('gold', '1', 'dup', 'eval'), ('sibling', '1', 'dup', 'weak'),
                                            ('outer', '0', 'outer', 'weak'), ('new', '2', 'new', 'weak'),
                                            ('masked', '2', 'masked', 'weak')]]
    data = '\n'.join(json.dumps(targets(u, -1 if u == 'masked' else .5)) for u in
                     ['gold', 'sibling', 'outer', 'new', 'masked']).encode()
    assert soft_training_uids(folds, data, '0') == ['new']
    assert soft_training_uids(folds, data, 'none') == ['new', 'outer']
    assert soft_table(data)['masked'][TARGET_COLUMNS[0]] is None
    with pytest.raises(ValueError, match='duplicate'):
        soft_table(data + b'\n' + json.dumps(targets('new')).encode())
    for invalid in (float('nan'), float('inf'), -2, 1.01):
        with pytest.raises(ValueError, match='invalid soft'):
            soft_table(json.dumps(targets('bad', invalid)).encode())


@pytest.mark.parametrize('plane', range(3))
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('flip_r,flip_c', [(False, False), (True, False), (False, True), (True, True)])
def test_canonical_anatomical_phantom_recovers_all_signed_permutations(plane, transpose, flip_r, flip_c):
    expected = np.arange(20, dtype=np.float32).reshape(4, 5)
    image = expected.copy()
    r, c = np.array(IOPS[plane][:3]), np.array(IOPS[plane][3:])
    spacing = (1., 2.)
    if flip_r:
        image, c = image[::-1], -c
    if flip_c:
        image, r = image[:, ::-1], -r
    if transpose:
        image, r, c, spacing = image.T, c, r, spacing[::-1]
    g = SliceGeom('p', tuple(r) + tuple(c), (0, 0, 0), 1, 'sop', spacing)
    actual, ps, _ = orient_slice(image, g, plane)
    np.testing.assert_array_equal(actual, expected)
    assert ps == (1., 2.)


def test_small_fov_is_padded_at_fixed_physical_scale():
    a, details = physical_resample(np.ones((4, 8), np.float32), (2., 1.), size=16, crop_mm=16.)
    assert details['resampled_shape'] == [8, 8]
    assert a[4:12, 4:12].min() == 1 and a.sum() == 64


def test_volume_caps_slices_and_records_fallback_and_rescale_semantics():
    headers = [SliceGeom(str(i), IOPS[2], (0, 0, float(i)), i + 1, str(i), (1., 1.),
                         slope=-2, intercept=5, photometric='MONOCHROME1') for i in range(70)]
    pixels = np.arange(64, dtype=np.float32).reshape(8, 8)
    volume, details = canonical_volume(headers, lambda g: pixels + int(g.path), 'Axial', size=8, crop_mm=8)
    assert len(volume) == 64 and details['n_raw'] == 70
    assert details['retained_indices'][0] == 0 and details['retained_indices'][-1] == 69
    assert details['anatomical_order_verified']
    # Negative rescale followed by MONOCHROME1 inversion restores increasing within-slice intensities.
    assert volume[0, -1, -1] > volume[0, 0, 0]
    fallback = [replace(g, ipp=None) for g in headers[:3]]
    _, details = canonical_volume(fallback, lambda g: pixels, 'Axial', size=8, crop_mm=8)
    assert details['order'] == 'instance_number_fallback' and not details['anatomical_order_verified']


@pytest.mark.parametrize('plane', range(3))
@pytest.mark.parametrize('reverse', [False, True])
def test_slice_direction_does_not_depend_on_iop_normal_sign(plane, reverse):
    headers, arrays = [], {}
    for i in range(7):
        ipp = [0., 0., 0.]
        ipp[plane] = float(i)
        iop = IOPS[plane]
        a = np.arange(64, dtype=np.float32).reshape(8, 8) + i * 100
        if reverse:
            iop = tuple(-v for v in iop[:3]) + iop[3:]
            a = a[:, ::-1]
        g = SliceGeom(str(i), iop, tuple(ipp), i, str(i), (1., 1.))
        headers.insert(0, g)
        arrays[g.path] = a
    volume, info = canonical_volume(headers, lambda g: arrays[g.path], PLANES[plane], size=8, crop_mm=8)
    sign = -1 if plane == 2 else 1
    assert np.all(np.diff(volume.mean(axis=(1, 2))) * sign > 0)
    assert info['n_raw'] == info['n'] == 7
    broken = [replace(g, iop=None) for g in headers]
    with pytest.raises(ValueError, match='missing_iop'):
        canonical_volume(broken, lambda g: arrays[g.path], PLANES[plane])
    with pytest.raises(ValueError, match='empty_intensity'):
        canonical_volume(headers, lambda g: np.zeros((8, 8)), PLANES[plane], size=8, crop_mm=8)


def fixture(tmp_path):
    rows = []
    for uid in ['1.2.1', '1.2.2']:
        for s, fs in [('fs-a', '1'), ('fs-b', '1'), ('nonfs', '0'), ('broken', '0')]:
            folder = tmp_path / 'dicom' / uid / s
            folder.mkdir(parents=True)
            for i in range(9):
                p = folder / f'{8-i}.dcm'
                _write_slice(p, uid, s, i + 1, i * 2)
                ds = pydicom.dcmread(p)
                ds.PixelSpacing = [1., 1.]
                if s == 'broken':
                    del ds.ImageOrientationPatient
                ds.save_as(p, enforce_file_format=True)
            rows.append({'StudyInstanceUID': uid, 'SeriesInstanceUID': s, 'Anatomical_Plane': 'Axial',
                         'Fluid_Sensitive': '0', 'Fat_Suppression': fs})
    csv_path = tmp_path / 'series.csv'
    with csv_path.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return dict(series_csv=csv_path, dicom_root=tmp_path / 'dicom', size=8, crop_mm=8.)


def test_cache_all_candidates_fs_slots_resume_integrity_budget_and_merge(tmp_path):
    kwargs = fixture(tmp_path)
    first = build_volume_cache(**kwargs, dest=tmp_path / 'one', study_ids=['1.2.1'])
    assert volume_manifest_ready(tmp_path / 'one/manifest.json') == (True, 'volume_cache_verified')
    meta = first['studies']['1.2.1']
    assert len(meta['candidates'][2]) == 2 and len(meta['candidates'][5]) == 1
    assert len(meta['series_failures']) == 1
    build_volume_cache(**kwargs, dest=tmp_path / 'one', study_ids=['1.2.1'])
    assert json.loads((tmp_path / 'one/build-summary.json').read_text())['reused'] == 1
    build_volume_cache(**kwargs, dest=tmp_path / 'two', study_ids=['1.2.2'])
    merge_volume_parts([tmp_path / 'one', tmp_path / 'two'], tmp_path / 'merged')
    assert volume_manifest_ready(tmp_path / 'merged/manifest.json', ['1.2.1', '1.2.2'])[0]
    assert not volume_manifest_ready(tmp_path / 'one/manifest.json', ['absent'])[0]
    with pytest.raises(CacheBudgetExceeded):
        build_volume_cache(**kwargs, dest=tmp_path / 'budget', max_cache_bytes=1)
    assert not list((tmp_path / 'budget').glob('*.npz'))
    (tmp_path / 'one/1.2.1.npz').write_bytes(b'corrupt')
    assert not volume_manifest_ready(tmp_path / 'one/manifest.json')[0]
    build_volume_cache(**kwargs, dest=tmp_path / 'one', study_ids=['1.2.1'])
    assert volume_manifest_ready(tmp_path / 'one/manifest.json')[0]
    with pytest.raises(ValueError, match='changed'):
        build_volume_cache(**(kwargs | {'size': 16}), dest=tmp_path / 'one')


def test_volume_sampling_true_positions_masks_and_deterministic_eval(tmp_path):
    m = build_volume_cache(**fixture(tmp_path), dest=tmp_path / 'cache')
    table = soft_table(json.dumps(targets('1.2.1')).encode())
    kwargs = dict(studies=m['studies'], cache=tmp_path / 'cache', size=8, targets=table)
    ds = VolumeStudy(['1.2.1'], 4, False, **kwargs)
    a, b = ds[0], ds[0]
    assert all(np.array_equal(x, y) for x, y in zip(a[:-1], b[:-1], strict=True))
    x, mask, pos, y, obs, uid = a
    assert x.shape == (6, 4, 3, 8, 8) and mask.tolist() == [False, False, True, False, False, True]
    assert np.allclose(pos[2], np.array([1, 3, 5, 7]) / 8)
    assert obs.sum() == 12
    train = VolumeStudy(['1.2.1'], 4, True, **kwargs)
    np.random.seed(2026)
    samples = [train[0][2].numpy().tobytes() for _ in range(20)]
    assert len(set(samples)) > 1
    missing = VolumeStudy(['1.2.2'], 4, False, **kwargs)[0]
    assert missing[4].sum() == 0
    masked = VolumeStudy(['1.2.1'], 4, False, **(kwargs | {'targets': {'1.2.1': dict.fromkeys(TARGET_COLUMNS, -1)}}))[0]
    assert masked[3].sum() == masked[4].sum() == 0


def test_readonly_mounted_view_preserves_pixels_without_copying_payloads(tmp_path):
    from rsna_knee.data.volume_mounts import mounted_volume_view
    from rsna_knee.hashing import sha256_file

    kwargs = fixture(tmp_path)
    parts = [tmp_path / 'part0', tmp_path / 'part1']
    for part, uid in zip(parts, ['1.2.1', '1.2.2'], strict=True):
        build_volume_cache(**kwargs, dest=part, study_ids=[uid])
    hashes = [sha256_file(p / 'manifest.json') for p in parts]
    view = tmp_path / 'view'
    merged = mounted_volume_view(parts, view, expected_hashes=hashes)
    assert volume_manifest_ready(view / 'manifest.json', ['1.2.1', '1.2.2'])[0]
    assert all(p.is_symlink() for p in view.glob('*.npz'))
    assert (view / '1.2.1.npz').resolve() == (parts[0] / '1.2.1.npz').resolve()
    assert mounted_volume_view(parts, view, expected_hashes=hashes) == merged
    original = json.loads((parts[0] / 'manifest.json').read_text())
    table = soft_table(json.dumps(targets('1.2.1')).encode())
    before = VolumeStudy(['1.2.1'], 4, False, studies=original['studies'], cache=parts[0], size=8, targets=table)[0]
    after = VolumeStudy(['1.2.1'], 4, False, studies=merged['studies'], cache=view, size=8, targets=table)[0]
    assert all(np.array_equal(a, b) for a, b in zip(before[:-1], after[:-1], strict=True))
    with pytest.raises(ValueError, match='hash_mismatch'):
        mounted_volume_view(parts, tmp_path / 'bad', expected_hashes=['incorrect', hashes[1]])
    assert not (tmp_path / 'bad').exists()
    with pytest.raises(ValueError, match='overlapping'):
        mounted_volume_view([parts[0], parts[0]], tmp_path / 'overlap', expected_hashes=[hashes[0]] * 2)
    data = bytearray((parts[0] / '1.2.1.npz').read_bytes())
    data[len(data) // 2] ^= 1
    (parts[0] / '1.2.1.npz').write_bytes(data)
    assert not volume_manifest_ready(view / 'manifest.json')[0]
