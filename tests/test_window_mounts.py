"""v5 mounted cache integrity and bounded disk use; no production metrics."""
import csv
import json

import pytest

from rsna_knee.data.cache_build import build_cache, manifest_ready, merge_cache_parts
from rsna_knee.data.window_mounts import mounted_window_view
from rsna_knee.hashing import sha256_file
from tests.test_cache_manifest import _write_slice


def test_readonly_window_view_matches_copy_merge_and_rejects_tampering(tmp_path):
    rows = []
    for uid in ('1.2.100', '1.2.200'):
        folder = tmp_path / 'dicom' / uid / '1.2.3'
        folder.mkdir(parents=True)
        for i in range(1, 5):
            _write_slice(folder / f'{i}.dcm', uid, '1.2.3', i, i * 3.)
        rows.append({'StudyInstanceUID': uid, 'SeriesInstanceUID': '1.2.3',
                     'Anatomical_Plane': 'Sagittal', 'Fluid_Sensitive': '1', 'Fat_Suppression': '1'})
    series = tmp_path / 'series.csv'
    with series.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    parts = [tmp_path / f'part{i}' for i in range(2)]
    for part, row in zip(parts, rows):
        build_cache(series_csv=series, dicom_root=tmp_path / 'dicom', dest=part,
                    study_ids=[row['StudyInstanceUID']], size=28, n_centers=3)
        assert manifest_ready(part / 'manifest.json')[0]
    hashes = [sha256_file(p / 'manifest.json') for p in parts]
    dest = tmp_path / 'mounted'
    merged = mounted_window_view(parts, dest, expected_hashes=hashes)
    assert merged == merge_cache_parts(parts, tmp_path / 'copied', move=False)
    assert manifest_ready(dest / 'manifest.json')[0]
    assert all((dest / f"{r['StudyInstanceUID']}.{s}").is_symlink() for r in rows for s in ('npz', 'json'))
    assert mounted_window_view(parts, dest, expected_hashes=hashes) == merged
    with pytest.raises(ValueError, match='overlapping'):
        mounted_window_view([parts[0], parts[0]], tmp_path / 'overlap', expected_hashes=[hashes[0]] * 2)
    data = parts[0] / '1.2.100.npz'
    original = data.read_bytes()
    data.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    assert not manifest_ready(dest / 'manifest.json')[0], 'linked payload corruption must fail QC'
    path = parts[0] / 'manifest.json'
    m = json.loads(path.read_text());m['n_requested'] += 1;path.write_text(json.dumps(m))
    with pytest.raises(ValueError, match='hash_mismatch'):
        mounted_window_view(parts, tmp_path / 'changed', expected_hashes=hashes)
