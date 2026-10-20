"""Resource pilots cannot claim AUC or widen the train-only image cohort."""
import json
import pytest
from rsna_knee.data.calibration_view import calibration_view
from rsna_knee.data.volume_cache import build_volume_cache, volume_manifest_ready
from rsna_knee.hashing import sha256_file
from tests.test_k01 import fixture


def test_train_subset_retains_pixel_qc_and_rejects_rebinding(tmp_path):
    source = tmp_path / 'source'
    build_volume_cache(**fixture(tmp_path), dest=source)
    digest = sha256_file(source / 'manifest.json')
    view = tmp_path / 'pilot'
    path = calibration_view([source], view, expected_hashes=[digest], uids=['1.2.1'])
    m = json.loads(path.read_text())
    assert set(m['studies']) == {'1.2.1'}
    assert set(p.name for p in view.glob('*.npz')) == {'1.2.1.npz'}
    assert (view / '1.2.1.npz').is_symlink()
    assert volume_manifest_ready(path)[0]
    with pytest.raises(ValueError, match='binding_changed'):
        calibration_view([source], view, expected_hashes=[digest], uids=['1.2.2'])
    with pytest.raises(ValueError, match='scope_missing'):
        calibration_view([source], tmp_path / 'missing', expected_hashes=[digest], uids=['gold-not-in-cache'])
    data = source / '1.2.1.npz'; original = data.read_bytes()
    data.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    assert not volume_manifest_ready(path)[0], 'subset must still hash actual pixel payloads'


def test_new_backbone_requires_its_pretrained_normalization(tmp_path):
    import torch
    from rsna_knee.submission.package import _kneenet_checkpoint_errors
    from pathlib import Path
    import shutil
    source = Path(__file__).resolve().parents[1] / 'state/external-code/goodpjw2008-reader/knee.py'
    shutil.copy(source, tmp_path / 'knee.py')
    blob = {'step': 1, 'reader_code_sha256': sha256_file(source),
            'kneenet': {'backbone': 'tf_efficientnetv2_s', 'dim': 384, 'depth': 2, 'res': 256}}
    # Normalization errors are rejected before any weight or import work.
    assert 'checkpoint_input_normalization' in _kneenet_checkpoint_errors(tmp_path / 'synthetic.pt', blob)
    blob['input_normalization'] = {'mean': [.485, .456, .406], 'std': [.229, .224, .225]}
    assert 'checkpoint_input_normalization' in _kneenet_checkpoint_errors(tmp_path / 'synthetic.pt', blob)
    blob['input_normalization'] = {'mean': [.5] * 3, 'std': [.5] * 3}
    assert _kneenet_checkpoint_errors(tmp_path / 'synthetic.pt', blob) == ['checkpoint_state_missing']
