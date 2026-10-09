import numpy as np
import pytest
from rsna_knee.runtime.train_payload import LazyStudyBatch, shard_path, load_volume


def test_compressed_batch_equals_legacy_uint8(tmp_path):
    rng=np.random.default_rng(2);a=rng.integers(0,256,(3,12,3,28,28),dtype=np.uint8)
    np.save(tmp_path/'legacy.npy',a);np.savez_compressed(tmp_path/'dense.npz',images=a)
    meta={'shape':list(a.shape),'slot_mask':[1,0,1]}
    legacy=LazyStudyBatch('legacy',meta,{},tmp_path)
    dense=LazyStudyBatch('dense',{**meta,'storage_format':'npz','shard_filename':'dense.npz'},{},tmp_path)
    assert np.array_equal(legacy['images'].numpy(),dense['images'].numpy())
    assert tuple(dense['images'].shape)==(1,3,12,3,28,28)


def test_storage_paths_and_archive_keys_are_strict(tmp_path):
    with pytest.raises(ValueError):shard_path('../outside',{},tmp_path)
    with pytest.raises(ValueError):shard_path('study',{'shard_filename':'other.npz','storage_format':'npz'},tmp_path)
    np.savez(tmp_path/'study.npz',unexpected=np.zeros(1))
    with pytest.raises(ValueError,match='only images'):load_volume('study',{'storage_format':'npz'},tmp_path)
