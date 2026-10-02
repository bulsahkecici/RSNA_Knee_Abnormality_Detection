from __future__ import annotations

import numpy as np

from rsna_knee.data.build import arrays_to_study
from rsna_knee.data.cache import load_shard
from rsna_knee.data.dicom import SliceGeom, order_slices
from rsna_knee.data.preprocess import decode_uint8, process_slice


def test_iop_ipp_not_sop_lex():
    slices = []
    for i, z in enumerate([30.0, 10.0, 20.0]):
        slices.append(
            SliceGeom(
                path=f"zzz{i}.dcm" if i == 0 else f"aaa{i}.dcm",
                iop=(1, 0, 0, 0, 1, 0),
                ipp=(0, 0, z),
                instance_number=None,
                sop=f"{9-i}.1.2",
                pixel_spacing=(0.5, 0.5),
            )
        )
    ordered, mode = order_slices(slices)
    assert mode == "iop_ipp"
    zs = [s.ipp[2] for s in ordered]
    assert zs == [10.0, 20.0, 30.0]


def test_monochrome1_and_slope(tmp_path):
    pix = np.arange(64, dtype=np.float32).reshape(8, 8)
    a, meta = process_slice(pix, slope=2.0, intercept=1.0, photometric="MONOCHROME1", spacing=(1.0, 1.0), size=8)
    b, _ = process_slice(pix, slope=2.0, intercept=1.0, photometric="MONOCHROME2", spacing=(1.0, 1.0), size=8)
    assert a.shape == (8, 8)
    assert not np.allclose(a, b)
    geoms = []
    arrays = []
    for i, z in enumerate([0.0, 1.0, 2.0]):
        g = SliceGeom(path=str(i), iop=(1, 0, 0, 0, 1, 0), ipp=(0, 0, z), instance_number=i, sop=str(i), pixel_spacing=(1.0, 1.0))
        geoms.append(g)
        arrays.append((np.full((8, 8), i + 1, dtype=np.float32), g))
    arrays_to_study("u", arrays, tmp_path, size=8)
    shard = np.array(load_shard("u", tmp_path))
    recon = decode_uint8(shard) if shard.max() > 1.5 else shard
    # Re-run preprocess on center slice and compare numerically (uint8 quantization)
    center, _ = process_slice(arrays[1][0], spacing=(1.0, 1.0), size=8)
    cached_center = recon[0, 1]
    assert np.mean(np.abs(cached_center - center)) < 0.02
