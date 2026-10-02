"""Build a study cache from in-memory slices (tests) or pydicom files."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rsna_knee.data.cache import write_shard
from rsna_knee.data.dicom import SliceGeom, order_slices
from rsna_knee.data.preprocess import adjacent_triplet, process_slice, select_centers


def arrays_to_study(
    uid: str,
    slices: list[tuple[np.ndarray, SliceGeom]],
    dest: Path,
    size: int = 32,
) -> Path:
    ordered, mode = order_slices([g for _, g in slices])
    geom_to_arr = {g.path: arr for arr, g in slices}
    stack = []
    metas = []
    for g in ordered:
        arr = geom_to_arr[g.path]
        proc, meta = process_slice(
            arr,
            slope=g.slope,
            intercept=g.intercept,
            photometric=g.photometric,
            spacing=g.pixel_spacing,
            size=size,
        )
        stack.append(proc)
        metas.append(meta)
    vol = np.stack(stack, axis=0)
    center = select_centers(len(vol), 1)[0]
    trip = adjacent_triplet(vol, center)
    # [1 slot, 3 slices, H, W]
    out = trip[None]
    return write_shard(uid, out, {"order": mode, "n": len(vol), "metas": metas}, root=dest, dtype="uint8")


def read_dicom_dir(series_dir: Path) -> list[tuple[np.ndarray, SliceGeom]]:
    try:
        import pydicom
    except ImportError as exc:
        raise RuntimeError("pydicom extra required") from exc
    out: list[tuple[np.ndarray, SliceGeom]] = []
    for path in series_dir.glob("*.dcm"):
        ds = pydicom.dcmread(str(path), force=True)
        pix = ds.pixel_array.astype(np.float32)
        iop = tuple(float(x) for x in ds.ImageOrientationPatient) if "ImageOrientationPatient" in ds else None
        ipp = tuple(float(x) for x in ds.ImagePositionPatient) if "ImagePositionPatient" in ds else None
        spacing = None
        if "PixelSpacing" in ds:
            spacing = (float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1]))
        geom = SliceGeom(
            path=str(path),
            iop=iop,
            ipp=ipp,
            instance_number=int(getattr(ds, "InstanceNumber", 0) or 0) or None,
            sop=str(getattr(ds, "SOPInstanceUID", path.name)),
            pixel_spacing=spacing,
            slope=float(getattr(ds, "RescaleSlope", 1.0) or 1.0),
            intercept=float(getattr(ds, "RescaleIntercept", 0.0) or 0.0),
            photometric=str(getattr(ds, "PhotometricInterpretation", "") or None),
            rows=int(getattr(ds, "Rows", pix.shape[0])),
            cols=int(getattr(ds, "Columns", pix.shape[1])),
        )
        out.append((pix, geom))
    return out
