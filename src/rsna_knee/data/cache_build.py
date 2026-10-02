"""Build study tensors from DICOM using the same preprocess as inference.

Output per study: float/uint8 array [slots, centers, 3, H, W].
A missing study is quarantined. A missing series is a slot mask, not a fake study of zeros.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from rsna_knee.data.cache import CacheIndex, write_shard
from rsna_knee.data.dicom import SliceGeom, order_slices
from rsna_knee.data.preprocess import PREPROCESS_VERSION, adjacent_triplet, preprocess_hash, process_slice
from rsna_knee.hashing import sha256_file, sha256_json
from rsna_knee.ontology import SERIES_ID_COL, STUDY_ID_COL

PILOT_SLOTS: tuple[tuple[str, str], ...] = (
    ("Sagittal", "fluid"),
    ("Coronal", "fluid"),
    ("Axial", "fluid"),
)


def select_series_for_slots(series_rows: list[dict[str, str]], slots: tuple[tuple[str, str], ...] = PILOT_SLOTS) -> list[dict[str, str] | None]:
    chosen: list[dict[str, str] | None] = []
    used: set[str] = set()
    for plane, kind in slots:
        match = None
        for row in series_rows:
            sid = row.get(SERIES_ID_COL, "")
            if sid in used:
                continue
            if row.get("Anatomical_Plane") != plane:
                continue
            fluid = row.get("Fluid_Sensitive") == "1"
            if kind == "fluid" and not fluid:
                continue
            if kind == "not_fluid" and fluid:
                continue
            match = row
            break
        if match is None:
            for row in series_rows:
                sid = row.get(SERIES_ID_COL, "")
                if sid not in used and row.get("Anatomical_Plane") == plane:
                    match = row
                    break
        if match is not None:
            used.add(match.get(SERIES_ID_COL, ""))
        chosen.append(match)
    return chosen


def _read_series_csv(path: Path) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped.setdefault(row[STUDY_ID_COL], []).append(row)
    return grouped


def _load_slices(series_dir: Path) -> tuple[list[tuple[np.ndarray, SliceGeom]], str | None]:
    try:
        import pydicom
    except ImportError:
        return [], "pydicom_missing"
    geoms: list[tuple[np.ndarray, SliceGeom]] = []
    files = sorted(series_dir.glob("*.dcm"))
    if not files:
        return [], "no_dicom_files"
    for path in files:
        try:
            ds = pydicom.dcmread(str(path), force=True)
            syntax = str(getattr(ds.file_meta, "TransferSyntaxUID", ""))
            try:
                pixels = np.asarray(ds.pixel_array, dtype=np.float32)
            except Exception as exc:  # noqa: BLE001
                return [], f"codec_or_decode:{syntax}:{exc.__class__.__name__}"
        except Exception as exc:  # noqa: BLE001
            return [], f"dicom_read:{exc.__class__.__name__}"
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
            photometric=str(getattr(ds, "PhotometricInterpretation", "") or "") or None,
        )
        geoms.append((pixels, geom))
    return geoms, None


def series_to_centers(pairs: list[tuple[np.ndarray, SliceGeom]], *, size: int, n_centers: int) -> tuple[np.ndarray, str]:
    ordered, mode = order_slices([g for _, g in pairs])
    lookup = {g.path: arr for arr, g in pairs}
    processed = []
    for geom in ordered:
        img, _meta = process_slice(
            lookup[geom.path],
            slope=geom.slope,
            intercept=geom.intercept,
            photometric=geom.photometric,
            spacing=geom.pixel_spacing,
            size=size,
        )
        processed.append(img)
    stack = np.stack(processed, axis=0)
    if n_centers <= 1:
        centers_idx = [stack.shape[0] // 2]
    else:
        centers_idx = np.linspace(0, stack.shape[0] - 1, n_centers).astype(int).tolist()
    triplets = [adjacent_triplet(stack, int(c)) for c in centers_idx]
    # [centers, 3, H, W]
    return np.stack(triplets, axis=0).astype(np.float32), mode


def build_cache(
    *,
    series_csv: Path,
    dicom_root: Path,
    dest: Path,
    study_ids: list[str] | None = None,
    limit: int | None = None,
    size: int = 224,
    n_centers: int = 3,
    layout: str = "study/series",
    resume: bool = True,
) -> dict[str, Any]:
    """Decode only studies that exist under dicom_root. Never downloads the archive."""
    grouped = _read_series_csv(series_csv)
    uids = study_ids or list(grouped)
    if limit is not None:
        uids = uids[:limit]
    dest.mkdir(parents=True, exist_ok=True)
    index = CacheIndex(dest)
    studies: dict[str, Any] = {}
    quarantine: dict[str, Any] = {}
    source_hashes = {"series_csv": sha256_file(series_csv)}
    pp_hash = preprocess_hash({"size": size, "n_centers": n_centers, "slots": PILOT_SLOTS})
    for uid in uids:
        prior = (index.data.get("studies") or {}).get(uid) if resume else None
        if prior and Path(str(prior.get("shard", ""))).is_file():
            studies[uid] = prior.get("meta") or {"uid": uid, "resumed": True}
            continue
        rows = grouped.get(uid)
        if not rows:
            quarantine[uid] = {"reason": "missing_study_metadata"}
            index.record_fail(uid, "missing_study_metadata")
            continue
        study_dir = dicom_root / uid if layout == "study/series" else dicom_root
        if layout == "study/series" and not study_dir.exists():
            quarantine[uid] = {"reason": "missing_study_dicom"}
            index.record_fail(uid, "missing_study_dicom")
            continue
        selected = select_series_for_slots(rows)
        slot_arrays: list[np.ndarray | None] = []
        slot_mask: list[float] = []
        order_mode = "none"
        fatal: dict[str, str] | None = None
        for slot in selected:
            if slot is None:
                slot_arrays.append(None)
                slot_mask.append(0.0)
                continue
            series_dir = study_dir / slot[SERIES_ID_COL]
            pairs, err = _load_slices(series_dir)
            if err and (err.startswith("codec") or err.startswith("dicom_read")):
                fatal = {"reason": err, "series": slot[SERIES_ID_COL]}
                break
            if err or not pairs:
                slot_arrays.append(None)
                slot_mask.append(0.0)
                continue
            try:
                centers, order_mode = series_to_centers(pairs, size=size, n_centers=n_centers)
            except Exception as exc:  # noqa: BLE001
                fatal = {"reason": f"geometry:{exc.__class__.__name__}", "series": slot[SERIES_ID_COL]}
                break
            slot_arrays.append(centers)
            slot_mask.append(1.0)
        if fatal:
            quarantine[uid] = fatal
            index.record_fail(uid, fatal["reason"], fatal.get("series"))
            continue
        present = [arr for arr in slot_arrays if arr is not None]
        if not present:
            quarantine[uid] = {"reason": "no_decodable_series"}
            index.record_fail(uid, "no_decodable_series")
            continue
        reference = present[0]
        tensor_slots = [arr if arr is not None else np.zeros_like(reference) for arr in slot_arrays]
        final_mask = slot_mask
        volume = np.stack(tensor_slots, axis=0)  # [slots, centers, 3, H, W]
        meta = {
            "uid": uid,
            "shape": list(volume.shape),
            "dtype": "uint8",
            "preprocess_version": PREPROCESS_VERSION,
            "preprocess_hash": pp_hash,
            "slot_mask": final_mask,
            "order": order_mode,
            "source_hashes": source_hashes,
            "synthetic": False,
        }
        path = write_shard(uid, volume, meta, root=dest, dtype="uint8")
        meta["sha256"] = sha256_file(path)
        studies[uid] = meta
        index.record_ok(uid, path, meta)
    manifest = {
        "preprocess_version": PREPROCESS_VERSION,
        "preprocess_hash": pp_hash,
        "source_hashes": source_hashes,
        "studies": studies,
        "quarantine": quarantine,
        "n_studies": len(studies),
        "n_quarantine": len(quarantine),
    }
    man_path = dest / "manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest_sha256"] = sha256_json({"studies": list(studies), "quarantine": quarantine, "preprocess_hash": pp_hash})
    manifest["path"] = str(man_path)
    return manifest
