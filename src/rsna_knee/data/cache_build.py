"""Versioned resumable DICOM cache with header-first selective decoding.

Output per study: uint8 array [slots, centers, 3, H, W]. Each window is three
physically adjacent slices of one series (IOP x IPP order); windows never cross
a series boundary. A missing study is quarantined; a missing series is a slot
mask, never a fake all-zero success. Use the same settings for training and
inference: the preprocess hash covers the selection policy and this code.

Storage: ``npz`` (v4, compressed, byte-budgeted) for new caches. Manifests from
the v3 ``npy`` builder remain readable by ``manifest_ready``; building v3 caches
is pinned to the archived v3 source bundle, not this module.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from rsna_knee.data.cache import (
    DEFAULT_MAX_CACHE_BYTES,
    ByteBudget,
    CacheBudgetExceeded,
    compressed_byte_total,
    read_shard,
    write_shard,
)
from rsna_knee.data.dicom import SliceGeom, order_slices, projection_key
from rsna_knee.data.preprocess import PREPROCESS_VERSION, preprocess_hash, process_slice
from rsna_knee.hashing import sha256_file, sha256_json
from rsna_knee.ontology import SERIES_ID_COL, STUDY_ID_COL

CACHE_BUILDER_VERSION = "cache-v4-compressed-dense"
LEGACY_BUILDER_VERSIONS = {"cache-v3-selective-inner": "npy"}
PILOT_SLOTS = (("Sagittal", "fluid"), ("Coronal", "fluid"), ("Axial", "fluid"))
STRUCTURAL_SLOTS = PILOT_SLOTS + (("Sagittal", "structural"), ("Coronal", "structural"))
DENSE_N_CENTERS = 12
DENSE_CENTER_RANGE = (0.05, 0.95)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def _safe_uid(uid: str) -> bool:
    return bool(uid) and len(uid) <= 240 and all(c.isalnum() or c in "._-" for c in uid) and uid not in {".", ".."}


def centers_for_series(n: int, k: int = 3, *, strategy: str = "interior", lower: float = 0.15, upper: float = 0.85) -> list[int]:
    if n < 1 or k < 1:
        raise ValueError("Series and center count must be positive")
    if strategy not in {"interior", "legacy"} or not 0 <= lower < upper <= 1:
        raise ValueError("Invalid center strategy or fractions")
    if k == 1:
        return [n // 2]
    if strategy == "legacy":
        return np.linspace(0, n - 1, k).astype(int).tolist()
    positions = np.linspace(lower, upper, k) * (n - 1)
    if n >= 3:
        positions = np.clip(positions, 1, n - 2)
    return np.rint(positions).astype(int).tolist()


def _selection_policy(size, n_centers, strategy, lower, upper, slots):
    here = Path(__file__).parent
    code = {name: sha256_file(here / name) for name in ("cache_build.py", "cache.py", "preprocess.py", "dicom.py")}
    return {"size": size, "n_centers": n_centers, "slots": [list(x) for x in slots], "storage_format": "npz",
            "center_strategy": strategy, "center_lower": lower, "center_upper": upper,
            "crop_mm": 160.0, "clip_percentiles": [0.5, 99.5], "dtype": "uint8",
            "builder_version": CACHE_BUILDER_VERSION, "code_sha256": code,
            "pixel_runtime_versions": {name: importlib.metadata.version(name) for name in ("numpy", "Pillow", "pydicom")}}


def _slot_rank(row: dict[str, str], kind: str, study_dir: Path | None) -> tuple:
    sid = row.get(SERIES_ID_COL, "")
    count = len(list((study_dir / sid).glob("*.dcm"))) if study_dir is not None and _safe_uid(sid) else 0
    desired = (row.get("Fluid_Sensitive") == "1") == (kind == "fluid")
    return (-int(count > 0) if study_dir is not None else 0, -int(desired),
            -int(row.get("Fat_Suppression") == "1"), -count, sid)


def select_series_for_slots(series_rows, slots=PILOT_SLOTS, *, study_dir=None):
    """Pick at most one series per slot; a series never fills two slots.

    Structural slots require explicit ``Fluid_Sensitive == "0"``. A fluid slot
    may fall back to a non-fluid series (v3 behavior) unless the same plane also
    has a structural slot, which then owns the non-fluid series.
    """
    chosen, used = [], set()
    for plane, kind in slots:
        if kind not in {"fluid", "structural"}:
            raise ValueError("Unknown slot kind")
        candidates = [r for r in series_rows if r.get("Anatomical_Plane") == plane
                      and r.get(SERIES_ID_COL) not in used
                      and (kind != "structural" or r.get("Fluid_Sensitive") == "0")
                      and not (kind == "fluid" and (plane, "structural") in slots
                               and r.get("Fluid_Sensitive") == "0")]
        match = min(candidates, key=lambda row, kind=kind: _slot_rank(row, kind, study_dir)) if candidates else None
        if match is not None:
            used.add(match[SERIES_ID_COL])
        chosen.append(match)
    return chosen


def _read_series_csv(path):
    grouped = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not {STUDY_ID_COL, SERIES_ID_COL, "Anatomical_Plane", "Fluid_Sensitive"} <= set(reader.fieldnames or []):
            raise ValueError("train_series.csv is missing required columns")
        for row in reader:
            if not _safe_uid(row[STUDY_ID_COL]) or not _safe_uid(row[SERIES_ID_COL]):
                raise ValueError("Unsafe or empty study/series identifier")
            grouped.setdefault(row[STUDY_ID_COL], []).append(row)
    return grouped


def _geom(ds, path):
    def number(name, default):
        raw = getattr(ds, name, default)
        value = float(default if raw is None else raw)
        if not math.isfinite(value):
            raise ValueError(f"Nonfinite {name}")
        return value
    iop = tuple(float(x) for x in ds.ImageOrientationPatient) if "ImageOrientationPatient" in ds else None
    ipp = tuple(float(x) for x in ds.ImagePositionPatient) if "ImagePositionPatient" in ds else None
    spacing = tuple(float(x) for x in ds.PixelSpacing[:2]) if "PixelSpacing" in ds else None
    if any(not math.isfinite(x) for values in (iop, ipp, spacing) if values is not None for x in values):
        raise ValueError("Nonfinite DICOM geometry")
    return SliceGeom(path=str(path), iop=iop, ipp=ipp,
                     instance_number=int(getattr(ds, "InstanceNumber", 0) or 0) or None,
                     sop=str(getattr(ds, "SOPInstanceUID", path.name)), pixel_spacing=spacing,
                     slope=number("RescaleSlope", 1.0), intercept=number("RescaleIntercept", 0.0),
                     photometric=str(getattr(ds, "PhotometricInterpretation", "") or "") or None,
                     rows=int(getattr(ds, "Rows", 0)), cols=int(getattr(ds, "Columns", 0)))


def _load_headers(series_dir):
    import pydicom
    files = sorted(series_dir.glob("*.dcm"))
    if not files:
        return [], "no_dicom_files"
    geoms = []
    for path in files:
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=False)
            geoms.append(_geom(ds, path))
        except Exception as exc:
            return [], f"dicom_header:{exc.__class__.__name__}"
    return geoms, None


def _decode_pixels(path):
    import pydicom
    ds = pydicom.dcmread(str(path), force=False)
    try:
        arr = np.asarray(ds.pixel_array, dtype=np.float32)
    except Exception as exc:
        raise RuntimeError(f"codec_or_decode:{getattr(ds.file_meta, 'TransferSyntaxUID', '')}:{exc.__class__.__name__}") from exc
    if arr.ndim != 2 or not np.isfinite(arr).all():
        raise ValueError("Expected a finite single-frame 2D image")
    return arr


def physical_geometry(ordered: list[SliceGeom]) -> dict[str, Any]:
    """Through-plane spacing/extent from ordered IOP x IPP projections (None when unavailable)."""
    keys = [projection_key(g.iop, g.ipp) for g in ordered]
    if len(ordered) < 2 or any(k is None for k in keys):
        return {"slice_spacing_mm_median": None, "slice_spacing_mm_max": None, "extent_mm": None}
    gaps = np.diff(np.asarray(keys, dtype=float))
    return {"slice_spacing_mm_median": round(float(np.median(gaps)), 4),
            "slice_spacing_mm_max": round(float(gaps.max()), 4),
            "extent_mm": round(float(keys[-1] - keys[0]), 3)}


def _triplets(ordered, *, size, n_centers, strategy, lower, upper, get_pixels):
    centers = centers_for_series(len(ordered), n_centers, strategy=strategy, lower=lower, upper=upper)
    triples = [[max(0, min(len(ordered) - 1, c + d)) for d in (-1, 0, 1)] for c in centers]
    selected = sorted({i for triple in triples for i in triple})
    processed, crop_modes = {}, Counter()
    for i in selected:
        geom = ordered[i]
        img, details = process_slice(get_pixels(geom), slope=geom.slope, intercept=geom.intercept,
                                     photometric=geom.photometric, spacing=geom.pixel_spacing, size=size)
        if img.shape != (size, size) or not np.isfinite(img).all() or img.min() < 0 or img.max() > 1.00001:
            raise ValueError("Invalid preprocessed pixel range or shape")
        processed[i] = img
        crop_modes[details["crop_mode"]] += 1
    tensor = np.stack([np.stack([processed[i] for i in triple]) for triple in triples]).astype(np.float32)
    return tensor, {"n_slices": len(ordered), "centers_idx": centers, "selected_indices": selected,
                    "triplet_indices": triples, "n_decoded_slices": len(selected), "crop_modes": dict(crop_modes),
                    "duplicate_centers": n_centers - len(set(centers)),
                    "duplicate_sop_count": len(ordered) - len({g.sop for g in ordered}),
                    **physical_geometry(ordered)}


def series_to_centers(pairs, *, size, n_centers, center_strategy="interior", center_lower=0.15, center_upper=0.85):
    ordered, mode = order_slices([g for _, g in pairs])
    lookup = {g.path: arr for arr, g in pairs}
    tensor, _ = _triplets(ordered, size=size, n_centers=n_centers, strategy=center_strategy,
                          lower=center_lower, upper=center_upper, get_pixels=lambda g: lookup[g.path])
    return tensor, mode


def _series_from_disk(series_dir, *, size, n_centers, strategy, lower, upper):
    headers, error = _load_headers(series_dir)
    if error:
        return None, {"error": error}
    ordered, order = order_slices(headers)
    try:
        tensor, details = _triplets(ordered, size=size, n_centers=n_centers, strategy=strategy,
                                    lower=lower, upper=upper, get_pixels=lambda g: _decode_pixels(Path(g.path)))
        details["order"] = order
        details["resolution"] = [ordered[0].rows, ordered[0].cols]
        return tensor, details
    except Exception as exc:
        reason = str(exc) if str(exc).startswith("codec_or_decode:") else f"geometry_or_pixels:{exc.__class__.__name__}"
        return None, {"error": reason}


def _dicom_bytes_sha(files):
    digest = hashlib.sha256()
    for path in files:
        digest.update(f"{path.parent.name}/{path.name}".encode())
        digest.update(sha256_file(path).encode())
    return digest.hexdigest()


def _check_shard(shard, meta, *, expected_shape, pp_hash):
    try:
        if not shard.is_file():
            return "shard_missing"
        if meta.get("preprocess_hash") != pp_hash or meta.get("preprocess_version") != PREPROCESS_VERSION:
            return "preprocess_mismatch"
        if meta.get("shape") != expected_shape or meta.get("dtype") != "uint8":
            return "metadata_shape_or_dtype_mismatch"
        mask = meta.get("slot_mask")
        if not isinstance(mask, list) or len(mask) != expected_shape[0] or any(v not in (0.0, 1.0) for v in mask) or not any(mask):
            return "invalid_slot_mask"
        if not meta.get("sha256") or sha256_file(shard) != meta["sha256"]:
            return "content_hash_mismatch"
        if shard.suffix == ".npz":
            if meta.get("storage_format") != "npz" or meta.get("shard_filename") != shard.name:
                return "storage_format_mismatch"
            if meta.get("compressed_bytes") != shard.stat().st_size:
                return "compressed_size_mismatch"
        elif meta.get("storage_format", "npy") != "npy":
            return "storage_format_mismatch"
        array = read_shard(shard)
        if list(array.shape) != expected_shape or array.dtype != np.uint8:
            return "array_shape_or_dtype_mismatch"
        for i, present in enumerate(mask):
            if (not present and np.any(array[i])) or (present and int(array[i].max()) == int(array[i].min())):
                return "slot_pixels_mask_mismatch"
        return None
    except Exception as exc:
        return f"invalid_shard:{exc.__class__.__name__}"


def _prior_reusable(prior, *, pp_hash, shape, source_hashes, slots, n_centers):
    if not prior:
        return None
    meta = dict(prior.get("meta") or {})
    if meta.get("source_hashes") != source_hashes or meta.get("slots") != slots or meta.get("n_centers") != n_centers:
        return None
    if _check_shard(Path(prior.get("shard") or ""), meta, expected_shape=shape, pp_hash=pp_hash):
        return None
    return meta


def manifest_ready(path):
    """Validate a complete cache manifest: v4 NPZ, or a legacy v3 NPY manifest."""
    try:
        if not path.is_file():
            return False, "cache_manifest_missing"
        payload = json.loads(path.read_text())
        if payload.get("scope") not in {"pilot", "full"}:
            return False, "cache_scope_missing"
        studies = payload.get("studies") or {}
        if not studies:
            return False, "cache_empty"
        policy = payload.get("preprocess_config")
        if not policy or preprocess_hash(policy) != payload.get("preprocess_hash"):
            return False, "manifest_preprocess_mismatch"
        storage = policy.get("storage_format", "npy")
        builder = payload.get("builder_version")
        if (storage, builder) != ("npz", CACHE_BUILDER_VERSION) and LEGACY_BUILDER_VERSIONS.get(builder) != storage:
            return False, "manifest_storage_format_mismatch"
        if storage == "npz":
            actual_bytes = compressed_byte_total(path.parent)
            if actual_bytes != payload.get("compressed_byte_total"):
                return False, "manifest_compressed_size_mismatch"
            if actual_bytes > payload.get("max_cache_bytes", 0) or payload.get("budget_error"):
                return False, "cache_budget_exceeded"
        shape = [len(policy["slots"]), policy["n_centers"], 3, policy["size"], policy["size"]]
        if len(studies) != payload.get("n_studies"):
            return False, "study_count_mismatch"
        requested = payload.get("requested_uids", [])
        if len(requested) != len(set(requested)) or len(requested) != payload.get("n_requested"):
            return False, "request_count_mismatch"
        if not set(studies) <= set(requested):
            return False, "unexpected_study"
        if payload.get("quarantine") or set(studies) != set(requested):
            return False, "requested_scope_incomplete"
        for uid, meta in studies.items():
            if not _safe_uid(uid) or meta.get("uid") != uid:
                return False, "invalid_study_uid"
            if meta.get("preprocess_config") != policy or meta.get("slots") != policy["slots"] or meta.get("n_centers") != policy["n_centers"]:
                return False, "study_policy_mismatch"
            reason = _check_shard(path.parent / f"{uid}.{storage}", meta, expected_shape=shape, pp_hash=payload["preprocess_hash"])
            if reason:
                return False, f"{reason}:{uid}"
            side = json.loads((path.parent / f"{uid}.json").read_text())
            if side != meta:
                return False, f"sidecar_manifest_mismatch:{uid}"
        return True, payload["scope"]
    except Exception as exc:
        return False, f"invalid_manifest:{exc.__class__.__name__}"


def _coverage(studies, quarantine, slots):
    coverage = {"planes": {}, "present_plane_counts": dict(Counter(str(int(sum(m["slot_mask"]))) for m in studies.values())),
                "one_plane_studies": sum(sum(m["slot_mask"]) == 1 for m in studies.values()),
                "quarantine_reasons": dict(Counter(v["reason"] for v in quarantine.values()))}
    order_counts = Counter()
    unique_planes = len({p for p, _ in slots}) == len(slots)
    for slot_id, (plane, kind) in enumerate(slots):
        info = [m["series_details"][slot_id] for m in studies.values()]
        spacing = [d["slice_spacing_mm_median"] for d in info if d.get("slice_spacing_mm_median") is not None]
        coverage["planes"][plane if unique_planes else f"{plane}:{kind}"] = {
            "present": sum(d["present"] for d in info), "missing": sum(not d["present"] for d in info),
            "fallback": sum(d["fallback"] and d["present"] for d in info),
            "decoded_slices": sum(d.get("n_decoded_slices", 0) for d in info),
            "available_slices": sum(d.get("n_slices", 0) for d in info),
            "duplicate_centers": sum(d.get("duplicate_centers", 0) for d in info),
            "slice_spacing_mm_median": float(np.median(spacing)) if spacing else None,
            "missing_reasons": dict(Counter(d.get("error", "unknown") for d in info if not d["present"]))}
        order_counts.update(d["order"] for d in info if d.get("order"))
    coverage["order_modes"] = dict(order_counts)
    return coverage


def build_cache(*, series_csv, dicom_root, dest, study_ids=None, limit=None, size=224, n_centers=DENSE_N_CENTERS,
                layout="study/series", resume=True, strict_series=True, center_strategy="interior",
                center_lower=DENSE_CENTER_RANGE[0], center_upper=DENSE_CENTER_RANGE[1], min_present_slots=1,
                slots=PILOT_SLOTS, max_cache_bytes=DEFAULT_MAX_CACHE_BYTES,
                progress_every=25, index_flush_every=50):
    slots = tuple(tuple(slot) for slot in slots)
    if (not slots or len(set(slots)) != len(slots)
            or any(len(slot) != 2 or slot[0] not in {"Sagittal", "Coronal", "Axial"}
                   or slot[1] not in {"fluid", "structural"} for slot in slots)):
        raise ValueError("Invalid or duplicate cache slots")
    if size < 1 or n_centers < 1 or not 1 <= min_present_slots <= len(slots):
        raise ValueError("Invalid cache dimensions or minimum plane count")
    if layout not in {"study/series", "series"} or progress_every < 1 or index_flush_every < 1:
        raise ValueError("Invalid layout or progress/checkpoint interval")
    centers_for_series(3, n_centers, strategy=center_strategy, lower=center_lower, upper=center_upper)
    grouped = _read_series_csv(series_csv)
    uids = list(grouped) if study_ids is None else list(study_ids)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        uids = uids[:limit]
    if not uids or len(set(uids)) != len(uids) or any(not _safe_uid(u) for u in uids):
        raise ValueError("Requested studies must be nonempty, safe and unique")
    dest.mkdir(parents=True, exist_ok=True)
    budget = ByteBudget(dest, max_cache_bytes)
    policy = _selection_policy(size, n_centers, center_strategy, center_lower, center_upper, slots)
    pp_hash, shape, slot_names = preprocess_hash(policy), [len(slots), n_centers, 3, size, size], policy["slots"]
    sources = {"series_csv": sha256_file(series_csv), "dicom_root": str(dicom_root.resolve()),
               "strict_series": str(strict_series), "min_present_slots": str(min_present_slots)}
    studies, quarantine, counts = {}, {}, Counter()
    budget_error = None
    started = time.perf_counter()

    def save_index():
        _atomic_json(dest / "index.json", {"version": PREPROCESS_VERSION, "preprocess_hash": pp_hash,
                     "studies": {u: {"shard": str(dest / f"{u}.npz"), "meta": m, "sha256": m["sha256"]} for u, m in studies.items()},
                     "failures": quarantine})

    def fail(uid, reason, **details):
        quarantine[uid] = {"reason": reason, **details}

    def report(done):
        elapsed = time.perf_counter() - started
        rate = done / max(elapsed, 1e-6)
        event = {"event": "cache_progress", "done": done, "total": len(uids), "built": counts["built"],
                 "reused": counts["reused"], "quarantined": len(quarantine), "elapsed_seconds": round(elapsed, 1),
                 "compressed_byte_total": budget.total, "max_cache_bytes": max_cache_bytes,
                 "studies_per_minute": round(rate * 60, 2), "eta_seconds": round((len(uids) - done) / rate, 1)}
        print(json.dumps(event), flush=True)
        with (dest / "progress.jsonl").open("a") as handle:
            handle.write(json.dumps(event) + "\n")

    print(json.dumps({"event": "cache_start", "n_requested": len(uids), "shape": shape,
                      "sampling": center_strategy, "preprocess_hash": pp_hash}), flush=True)
    try:
        for done, uid in enumerate(uids, 1):
            try:
                if budget.total > max_cache_bytes:
                    raise CacheBudgetExceeded("cache_budget_exceeded:existing_payload")
                rows = grouped.get(uid)
                if not rows:
                    fail(uid, "missing_study_metadata")
                    continue
                study_dir = dicom_root / uid if layout == "study/series" else dicom_root
                if not study_dir.is_dir():
                    fail(uid, "missing_study_dicom")
                    continue
                selected = select_series_for_slots(rows, slots=slots, study_dir=study_dir)
                files = [p for row in selected if row is not None for p in sorted((study_dir / row[SERIES_ID_COL]).glob("*.dcm"))]
                study_sources = {**sources, "dicom_bytes": _dicom_bytes_sha(files),
                                 "selected_series": sha256_json([r[SERIES_ID_COL] if r else None for r in selected])}
                prior = None
                if resume:
                    try:
                        prior = {"shard": str(dest / f"{uid}.npz"), "meta": json.loads((dest / f"{uid}.json").read_text())}
                    except (OSError, ValueError):
                        pass
                meta = _prior_reusable(prior, pp_hash=pp_hash, shape=shape, source_hashes=study_sources,
                                       slots=slot_names, n_centers=n_centers)
                if meta is not None:
                    studies[uid] = meta
                    counts["reused"] += 1
                    continue
                arrays, masks, series_details, fatal = [], [], [], None
                for slot_id, row in enumerate(selected):
                    plane, kind = slots[slot_id]
                    details = {"plane": plane, "requested_kind": kind, "series_uid": row[SERIES_ID_COL] if row else None,
                               "fluid_sensitive": row.get("Fluid_Sensitive") if row else None,
                               "fat_suppression": row.get("Fat_Suppression") if row else None,
                               "fallback": bool(row and kind == "fluid" and row.get("Fluid_Sensitive") != "1")}
                    if row is None:
                        arr = None
                        details["error"] = "missing_plane_metadata"
                    else:
                        arr, info = _series_from_disk(study_dir / row[SERIES_ID_COL], size=size, n_centers=n_centers,
                                                      strategy=center_strategy, lower=center_lower, upper=center_upper)
                        details.update(info)
                    if arr is not None and float(arr.max()) <= float(arr.min()):
                        arr = None
                        details["error"] = "constant_preprocessed_series"
                    arrays.append(arr)
                    masks.append(float(arr is not None))
                    details["present"] = arr is not None
                    series_details.append(details)
                    error = details.get("error")
                    if error and error not in {"no_dicom_files", "missing_plane_metadata"} and strict_series:
                        fatal = error
                        break
                if fatal:
                    fail(uid, fatal, series_details=series_details)
                    continue
                if sum(masks) < min_present_slots:
                    fail(uid, "insufficient_decodable_planes", slot_mask=masks, series_details=series_details)
                    continue
                reference = next(a for a in arrays if a is not None)
                volume = np.stack([a if a is not None else np.zeros_like(reference) for a in arrays])
                meta = {"uid": uid, "shape": list(volume.shape), "dtype": "uint8", "preprocess_version": PREPROCESS_VERSION,
                        "storage_format": "npz", "shard_filename": f"{uid}.npz",
                        "preprocess_hash": pp_hash, "preprocess_config": policy, "slot_mask": masks,
                        "source_hashes": study_sources, "slots": slot_names, "n_centers": n_centers,
                        "order": next((d.get("order") for d in series_details if d.get("order")), "none"),
                        "series_details": series_details, "synthetic": False}
                path = write_shard(uid, volume, meta, root=dest, dtype="uint8", storage_format="npz", budget=budget)
                meta["sha256"] = sha256_file(path)
                meta["compressed_bytes"] = path.stat().st_size
                _atomic_json(dest / f"{uid}.json", meta)
                reason = _check_shard(path, meta, expected_shape=shape, pp_hash=pp_hash)
                if reason:
                    fail(uid, reason)
                    continue
                studies[uid] = meta
                counts["built"] += 1
            except CacheBudgetExceeded as exc:
                budget_error = str(exc)
                fail(uid, "cache_budget_exceeded", detail=budget_error)
                break
            except (OSError, ValueError, RuntimeError) as exc:
                fail(uid, f"study_error:{exc.__class__.__name__}", detail=str(exc)[:300])
            finally:
                if done % index_flush_every == 0:
                    save_index()
                if done % progress_every == 0 or done == len(uids):
                    report(done)
    finally:
        save_index()
    coverage = _coverage(studies, quarantine, slots)
    complete = set(studies) == set(grouped) and not quarantine and limit is None
    manifest = {"preprocess_version": PREPROCESS_VERSION, "builder_version": CACHE_BUILDER_VERSION,
                "preprocess_hash": pp_hash, "preprocess_config": policy, "source_hashes": sources,
                "compressed_byte_total": compressed_byte_total(dest), "max_cache_bytes": max_cache_bytes,
                "budget_error": budget_error,
                "studies": studies, "quarantine": quarantine, "n_studies": len(studies), "n_quarantine": len(quarantine),
                "n_requested": len(uids), "requested_uids": uids, "scope": "full" if complete else "pilot",
                "requested_complete": set(studies) == set(uids) and not quarantine,
                "coverage_note": "complete series CSV" if complete else "requested subset of the series CSV", "coverage": coverage}
    _atomic_json(dest / "manifest.json", manifest)
    _atomic_json(dest / "coverage.json", coverage)
    _atomic_json(dest / "build-summary.json", {"built": counts["built"], "reused": counts["reused"],
                 "elapsed_seconds": round(time.perf_counter() - started, 3), "n_studies": len(studies),
                 "n_quarantine": len(quarantine), "preprocess_hash": pp_hash})
    if budget_error:
        raise CacheBudgetExceeded(budget_error)
    return {**manifest, "manifest_sha256": sha256_file(dest / "manifest.json"), "path": str(dest / "manifest.json")}
