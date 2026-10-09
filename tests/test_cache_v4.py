"""Compressed dense v4 cache: slot selection, budget, resume, legacy NPY manifests."""
from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from rsna_knee.data.cache import CacheBudgetExceeded, compressed_byte_total, load_shard, write_shard
from rsna_knee.data.cache_build import (
    PILOT_SLOTS,
    STRUCTURAL_SLOTS,
    build_cache,
    manifest_ready,
    select_series_for_slots,
)
from rsna_knee.data.preprocess import preprocess_hash
from rsna_knee.ontology import SERIES_ID_COL, STUDY_ID_COL
from rsna_knee.runtime.train_payload import LazyStudyBatch
from tests.test_cache_manifest import _write_slice

SERIES = {  # name: (plane, fluid, n_slices)
    "sag-fs": ("Sagittal", "1", 7),
    "sag-t1": ("Sagittal", "0", 5),
    "cor-fs": ("Coronal", "1", 6),
    "ax-fs": ("Axial", "1", 4),
}


def _study(tmp_path, study="1.2.3", series=SERIES):
    rows = []
    for name, (plane, fluid, n) in series.items():
        folder = tmp_path / "dicom" / study / name
        folder.mkdir(parents=True)
        for instance in range(1, n + 1):
            _write_slice(folder / f"s{n - instance}.dcm", study, name, instance, 3.0 * instance)
        rows.append({STUDY_ID_COL: study, SERIES_ID_COL: name, "Anatomical_Plane": plane,
                     "Fluid_Sensitive": fluid, "Fat_Suppression": fluid})
    path = tmp_path / "train_series.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path, rows


def _build(tmp_path, csv_path, dest="cache", **kwargs):
    options = dict(series_csv=csv_path, dicom_root=tmp_path / "dicom", dest=tmp_path / dest, size=28,
                   n_centers=12, slots=STRUCTURAL_SLOTS)
    options.update(kwargs)
    return build_cache(**options)


def test_structural_slots_never_take_fluid_series_and_missing_slot_is_masked(tmp_path):
    csv_path, rows = _study(tmp_path)
    chosen = select_series_for_slots(rows, slots=STRUCTURAL_SLOTS)
    assert [r[SERIES_ID_COL] if r else None for r in chosen] == ["sag-fs", "cor-fs", "ax-fs", "sag-t1", None]
    manifest = _build(tmp_path, csv_path)
    meta = manifest["studies"]["1.2.3"]
    assert meta["slot_mask"] == [1.0, 1.0, 1.0, 1.0, 0.0]
    assert meta["shape"] == [5, 12, 3, 28, 28]
    assert meta["series_details"][4]["error"] == "missing_plane_metadata"
    assert manifest["coverage"]["planes"]["Coronal:structural"]["missing"] == 1
    volume = load_shard("1.2.3", tmp_path / "cache")
    assert volume.dtype == np.uint8 and not volume[4].any() and volume[3].max() > volume[3].min()


def test_fluid_slot_falls_back_only_without_structural_slot_for_plane():
    rows = [{STUDY_ID_COL: "s", SERIES_ID_COL: "t1", "Anatomical_Plane": "Coronal", "Fluid_Sensitive": "0"}]
    assert select_series_for_slots(rows, slots=PILOT_SLOTS)[1][SERIES_ID_COL] == "t1"
    chosen = select_series_for_slots(rows, slots=STRUCTURAL_SLOTS)
    assert chosen[1] is None and chosen[4][SERIES_ID_COL] == "t1"


def test_dense_windows_stay_inside_one_series_and_record_physical_geometry(tmp_path):
    csv_path, _ = _study(tmp_path)
    details = _build(tmp_path, csv_path)["studies"]["1.2.3"]["series_details"]
    for slot in details[:4]:
        n = slot["n_slices"]
        for triple in slot["triplet_indices"]:
            assert all(0 <= i < n for i in triple) and triple[0] <= triple[1] <= triple[2]
            assert triple[2] - triple[0] <= 2
        assert slot["slice_spacing_mm_median"] == 3.0
        assert slot["extent_mm"] == pytest.approx(3.0 * (n - 1))
        # 12 windows over a 4-7 slice series must repeat centers; this is recorded, not hidden.
        assert slot["duplicate_centers"] == 12 - len(set(slot["centers_idx"]))
        assert slot["order"] == "iop_ipp"


def test_npz_cache_is_ready_resumes_and_detects_tampering(tmp_path):
    csv_path, _ = _study(tmp_path)
    first = _build(tmp_path, csv_path)
    path = tmp_path / "cache" / "manifest.json"
    assert first["scope"] == "full" and manifest_ready(path) == (True, "full")
    assert first["compressed_byte_total"] == compressed_byte_total(tmp_path / "cache") > 0
    second = _build(tmp_path, csv_path)
    summary = json.loads((tmp_path / "cache" / "build-summary.json").read_text())
    assert summary["reused"] == 1 and summary["built"] == 0
    assert second["preprocess_hash"] == first["preprocess_hash"]
    shard = tmp_path / "cache" / "1.2.3.npz"
    np.savez_compressed(shard, images=np.zeros((5, 12, 3, 28, 28), dtype=np.uint8))
    ready, reason = manifest_ready(path)
    assert not ready and reason.startswith(("content_hash_mismatch", "manifest_compressed_size_mismatch"))


def test_policy_change_invalidates_hash(tmp_path):
    csv_path, _ = _study(tmp_path)
    dense = _build(tmp_path, csv_path, dest="a")
    sparse = _build(tmp_path, csv_path, dest="b", n_centers=6, slots=PILOT_SLOTS)
    assert dense["preprocess_hash"] != sparse["preprocess_hash"]
    assert sparse["studies"]["1.2.3"]["shape"] == [3, 6, 3, 28, 28]


def test_byte_budget_stops_without_partial_shard(tmp_path):
    csv_path, _ = _study(tmp_path)
    with pytest.raises(CacheBudgetExceeded):
        _build(tmp_path, csv_path, max_cache_bytes=100)
    cache = tmp_path / "cache"
    assert not list(cache.glob("*.npz")) and not list(cache.glob(".*.tmp.npz"))
    manifest = json.loads((cache / "manifest.json").read_text())
    assert manifest["budget_error"] and manifest["quarantine"]["1.2.3"]["reason"] == "cache_budget_exceeded"
    assert manifest_ready(cache / "manifest.json")[0] is False


def test_v4_shard_feeds_training_batch_with_five_slots(tmp_path):
    csv_path, _ = _study(tmp_path)
    meta = _build(tmp_path, csv_path)["studies"]["1.2.3"]
    batch = LazyStudyBatch("1.2.3", meta, {}, tmp_path / "cache")
    assert tuple(batch["images"].shape) == (1, 5, 12, 3, 28, 28)
    assert batch["slot_mask"].tolist() == [[1.0, 1.0, 1.0, 1.0, 0.0]]
    assert float(batch["images"].max()) <= 1.0


def test_legacy_v3_npy_manifest_remains_readable(tmp_path):
    cache = tmp_path / "v3"
    policy = {"size": 8, "n_centers": 1, "slots": [list(s) for s in PILOT_SLOTS],
              "builder_version": "cache-v3-selective-inner"}
    pp_hash = preprocess_hash(policy)
    volume = np.zeros((3, 1, 3, 8, 8), dtype=np.uint8)
    volume[0] = np.arange(64, dtype=np.uint8).reshape(8, 8)
    meta = {"uid": "u1", "shape": list(volume.shape), "dtype": "uint8", "preprocess_hash": pp_hash,
            "preprocess_config": policy, "slots": policy["slots"], "n_centers": 1, "slot_mask": [1.0, 0.0, 0.0]}
    shard = write_shard("u1", volume, meta, root=cache, storage_format="npy")
    from rsna_knee.hashing import sha256_file
    meta = json.loads((cache / "u1.json").read_text())
    meta["sha256"] = sha256_file(shard)
    (cache / "u1.json").write_text(json.dumps(meta))
    manifest = {"scope": "pilot", "builder_version": "cache-v3-selective-inner", "preprocess_config": policy,
                "preprocess_hash": pp_hash, "studies": {"u1": meta}, "requested_uids": ["u1"],
                "n_studies": 1, "n_requested": 1, "quarantine": {}}
    (cache / "manifest.json").write_text(json.dumps(manifest))
    assert manifest_ready(cache / "manifest.json") == (True, "pilot")
    assert load_shard("u1", cache).shape == (3, 1, 3, 8, 8)


def test_parallel_build_matches_sequential(tmp_path):
    csv_path, _ = _study(tmp_path)
    sequential = _build(tmp_path, csv_path, dest="seq")
    parallel = _build(tmp_path, csv_path, dest="par", workers=2)
    a, b = sequential["studies"]["1.2.3"], parallel["studies"]["1.2.3"]
    assert a["sha256"] == b["sha256"] and a["slot_mask"] == b["slot_mask"]
    assert sequential["preprocess_hash"] == parallel["preprocess_hash"]
    assert manifest_ready(tmp_path / "par" / "manifest.json") == (True, "full")


def test_series_intensity_window_preserves_between_slice_contrast(tmp_path):
    csv_path, _ = _study(tmp_path)
    series = _build(tmp_path, csv_path, dest="series", slots=PILOT_SLOTS, intensity="series", crop_mm=3.0)
    per_slice = _build(tmp_path, csv_path, dest="slice", slots=PILOT_SLOTS, crop_mm=3.0)
    assert series["preprocess_config"]["intensity"] == "series" and series["preprocess_config"]["crop_mm"] == 3.0
    assert series["preprocess_hash"] != per_slice["preprocess_hash"]
    detail = series["studies"]["1.2.3"]["series_details"][0]
    assert detail["intensity_window"] is not None
    window = load_shard("1.2.3", tmp_path / "series")[0, 6].astype(float)  # a [prev, centre, next] window
    flat = load_shard("1.2.3", tmp_path / "slice")[0, 6].astype(float)
    # Synthetic slices differ by a constant offset per instance: a shared window keeps that ordering,
    # per-slice normalization erases it.
    assert window[0].mean() < window[1].mean() < window[2].mean()
    assert abs(flat[0].mean() - flat[2].mean()) < 1.0


def test_shard_parts_merge_into_one_verified_manifest(tmp_path):
    from rsna_knee.data.cache_build import merge_cache_parts

    csv_a, _ = _study(tmp_path, study="1.2.3")
    _study(tmp_path / "b", study="1.2.4")
    # second study lives in another DICOM root; build each part from its own CSV/root
    _build(tmp_path, csv_a, dest="part0", slots=PILOT_SLOTS)
    build_cache(series_csv=tmp_path / "b" / "train_series.csv", dicom_root=tmp_path / "b" / "dicom",
                    dest=tmp_path / "part1", size=28, n_centers=12, slots=PILOT_SLOTS)
    merged = merge_cache_parts([tmp_path / "part0", tmp_path / "part1"], tmp_path / "merged")
    assert merged["n_studies"] == 2 and set(merged["requested_uids"]) == {"1.2.3", "1.2.4"}
    assert manifest_ready(tmp_path / "merged" / "manifest.json") == (True, "full")
    with pytest.raises(ValueError, match="overlap"):
        merge_cache_parts([tmp_path / "merged", tmp_path / "merged"], tmp_path / "again", move=False)
    other = _build(tmp_path, csv_a, dest="other", slots=STRUCTURAL_SLOTS)
    assert other["n_studies"] == 1
    with pytest.raises(ValueError, match="policies"):
        merge_cache_parts([tmp_path / "merged", tmp_path / "other"], tmp_path / "bad", move=False)
