"""Offline hidden-test inference. Same preprocess as cache build. No invented scores."""

from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rsna_knee.data.cache_build import PILOT_SLOTS, build_cache
from rsna_knee.models.dinov2 import ENCODER_NAME, Dinov2ViTS14
from rsna_knee.models.study import StudyModel
from rsna_knee.ontology import STUDY_ID_COL, TARGET_COLUMNS
from rsna_knee.submission.infer import write_submission_csv
from rsna_knee.training.checkpoint import torch_load


def project_runtime(seconds_per_study: float | None, n_measured: int, n_hidden: int = 1300) -> dict[str, Any]:
    if seconds_per_study is None or n_measured < 30:
        return {
            "guaranteed_under_9h": False,
            "projected_seconds": None,
            "reason": "benchmark_too_small" if n_measured < 30 else "no_measurement",
            "n_measured": n_measured,
            "n_hidden": n_hidden,
        }
    projected = float(seconds_per_study) * n_hidden
    return {
        "guaranteed_under_9h": projected < 9 * 3600,
        "projected_seconds": projected,
        "n_measured": n_measured,
        "n_hidden": n_hidden,
        "reason": "measured_batch",
    }


def _load_model(checkpoint: Path, img_size: int) -> StudyModel:
    blob = torch_load(checkpoint)
    if blob.get("encoder_name") != ENCODER_NAME:
        raise RuntimeError(f"checkpoint encoder is {blob.get('encoder_name')}, expected {ENCODER_NAME}")
    if int(blob.get("img_size", img_size)) != img_size:
        raise RuntimeError("checkpoint image size does not match inference")
    model = StudyModel(Dinov2ViTS14(img_size=img_size), freeze_encoder=True,
                       pooling=blob.get("pooling", "mean"))
    if blob.get("pooling_version", model.pooling_version) != model.pooling_version:
        raise RuntimeError("checkpoint pooling version is unsupported")
    model.load_state_dict(blob["model"])
    model.eval()
    return model


def run_offline_inference(
    *,
    checkpoint: Path,
    series_csv: Path,
    dicom_root: Path,
    study_csv: Path,
    dest: Path,
    work: Path,
    img_size: int = 224,
    n_centers: int | None = None,
    preprocess_config: Path | None = None,
    seconds_per_study: float | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint missing: {checkpoint}")
    if not study_csv.is_file() or not series_csv.is_file():
        raise FileNotFoundError("hidden-test metadata is missing")
    if not dicom_root.exists():
        raise FileNotFoundError(f"DICOM root missing: {dicom_root}")
    started = time.perf_counter()
    with study_csv.open(newline="", encoding="utf-8") as handle:
        studies = [row[STUDY_ID_COL] for row in csv.DictReader(handle)]
    if not studies:
        raise RuntimeError("study metadata has no rows")
    cache_options = inference_cache_options(img_size, n_centers, preprocess_config)
    manifest = build_cache(
        series_csv=series_csv,
        dicom_root=dicom_root,
        dest=work,
        study_ids=studies,
        **cache_options,
    )
    missing = [uid for uid in studies if uid not in manifest["studies"]]
    if missing:
        raise RuntimeError(f"inference refused silent zeros; missing or quarantined: {missing[:8]}")
    from rsna_knee.evaluation.heldout import select_device

    dev = select_device(device)
    model = _load_model(checkpoint, img_size).to(dev)
    rows = []
    with torch.no_grad():
        for uid in studies:
            volume = np.load(work / f"{uid}.npy")
            if volume.dtype == np.uint8:
                volume = volume.astype(np.float32) / 255.0
            images = torch.from_numpy(volume).unsqueeze(0).to(dev)
            slot = torch.tensor(manifest["studies"][uid]["slot_mask"], dtype=torch.float32).unsqueeze(0).to(dev)
            centers = images.shape[2]
            center = torch.ones((1, images.shape[1], centers), dtype=torch.float32, device=dev)
            center = center * slot[:, :, None]
            logits = model(images, slot, center)
            probs = torch.sigmoid(logits)[0].detach().cpu().tolist()
            row = {STUDY_ID_COL: uid}
            row.update({name: float(p) for name, p in zip(TARGET_COLUMNS, probs, strict=True)})
            rows.append(row)
    write_submission_csv(rows, dest)
    elapsed = time.perf_counter() - started
    measured = elapsed / len(rows) if rows else None
    runtime = project_runtime(measured, n_measured=len(rows))
    runtime["measured_seconds"] = elapsed
    runtime["seconds_per_study"] = measured
    runtime["device"] = str(dev)
    runtime["stopwatch"] = True
    runtime["includes_decode_and_load"] = True
    if seconds_per_study is not None:
        runtime["caller_seconds_per_study_ignored"] = seconds_per_study
    return {
        "submission": str(dest),
        "n_studies": len(rows),
        "slots": [list(s) for s in PILOT_SLOTS],
        "runtime": runtime,
        "quarantine": manifest["quarantine"],
        "series_errors": {uid: meta["series_errors"] for uid, meta in manifest["studies"].items() if meta.get("series_errors")},
        "checkpoint": str(checkpoint),
    }


def inference_cache_options(img_size: int, n_centers: int | None, config: Path | None) -> dict[str, Any]:
    """Use versioned training preprocessing; reject conflicting caller settings."""
    if config is None:
        if n_centers is not None and (type(n_centers) is not int or n_centers < 1):
            raise ValueError("invalid_inference_preprocess_layout")
        return {"size": img_size, "n_centers": 3 if n_centers is None else n_centers, "strict_series": False}
    policy = json.loads(config.read_text())
    centers = policy.get("n_centers")
    if type(centers) is not int or centers < 1 or policy.get("size") != img_size:
        raise ValueError("invalid_inference_preprocess_layout")
    if n_centers is not None and centers != n_centers:
        raise ValueError("inference_center_count_conflicts_with_training")
    if policy.get("dtype") != "uint8" or policy.get("center_strategy") != "interior":
        raise ValueError("unsupported_inference_preprocess_policy")
    expected_slots = [["Sagittal", "fluid"], ["Coronal", "fluid"], ["Axial", "fluid"]]
    if policy.get("slots") != expected_slots or policy.get("crop_mm") != 160.0 or policy.get("clip_percentiles") != [0.5, 99.5]:
        raise ValueError("unsupported_inference_pixel_policy")
    code_hashes = policy.get("code_sha256", {})
    data = Path(__file__).parents[1] / "data"
    for name in ["cache_build.py", "cache.py", "preprocess.py", "dicom.py"]:
        if hashlib.sha256((data / name).read_bytes()).hexdigest() != code_hashes.get(name):
            raise ValueError("inference_preprocess_source_hash_mismatch")
    lower, upper = policy.get("center_lower"), policy.get("center_upper")
    if not isinstance(lower, (int, float)) or not isinstance(upper, (int, float)) or not 0 <= lower < upper <= 1:
        raise ValueError("invalid_inference_center_fractions")
    return {"size": img_size, "n_centers": centers, "strict_series": True,
            "center_strategy": "interior", "center_lower": lower, "center_upper": upper,
            "min_present_slots": 1}
