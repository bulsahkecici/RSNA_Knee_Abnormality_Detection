"""Offline hidden-test inference. Same preprocess as cache build. No invented scores."""

from __future__ import annotations

import csv
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
    model = StudyModel(Dinov2ViTS14(img_size=img_size), freeze_encoder=True)
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
    n_centers: int = 3,
    seconds_per_study: float | None = None,
) -> dict[str, Any]:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint missing: {checkpoint}")
    if not study_csv.is_file() or not series_csv.is_file():
        raise FileNotFoundError("hidden-test metadata is missing")
    if not dicom_root.exists():
        raise FileNotFoundError(f"DICOM root missing: {dicom_root}")
    with study_csv.open(newline="", encoding="utf-8") as handle:
        studies = [row[STUDY_ID_COL] for row in csv.DictReader(handle)]
    if not studies:
        raise RuntimeError("study metadata has no rows")
    manifest = build_cache(
        series_csv=series_csv,
        dicom_root=dicom_root,
        dest=work,
        study_ids=studies,
        size=img_size,
        n_centers=n_centers,
    )
    missing = [uid for uid in studies if uid not in manifest["studies"]]
    if missing:
        raise RuntimeError(f"inference refused silent zeros; missing or quarantined: {missing[:8]}")
    model = _load_model(checkpoint, img_size)
    rows = []
    with torch.no_grad():
        for uid in studies:
            volume = np.load(work / f"{uid}.npy")
            if volume.dtype == np.uint8:
                volume = volume.astype(np.float32) / 255.0
            images = torch.from_numpy(volume).unsqueeze(0)
            slot = torch.tensor(manifest["studies"][uid]["slot_mask"], dtype=torch.float32).unsqueeze(0)
            centers = images.shape[2]
            center = torch.ones((1, images.shape[1], centers), dtype=torch.float32)
            center = center * slot[:, :, None]
            logits = model(images, slot, center)
            probs = torch.sigmoid(logits)[0].tolist()
            row = {STUDY_ID_COL: uid}
            row.update({name: float(p) for name, p in zip(TARGET_COLUMNS, probs, strict=True)})
            rows.append(row)
    write_submission_csv(rows, dest)
    return {
        "submission": str(dest),
        "n_studies": len(rows),
        "slots": [list(s) for s in PILOT_SLOTS],
        "runtime": project_runtime(seconds_per_study, n_measured=len(rows)),
        "quarantine": manifest["quarantine"],
    }
