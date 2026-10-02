"""Offline bundle + inference that does not require internet, Drive, or LM Studio."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from rsna_knee.models.classifier import StudyClassifier
from rsna_knee.ontology import STUDY_ID_COL, TARGET_COLUMNS
from rsna_knee.training.losses import sigmoid


def infer_study(clf: StudyClassifier, images: np.ndarray, slot_mask: np.ndarray, slice_mask: np.ndarray) -> dict[str, float]:
    logits = clf.forward_numpy(images[None], slot_mask[None], slice_mask[None])[0]
    probs = sigmoid(logits)
    return {name: float(np.clip(p, 0.0, 1.0)) for name, p in zip(TARGET_COLUMNS, probs, strict=True)}


def write_submission_csv(rows: list[dict[str, Any]], path: Path) -> Path:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    cols = [STUDY_ID_COL, *TARGET_COLUMNS]
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row[c] for c in cols})
    tmp.replace(path)
    return path
