"""Submission contract: all UIDs, 12 finite probabilities, sample column order."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from rsna_knee.ontology import STUDY_ID_COL, TARGET_COLUMNS


def validate_submission(path: Path, required_uids: Sequence[str] | None = None) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        cols = list(reader.fieldnames or [])
        rows = list(reader)
    errors: list[str] = []
    expected = [STUDY_ID_COL, *TARGET_COLUMNS]
    if cols != expected:
        errors.append(f"columns:{cols}")
    seen = []
    for row in rows:
        uid = row.get(STUDY_ID_COL, "")
        seen.append(uid)
        for t in TARGET_COLUMNS:
            try:
                v = float(row[t])
            except Exception:
                errors.append(f"nonfloat:{uid}:{t}")
                continue
            if not np.isfinite(v) or v < 0 or v > 1:
                errors.append(f"not_prob:{uid}:{t}:{v}")
    if required_uids is not None:
        req = list(required_uids)
        if set(seen) != set(req) or len(seen) != len(req):
            errors.append("uid_set_mismatch")
        if seen != req:
            errors.append("uid_order_differs_from_test")
    # Visible 3 UIDs must not be hardcoded as the only legal set when required_uids is the live test list.
    return {
        "ok": not errors,
        "n_rows": len(rows),
        "errors": errors,
        "hardcoded_three_uids": False,
    }
