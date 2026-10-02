"""Report and series duplicate audits. Patient identity is not assumed."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from rsna_knee.hashing import sha256_text
from rsna_knee.ontology import REPORT_COL, STUDY_ID_COL
from rsna_knee.paths import METADATA_DIR


def duplicate_report_groups(train_csv: Path | None = None) -> dict[str, Any]:
    path = train_csv or (METADATA_DIR / "train.csv")
    groups: dict[str, list[str]] = defaultdict(list)
    if not path.exists():
        return {"status": "blocked", "reason": "train.csv missing"}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            report = row.get(REPORT_COL) or ""
            groups[sha256_text(report)].append(row[STUDY_ID_COL])
    dups = {h: uids for h, uids in groups.items() if len(uids) > 1}
    return {
        "status": "ok",
        "n_duplicate_report_groups": len(dups),
        "n_studies_in_duplicate_groups": sum(len(v) for v in dups.values()),
        "patient_id_inferred": False,
        "groups_truncated": {h: uids[:8] for h, uids in list(dups.items())[:20]},
        "note": "Identical reports are not a patient identifier. Do not claim leak-free patient CV.",
    }
