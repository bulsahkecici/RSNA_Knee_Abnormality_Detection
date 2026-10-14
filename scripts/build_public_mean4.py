"""Build public-mean4-v0: the vetted reader's label recipe as a versioned soft table.

Usage: python scripts/build_public_mean4.py

Mean of four CC0 report-label tables (stevenleehans v4_blend, pilkwang v2, lixin73 sol56,
laymond qwen, all derived from competition reports only), averaged over the sources available for
each (study, finding), clipped to [0.01, 0.99] — the `v0` recipe of
goodpjw2008/rsna-knee-2-5d-convnext-reader make_labels.py. Gold-58 rows are dropped. Only numeric
columns are read; nothing else from the files is printed or stored. Inputs are downloaded with
`kaggle datasets download <dataset> -f <file>` into state/external-labels/<owner>_<dataset>/.
Writes state/soft-labels/public-mean4-v0.jsonl and .json (provenance and sha256).
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

from rsna_knee.data.folds import load_folds
from rsna_knee.ontology import TARGET_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "state/external-labels"
SOURCES = {  # name: (folder, file, column suffix)
    "steven": ("stevenleehans_rsna-knee-llm-report-labels", "llm_labels_v4_blend.csv", ""),
    "pilkwang": ("pilkwang_rsna-knee-llm-labels", "report_labels_v2.csv", ""),
    "lixin": ("lixin73_rsna-knee-llm-report-labels-sol56", "labels_llm_gpt56sol.csv", ""),
    "qwen": ("laymond_rsna-knee-abnormality-qwen3-8b-weak-labels", "qwen_knee_weak_labels.csv", "__label"),
}
NAME = "public-mean4-v0"


def _number(text: str) -> float | None:
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def main() -> None:
    gold = {r["StudyInstanceUID"] for r in load_folds(ROOT / "state/folds-v2-allgold.csv") if r.get("gold_set")}
    tables, inputs = {}, {}
    for name, (folder, file, suffix) in SOURCES.items():
        path = EXT / folder / file
        inputs[name] = {"file": f"{folder}/{file}", "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        with path.open(newline="") as handle:
            tables[name] = {row["StudyInstanceUID"]: {t: _number(row[t + suffix]) for t in TARGET_COLUMNS}
                            for row in csv.DictReader(handle)}
    uids = sorted(set().union(*tables.values()) - gold)
    rows, missing = [], 0
    for uid in uids:
        targets = {}
        for t in TARGET_COLUMNS:
            values = [tables[n][uid][t] for n in tables if uid in tables[n] and tables[n][uid][t] is not None]
            if values:
                targets[t] = round(min(0.99, max(0.01, sum(values) / len(values))), 6)
            else:
                targets[t] = None  # masked, as in the reader's NaN-free table this never happens in practice
                missing += 1
        rows.append({"StudyInstanceUID": uid, "soft_targets": targets})
    body = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    out = ROOT / "state/soft-labels" / f"{NAME}.jsonl"
    out.write_text(body)
    sha = hashlib.sha256(body.encode()).hexdigest()
    (out.with_suffix(".json")).write_text(json.dumps({"name": NAME, "sha256": sha, "n_rows": len(rows),
                                                      "gold_rows": 0, "masked_cells": missing, "clip": [0.01, 0.99],
                                                      "recipe": "goodpjw2008 reader make_labels.py v0", "inputs": inputs},
                                                     indent=2))
    print(json.dumps({"name": NAME, "sha256": sha, "n_rows": len(rows), "masked_cells": missing}))


if __name__ == "__main__":
    main()
