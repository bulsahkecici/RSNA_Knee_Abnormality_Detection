"""Build a versioned soft-target table from public CC0 numeric label sets (training only).

Pre-registered rule (fixed before any image result, not tuned on gold): one table
per author — stevenleehans ``llm_labels_v4_blend.csv`` and yunusgmsoy
``yunus_llm_labels.csv`` — averaged with equal weight per study and target.
Every gold study (both gold sets in the fold manifest) is dropped, so these rows
can never stand in for gold. The output is a separate label source with its own
sha256; evaluation keeps the fixed hard masked labels.

Usage: python scripts/build_soft_labels.py --version public-mean-v1
Writes state/soft-labels/<version>.jsonl and <version>.json (provenance).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from rsna_knee.hashing import sha256_file
from rsna_knee.ontology import TARGET_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "stevenleehans/llm_labels_v4_blend": ROOT / "state/public-labels/stevenleehans/files/llm_labels_v4_blend.csv",
    "yunusgmsoy/yunus_llm_labels": ROOT / "state/public-labels/yunusgmsoy/files/yunus_llm_labels.csv",
}
FOLDS = ROOT / "state/folds-v2-allgold.csv"


def read(path: Path) -> dict[str, list[float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {r["StudyInstanceUID"]: [float(r[t]) for t in TARGET_COLUMNS] for r in csv.DictReader(handle)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    with FOLDS.open(newline="", encoding="utf-8") as handle:
        folds = list(csv.DictReader(handle))
    gold = {r["StudyInstanceUID"] for r in folds if r.get("is_gold") == "1"}
    tables = {name: read(path) for name, path in SOURCES.items()}
    uids = sorted(set.intersection(*(set(t) for t in tables.values())) & {r["StudyInstanceUID"] for r in folds} - gold)
    out_dir = ROOT / "state/soft-labels"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.version}.jsonl"
    with out.open("w", encoding="utf-8") as handle:
        for uid in uids:
            mean = np.mean([tables[name][uid] for name in SOURCES], axis=0)
            if not np.all((mean >= 0) & (mean <= 1)):
                raise ValueError(f"source value outside [0, 1] for {uid}")
            row = {"StudyInstanceUID": uid, "soft_targets": {t: round(float(v), 6) for t, v in zip(TARGET_COLUMNS, mean, strict=True)}}
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    provenance = {
        "version": args.version, "path": str(out.relative_to(ROOT)), "sha256": sha256_file(out), "n_rows": len(uids),
        "rule": "equal mean of one public CC0 table per author; gold studies excluded; fixed before image results",
        "sources": {name: {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path), "license": "CC0-1.0"}
                    for name, path in SOURCES.items()},
        "folds_sha256": sha256_file(FOLDS), "n_gold_excluded": len(gold),
        "use": "training targets only; evaluation targets remain the fixed hard masked labels",
    }
    (out_dir / f"{args.version}.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(provenance, indent=1))


if __name__ == "__main__":
    main()
