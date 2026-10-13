"""Build a versioned pseudo-label soft table from reviewed out-of-fold teachers.

Usage:
  python scripts/build_pseudo_labels.py pseudo-v1-all --teachers D3LR2 LR2F1 LR2F2 LR2F3 LR2F4
  python scripts/build_pseudo_labels.py pseudo-v1-nofold0 --teachers LR2I1 LR2I2 LR2I3 LR2I4 --blind-to 0

Each teacher contributes its plain (no-TTA) predictions on the weak studies of its own held-out
outer fold. Target = ALPHA * public-mean-v1 + (1 - ALPHA) * teacher probability, per finding; the
mix ratio was fixed on 2026-10-13 before any teacher ran. Gold studies never appear in the table.

Checks (any failure aborts): every teacher passed scripts/review_experiment.py; teachers hold out
pairwise different folds; a teacher never trained on the studies it labels; with --blind-to F, no
teacher trained on fold F and the table has no fold-F rows (so a fold-F student's weak score stays
clean). Writes state/soft-labels/<name>.jsonl and <name>.json (provenance, sha256).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from rsna_knee.data.folds import load_folds
from rsna_knee.ontology import TARGET_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "state/soft-labels/public-mean-v1.jsonl"
BASE_SHA256 = "7d045805be5a8643dc6e1397ad8f71e7875003799a27fd11a05a6d5a9d916e04"
FOLDS = ROOT / "state/folds-v2-allgold.csv"
ALPHA = 0.5  # weight of the public table; fixed before any teacher result existed


def _teacher(exp: str) -> dict:
    out = ROOT / "artifacts/colab" / f"p1-{exp}" / "output"
    review = json.loads((out / "review.json").read_text())  # exists only after a passing review
    ran = json.loads((out / "experiment.json").read_text())
    metrics = json.loads((out / "job/metrics.json").read_text())
    config = yaml.safe_load((ROOT / ran["config_file"]).read_text())
    rows = [json.loads(x) for x in (out / "job/predictions.jsonl").read_text().splitlines()]
    weak = {r["StudyInstanceUID"]: [float(r[t]) for t in TARGET_COLUMNS] for r in rows if r["split"] == "weak"}
    return {"id": exp, "fold": str(ran["outer_fold"]), "excluded": [str(f) for f in config.get("exclude_folds") or []],
            "train_uids": set(metrics["train_uids"]), "weak": weak, "review_experiment": review["experiment"],
            "checkpoint_sha256": metrics["oof_provenance"]["checkpoint_sha256"]}


def build(name: str, teachers: list[str], blind_to: str | None) -> Path:
    if hashlib.sha256(BASE.read_bytes()).hexdigest() != BASE_SHA256:
        raise SystemExit("public-mean-v1 table changed")
    base = {r["StudyInstanceUID"]: r["soft_targets"] for r in map(json.loads, BASE.read_text().splitlines())}
    folds = load_folds(FOLDS)
    fold_of = {r["StudyInstanceUID"]: str(r["fold"]) for r in folds}
    gold = {r["StudyInstanceUID"] for r in folds if r.get("gold_set")}
    loaded = [_teacher(t) for t in teachers]
    held = [t["fold"] for t in loaded]
    if len(set(held)) != len(held) or "none" in held:
        raise SystemExit(f"teachers must hold out pairwise different folds, got {held}")
    rows, provenance = {}, []
    for t in loaded:
        if t["review_experiment"] != t["id"]:
            raise SystemExit(f"{t['id']}: review belongs to another experiment")
        if blind_to is not None and (blind_to not in t["excluded"] or any(fold_of[u] == blind_to for u in t["train_uids"])):
            raise SystemExit(f"{t['id']} trained on fold {blind_to}")
        for uid, probs in t["weak"].items():
            if uid in gold or fold_of.get(uid) != t["fold"] or uid in t["train_uids"]:
                raise SystemExit(f"{t['id']}: {uid} is gold, outside its held-out fold, or in its training set")
            if uid in rows:
                raise SystemExit(f"{uid} labelled twice")
            public = base.get(uid)
            mixed = {}
            for target, p in zip(TARGET_COLUMNS, probs, strict=True):
                q = None if public is None else public.get(target)
                mixed[target] = round(p if q is None else ALPHA * float(q) + (1 - ALPHA) * p, 6)
            rows[uid] = {"StudyInstanceUID": uid, "soft_targets": mixed}
        provenance.append({"teacher": t["id"], "held_out_fold": t["fold"], "excluded_folds": t["excluded"],
                           "checkpoint_sha256": t["checkpoint_sha256"], "n_rows": len(t["weak"])})
    if blind_to is not None and any(fold_of[u] == blind_to for u in rows):
        raise SystemExit("table contains rows of the blinded fold")
    out = ROOT / "state/soft-labels" / f"{name}.jsonl"
    body = "".join(json.dumps(rows[u], sort_keys=True) + "\n" for u in sorted(rows))
    out.write_text(body)
    sha = hashlib.sha256(body.encode()).hexdigest()
    no_public = sum(1 for u in rows if u not in base)
    (out.with_suffix(".json")).write_text(json.dumps({
        "name": name, "sha256": sha, "n_rows": len(rows), "alpha_public": ALPHA, "base": "public-mean-v1",
        "base_sha256": BASE_SHA256, "blind_to_fold": blind_to, "rows_without_public_label": no_public,
        "folds_covered": sorted(set(held)), "teachers": provenance, "gold_rows": 0}, indent=2))
    print(json.dumps({"name": name, "sha256": sha, "n_rows": len(rows), "folds": sorted(set(held)),
                      "rows_without_public_label": no_public}))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("--teachers", nargs="+", required=True)
    parser.add_argument("--blind-to", help="fold no teacher may have trained on; its rows are left out")
    args = parser.parse_args()
    build(args.name, args.teachers, args.blind_to)


if __name__ == "__main__":
    main()
