"""Ingest one Colab result archive, verify its identity, and review it against its control arm.

Usage: python scripts/review_experiment.py D1 --zip ~/Downloads/colab-results.zip
       python scripts/review_experiment.py L1            # already ingested; re-review only

Checks (any failure aborts, nothing is promoted): exit_reason ok; real metrics;
experiment.json equals the generator's notebooks/generated/<id>.build.json for every
pinned hash; train UIDs equal the pinned list and pass the folds-v2 leakage check;
predictions are unique, finite probabilities covering every gold and weak study.

Reports gold AUC on all 58 and separately on gold set A (historic eval, reused for
selection) and set B (formerly trained on by older models), weak AUC against our
fixed hard masked labels, per-class AUC, timing, and — when the control arm has been
reviewed — paired bootstrap deltas. Local validation only; never a leaderboard score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import yaml

from rsna_knee.data.folds import assert_train_uids, load_folds
from rsna_knee.evaluation.metrics import per_class_auc
from rsna_knee.ontology import TARGET_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "state/campaigns/campaign-20261003-v2/numeric-training.jsonl"
FOLDS = ROOT / "state/folds-v2-allgold.csv"
PINNED = ("git_commit", "source_sha256", "labels_sha256", "folds_sha256", "cache_manifest_sha256",
          "weights_sha256", "soft_labels_sha256", "train_uids_sha256", "config_sha256")
N_BOOT = 2000


def out_dir(exp: str) -> Path:
    return ROOT / "artifacts/colab" / f"p1-{exp}" / "output"


def ingest(exp: str, archive: Path) -> Path:
    dest = out_dir(exp)
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        for name in bundle.namelist():
            rel = Path(name)
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError(f"unsafe archive member {name}")
        bundle.extractall(dest)
    return dest


def labels_table() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    table = {}
    for line in LABELS.read_text().splitlines():
        row = json.loads(line)
        y = np.array([np.nan if row["targets"][t] is None else float(row["targets"][t]) for t in TARGET_COLUMNS])
        mask = np.array([float(row["y_mask"][t]) for t in TARGET_COLUMNS])
        table[row["StudyInstanceUID"]] = (np.where(mask > 0, y, np.nan), row.get("source"))
    return table


def macro(y: np.ndarray, p: np.ndarray) -> tuple[float | None, dict]:
    result = per_class_auc(y, p)
    return result["macro_auc"], result


def load_predictions(folder: Path) -> dict[str, np.ndarray]:
    rows = [json.loads(x) for x in (folder / "job/predictions.jsonl").read_text().splitlines()]
    preds = {r["StudyInstanceUID"]: (r["split"], np.array([float(r[t]) for t in TARGET_COLUMNS])) for r in rows}
    if len(preds) != len(rows):
        raise ValueError("duplicate prediction UID")
    for _, p in preds.values():
        if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
            raise ValueError("invalid prediction probability")
    return preds


def verify(exp: str, folder: Path) -> dict:
    build = json.loads((ROOT / "notebooks/generated" / f"{exp}.build.json").read_text())
    ran = json.loads((folder / "experiment.json").read_text())
    mismatched = [k for k in PINNED if build.get(k) != ran.get(k)]
    if mismatched:
        raise ValueError(f"result was not produced by the generated notebook: {mismatched}")
    if json.loads((folder / "campaign-result.json").read_text())["exit_reason"] != "ok":
        raise ValueError("training did not finish ok")
    metrics = json.loads((folder / "job/metrics.json").read_text())
    if metrics.get("kind") != "real":
        raise ValueError("metrics are not real")
    uids = sorted(metrics["train_uids"])
    if hashlib.sha256("\n".join(uids).encode()).hexdigest() != build["train_uids_sha256"]:
        raise ValueError("trained UIDs differ from the pinned list")
    assert_train_uids(set(uids), load_folds(FOLDS))
    return metrics


def score(preds: dict, labels: dict, uids: list[str]) -> tuple[float | None, dict, np.ndarray, np.ndarray]:
    y = np.stack([labels[u][0] for u in uids])
    p = np.stack([preds[u][1] for u in uids])
    value, detail = macro(y, p)
    return value, detail, y, p


def paired_delta(y, p_new, p_old, seed=2026) -> dict:
    rng = np.random.default_rng(seed)
    base = (macro(y, p_new)[0] or np.nan) - (macro(y, p_old)[0] or np.nan)
    deltas = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(y), len(y))
        a, b = macro(y[idx], p_new[idx])[0], macro(y[idx], p_old[idx])[0]
        if a is not None and b is not None:
            deltas.append(a - b)
    lo, hi = np.percentile(deltas, [2.5, 97.5]) if deltas else (np.nan, np.nan)
    return {"delta": round(float(base), 5), "ci95": [round(float(lo), 5), round(float(hi), 5)], "n_boot": len(deltas)}


def review(exp: str) -> dict:
    folder = out_dir(exp)
    metrics = verify(exp, folder)
    preds, labels = load_predictions(folder), labels_table()
    folds = load_folds(FOLDS)
    gold_sets = {"gold_all": [r["StudyInstanceUID"] for r in folds if r.get("gold_set")]}
    for name in ("A_historic_eval", "B_former_train"):
        gold_sets[f"gold_{name}"] = [r["StudyInstanceUID"] for r in folds if r.get("gold_set") == name]
    weak = sorted(u for u, (split, _) in preds.items() if split == "weak")
    missing = [u for u in gold_sets["gold_all"] if u not in preds]
    if missing:
        raise ValueError(f"{len(missing)} gold studies have no prediction")
    report: dict = {"experiment": exp, "kind": "local_validation_not_leaderboard", "splits": {}}
    arrays = {}
    for name, uids in {**gold_sets, "weak_hard_labels": weak}.items():
        value, detail, y, p = score(preds, labels, sorted(uids))
        arrays[name] = (sorted(uids), y)
        report["splits"][name] = {"n": len(uids), "macro_auc": None if value is None else round(value, 5),
                                  "per_class": {t: (None if v["auc"] is None else round(v["auc"], 4))
                                                for t, v in (detail.get("per_class") or {}).items()}}
    timing = folder / "timing.json"
    report["timing"] = json.loads(timing.read_text()) if timing.is_file() else None
    report["train_metrics"] = {k: metrics.get(k) for k in ("epoch_wall_seconds", "seconds_per_train_study",
                                                           "peak_cuda_memory_bytes", "evaluation_seconds",
                                                           "skipped_updates", "attempted_updates", "config_id",
                                                           "init_source", "label_source")}
    config = next((ROOT / "configs/experiments").glob(f"P*-{exp}-*.yaml"), None)
    control = yaml.safe_load(config.read_text()).get("control") if config else None
    if control and (out_dir(control) / "review.json").is_file():
        old = load_predictions(out_dir(control))
        report["vs_control"] = {"control": control}
        for name in ("gold_all", "gold_A_historic_eval", "gold_B_former_train", "weak_hard_labels"):
            uids, y = arrays[name]
            p_new = np.stack([preds[u][1] for u in uids])
            p_old = np.stack([old[u][1] for u in uids])
            report["vs_control"][name] = paired_delta(y, p_new, p_old)
    (folder / "review.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment")
    parser.add_argument("--zip", type=Path)
    args = parser.parse_args()
    if args.zip:
        ingest(args.experiment, args.zip)
    report = review(args.experiment)
    summary = {k: v["macro_auc"] for k, v in report["splits"].items()}
    print(json.dumps({"experiment": args.experiment, "macro_auc": summary, "vs_control": report.get("vs_control"),
                      "timing": report["timing"]}, indent=1))


if __name__ == "__main__":
    main()
