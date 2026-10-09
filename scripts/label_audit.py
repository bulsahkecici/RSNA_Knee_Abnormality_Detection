"""Numeric label disagreement audit: our local LLM states vs public numeric label sources vs gold.

Local only. Reads numeric/categorical fields; never reads or prints evidence spans or
report text. Aggregates go to state/label-audit/<tag>.json; per-study disagreement
UID lists stay in state/ (gitignored). Gold has 58 studies that were also used for
earlier selection, so gold agreement here is a label-quality signal, not a test score.

Usage: python scripts/label_audit.py --tag 20261009
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from rsna_knee.hashing import sha256_file
from rsna_knee.ontology import TARGET_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
OURS = ROOT / "data/labels/labels-full-20261003-v2-labels.jsonl"
OURS_GOLD = ROOT / "data/labels/labels-goldmeasure-20261009-labels.jsonl"  # gold reports, measurement only
PUBLIC = {
    "steven_full": ROOT / "state/public-labels/stevenleehans/files/llm_labels_full.csv",
    "steven_v2": ROOT / "state/public-labels/stevenleehans/files/llm_labels_v2.csv",
    "steven_v4_blend": ROOT / "state/public-labels/stevenleehans/files/llm_labels_v4_blend.csv",
    "yunus": ROOT / "state/public-labels/yunusgmsoy/files/yunus_llm_labels.csv",
}
TRAIN_CSV = ROOT / "data/metadata/train.csv"
HIGH, LOW = 0.8, 0.2


def our_states(path: Path = OURS) -> dict[str, dict[str, str]]:
    out = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("status") not in {"ok", "valid", None}:
                continue
            targets = (row.get("payload") or {}).get("targets") or {}
            if set(targets) != set(TARGET_COLUMNS):
                continue
            out[row["payload"]["StudyInstanceUID"]] = {t: targets[t].get("state") for t in TARGET_COLUMNS}
    return out


def public_scores(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {r["StudyInstanceUID"]: np.array([float(r[t]) for t in TARGET_COLUMNS]) for r in csv.DictReader(handle)}


def gold_table() -> dict[str, np.ndarray]:
    gold = {}
    with TRAIN_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            values = [row.get(t) for t in TARGET_COLUMNS]
            if all(v not in (None, "", "nan") for v in values):
                gold[row["StudyInstanceUID"]] = np.array([float(v) for v in values])
    return gold


def auc(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return float(roc_auc_score(y, p)) if len(set(y.tolist())) == 2 else None


def macro(values):
    vals = [v for v in values if v is not None]
    return round(float(np.mean(vals)), 4) if vals else None


POLICIES = {  # state -> score; None marks a target the policy masks
    "current_training": {"criterion_positive": 1.0, "explicit_negative": 0.0, "borderline_negative_hint": 0.0},
    "unmentioned_low": {"criterion_positive": 1.0, "explicit_negative": 0.0, "borderline_negative_hint": 0.0,
                        "uncertain": 0.5, "not_mentioned": 0.25},
    "borderline_positive": {"criterion_positive": 1.0, "explicit_negative": 0.0, "borderline_negative_hint": 1.0,
                            "uncertain": 0.5, "not_mentioned": 0.25},
    "borderline_half": {"criterion_positive": 1.0, "explicit_negative": 0.0, "borderline_negative_hint": 0.5,
                        "uncertain": 0.5, "not_mentioned": 0.25},
}


def ours_vs_gold(states: dict[str, dict[str, str]], gold: dict[str, np.ndarray]) -> dict:
    """AUC of our extraction against gold under several state->score policies.

    current_training scores only the observed (unmasked) targets, as training sees them;
    the other policies score every gold study.
    """
    uids = sorted(set(states) & set(gold))
    out = {"n": len(uids), "state_vs_gold_positive_rate": {}}
    for name, mapping in POLICIES.items():
        per, coverage = {}, {}
        for i, t in enumerate(TARGET_COLUMNS):
            pairs = [(gold[u][i], mapping.get(states[u][t])) for u in uids if gold[u][i] in (0.0, 1.0)]
            pairs = [(y, s) for y, s in pairs if s is not None]
            coverage[t] = len(pairs)
            value = auc([y for y, _ in pairs], [s for _, s in pairs])
            per[t] = None if value is None else round(value, 4)
        out[name] = {"macro_auc": macro(per.values()), "per_target": per, "n_scored": coverage}
    for i, t in enumerate(TARGET_COLUMNS):
        by_state: dict[str, list[float]] = {}
        for u in uids:
            by_state.setdefault(states[u][t] or "missing", []).append(float(gold[u][i]))
        out["state_vs_gold_positive_rate"][t] = {s: [len(v), round(float(np.mean(v)), 2)] for s, v in sorted(by_state.items())}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    ours, gold = our_states(), gold_table()
    public = {name: public_scores(path) for name, path in PUBLIC.items()}
    report: dict = {"inputs": {"ours": {"path": str(OURS.relative_to(ROOT)), "sha256": sha256_file(OURS), "n": len(ours)},
                               **{n: {"path": str(p.relative_to(ROOT)), "sha256": sha256_file(p), "n": len(public[n])}
                                  for n, p in PUBLIC.items()},
                               "gold": {"n": len(gold), "train_csv_sha256": sha256_file(TRAIN_CSV)}},
                    "thresholds": {"high": HIGH, "low": LOW}}
    # 1. Each source against gold (58). Ours has no extraction on gold studies (excluded at labeling time).
    gold_uids = sorted(gold)
    y_gold = np.stack([gold[u] for u in gold_uids])
    vs_gold = {}
    for name, table in public.items():
        missing = [u for u in gold_uids if u not in table]
        p = np.stack([table[u] for u in gold_uids if u in table])
        y = np.stack([gold[u] for u in gold_uids if u in table])
        per = {t: auc(y[:, i], p[:, i]) for i, t in enumerate(TARGET_COLUMNS)}
        vs_gold[name] = {"n": len(p), "missing": len(missing), "macro_auc": macro(per.values()),
                         "per_target": {t: None if v is None else round(v, 4) for t, v in per.items()}}
    report["vs_gold"] = vs_gold
    if OURS_GOLD.is_file():
        report["inputs"]["ours_gold"] = {"path": str(OURS_GOLD.relative_to(ROOT)), "sha256": sha256_file(OURS_GOLD)}
        report["ours_vs_gold"] = ours_vs_gold(our_states(OURS_GOLD), gold)
    report["gold_positives"] = {t: int(y_gold[:, i].sum()) for i, t in enumerate(TARGET_COLUMNS)}
    # 2. Ours vs public on weak studies, split by our per-target state.
    agreement, disagreements = {}, {}
    for name, table in public.items():
        per_target = {}
        for i, t in enumerate(TARGET_COLUMNS):
            by_state: dict[str, list[float]] = {}
            for uid, states in ours.items():
                if uid in table and uid not in gold:
                    by_state.setdefault(states[t] or "missing", []).append(float(table[uid][i]))
            obs_y, obs_p = [], []
            for state, label in (("criterion_positive", 1), ("explicit_negative", 0)):
                obs_y += [label] * len(by_state.get(state, []))
                obs_p += by_state.get(state, [])
            per_target[t] = {
                "auc_public_vs_our_explicit": None if auc(obs_y, obs_p) is None else round(auc(obs_y, obs_p), 4),
                "states": {s: {"n": len(v), "public_mean": round(float(np.mean(v)), 3),
                               "frac_public_high": round(float(np.mean(np.array(v) >= HIGH)), 3),
                               "frac_public_low": round(float(np.mean(np.array(v) <= LOW)), 3)}
                           for s, v in sorted(by_state.items())},
            }
            if name == "steven_full":
                pos_low = [u for u, s in ours.items() if u in table and u not in gold
                           and s[t] == "criterion_positive" and table[u][i] <= LOW]
                neg_high = [u for u, s in ours.items() if u in table and u not in gold
                            and s[t] == "explicit_negative" and table[u][i] >= HIGH]
                disagreements[t] = {"ours_positive_public_low": pos_low, "ours_negative_public_high": neg_high}
        agreement[name] = per_target
    report["ours_vs_public"] = agreement
    report["disagreement_counts_vs_steven_full"] = {t: {k: len(v) for k, v in d.items()} for t, d in disagreements.items()}
    report["our_state_counts"] = {t: dict(Counter(s[t] for s in ours.values())) for t in TARGET_COLUMNS}
    # 3. Public sources among themselves on all non-gold studies.
    names = list(public)
    common = sorted(set.intersection(*(set(public[n]) for n in names)) - set(gold))
    corr = {}
    for i, t in enumerate(TARGET_COLUMNS):
        mat = np.stack([[public[n][u][i] for u in common] for n in names])
        c = np.corrcoef(mat)
        corr[t] = {f"{names[a]}~{names[b]}": round(float(c[a, b]), 3) for a in range(len(names)) for b in range(a + 1, len(names))}
    report["public_pairwise_pearson"] = corr
    out = ROOT / "state/label-audit"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.tag}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / f"{args.tag}-disagreement-uids.json").write_text(json.dumps(disagreements, indent=1), encoding="utf-8")
    ours_gold = report.get("ours_vs_gold", {})
    print(json.dumps({"vs_gold_macro": {n: v["macro_auc"] for n, v in vs_gold.items()},
                      "ours_vs_gold_macro": {k: v["macro_auc"] for k, v in ours_gold.items() if isinstance(v, dict) and "macro_auc" in v},
                      "disagreements": report["disagreement_counts_vs_steven_full"]}, indent=1))


if __name__ == "__main__":
    main()
