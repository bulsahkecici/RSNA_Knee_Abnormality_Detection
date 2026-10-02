"""Soft-label calibration is gold-only. LLM self-confidence is not a probability."""

from __future__ import annotations

from typing import Any

import numpy as np

from rsna_knee.ontology import TARGET_COLUMNS


def shrink_probability(p: float, n: int, prior: float = 0.5, k: float = 8.0) -> float:
    """Controlled shrinkage when gold support is small."""
    w = n / (n + k)
    return float(w * p + (1 - w) * prior)


def audit_prediction_spread(probs: np.ndarray) -> dict[str, Any]:
    if probs.size == 0:
        return {"n": 0, "std": None, "collapsed": True}
    std = float(np.std(probs))
    return {"n": int(probs.size), "std": std, "mean": float(np.mean(probs)), "collapsed": std < 1e-4}


def calibrate_from_gold(
    gold: dict[str, list[float]],
    pred: dict[str, list[float]],
) -> dict[str, Any]:
    report: dict[str, Any] = {"method": "gold_only_shrinkage", "targets": {}}
    for name in TARGET_COLUMNS:
        g = np.asarray(gold.get(name, []), dtype=float)
        p = np.asarray(pred.get(name, []), dtype=float)
        if g.size < 8 or p.size != g.size:
            report["targets"][name] = {"status": "insufficient_gold", "n": int(g.size)}
            continue
        # Platt-like one-parameter shrink of mean toward gold prevalence; not a fake AUC.
        prev = float(g.mean())
        mean_p = float(p.mean())
        report["targets"][name] = {
            "status": "ok",
            "n": int(g.size),
            "gold_prevalence": prev,
            "pred_mean": mean_p,
            "shrunk_mean": shrink_probability(mean_p, int(g.size), prior=prev),
            "spread": audit_prediction_spread(p),
        }
    return report
