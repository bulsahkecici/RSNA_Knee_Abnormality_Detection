"""OOF collection with gold vs weak split."""

from __future__ import annotations

from typing import Any

import numpy as np

from rsna_knee.evaluation.metrics import per_class_auc
from rsna_knee.ontology import TARGET_COLUMNS


def split_oof(rows: list[dict[str, Any]]) -> dict[str, Any]:
    gold = [r for r in rows if r.get("gold_split") == "eval" or r.get("is_gold_eval")]
    weak = [r for r in rows if r not in gold]
    def pack(group: list[dict[str, Any]], kind: str) -> dict[str, Any]:
        if not group:
            return {"kind": kind, "n": 0, "macro_auc": None, "per_class": {}}
        y = np.stack([r["y"] for r in group])
        p = np.stack([r["p"] for r in group])
        mask = np.stack([r.get("mask", np.isfinite(y[0])) for r in group]) if group else None
        metrics = per_class_auc(y, p, mask)
        metrics["kind"] = kind
        metrics["n"] = len(group)
        metrics["targets"] = list(TARGET_COLUMNS)
        return metrics
    return {"gold_oof": pack(gold, "gold_oof"), "weak_oof": pack(weak, "weak_oof")}
