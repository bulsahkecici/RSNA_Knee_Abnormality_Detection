"""Real y,p metrics. Undefined AUC is not hidden as 0.5."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

from rsna_knee.errors import FakeResultError
from rsna_knee.ontology import TARGET_COLUMNS


def per_class_auc(y: np.ndarray, p: np.ndarray, mask: np.ndarray | None = None) -> dict[str, Any]:
    """y,p [N] or [N,12]."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
        p = p[:, None]
    n_t = y.shape[1]
    out: dict[str, Any] = {}
    aucs: list[float] = []
    defined = 0
    for i in range(n_t):
        name = TARGET_COLUMNS[i] if i < len(TARGET_COLUMNS) else str(i)
        yy = y[:, i]
        pp = p[:, i]
        if mask is not None:
            m = mask[:, i].astype(bool)
            yy, pp = yy[m], pp[m]
        valid = np.isfinite(yy) & np.isfinite(pp)
        yy, pp = yy[valid], pp[valid]
        n_pos = int((yy == 1).sum())
        n_neg = int((yy == 0).sum())
        if n_pos == 0 or n_neg == 0 or yy.size < 2:
            out[name] = {"auc": None, "undefined": True, "n_pos": n_pos, "n_neg": n_neg, "n": int(yy.size)}
            continue
        auc = float(roc_auc_score(yy, pp))
        aucs.append(auc)
        defined += 1
        out[name] = {"auc": auc, "undefined": False, "n_pos": n_pos, "n_neg": n_neg, "n": int(yy.size)}
    macro = float(np.mean(aucs)) if aucs else None
    return {"per_class": out, "macro_auc": macro, "n_defined": defined, "n_targets": n_t}


def diagnostics(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(p)
    spread = float(np.std(p[finite])) if finite.any() else None
    try:
        mask = np.isfinite(y) & np.isfinite(p)
        ll = float(log_loss(y[mask], np.clip(p[mask], 1e-6, 1 - 1e-6), labels=[0, 1])) if mask.any() else None
    except Exception:
        ll = None
    return {"log_loss": ll, "pred_std": spread, "pred_mean": float(np.mean(p[finite])) if finite.any() else None}


def reject_fake_production(metrics: dict[str, Any], *, production: bool) -> None:
    kind = metrics.get("kind")
    if production and kind in {"synthetic", "simulated", "fake"}:
        raise FakeResultError("synthetic metrics cannot pass a production competition gate")
    if production and metrics.get("macro_auc") is not None and metrics.get("n_defined", 1) == 0:
        raise FakeResultError("macro AUC with zero defined classes is invalid")


def bootstrap_study(y: np.ndarray, p: np.ndarray, n: int = 200, seed: int = 0) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    stats = []
    n_s = y.shape[0]
    for _ in range(n):
        idx = rng.integers(0, n_s, size=n_s)
        m = per_class_auc(y[idx], p[idx])
        if m["macro_auc"] is not None:
            stats.append(m["macro_auc"])
    if not stats:
        return {"mean": None, "low": None, "high": None, "n": 0}
    arr = np.asarray(stats)
    return {"mean": float(arr.mean()), "low": float(np.percentile(arr, 2.5)), "high": float(np.percentile(arr, 97.5)), "n": len(stats)}
