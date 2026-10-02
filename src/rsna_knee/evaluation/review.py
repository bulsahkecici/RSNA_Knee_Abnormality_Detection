"""Experiment review: AUC>=0.86 is not a champion by itself."""

from __future__ import annotations

from typing import Any


def review_run(metrics: dict[str, Any], *, audit: dict[str, Any], source: dict[str, Any], runtime: dict[str, Any]) -> dict[str, Any]:
    auc = metrics.get("macro_auc")
    kind = metrics.get("kind")
    reasons: list[str] = []
    if kind != "real":
        reasons.append("metrics_not_real")
    if auc is None:
        reasons.append("macro_auc_undefined")
    audit_gate = audit.get("overall", "FAIL")
    if audit_gate != "PASS":
        reasons.append(f"audit_{audit_gate}")
    if not source.get("legitimate", False):
        reasons.append("source_not_legitimate")
    if not runtime.get("finite_outputs", False):
        reasons.append("outputs_not_finite")
    champion = not reasons and auc is not None
    # Explicit: 0.86 alone is insufficient.
    if auc is not None and auc >= 0.86 and reasons:
        note = "AUC>=0.86 is not sufficient for champion or submit."
    else:
        note = "decision uses audit, baseline delta, uncertainty, source, runtime, finite outputs."
    return {
        "champion": champion,
        "blockers": reasons,
        "note": note,
        "auc": auc,
        "kind": kind,
    }
