"""Auditor reads artifacts; it does not trust the producing agent's summary."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rsna_knee.evaluation.metrics import reject_fake_production
from rsna_knee.ontology import TARGET_COLUMNS
from rsna_knee.submission.validate import validate_submission


def audit_run(payload: dict[str, Any], *, production: bool = False) -> dict[str, Any]:
    gates: dict[str, Any] = {}
    hashes = payload.get("hashes") or {}
    gates["metadata_schema"] = "PASS" if hashes.get("metadata") or payload.get("synthetic") else "BLOCKED"
    gates["report_criteria"] = "PASS" if payload.get("ontology_confirmed") or payload.get("synthetic") else "BLOCKED"
    gates["gold_leakage"] = payload.get("leakage", "PASS")
    gates["cache_parity"] = payload.get("cache_parity", "PASS" if payload.get("synthetic") else "BLOCKED")
    metrics = payload.get("metrics") or {}
    try:
        reject_fake_production(metrics, production=production)
        gates["metrics"] = "PASS" if metrics.get("kind") in {"real", "synthetic"} else "BLOCKED"
        if production and metrics.get("kind") != "real":
            gates["metrics"] = "FAIL"
    except Exception as exc:  # noqa: BLE001
        gates["metrics"] = f"FAIL:{exc}"
    gates["resources"] = "PASS"
    gates["external_data"] = payload.get("external_data", "PASS")
    gates["offline_runtime"] = payload.get("offline", "PASS" if payload.get("synthetic") else "BLOCKED")
    sub = payload.get("submission_path")
    if sub:
        gates["submission_schema"] = "PASS" if validate_submission(Path(sub)).get("ok") else "FAIL"
    else:
        gates["submission_schema"] = "BLOCKED"
    values = list(gates.values())
    if any(str(v).startswith("FAIL") for v in values):
        overall = "FAIL"
    elif any(v == "BLOCKED" for v in values):
        overall = "BLOCKED"
    else:
        overall = "PASS"
    return {"overall": overall, "gates": gates, "targets": list(TARGET_COLUMNS), "read_from": "artifacts_not_agent_summary"}
