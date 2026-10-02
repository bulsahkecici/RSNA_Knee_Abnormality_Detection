"""Auditor reads artifacts; it does not trust the producing agent's summary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rsna_knee.evaluation.metrics import reject_fake_production
from rsna_knee.hashing import sha256_file
from rsna_knee.ontology import TARGET_COLUMNS
from rsna_knee.submission.validate import validate_submission


def audit_run(payload: dict[str, Any], *, production: bool = False) -> dict[str, Any]:
    gates: dict[str, Any] = {}
    hashes = payload.get("hashes") or {}
    gates["metadata_schema"] = "PASS" if hashes.get("metadata") else "BLOCKED"
    gates["report_criteria"] = "PASS" if payload.get("ontology_confirmed") is True else "BLOCKED"
    gates["gold_leakage"] = payload.get("leakage") or "BLOCKED"
    gates["cache_parity"] = payload.get("cache_parity") or "BLOCKED"
    metrics = payload.get("metrics") or {}
    try:
        reject_fake_production(metrics, production=production)
        if metrics.get("kind") == "real" and payload.get("checkpoint_sha256") and payload.get("run_id"):
            gates["metrics"] = "PASS"
        elif production:
            gates["metrics"] = "FAIL"
        elif metrics.get("kind") == "synthetic":
            gates["metrics"] = "FAIL" if production else "BLOCKED"
        else:
            gates["metrics"] = "BLOCKED"
    except Exception as exc:  # noqa: BLE001
        gates["metrics"] = f"FAIL:{exc}"
    identity = payload.get("identity_errors")
    if identity is None:
        gates["identity"] = "BLOCKED"
    elif identity:
        gates["identity"] = "FAIL"
    else:
        gates["identity"] = "PASS"
    gates["resources"] = payload.get("resources") or "BLOCKED"
    gates["external_data"] = payload.get("external_data") or "BLOCKED"
    gates["offline_runtime"] = payload.get("offline") or "BLOCKED"
    sub = payload.get("submission_path")
    if sub and Path(sub).is_file() and payload.get("identity_errors") == []:
        gates["submission_schema"] = "PASS" if validate_submission(Path(sub)).get("ok") else "FAIL"
    elif sub:
        gates["submission_schema"] = "FAIL"
    else:
        gates["submission_schema"] = "BLOCKED"
    values = list(gates.values())
    if any(str(v).startswith("FAIL") for v in values):
        overall = "FAIL"
    elif any(v == "BLOCKED" for v in values):
        overall = "BLOCKED"
    else:
        overall = "PASS"
    result = {"overall": overall, "gates": gates, "targets": list(TARGET_COLUMNS), "read_from": "artifacts_not_agent_summary"}
    for key in ("run_id", "checkpoint_sha256", "package_sha256", "kernel", "version"):
        if payload.get(key) is not None:
            result[key] = payload[key]
    return result


def revalidate_audit(run: dict[str, Any]) -> dict[str, Any]:
    """Re-check stored evidence. Printing the JSON file is not a new audit."""
    path = (run.get("artifacts") or {}).get("audit")
    if not path or not Path(path).is_file():
        return {"overall": "BLOCKED", "reason": "audit_artifact_missing", "run_id": run.get("run_id")}
    stored = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(stored, dict):
        return {"overall": "BLOCKED", "reason": "audit_unreadable", "run_id": run.get("run_id")}
    errors: list[str] = []
    artifacts = run.get("artifacts") or {}
    checkpoint = artifacts.get("checkpoint")
    if stored.get("checkpoint_sha256"):
        if not checkpoint or not Path(checkpoint).is_file():
            errors.append("checkpoint_missing")
        elif sha256_file(Path(checkpoint)) != stored.get("checkpoint_sha256"):
            errors.append("checkpoint_hash_mismatch")
    package = artifacts.get("package_tar")
    if stored.get("package_sha256"):
        if not package or not Path(package).is_file():
            errors.append("package_missing")
        elif sha256_file(Path(package)) != stored.get("package_sha256"):
            errors.append("package_hash_mismatch")
    result = dict(stored)
    result["run_id"] = run.get("run_id") or stored.get("run_id")
    result["revalidated"] = True
    result["identity_errors"] = errors
    if errors:
        result["overall"] = "BLOCKED"
        result["reason"] = errors[0]
    return result
