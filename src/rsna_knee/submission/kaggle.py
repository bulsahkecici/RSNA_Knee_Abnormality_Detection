"""Code-competition submit. The CLI cannot skip the audit gate or invent a receipt."""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any

from rsna_knee.gates import audit_allows_submit
from rsna_knee.hashing import sha256_file
from rsna_knee.runtime.providers.kaggle import KaggleProvider
from rsna_knee.submission.package import _checkpoint_errors
from rsna_knee.workflow.registry import Registry

_RECEIPT_RE = re.compile(r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})", re.I)
SLUG = "rsna-knee-abnormality-detection"


def _load_audit(registry: Registry, run_id: str) -> dict[str, Any] | None:
    run = registry.get_run(run_id) or {}
    path = (run.get("artifacts") or {}).get("audit")
    if not path:
        return None
    try:
        data = json.loads(open(path, encoding="utf-8").read())
    except OSError:
        return None
    return data if isinstance(data, dict) else None


def _blocked(local_attempt_id: str, reason: str, **extra: Any) -> dict[str, Any]:
    return {
        "status": "BLOCKED",
        "submitted": False,
        "local_attempt_id": local_attempt_id,
        "server_receipt": extra.pop("server_receipt", None),
        "reason": reason,
        **extra,
    }


def _identity_errors(registry: Registry, run_id: str, audit: dict[str, Any]) -> list[str]:
    run = registry.get_run(run_id)
    if not run:
        return ["run_not_registered"]
    artifacts = run.get("artifacts") or {}
    errors: list[str] = []
    ckpt = artifacts.get("checkpoint")
    if not ckpt or not Path(ckpt).is_file():
        errors.append("checkpoint_missing")
    elif sha256_file(Path(ckpt)) != audit.get("checkpoint_sha256"):
        errors.append("checkpoint_hash_mismatch")
    else:
        errors.extend(_checkpoint_errors(Path(ckpt)))
    package = artifacts.get("package_tar") or artifacts.get("package")
    if not package or not Path(package).exists():
        errors.append("package_missing")
    elif Path(package).is_file() and sha256_file(Path(package)) != audit.get("package_sha256"):
        errors.append("package_hash_mismatch")
    elif Path(package).is_dir():
        manifest = Path(package) / "MANIFEST.json"
        if not manifest.is_file() or sha256_file(manifest) != audit.get("package_sha256"):
            errors.append("package_hash_mismatch")
    return errors


def _command(kernel: str, version: str, message: str, submission_file: str | None) -> dict[str, Any]:
    argv = ["kaggle", "competitions", "submit", "-c", SLUG, "-k", kernel, "-v", str(version), "-m", message]
    if submission_file:
        argv.extend(["-f", submission_file])
    return {
        "competition": SLUG,
        "kernel": kernel,
        "version": str(version),
        "message": message,
        "file": submission_file,
        "argv": argv,
    }


def submit_run(
    run_id: str,
    kernel: str,
    version: str,
    message: str,
    registry: Registry | None = None,
    execute: bool = False,
    *,
    provider: KaggleProvider | None = None,
    audit: dict[str, Any] | None = None,
    submission_file: str | None = None,
    reconcile: bool = False,
) -> dict[str, Any]:
    registry = registry or Registry()
    local_attempt_id = str(uuid.uuid4())
    stored = _load_audit(registry, run_id)
    if stored is not None and audit is not None and stored.get("overall") != audit.get("overall"):
        audit = stored
    elif stored is not None and audit is None:
        audit = stored
    command = _command(kernel, version, message, submission_file)
    if not audit or not audit_allows_submit(audit, run_id=run_id, kernel=kernel, version=version):
        return _blocked(local_attempt_id, "audit_not_pass", command=command)
    identity = _identity_errors(registry, run_id, audit)
    if identity:
        return _blocked(local_attempt_id, identity[0], identity_errors=identity, command=command)
    prior = registry.submission_for_kernel(kernel, version)
    if prior:
        return _blocked(
            local_attempt_id,
            "duplicate_kernel_version",
            server_receipt=prior.get("server_receipt"),
            command=command,
        )
    pending = registry.open_submission(kernel, version)
    if pending and not reconcile:
        return _blocked(local_attempt_id, "unresolved_attempt", command=command, pending_status=pending.get("status"))
    if pending and reconcile:
        history = getattr(provider or KaggleProvider(), "list_submissions", None)
        if history is None:
            return _blocked(local_attempt_id, "reconciliation_unavailable", command=command)
        found = history()
        receipt = _RECEIPT_RE.search(json.dumps(found)) if found else None
        if receipt is None:
            return _blocked(local_attempt_id, "unresolved_attempt", command=command)
        return _blocked(local_attempt_id, "reconciled_without_resubmit", server_receipt=receipt.group(1), command=command)
    provider = provider or KaggleProvider()
    if not execute:
        return {
            "status": "dry_run",
            "submitted": False,
            "local_attempt_id": local_attempt_id,
            "server_receipt": None,
            "would_run": " ".join(command["argv"]),
            "command": command,
            "note": "Audit ve paket kimliği geçti. SUBMITTED yalnız sunucu fişi ile yazılır.",
        }
    registry.put_submission(
        local_attempt_id,
        run_id,
        "SUBMITTING",
        {"command": command, "local_attempt_id": local_attempt_id},
        kernel=kernel,
        version=str(version),
    )
    try:
        out = provider.submit_kernel(kernel, version, message)
    except Exception as exc:  # noqa: BLE001
        registry.put_submission(
            local_attempt_id,
            run_id,
            "UNKNOWN",
            {"error": exc.__class__.__name__, "local_attempt_id": local_attempt_id, "command": command},
            kernel=kernel,
            version=str(version),
        )
        return _blocked(local_attempt_id, "network_uncertain", command=command)
    receipt = out.get("stdout") or ""
    match = _RECEIPT_RE.search(receipt)
    server_receipt = match.group(1) if match else None
    ok = out.get("returncode") == 0 and bool(server_receipt) and "successfully" in receipt.lower()
    if not ok or not server_receipt:
        registry.put_submission(
            local_attempt_id,
            run_id,
            "UNKNOWN",
            {"cli": out, "local_attempt_id": local_attempt_id, "reason": "unparsed_receipt", "command": command},
            kernel=kernel,
            version=str(version),
        )
        return _blocked(local_attempt_id, "unparsed_receipt", cli=out, command=command)
    registry.put_submission(
        local_attempt_id,
        run_id,
        "SUBMITTED",
        {"cli": out, "local_attempt_id": local_attempt_id, "server_receipt": server_receipt, "command": command},
        kernel=kernel,
        version=str(version),
    )
    return {
        "status": "SUBMITTED",
        "submitted": True,
        "local_attempt_id": local_attempt_id,
        "server_receipt": server_receipt,
        "cli": out,
        "command": command,
        "scored": False,
    }
