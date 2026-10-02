"""Code-competition submit. The CLI cannot skip the audit gate or invent a receipt."""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from rsna_knee.gates import audit_allows_submit
from rsna_knee.runtime.providers.kaggle import KaggleProvider
from rsna_knee.workflow.registry import Registry

_RECEIPT_RE = re.compile(r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})", re.I)


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
) -> dict[str, Any]:
    registry = registry or Registry()
    local_attempt_id = str(uuid.uuid4())
    audit = audit if audit is not None else _load_audit(registry, run_id)
    if not audit or not audit_allows_submit(audit):
        return {
            "status": "BLOCKED",
            "submitted": False,
            "local_attempt_id": local_attempt_id,
            "server_receipt": None,
            "reason": "audit_not_pass",
        }
    prior = registry.submission_for_kernel(kernel, version)
    if prior:
        return {
            "status": "BLOCKED",
            "submitted": False,
            "local_attempt_id": local_attempt_id,
            "server_receipt": prior.get("server_receipt"),
            "reason": "duplicate_kernel_version",
        }
    provider = provider or KaggleProvider()
    if not execute:
        return {
            "status": "dry_run",
            "submitted": False,
            "local_attempt_id": local_attempt_id,
            "server_receipt": None,
            "would_run": (
                "kaggle competitions submit -c rsna-knee-abnormality-detection "
                f"-k {kernel} -v {version} -m {message!r}"
            ),
            "note": "Audit geçti. SUBMITTED yalnız sunucu fişi ile yazılır.",
        }
    out = provider.submit_kernel(kernel, version, message)
    receipt = out.get("stdout") or ""
    match = _RECEIPT_RE.search(receipt)
    server_receipt = match.group(1) if match else None
    ok = out.get("returncode") == 0 and bool(server_receipt) and "successfully" in receipt.lower()
    if not ok or not server_receipt:
        return {
            "status": "BLOCKED",
            "submitted": False,
            "local_attempt_id": local_attempt_id,
            "server_receipt": None,
            "reason": "unparsed_receipt",
            "cli": out,
        }
    registry.put_submission(
        server_receipt,
        run_id,
        "SUBMITTED",
        {"cli": out, "local_attempt_id": local_attempt_id, "server_receipt": server_receipt},
        kernel=kernel,
        version=str(version),
    )
    return {
        "status": "SUBMITTED",
        "submitted": True,
        "local_attempt_id": local_attempt_id,
        "server_receipt": server_receipt,
        "cli": out,
        "scored": False,
    }
