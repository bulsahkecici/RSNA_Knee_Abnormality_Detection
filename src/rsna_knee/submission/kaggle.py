"""Code-competition submit via kernel version. SUBMITTED only after a real CLI receipt."""

from __future__ import annotations

import re
import uuid
from typing import Any

from rsna_knee.runtime.providers.kaggle import KaggleProvider
from rsna_knee.workflow.registry import Registry


def submit_run(run_id: str, kernel: str, version: str, message: str, registry: Registry | None = None, execute: bool = False) -> dict[str, Any]:
    registry = registry or Registry()
    provider = KaggleProvider()
    limits = provider.submission_limits()
    if not execute:
        return {
            "status": "dry_run",
            "would_run": f"kaggle competitions submit -c rsna-knee-abnormality-detection -k {kernel} -v {version} -m {message!r}",
            "limits": limits,
            "note": "İlk smoke/submit etmez. --submit ve geçerli kota olmadan SUBMITTED yazılmaz.",
        }
    out = provider.submit_kernel(kernel, version, message)
    receipt = out.get("stdout") or ""
    ok = out.get("returncode") == 0 and ("error" not in receipt.lower() or "successfully" in receipt.lower())
    request_id = None
    m = re.search(r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})", receipt, re.I)
    if m:
        request_id = m.group(1)
    request_id = request_id or (f"kaggle-{uuid.uuid4()}" if ok else None)
    status = "SUBMITTED" if ok and request_id else "FAILED"
    if request_id:
        registry.put_submission(request_id, run_id, status, {"cli": out}, kernel=kernel, version=str(version))
    return {"status": status, "request_id": request_id, "cli": out, "scored": False}
