"""After the official Colab extension is connected, run the job that is already on disk.

This does not allocate a GPU. A missing bundle or missing studies is not success.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rsna_knee.runtime.transport import FileTransport
from rsna_knee.runtime.worker import handshake, load_job_bundle, run_job


def run_after_handshake(job_bundle: Path, inbox: Path | None = None) -> dict[str, Any]:
    hs = handshake()
    if not job_bundle.is_file():
        return {
            "status": "BLOCKED",
            "reason": "job_bundle_missing",
            "handshake": hs,
            "exit_reason": "no_items",
        }
    job = load_job_bundle(job_bundle)
    (job_bundle.parent / "handshake.json").write_text(json.dumps(hs, default=str, indent=2), encoding="utf-8")
    transport = FileTransport(inbox) if inbox else FileTransport(job_bundle.parent / "transport")
    record = run_job(job, transport, expected_token=job.get("fencing_token"))
    return {
        "status": "BLOCKED" if record.exit_reason != "ok" else "ok",
        "exit_reason": record.exit_reason,
        "handshake": record.handshake,
        "job_id": record.job_id,
        "attempt_id": record.attempt_id,
        "fencing_token": record.fencing_token,
    }
