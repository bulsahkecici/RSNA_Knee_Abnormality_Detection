"""Remote worker: handshake, job consume, heartbeat, fenced results."""

from __future__ import annotations

import json
import os
import platform
from pathlib import Path
from typing import Any

from rsna_knee.hashing import sha256_file, sha256_json
from rsna_knee.runtime.transport import FileTransport
from rsna_knee.training.train import train_tiny
from rsna_knee.workflow.state import ResultMetrics, ResultRecord, SCHEMA_VERSION


def handshake() -> dict[str, Any]:
    gpu = "cpu"
    vram = None
    dtype = {"fp16": False, "bf16": False}
    try:
        import torch

        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory
            dtype["fp16"] = True
            dtype["bf16"] = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
        elif torch.backends.mps.is_available():
            gpu = "mps"
    except Exception:
        pass
    return {
        "gpu": gpu,
        "vram": vram,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dtype": dtype,
        "disk_free": _disk_free(),
        "pid": os.getpid(),
    }


def _disk_free() -> int | None:
    try:
        st = os.statvfs(".")
        return st.f_bavail * st.f_frsize
    except Exception:
        return None


def accept_job(job: dict[str, Any], expected_token: str | None) -> dict[str, Any] | None:
    if expected_token and job.get("fencing_token") != expected_token:
        return None
    return job


def run_job(job: dict[str, Any], transport: FileTransport, items: list[dict[str, Any]] | None = None) -> ResultRecord:
    hs = handshake()
    transport.write_heartbeat(job["job_id"], job["attempt_id"], {"stage": "start", "handshake": hs})
    synthetic = bool(job.get("synthetic"))
    metrics: dict[str, Any]
    if items:
        out = train_tiny(items, epochs=1, ckpt_dir=None)
        metrics = {
            "kind": "synthetic" if synthetic else "real",
            "macro_auc": None,
            "notes": "tiny train step completed; AUC not claimed without labeled eval",
            "steps": out["step"],
        }
    else:
        metrics = {"kind": "unavailable" if not synthetic else "synthetic", "macro_auc": None, "notes": "no items"}
    rec = ResultRecord(
        schema_version=SCHEMA_VERSION,
        job_id=job["job_id"],
        run_id=job["run_id"],
        attempt_id=job["attempt_id"],
        fencing_token=job["fencing_token"],
        handshake=hs,
        metrics=ResultMetrics.model_validate(metrics),
        checkpoint_uris=[],
        artifact_hashes={"job": sha256_json(job)},
        exit_reason="ok",
    )
    transport.write_result(rec.model_dump())
    return rec


def load_job_bundle(path: Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    data["_bundle_sha256"] = sha256_file(Path(path))
    return data
