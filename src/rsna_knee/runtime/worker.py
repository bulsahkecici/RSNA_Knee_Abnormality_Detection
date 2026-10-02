"""Remote worker: handshake, job consume, heartbeat, fenced results."""

from __future__ import annotations

import json
import os
import platform
from pathlib import Path
from typing import Any

from rsna_knee.hashing import sha256_file, sha256_json
from rsna_knee.runtime.jobs import validate_job
from rsna_knee.runtime.transport import FileTransport
from rsna_knee.training.train import train_tiny
from rsna_knee.workflow.state import SCHEMA_VERSION, ResultMetrics, ResultRecord


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


def run_job(
    job: dict[str, Any],
    transport: FileTransport,
    items: list[dict[str, Any]] | None = None,
    *,
    expected_token: str | None = None,
) -> ResultRecord:
    errors = validate_job(job, expected_token=expected_token)
    synthetic = bool(job.get("synthetic"))
    encoder = job.get("encoder")
    hs = handshake()
    exit_reason = "ok"
    checkpoint_uris: list[str] = []
    metrics: dict[str, Any]
    if errors:
        exit_reason = "rejected"
        metrics = {"kind": "unavailable", "macro_auc": None, "notes": ",".join(errors)}
    elif items is None:
        exit_reason = "no_items"
        metrics = {"kind": "unavailable", "macro_auc": None, "notes": "worker has no studies"}
    elif not synthetic and encoder != "tiny_test_encoder":
        ckpt = job.get("checkpoint")
        if not ckpt or not Path(ckpt).is_file():
            exit_reason = "no_checkpoint"
            metrics = {"kind": "unavailable", "macro_auc": None, "notes": "production job has no checkpoint"}
        else:
            exit_reason = "ok"
            checkpoint_uris = [str(ckpt)]
            metrics = {"kind": "real", "macro_auc": None, "notes": "checkpoint present; AUC is not invented here"}
    else:
        out = train_tiny(items, epochs=1, ckpt_dir=None)
        if out.get("step", 0) <= 0:
            exit_reason = "no_checkpoint"
            metrics = {"kind": "unavailable", "macro_auc": None, "notes": "training produced no step"}
        else:
            metrics = {
                "kind": "synthetic" if synthetic else "real",
                "macro_auc": None,
                "notes": "tiny test step only; not a DINOv2 result",
                "steps": out["step"],
            }
    if exit_reason == "ok" and not synthetic and not checkpoint_uris and encoder != "tiny_test_encoder":
        exit_reason = "no_checkpoint"
    transport.write_heartbeat(job["job_id"], job["attempt_id"], {"stage": exit_reason, "handshake": hs})
    rec = ResultRecord(
        schema_version=SCHEMA_VERSION,
        job_id=job["job_id"],
        run_id=job["run_id"],
        attempt_id=job["attempt_id"],
        fencing_token=job["fencing_token"],
        handshake=hs,
        metrics=ResultMetrics.model_validate(metrics),
        checkpoint_uris=checkpoint_uris,
        artifact_hashes={"job": sha256_json(job)},
        exit_reason=exit_reason,
    )
    transport.write_result(rec.model_dump())
    return rec


def load_job_bundle(path: Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    data["_bundle_sha256"] = sha256_file(Path(path))
    return data
