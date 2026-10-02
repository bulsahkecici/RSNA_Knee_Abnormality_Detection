"""Accept a worker result only when the fencing token matches the expected token."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rsna_knee.hashing import sha256_file
from rsna_knee.workflow.registry import Registry
from rsna_knee.workflow.state import PipelineState, Stage


def ingest_result(
    *,
    expected_token: str,
    result: dict[str, Any] | None,
    heartbeat_stale: bool,
) -> dict[str, Any]:
    if not result:
        if heartbeat_stale:
            return {"status": "heartbeat_stale", "accepted": False, "complete": False}
        return {"status": "missing_result", "accepted": False, "complete": False}
    incoming = result.get("fencing_token")
    if incoming != expected_token:
        return {
            "status": "token_mismatch",
            "accepted": False,
            "complete": False,
            "expected_token": expected_token,
        }
    exit_reason = result.get("exit_reason")
    if exit_reason != "ok":
        return {
            "status": "rejected_exit",
            "accepted": False,
            "complete": False,
            "exit_reason": exit_reason,
            "heartbeat_stale": heartbeat_stale,
        }
    return {
        "status": "ingested",
        "accepted": True,
        "complete": True,
        "heartbeat_stale": heartbeat_stale,
        "fencing_token": expected_token,
    }


def accept_into_registry(
    registry: Registry,
    job: dict[str, Any],
    result: dict[str, Any],
    *,
    job_path: Path | None = None,
) -> dict[str, Any]:
    """Bind an accepted worker result to the same run the CLI evaluate command reads."""
    decision = ingest_result(
        expected_token=str(job.get("fencing_token") or ""),
        result=result,
        heartbeat_stale=False,
    )
    if not decision.get("accepted"):
        return decision
    existing = registry.get_run(str(job["run_id"]))
    if existing:
        state = PipelineState.model_validate(existing)
    else:
        state = PipelineState(
            run_id=str(job["run_id"]),
            profile=str((job.get("train") or {}).get("profile") or "pilot"),
            synthetic=bool(job.get("synthetic")),
            fencing_token=str(job.get("fencing_token") or ""),
        )
    uris = list(result.get("checkpoint_uris") or [])
    if uris:
        checkpoint = Path(uris[0])
        state.artifacts["checkpoint"] = str(checkpoint)
        if checkpoint.is_file():
            state.hashes["train"] = sha256_file(checkpoint)
    if len(uris) > 1:
        metrics = Path(uris[1])
        state.artifacts["metrics"] = str(metrics)
        if metrics.is_file():
            state.hashes["evaluate"] = sha256_file(metrics)
    if job_path is not None:
        state.artifacts["job"] = str(job_path)
    state.stage = Stage.TRAINING
    state.artifacts["encoder"] = str(job.get("encoder") or "")
    registry.upsert_run(state.run_id, state.profile, state.stage, state.model_dump(), state.synthetic)
    decision["run_id"] = state.run_id
    decision["checkpoint"] = state.artifacts.get("checkpoint")
    return decision
