"""Accept a worker result only when the fencing token matches the expected token."""

from __future__ import annotations

from typing import Any


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
