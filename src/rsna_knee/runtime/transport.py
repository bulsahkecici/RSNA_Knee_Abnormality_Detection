"""File-poll transport. Controller SQLite stays local. Remote results are immutable per-attempt files."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from rsna_knee.hashing import canonical_json
from rsna_knee.paths import STATE_DIR, ensure_runtime_dirs


class FileTransport:
    def __init__(self, root: Path | None = None):
        ensure_runtime_dirs()
        override = os.environ.get("RSNA_ROOTS")
        default = Path(override) / "transport" if override else STATE_DIR / "transport"
        self.root = root or default
        self.outbox = self.root / "outbox"
        self.inbox = self.root / "inbox"
        self.outbox.mkdir(parents=True, exist_ok=True)
        self.inbox.mkdir(parents=True, exist_ok=True)

    def write_job(self, job: dict[str, Any]) -> Path:
        attempt = job["attempt_id"]
        path = self.outbox / f"{job['job_id']}__{attempt}__job.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(canonical_json(job), encoding="utf-8")
        tmp.replace(path)
        return path

    def write_result(self, result: dict[str, Any]) -> Path:
        attempt = result["attempt_id"]
        path = self.inbox / f"{result['job_id']}__{attempt}__result.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(canonical_json(result), encoding="utf-8")
        tmp.replace(path)
        return path

    def read_result(self, job_id: str, attempt_id: str) -> dict[str, Any] | None:
        path = self.inbox / f"{job_id}__{attempt_id}__result.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def heartbeat_path(self, job_id: str, attempt_id: str) -> Path:
        return self.inbox / f"{job_id}__{attempt_id}__heartbeat.json"

    def write_heartbeat(self, job_id: str, attempt_id: str, payload: dict[str, Any]) -> Path:
        path = self.heartbeat_path(job_id, attempt_id)
        payload = dict(payload)
        payload["ts"] = time.time()
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(canonical_json(payload), encoding="utf-8")
        tmp.replace(path)
        return path

    def heartbeat_stale(self, job_id: str, attempt_id: str, max_age: float = 120.0, now: float | None = None) -> bool:
        path = self.heartbeat_path(job_id, attempt_id)
        if not path.exists():
            return True
        data = json.loads(path.read_text(encoding="utf-8"))
        ts = float(data.get("ts") or 0)
        now = time.time() if now is None else now
        return (now - ts) > max_age
