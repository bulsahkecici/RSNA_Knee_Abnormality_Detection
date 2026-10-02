"""SQLite job/stage/artifact registry. Local controller only — never a shared Drive DB."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from rsna_knee.hashing import canonical_json
from rsna_knee.paths import STATE_DIR, ensure_runtime_dirs
from rsna_knee.workflow.state import Stage

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY,
  profile TEXT NOT NULL,
  stage TEXT NOT NULL,
  synthetic INTEGER NOT NULL DEFAULT 0,
  created_ts REAL NOT NULL,
  updated_ts REAL NOT NULL,
  payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL,
  ts REAL NOT NULL,
  stage TEXT NOT NULL,
  kind TEXT NOT NULL,
  payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS artifacts (
  artifact_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  kind TEXT NOT NULL,
  path TEXT NOT NULL,
  sha256 TEXT NOT NULL,
  count INTEGER,
  source_hashes TEXT NOT NULL,
  created_ts REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
  job_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  attempt_id TEXT NOT NULL,
  fencing_token TEXT NOT NULL,
  lease_id TEXT,
  status TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  created_ts REAL NOT NULL,
  updated_ts REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS submissions (
  request_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  kernel TEXT,
  version TEXT,
  status TEXT NOT NULL,
  receipt_json TEXT NOT NULL,
  created_ts REAL NOT NULL,
  updated_ts REAL NOT NULL
);
"""


class Registry:
    def __init__(self, path: Path | None = None):
        ensure_runtime_dirs()
        self.path = path or (STATE_DIR / "registry.sqlite")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._conn:
            self._conn.executescript(SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def upsert_run(self, run_id: str, profile: str, stage: Stage | str, payload: dict[str, Any], synthetic: bool = False) -> None:
        now = time.time()
        stage_s = str(stage)
        with self._lock, self._conn:
            existing = self._conn.execute("SELECT run_id FROM runs WHERE run_id=?", (run_id,)).fetchone()
            blob = canonical_json(payload)
            if existing:
                self._conn.execute(
                    "UPDATE runs SET profile=?, stage=?, synthetic=?, updated_ts=?, payload_json=? WHERE run_id=?",
                    (profile, stage_s, int(synthetic), now, blob, run_id),
                )
            else:
                self._conn.execute(
                    "INSERT INTO runs(run_id, profile, stage, synthetic, created_ts, updated_ts, payload_json) VALUES (?,?,?,?,?,?,?)",
                    (run_id, profile, stage_s, int(synthetic), now, now, blob),
                )
            self._conn.execute(
                "INSERT INTO events(run_id, ts, stage, kind, payload_json) VALUES (?,?,?,?,?)",
                (run_id, now, stage_s, "stage", blob),
            )

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if not row:
            return None
        payload = json.loads(row["payload_json"])
        payload.update(
            {
                "run_id": row["run_id"],
                "profile": row["profile"],
                "stage": row["stage"],
                "synthetic": bool(row["synthetic"]),
                "created_ts": row["created_ts"],
                "updated_ts": row["updated_ts"],
            }
        )
        return payload

    def latest_run(self) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT run_id FROM runs ORDER BY updated_ts DESC LIMIT 1").fetchone()
        return self.get_run(row["run_id"]) if row else None

    def add_artifact(
        self,
        artifact_id: str,
        run_id: str,
        kind: str,
        path: str,
        sha256: str,
        source_hashes: dict[str, str],
        count: int | None = None,
    ) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """INSERT OR REPLACE INTO artifacts(artifact_id, run_id, kind, path, sha256, count, source_hashes, created_ts)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (artifact_id, run_id, kind, path, sha256, count, canonical_json(source_hashes), time.time()),
            )

    def artifacts_for(self, run_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute("SELECT * FROM artifacts WHERE run_id=?", (run_id,)).fetchall()
        out = []
        for row in rows:
            out.append(
                {
                    "artifact_id": row["artifact_id"],
                    "kind": row["kind"],
                    "path": row["path"],
                    "sha256": row["sha256"],
                    "count": row["count"],
                    "source_hashes": json.loads(row["source_hashes"]),
                }
            )
        return out

    def put_job(self, job_id: str, run_id: str, attempt_id: str, fencing_token: str, status: str, payload: dict[str, Any], lease_id: str | None = None) -> None:
        now = time.time()
        with self._lock, self._conn:
            self._conn.execute(
                """INSERT OR REPLACE INTO jobs(job_id, run_id, attempt_id, fencing_token, lease_id, status, payload_json, created_ts, updated_ts)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (job_id, run_id, attempt_id, fencing_token, lease_id, status, canonical_json(payload), now, now),
            )

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if not row:
            return None
        payload = json.loads(row["payload_json"])
        payload.update({"status": row["status"], "fencing_token": row["fencing_token"], "attempt_id": row["attempt_id"]})
        return payload

    def put_submission(self, request_id: str, run_id: str, status: str, receipt: dict[str, Any], kernel: str | None = None, version: str | None = None) -> None:
        now = time.time()
        with self._lock, self._conn:
            existing = self._conn.execute("SELECT request_id FROM submissions WHERE request_id=?", (request_id,)).fetchone()
            blob = canonical_json(receipt)
            if existing:
                self._conn.execute(
                    "UPDATE submissions SET status=?, receipt_json=?, updated_ts=?, kernel=?, version=? WHERE request_id=?",
                    (status, blob, now, kernel, version, request_id),
                )
            else:
                self._conn.execute(
                    """INSERT INTO submissions(request_id, run_id, kernel, version, status, receipt_json, created_ts, updated_ts)
                       VALUES (?,?,?,?,?,?,?,?)""",
                    (request_id, run_id, kernel, version, status, blob, now, now),
                )

    def submission_for_kernel(self, kernel: str, version: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM submissions WHERE kernel=? AND version=? AND status='SUBMITTED'",
            (kernel, str(version)),
        ).fetchone()
        if not row:
            return None
        rec = json.loads(row["receipt_json"])
        rec.update(
            {
                "request_id": row["request_id"],
                "status": row["status"],
                "run_id": row["run_id"],
                "kernel": row["kernel"],
                "version": row["version"],
            }
        )
        return rec

    def get_submission(self, request_id: str) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT * FROM submissions WHERE request_id=?", (request_id,)).fetchone()
        if not row:
            return None
        rec = json.loads(row["receipt_json"])
        rec.update({"status": row["status"], "run_id": row["run_id"], "kernel": row["kernel"], "version": row["version"]})
        return rec

    def events(self, run_id: str, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT ts, stage, kind FROM events WHERE run_id=? ORDER BY id DESC LIMIT ?",
            (run_id, limit),
        ).fetchall()
        return [{"ts": r["ts"], "stage": r["stage"], "kind": r["kind"]} for r in rows]
