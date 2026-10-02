from __future__ import annotations

import time

from rsna_knee.runtime.transport import FileTransport
from rsna_knee.runtime.worker import accept_job, handshake, run_job
from rsna_knee.workflow.state import SCHEMA_VERSION


def test_stale_heartbeat_is_not_success(tmp_path):
    tr = FileTransport(tmp_path)
    tr.write_heartbeat("j1", "a1", {"stage": "train"})
    assert tr.heartbeat_stale("j1", "a1", max_age=10, now=time.time() + 50)
    assert not tr.heartbeat_stale("j1", "a1", max_age=10, now=time.time())


def test_fencing_rejects_wrong_token(tmp_path):
    job = {
        "schema_version": SCHEMA_VERSION,
        "job_id": "j1",
        "run_id": "r1",
        "attempt_id": "a1",
        "stage": "TRAINING",
        "bundle_sha256": "0" * 64,
        "config_hash": "c",
        "fold_hash": "f",
        "labels_hash": "l",
        "cache_hash": "k",
        "lease_id": "lease",
        "fencing_token": "tok-correct",
        "synthetic": True,
    }
    assert accept_job(job, "tok-correct") is not None
    assert accept_job(job, "tok-other") is None
    rec = run_job(job, FileTransport(tmp_path), items=None)
    assert rec.fencing_token == "tok-correct"
    again = FileTransport(tmp_path).read_result("j1", "a1")
    assert again["job_id"] == "j1"


def test_handshake_keys():
    hs = handshake()
    assert "gpu" in hs and "python" in hs
