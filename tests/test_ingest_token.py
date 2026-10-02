from __future__ import annotations

from rsna_knee.runtime.ingest import ingest_result


def test_token_is_not_replaced_and_stale_heartbeat_keeps_finished_result():
    result = {"fencing_token": "expected", "exit_reason": "ok", "checkpoint_uris": ["ckpt.pt"]}
    stale = ingest_result(expected_token="expected", result=result, heartbeat_stale=True)
    assert stale["accepted"] is True
    assert stale["fencing_token"] == "expected"
    mismatch = ingest_result(expected_token="expected", result={**result, "fencing_token": "other"}, heartbeat_stale=False)
    assert mismatch["accepted"] is False
    assert mismatch["expected_token"] == "expected"
    assert mismatch["status"] == "token_mismatch"


def test_missing_result_with_stale_heartbeat_stays_incomplete():
    out = ingest_result(expected_token="expected", result=None, heartbeat_stale=True)
    assert out["accepted"] is False
    assert out["status"] == "heartbeat_stale"
