from __future__ import annotations

import json
from pathlib import Path

from rsna_knee.pipeline import run_pipeline
from rsna_knee.submission.validate import validate_submission


def test_synthetic_pipeline_writes_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("RSNA_LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
    state = run_pipeline("smoke", synthetic=True, resume=False)
    assert state.synthetic is True
    assert Path(state.artifacts["labels"]).exists()
    assert Path(state.artifacts["metrics"]).exists()
    metrics = json.loads(Path(state.artifacts["metrics"]).read_text())
    assert metrics["kind"] == "synthetic"
    sub = state.artifacts.get("submission")
    assert sub and Path(sub).exists()
    val = validate_submission(Path(sub))
    assert val["ok"]
    # cannot be used as real competition gate
    from rsna_knee.agents.auditor import audit_run

    audit = audit_run({"synthetic": True, "metrics": metrics, "hashes": state.hashes, "ontology_confirmed": True}, production=True)
    assert audit["overall"] in {"FAIL", "BLOCKED"}
