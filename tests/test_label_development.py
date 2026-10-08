"""Development extraction must exclude held-out gold and bound failing batches."""

import csv
import json
from types import SimpleNamespace

from rsna_knee import pipeline
from rsna_knee.ontology import TARGET_COLUMNS
from rsna_knee.roots import roots_for
from rsna_knee.workflow.registry import Registry
from rsna_knee.workflow.state import PipelineState, Stage


def setup(tmp_path, monkeypatch, n=12):
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    folds = [
        {"StudyInstanceUID": "gold", "fold": "0", "is_gold": "1", "gold_split": "eval", "group_id": "held"},
        {"StudyInstanceUID": "duplicate", "fold": "0", "is_gold": "0", "gold_split": "weak", "group_id": "held"},
        {"StudyInstanceUID": "gold-train", "fold": "1", "is_gold": "1", "gold_split": "train_image_ok", "group_id": "gold-train"},
    ]
    folds += [{"StudyInstanceUID": f"dev-{i}", "fold": "1", "is_gold": "0", "gold_split": "weak", "group_id": f"dev-{i}"} for i in range(n)]
    with roots_for(False).folds.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(folds[0]))
        w.writeheader()
        w.writerows(folds)
    with (metadata / "train.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=["StudyInstanceUID", "Report"])
        w.writeheader()
        w.writerows({"StudyInstanceUID": r["StudyInstanceUID"], "Report": "Fixture report."} for r in folds)
    monkeypatch.setattr(pipeline, "METADATA_DIR", metadata)
    monkeypatch.setattr(pipeline, "LMStudioClient", lambda **kwargs: SimpleNamespace(
        list_models=lambda: [], resolve_model=lambda *a: SimpleNamespace(id="local", raw={}),
        probe_json_schema=lambda *a, **kw: SimpleNamespace(mode="json_schema", as_dict=lambda: {}), close=lambda: None))
    return PipelineState(run_id="dev-test", synthetic=False, profile="pilot"), Registry(tmp_path / "reg.sqlite")


def test_development_excludes_all_gold_and_gold_eval_duplicate_groups(tmp_path, monkeypatch):
    state, registry = setup(tmp_path, monkeypatch)
    called = []
    def extract(client, **kw):
        called.append(kw["uid"])
        return {"status": "ok", "payload": {"StudyInstanceUID": kw["uid"], "targets": {
            t: {"state": "not_mentioned", "evidence_span": "", "criterion_mapping": "unmentioned"} for t in TARGET_COLUMNS}}}
    monkeypatch.setattr(pipeline, "extract_one", extract)
    out = pipeline.stage_labels(state, registry, limit=2, live=True, development_only=True)
    assert called == ["dev-0", "dev-1"]
    assert out.artifacts["label_selection"] == "non_gold_development_only"


def test_limit_bounds_attempts_and_quality_gate_stops_bad_bulk(tmp_path, monkeypatch):
    state, registry = setup(tmp_path, monkeypatch)
    called = []
    def extract(client, **kw):
        called.append(kw["uid"])
        return {"status": "quarantine", "reason": "ACL:evidence_not_in_report"}
    monkeypatch.setattr(pipeline, "extract_one", extract)
    out = pipeline.stage_labels(state, registry, limit=3, live=True, development_only=True)
    assert len(called) == 3
    assert out.stage == Stage.LABEL_PILOT
    called.clear()
    out = pipeline.stage_labels(state, registry, limit=100, live=True, development_only=True)
    assert len(called) == 10
    assert out.stage == Stage.BLOCKED
    assert out.blocked_reason == "label_quality_failure_rate"
    assert json.loads(out.artifacts["label_counts"])["quarantined"] == 10
