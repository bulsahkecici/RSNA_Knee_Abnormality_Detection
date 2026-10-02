from __future__ import annotations

import json

from rsna_knee.roots import roots_for
from rsna_knee.workflow.graph import sequential_run
from rsna_knee.workflow.locks import ControllerLock
from rsna_knee.workflow.registry import Registry
from rsna_knee.workflow.state import PipelineState, Stage


def test_resume_skips_only_matching_stage_and_invalidates_downstream(tmp_path):
    calls: list[str] = []

    def folds(state, registry):
        calls.append("folds")
        state.hashes["folds"] = "fold-hash"
        state.stage = Stage.SPLIT_FROZEN
        state.artifacts["marker"] = "kept"
        return state

    def labels(state, registry):
        calls.append("labels")
        state.hashes["labels"] = "label-hash"
        state.stage = Stage.LABELS_READY
        return state

    registry = Registry(tmp_path / "reg.sqlite")
    state = PipelineState(run_id="run-resume", synthetic=False, resume=False, hashes={"metadata": "m1", "config": "c"})
    lock = ControllerLock(tmp_path / "lock")
    first = sequential_run(state, registry, [("folds", folds), ("labels", labels)], lock=lock)
    assert calls == ["folds", "labels"]
    assert first.artifacts["marker"] == "kept"

    calls.clear()
    again = PipelineState.model_validate(registry.get_run("run-resume"))
    again.resume = True
    sequential_run(again, registry, [("folds", folds), ("labels", labels)], lock=lock)
    assert calls == []

    calls.clear()
    changed = PipelineState.model_validate(registry.get_run("run-resume"))
    changed.resume = True
    changed.hashes["metadata"] = "m2"
    sequential_run(changed, registry, [("folds", folds), ("labels", labels)], lock=lock)
    assert calls == ["folds", "labels"]


def test_pause_control_does_not_replace_payload(tmp_path):
    def blocked(state, registry):
        state.stage = Stage.CACHE_READY
        state.artifacts["folds"] = "real-folds"
        return state

    registry = Registry(tmp_path / "reg.sqlite")
    state = PipelineState(
        run_id="run-pause",
        synthetic=False,
        hashes={"config": "c"},
        artifacts={"folds": "real-folds"},
    )
    path = roots_for(False).control
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"run_id": "run-pause", "command": "pause"}), encoding="utf-8")
    try:
        out = sequential_run(state, registry, [("cache", blocked)], lock=ControllerLock(tmp_path / "lock2"))
        assert out.stage == Stage.PAUSED
        stored = registry.get_run("run-pause")
        assert stored["artifacts"]["folds"] == "real-folds"
        assert "real-folds" in json.dumps(stored)
    finally:
        path.unlink(missing_ok=True)
