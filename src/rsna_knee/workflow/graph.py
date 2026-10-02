"""Durable pipeline graph.

LangGraph is used when installed. The same stage functions run without it so
tests and CLI stay functional. Completed hashes are checked before advancing.
"""

from __future__ import annotations

import json
import signal
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rsna_knee.hashing import sha256_file, sha256_json
from rsna_knee.paths import STATE_DIR
from rsna_knee.roots import roots_for
from rsna_knee.workflow.locks import ControllerLock, new_fencing_token, new_run_id
from rsna_knee.workflow.registry import Registry
from rsna_knee.workflow.state import PipelineState, Stage

_PAUSE = {"flag": False}


def _install_sigint() -> None:
    def handler(signum: int, frame: Any) -> None:  # noqa: ARG001
        _PAUSE["flag"] = True

    try:
        signal.signal(signal.SIGINT, handler)
    except (ValueError, OSError):
        pass


def paused() -> bool:
    return bool(_PAUSE["flag"])


def request_pause() -> None:
    _PAUSE["flag"] = True


def clear_pause() -> None:
    _PAUSE["flag"] = False


StageFn = Callable[[PipelineState, Registry], PipelineState]


def _try_langgraph():
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        from langgraph.graph import END, START, StateGraph

        return SqliteSaver, StateGraph, START, END
    except Exception:
        return None


def build_graph(fns: dict[str, StageFn]):
    packed = _try_langgraph()
    if packed is None:
        return None
    SqliteSaver, StateGraph, START, END = packed

    def wrap(fn: StageFn):
        def node(state: dict[str, Any]) -> dict[str, Any]:
            ps = PipelineState.model_validate(state)
            registry = Registry()
            out = fn(ps, registry)
            return out.model_dump()

        return node

    builder = StateGraph(dict)
    order = [
        "preflight",
        "metadata",
        "folds",
        "labels",
        "cache",
        "runtime",
        "train",
        "evaluate",
        "audit",
        "package",
    ]
    for name in order:
        if name in fns:
            builder.add_node(name, wrap(fns[name]))
    builder.add_edge(START, order[0])
    for a, b in zip(order, order[1:], strict=False):
        if a in fns and b in fns:
            builder.add_edge(a, b)
    builder.add_edge(order[-1], END)
    db = STATE_DIR / "langgraph.sqlite"
    db.parent.mkdir(parents=True, exist_ok=True)

    class _Compiled:
        def __init__(self):
            self.conn_cm = SqliteSaver.from_conn_string(str(db))
            self.checkpointer = self.conn_cm.__enter__()
            self.graph = builder.compile(checkpointer=self.checkpointer)

        def invoke(self, state: dict[str, Any], thread_id: str) -> dict[str, Any]:
            return self.graph.invoke(state, {"configurable": {"thread_id": thread_id}})

    return _Compiled()


def hash_if_exists(path: str | Path | None) -> str | None:
    if not path:
        return None
    p = Path(path)
    if p.exists() and p.is_file():
        return sha256_file(p)
    return None


def invalidate_if_changed(prev: str | None, current: str | None) -> bool:
    if prev is None or current is None:
        return False
    return prev != current


STOP_STAGES = {
    Stage.FAILED,
    Stage.NEEDS_AUTH,
    Stage.NEEDS_RUNTIME,
    Stage.QUEUED,
    Stage.PAUSED,
    Stage.BLOCKED,
}

_INPUT_KEYS = {
    "preflight": ("config",),
    "metadata": ("config",),
    "folds": ("metadata",),
    "labels": ("metadata", "folds"),
    "cache": ("folds",),
    "runtime": ("config",),
    "train": ("cache", "folds", "labels"),
    "evaluate": ("train",),
    "audit": ("train",),
    "package": ("audit",),
}


def input_hash_for(name: str, state: PipelineState) -> str:
    blob = {key: state.hashes.get(key) for key in _INPUT_KEYS.get(name, ())}
    blob["synthetic"] = state.synthetic
    blob["profile"] = state.profile
    return sha256_json(blob)


def control_requests_pause(state: PipelineState) -> bool:
    path = roots_for(state.synthetic).control
    if not path.is_file():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return data.get("command") == "pause" and data.get("run_id") == state.run_id


def invalidate_from(state: PipelineState, name: str, names: list[str]) -> None:
    if name not in names:
        return
    for later in names[names.index(name) :]:
        state.stage_records.pop(later, None)


def sequential_run(
    state: PipelineState,
    registry: Registry,
    fns: list[tuple[str, StageFn]],
    lock: ControllerLock | None = None,
) -> PipelineState:
    _install_sigint()
    own_lock = lock or ControllerLock(roots_for(state.synthetic).lock)
    own_lock.acquire()
    names = [name for name, _fn in fns]
    try:
        if not state.fencing_token:
            state.fencing_token = new_fencing_token(state.run_id)
        registry.upsert_run(state.run_id, state.profile, state.stage, state.model_dump(), state.synthetic)
        for name, fn in fns:
            if paused() or control_requests_pause(state):
                state.stage = Stage.PAUSED
                state.pause_requested = True
                state.next_action_tr = "Duraklatma kaydı alındı. Aynı run_id ile rsna pipeline run --resume devam eder."
                registry.upsert_run(state.run_id, state.profile, state.stage, state.model_dump(), state.synthetic)
                return state
            incoming = input_hash_for(name, state)
            record = state.stage_records.get(name) or {}
            if state.resume and record.get("status") == "complete" and record.get("input_hash") == incoming:
                continue
            if record and record.get("input_hash") not in {None, incoming}:
                invalidate_from(state, name, names)
            state = fn(state, registry)
            registry.upsert_run(state.run_id, state.profile, state.stage, state.model_dump(), state.synthetic)
            if state.stage in STOP_STAGES:
                return state
            state.stage_records[name] = {"status": "complete", "input_hash": incoming}
            registry.upsert_run(state.run_id, state.profile, state.stage, state.model_dump(), state.synthetic)
        return state
    finally:
        own_lock.release()


def new_state(profile: str, synthetic: bool = False, run_id: str | None = None) -> PipelineState:
    rid = run_id or new_run_id("syn" if synthetic else "run")
    return PipelineState(
        run_id=rid,
        profile=profile,
        synthetic=synthetic,
        stage=Stage.CREATED,
        hashes={"config": sha256_json({"profile": profile, "synthetic": synthetic})},
        next_action_tr="rsna doctor ardından rsna smoke --synthetic",
    )
