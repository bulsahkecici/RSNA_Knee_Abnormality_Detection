"""Pipeline states, events, and typed records."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0.0"


class Stage(StrEnum):
    CREATED = "CREATED"
    PREFLIGHT = "PREFLIGHT"
    METADATA_READY = "METADATA_READY"
    SPLIT_FROZEN = "SPLIT_FROZEN"
    LABEL_PILOT = "LABEL_PILOT"
    LABELS_READY = "LABELS_READY"
    CACHE_PILOT = "CACHE_PILOT"
    CACHE_READY = "CACHE_READY"
    WAITING_RUNTIME = "WAITING_RUNTIME"
    TRAINING = "TRAINING"
    EVALUATING = "EVALUATING"
    AUDITING = "AUDITING"
    PACKAGING = "PACKAGING"
    READY_TO_SUBMIT = "READY_TO_SUBMIT"
    SUBMITTED = "SUBMITTED"
    QUEUED = "QUEUED"
    NEEDS_AUTH = "NEEDS_AUTH"
    NEEDS_RUNTIME = "NEEDS_RUNTIME"
    PAUSED = "PAUSED"
    FAILED = "FAILED"
    SCORED = "SCORED"


TERMINAL = {Stage.SUBMITTED, Stage.SCORED, Stage.FAILED}
RESUME_OK = {
    Stage.CREATED,
    Stage.PREFLIGHT,
    Stage.METADATA_READY,
    Stage.SPLIT_FROZEN,
    Stage.LABEL_PILOT,
    Stage.LABELS_READY,
    Stage.CACHE_PILOT,
    Stage.CACHE_READY,
    Stage.WAITING_RUNTIME,
    Stage.TRAINING,
    Stage.EVALUATING,
    Stage.AUDITING,
    Stage.PACKAGING,
    Stage.READY_TO_SUBMIT,
    Stage.QUEUED,
    Stage.NEEDS_AUTH,
    Stage.NEEDS_RUNTIME,
    Stage.PAUSED,
}


class JobRecord(BaseModel):
    model_config = ConfigDict(extra="allow")
    schema_version: str = SCHEMA_VERSION
    job_id: str
    run_id: str
    attempt_id: str
    stage: str
    commit_sha: str | None = None
    bundle_sha256: str
    config_hash: str
    fold_hash: str
    labels_hash: str
    cache_hash: str
    input_uris: list[str] = Field(default_factory=list)
    desired_gpu_profile: str = "A100"
    budgets: dict[str, Any] = Field(default_factory=dict)
    lease_id: str
    fencing_token: str
    synthetic: bool = False


class ResultMetrics(BaseModel):
    model_config = ConfigDict(extra="allow")
    kind: Literal["real", "synthetic", "unavailable"]
    macro_auc: float | None = None
    per_class: dict[str, Any] = Field(default_factory=dict)
    notes: str | None = None


class ResultRecord(BaseModel):
    model_config = ConfigDict(extra="allow")
    schema_version: str = SCHEMA_VERSION
    job_id: str
    run_id: str
    attempt_id: str
    fencing_token: str
    handshake: dict[str, Any] = Field(default_factory=dict)
    metrics: ResultMetrics
    checkpoint_uris: list[str] = Field(default_factory=list)
    artifact_hashes: dict[str, str] = Field(default_factory=dict)
    exit_reason: str
    heartbeat_ts: float | None = None


class PipelineState(BaseModel):
    """LangGraph-compatible typed state. Values persist via checkpointer + registry."""

    model_config = ConfigDict(extra="allow")
    run_id: str
    stage: Stage = Stage.CREATED
    profile: str = "pilot"
    resume: bool = True
    synthetic: bool = False
    hashes: dict[str, str] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)
    next_action_tr: str = ""
    blocked_reason: str | None = None
    error: str | None = None
    fencing_token: str | None = None
    lease_id: str | None = None
    gpu_desired: str | None = None
    gpu_actual: str | None = None
    last_heartbeat: float | None = None
    events: list[dict[str, Any]] = Field(default_factory=list)
