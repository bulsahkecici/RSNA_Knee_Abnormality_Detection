"""YAML config loading with env overlay."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from rsna_knee.paths import CONFIG_DIR, ROOT, find_repo_root


class AllocationConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    preferred_gpu_order: list[str] = Field(default_factory=lambda: ["A100", "L4", "T4"])
    max_wait_per_gpu_seconds: int = 300
    max_total_wait_seconds: int = 900
    poll_interval_seconds: int = 15
    respect_retry_after: bool = True
    max_concurrent_runtime_requests: int = 1
    skip_unsupported: bool = True
    accept_usable_allocated_gpu: bool = True
    automatic_acquisition: str = "capability_gated"
    when_exhausted: str = "queue_and_report"


class FallbackTrainingConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    kaggle_enabled: bool = True
    local_mps_full_training_enabled: bool = False
    cpu_full_training_enabled: bool = False


class TrainingRuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    effective_study_batch: int = 8
    concurrent_gpu_jobs: int = 1
    preserve_model_resolution: bool = True
    pilot_minutes_per_run: int = 60


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    provider: str = "colab"
    allocation: AllocationConfig = Field(default_factory=AllocationConfig)
    fallback_training: FallbackTrainingConfig = Field(default_factory=FallbackTrainingConfig)
    training: TrainingRuntimeConfig = Field(default_factory=TrainingRuntimeConfig)


class LmStudioConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    backend: str = "lmstudio"
    base_url: str = "http://127.0.0.1:1234/v1"
    preferred_model_id: str = "qwen/qwen3.8-27b"
    alternate_model_id: str = "qwen/qwen3.5-9b"
    context_tokens: int = 8192
    output_tokens: int = 2048
    max_in_flight: int = 1
    temperature: float = 0.2
    thinking: str = "off"
    json_mode: str = "json_schema"
    timeout_seconds: float = 180.0
    max_retries: int = 2
    auto_downgrade_on_malformed: bool = False


class LabelsConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    ontology_version: str = "2026-08-06.host-733343.v1"
    pilot_limit: int = 100
    gold_eval_fraction: float = 0.5
    prompt_dev_uses_gold: bool = False
    weak_label_loss_weight: float = 0.25
    gold_loss_weight: float = 1.0
    mask_unmentioned: bool = True
    mask_uncertain: bool = True
    mask_sentinel_negative_one: bool = True


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    competition_slug: str = "rsna-knee-abnormality-detection"
    local_full_archive: bool = False
    cache_pilot_studies: int = 20
    image_size: int = 224
    n_slices_per_slot: int = 3
    n_folds: int = 5
    seed: int = 2026
    dtype: str = "uint8"


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    project_name: str = "rsna-knee"
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    lmstudio: LmStudioConfig = Field(default_factory=LmStudioConfig)
    labels: LabelsConfig = Field(default_factory=LabelsConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    raw: dict[str, Any] = Field(default_factory=dict)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_yaml_stack(paths: list[Path]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for path in paths:
        merged = deep_merge(merged, _read_yaml(path))
    return merged


def apply_env_overrides(cfg: dict[str, Any]) -> dict[str, Any]:
    lm = cfg.setdefault("lmstudio", {})
    if os.getenv("RSNA_LMSTUDIO_BASE_URL"):
        lm["base_url"] = os.environ["RSNA_LMSTUDIO_BASE_URL"]
    if os.getenv("RSNA_LMSTUDIO_MODEL_ID"):
        lm["preferred_model_id"] = os.environ["RSNA_LMSTUDIO_MODEL_ID"]
    runtime = cfg.setdefault("runtime", {})
    if os.getenv("RSNA_PROVIDER"):
        runtime["provider"] = os.environ["RSNA_PROVIDER"]
    return cfg


def load_config(profile: str = "default", extra: list[Path] | None = None) -> AppConfig:
    root = find_repo_root()
    config_dir = root / "configs" if (root / "configs").exists() else CONFIG_DIR
    paths = [
        config_dir / "default.yaml",
        config_dir / "competition.yaml",
        config_dir / "labels.yaml",
        config_dir / "runtime.yaml",
        config_dir / "resources.yaml",
    ]
    if extra:
        paths.extend(extra)
    merged = apply_env_overrides(load_yaml_stack(paths))
    cfg = AppConfig.model_validate(merged)
    cfg.raw = merged
    cfg.raw["_root"] = str(root)
    cfg.raw["_profile"] = profile
    return cfg


def load_experiment(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    data = _read_yaml(p)
    if not data:
        raise FileNotFoundError(f"Experiment config missing: {p}")
    data["_path"] = str(p)
    return data
