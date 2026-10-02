"""Select among predefined experiment YAMLs. No arbitrary exec of LLM code."""

from __future__ import annotations


from rsna_knee.config import load_experiment
from rsna_knee.paths import ROOT

QUEUE = [
    "configs/experiments/EXP-001-reference.yaml",
    "configs/experiments/EXP-002-frozen.yaml",
    "configs/experiments/EXP-003-local-labels.yaml",
    "configs/experiments/EXP-004-finetune.yaml",
]


def listed_experiments() -> list[dict]:
    return [load_experiment(ROOT / p) for p in QUEUE]


def choose(experiment_id: str) -> dict:
    for item in listed_experiments():
        if item.get("experiment_id") == experiment_id:
            return item
    raise KeyError(experiment_id)
