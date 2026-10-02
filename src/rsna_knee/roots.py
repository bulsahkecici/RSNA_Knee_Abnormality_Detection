"""Separate roots so synthetic smoke cannot touch real folds, labels, or cache."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rsna_knee.paths import CACHE_DIR, LABELS_DIR, ROOT, RUNS_DIR, STATE_DIR, SUBMISSIONS_DIR


@dataclass(frozen=True)
class RunRoots:
    synthetic: bool
    state: Path
    folds: Path
    labels: Path
    cache: Path
    runs: Path
    submissions: Path
    registry: Path
    control: Path
    lock: Path

    def ensure(self) -> None:
        for path in (self.state, self.labels, self.cache, self.runs, self.submissions):
            path.mkdir(parents=True, exist_ok=True)


def roots_for(synthetic: bool) -> RunRoots:
    if synthetic:
        base = ROOT / "state" / "synthetic"
        return RunRoots(
            synthetic=True,
            state=base,
            folds=base / "folds.csv",
            labels=base / "labels",
            cache=base / "cache",
            runs=base / "runs",
            submissions=base / "submissions",
            registry=base / "registry.sqlite",
            control=base / "control.json",
            lock=base / "controller.lock",
        )
    return RunRoots(
        synthetic=False,
        state=STATE_DIR,
        folds=STATE_DIR / "folds.csv",
        labels=LABELS_DIR,
        cache=CACHE_DIR,
        runs=RUNS_DIR,
        submissions=SUBMISSIONS_DIR,
        registry=STATE_DIR / "registry.sqlite",
        control=STATE_DIR / "control.json",
        lock=STATE_DIR / "controller.lock",
    )
