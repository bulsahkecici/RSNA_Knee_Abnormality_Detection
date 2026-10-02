from __future__ import annotations

from rsna_knee.hashing import sha256_file
from rsna_knee.paths import CACHE_DIR, LABELS_DIR, STATE_DIR
from rsna_knee.pipeline import run_pipeline


def test_synthetic_smoke_does_not_touch_real_roots():
    targets = [
        STATE_DIR / "folds.csv",
        LABELS_DIR / "real-labels.jsonl",
        CACHE_DIR / "index.json",
        CACHE_DIR / "manifest.json",
        STATE_DIR / "registry.sqlite",
        STATE_DIR / "control.json",
    ]
    before = {str(path): sha256_file(path) if path.is_file() else None for path in targets}
    state = run_pipeline("smoke", synthetic=True, resume=False)
    after = {str(path): sha256_file(path) if path.is_file() else None for path in targets}
    assert after == before
    assert "state/synthetic" in state.artifacts["folds"]
    assert state.artifacts["submission"]
    assert "state/synthetic" in state.artifacts["submission"]
    assert state.stage != "READY_TO_SUBMIT"
