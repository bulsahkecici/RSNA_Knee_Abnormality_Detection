from __future__ import annotations

from rsna_knee.hashing import sha256_text
from rsna_knee.paths import CACHE_DIR, LABELS_DIR, STATE_DIR
from rsna_knee.pipeline import run_pipeline


def test_synthetic_smoke_does_not_touch_real_roots(tmp_path, monkeypatch):
    folds = STATE_DIR / "folds.csv"
    labels = LABELS_DIR / "real-labels.jsonl"
    cache = CACHE_DIR / "index.json"
    folds.parent.mkdir(parents=True, exist_ok=True)
    labels.parent.mkdir(parents=True, exist_ok=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    sentinel = "StudyInstanceUID,fold\nREAL,0\n"
    folds.write_text(sentinel, encoding="utf-8")
    labels.write_text("{}\n", encoding="utf-8")
    before_folds = sha256_text(sentinel)
    before_labels = labels.read_text(encoding="utf-8")
    before_cache = cache.read_text(encoding="utf-8") if cache.exists() else None
    state = run_pipeline("smoke", synthetic=True, resume=False)
    assert sha256_text(folds.read_text(encoding="utf-8")) == before_folds
    assert labels.read_text(encoding="utf-8") == before_labels
    after_cache = cache.read_text(encoding="utf-8") if cache.exists() else None
    assert after_cache == before_cache
    assert "state/synthetic" in state.artifacts["folds"]
    assert state.artifacts["submission"]
    assert "state/synthetic" in state.artifacts["submission"]
