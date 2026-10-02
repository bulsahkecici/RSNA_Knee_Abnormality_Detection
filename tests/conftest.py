"""Keep pytest off the real folds, labels, cache, registry, and control files."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def isolate_real_roots(tmp_path, monkeypatch):
    base = tmp_path / "real-roots"
    base.mkdir()
    monkeypatch.setenv("RSNA_ROOTS", str(base))
    monkeypatch.setenv("RSNA_REGISTRY", str(base / "registry.sqlite"))
    monkeypatch.setenv("RSNA_RUNS", str(base / "runs"))
