from __future__ import annotations

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _chdir_root(monkeypatch):
    monkeypatch.chdir(ROOT)
    os.environ.setdefault("RSNA_LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
