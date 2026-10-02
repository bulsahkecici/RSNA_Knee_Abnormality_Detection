"""Workspace paths. Runtime directories are created on demand."""

from __future__ import annotations

from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    here = (start or Path.cwd()).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "src" / "rsna_knee").exists():
            return candidate
        if (candidate / "CURSOR_RSNA_MASTER_PROMPT.md").exists() and (candidate / "pyproject.toml").exists():
            return candidate
    return here


ROOT = find_repo_root()
CONFIG_DIR = ROOT / "configs"
SCHEMA_DIR = ROOT / "schemas"
PROMPT_DIR = ROOT / "prompts"
DATA_DIR = ROOT / "data"
METADATA_DIR = DATA_DIR / "metadata"
LABELS_DIR = DATA_DIR / "labels"
CACHE_DIR = DATA_DIR / "cache"
ARTIFACTS_DIR = ROOT / "artifacts"
RUNS_DIR = ARTIFACTS_DIR / "runs"
SUBMISSIONS_DIR = ARTIFACTS_DIR / "submissions"
REPORTS_DIR = ROOT / "reports"
STATE_DIR = ROOT / "state"
TESTS_FIXTURES = ROOT / "tests" / "fixtures"


def ensure_runtime_dirs() -> None:
    for path in (
        METADATA_DIR,
        LABELS_DIR,
        CACHE_DIR,
        RUNS_DIR,
        SUBMISSIONS_DIR,
        REPORTS_DIR,
        STATE_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
        gitkeep = path / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.write_text("", encoding="utf-8")
