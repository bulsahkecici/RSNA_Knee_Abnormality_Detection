"""Package an offline inference bundle with hashes. No .env."""

from __future__ import annotations

import json
import shutil
import tarfile
from pathlib import Path
from typing import Any

from rsna_knee.hashing import sha256_file
from rsna_knee.paths import ROOT, SUBMISSIONS_DIR, ensure_runtime_dirs

INCLUDE = [
    "src",
    "configs",
    "schemas",
    "prompts",
    "pyproject.toml",
    "notebooks/04_kaggle_inference.ipynb",
]


def package_run(run_id: str, extra_files: list[Path] | None = None) -> dict[str, Any]:
    ensure_runtime_dirs()
    dest_dir = SUBMISSIONS_DIR / run_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    for rel in INCLUDE:
        src = ROOT / rel
        if not src.exists():
            continue
        target = dest_dir / rel
        if src.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(src, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".venv"))
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
        if target.is_file():
            files[rel] = sha256_file(target)
    if extra_files:
        for path in extra_files:
            if path.exists() and path.is_file():
                shutil.copy2(path, dest_dir / path.name)
                files[path.name] = sha256_file(dest_dir / path.name)
    manifest = {
        "run_id": run_id,
        "files": files,
        "internet_required": False,
        "lmstudio_required": False,
        "drive_required": False,
        "note": "Scoring notebook must run offline. Bundle wheels for the actual OS/Python/CUDA separately after measuring the worker.",
    }
    man_path = dest_dir / "MANIFEST.json"
    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tar_path = SUBMISSIONS_DIR / f"{run_id}.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(dest_dir, arcname=run_id)
    return {"dir": str(dest_dir), "tar": str(tar_path), "sha256": sha256_file(tar_path), "manifest": manifest}
