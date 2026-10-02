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


def _kaggle_username() -> str | None:
    import json as _json
    import os

    env = os.environ.get("KAGGLE_USERNAME")
    if env:
        return env
    path = Path.home() / ".kaggle" / "kaggle.json"
    if not path.is_file():
        return None
    try:
        return _json.loads(path.read_text(encoding="utf-8")).get("username")
    except (OSError, ValueError):
        return None


def _hash_tree(root: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.json" and "__pycache__" not in path.parts:
            files[str(path.relative_to(root))] = sha256_file(path)
    return files


def package_run(
    run_id: str,
    extra_files: list[Path] | None = None,
    *,
    dest_root: Path | None = None,
    synthetic: bool = False,
    checkpoint: Path | None = None,
) -> dict[str, Any]:
    ensure_runtime_dirs()
    dest_root = dest_root or SUBMISSIONS_DIR
    dest_dir = dest_root / run_id
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    for rel in INCLUDE:
        src = ROOT / rel
        if not src.exists():
            continue
        target = dest_dir / rel
        if src.is_dir():
            shutil.copytree(src, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".venv"))
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
    copied_checkpoint = None
    if checkpoint and checkpoint.is_file():
        copied_checkpoint = dest_dir / "checkpoint" / checkpoint.name
        copied_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checkpoint, copied_checkpoint)
    if extra_files:
        for path in extra_files:
            if path.exists() and path.is_file():
                shutil.copy2(path, dest_dir / path.name)
    user = _kaggle_username()
    kernel_meta = {
        "id": f"{user}/rsna-knee-infer" if user else None,
        "title": "rsna-knee-infer",
        "code_file": "notebooks/04_kaggle_inference.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "dataset_sources": [],
        "kernel_sources": [],
        "push_ready": bool(user) and copied_checkpoint is not None and not synthetic,
    }
    (dest_dir / "kernel-metadata.json").write_text(json.dumps(kernel_meta, indent=2), encoding="utf-8")
    files = _hash_tree(dest_dir)
    manifest = {
        "run_id": run_id,
        "synthetic": synthetic,
        "encoder": "tiny_test_encoder" if synthetic else "dinov2_vits14",
        "checkpoint": None if copied_checkpoint is None else str(copied_checkpoint.relative_to(dest_dir)),
        "files": files,
        "internet_required": False,
        "lmstudio_required": False,
        "drive_required": False,
        "kernel_metadata": "kernel-metadata.json",
        "note": "Scoring notebook must run offline. Default audit is not PASS.",
    }
    man_path = dest_dir / "MANIFEST.json"
    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tar_path = dest_root / f"{run_id}.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(dest_dir, arcname=run_id)
    return {
        "dir": str(dest_dir),
        "tar": str(tar_path),
        "sha256": sha256_file(tar_path),
        "manifest": manifest,
        "identity_errors": verify_package(dest_dir),
    }


def verify_package(dest_dir: Path) -> list[str]:
    man_path = dest_dir / "MANIFEST.json"
    if not man_path.is_file():
        return ["manifest_missing"]
    manifest = json.loads(man_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if manifest.get("synthetic") is True:
        errors.append("synthetic_manifest")
    if not manifest.get("checkpoint"):
        errors.append("checkpoint_missing")
    else:
        ckpt = dest_dir / manifest["checkpoint"]
        if not ckpt.is_file():
            errors.append("checkpoint_file_missing")
    kernel = dest_dir / "kernel-metadata.json"
    if not kernel.is_file():
        errors.append("kernel_metadata_missing")
    else:
        meta = json.loads(kernel.read_text(encoding="utf-8"))
        if meta.get("enable_internet") is not False:
            errors.append("internet_not_disabled")
        if "rsna-knee-abnormality-detection" not in (meta.get("competition_sources") or []):
            errors.append("competition_source_missing")
    for rel, digest in (manifest.get("files") or {}).items():
        path = dest_dir / rel
        if not path.is_file():
            errors.append(f"missing_file:{rel}")
        elif sha256_file(path) != digest:
            errors.append(f"hash_mismatch:{rel}")
    return errors
