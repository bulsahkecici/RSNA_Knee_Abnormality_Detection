"""Prepare a Kaggle kernel that calls the real cache or inference code.

push/status/output are the documented CLI. This module does not push unless execute=True.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from rsna_knee.paths import ROOT, SUBMISSIONS_DIR
from rsna_knee.runtime.providers.kaggle import KaggleProvider


def _username() -> str | None:
    env = os.environ.get("KAGGLE_USERNAME")
    if env:
        return env
    path = Path.home() / ".kaggle" / "kaggle.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("username")
    except (OSError, ValueError):
        return None


def prepare_cache_kernel(dest: Path | None = None) -> dict[str, Any]:
    dest = dest or (SUBMISSIONS_DIR / "cache_kernel")
    dest.mkdir(parents=True, exist_ok=True)
    script = dest / "cache_entry.py"
    script.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "import os",
                "from rsna_knee.data.cache_build import build_cache",
                "from rsna_knee.data.metadata import discover_input_root",
                "root = discover_input_root()",
                "if root is None:",
                "    raise SystemExit('BLOCKED: competition input is not mounted')",
                "limit = os.environ.get('RSNA_CACHE_LIMIT')",
                "manifest = build_cache(",
                "    series_csv=root / 'train_series.csv',",
                "    dicom_root=root / 'train',",
                "    dest=Path('/kaggle/working/cache'),",
                "    limit=int(limit) if limit else None,",
                ")",
                "print(manifest['n_studies'], manifest['n_quarantine'], manifest['path'])",
                "",
            ]
        ),
        encoding="utf-8",
    )
    user = _username()
    meta = {
        "id": f"{user}/rsna-knee-cache" if user else None,
        "title": "rsna-knee-cache",
        "code_file": "cache_entry.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": False,
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "dataset_sources": [],
        "kernel_sources": [],
    }
    (dest / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    kernel_ref = meta["id"] or "OWNER/rsna-knee-cache"
    return {
        "status": "kernel_prepared",
        "dir": str(dest),
        "push_argv": ["kaggle", "kernels", "push", "-p", str(dest)],
        "status_argv": ["kaggle", "kernels", "status", kernel_ref],
        "output_argv": ["kaggle", "kernels", "output", kernel_ref, "-p", str(dest / "output")],
        "executed": False,
        "downloaded_full_archive": False,
    }


def push_status_output(folder: Path, kernel: str, *, execute: bool = False, provider: KaggleProvider | None = None) -> dict[str, Any]:
    provider = provider or KaggleProvider()
    plan = {
        "push_argv": ["kaggle", "kernels", "push", "-p", str(folder)],
        "status_argv": ["kaggle", "kernels", "status", kernel],
        "output_argv": ["kaggle", "kernels", "output", kernel, "-p", str(folder / "output")],
        "executed": execute,
    }
    if not execute:
        return plan
    plan["push"] = provider.push_kernel(str(folder))
    plan["status"] = provider.poll({"kernel": kernel})
    plan["output"] = provider.kernel_output(kernel, str(folder / "output"))
    return plan


def repo_notebook(name: str) -> Path:
    return ROOT / "notebooks" / name
