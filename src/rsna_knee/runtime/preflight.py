"""Environment doctor / preflight. Measures the real machine."""

from __future__ import annotations

import platform
import shutil
import sys
from typing import Any

import httpx

from rsna_knee import __version__
from rsna_knee.config import load_config
from rsna_knee.paths import find_repo_root
from rsna_knee.runtime.providers.colab import ColabProvider
from rsna_knee.runtime.providers.kaggle import KaggleProvider
from rsna_knee.runtime.providers.local import LocalProvider
from rsna_knee.workflow.resources import snapshot


def _lmstudio(base: str) -> dict[str, Any]:
    url = base.rstrip("/") + "/models"
    try:
        resp = httpx.get(url, timeout=2.0)
        resp.raise_for_status()
        ids = [m.get("id") for m in resp.json().get("data", [])]
        return {"ok": True, "models": ids, "base_url": base}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc), "base_url": base}


def doctor(*, probe_remote: bool = True) -> dict[str, Any]:
    cfg = load_config()
    snap = snapshot()
    cuda = False
    mps = False
    torch_v = None
    try:
        import torch

        torch_v = torch.__version__
        cuda = bool(torch.cuda.is_available())
        mps = bool(torch.backends.mps.is_available())
    except Exception:
        pass
    kaggle = shutil.which("kaggle")
    report = {
        "rsna_knee": __version__,
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "ram_gb": round(snap.physical_ram_gb, 2),
        "available_ram_gb": round(snap.available_ram_gb, 2),
        "cpu_count": snap.cpu_count,
        "torch": torch_v,
        "cuda": cuda,
        "mps": mps,
        "cuda_expected_on_macos": False,
        "kaggle_cli": kaggle,
        "lmstudio": _lmstudio(cfg.lmstudio.base_url) if probe_remote else {"ok": None, "probed": False},
        "preferred_lm_model": cfg.lmstudio.preferred_model_id,
        "providers": {
            "local": LocalProvider().capabilities().as_dict(),
            "colab": ColabProvider().capabilities().as_dict(),
            "kaggle": KaggleProvider().capabilities(probe=probe_remote).as_dict(),
        },
        "repo": str(find_repo_root()),
        "notes": [
            "Büyük görüntü eğitimi varsayılan olarak remote GPU'dadır.",
            "Colab otomatik GPU tahsisi capability_gated: headless API yoksa handoff.",
        ],
    }
    return report
