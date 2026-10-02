"""Colab bootstrap. Run inside a connected Colab kernel after handshake, not as a Mac GPU allocator."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def handshake() -> dict:
    info = {
        "python": sys.version,
        "platform": platform.platform(),
        "disk": shutil.disk_usage("/").free,
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.is_available()
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        info["bf16"] = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    except Exception as exc:  # noqa: BLE001
        info["torch_error"] = str(exc)
    Path("/content/rsna_handshake.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))
    return info


def pip_install() -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pydicom", "numpy", "pandas", "pyyaml", "scikit-learn"])


if __name__ == "__main__":
    pip_install()
    handshake()
