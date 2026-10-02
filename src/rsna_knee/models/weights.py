"""DINOv2 weight trust. A matching architecture is not a pretrained model."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from rsna_knee.hashing import sha256_file
from rsna_knee.paths import CONFIG_DIR

OFFICIAL_WEIGHTS_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"
# Published checksums are not invented here. Add a hash only after the file is downloaded
# and hashed locally into configs/dinov2_vits14_allowlist.json or RSNA_DINOV2_ALLOWLIST.
PUBLISHED_SHA256: set[str] = set()


def allowlist_path() -> Path | None:
    env = os.environ.get("RSNA_DINOV2_ALLOWLIST")
    if env:
        return Path(env)
    default = CONFIG_DIR / "dinov2_vits14_allowlist.json"
    return default if default.is_file() else None


def allowlisted_sha256() -> set[str]:
    path = allowlist_path()
    if path is None or not path.is_file():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    values = data.get("sha256") if isinstance(data, dict) else data
    if not isinstance(values, list):
        return set()
    return {str(item) for item in values if item}


def weight_trust(path: Path) -> dict[str, Any]:
    digest = sha256_file(path)
    allowlisted = digest in allowlisted_sha256()
    official = digest in PUBLISHED_SHA256
    return {
        "sha256": digest,
        "allowlisted": allowlisted,
        "official": official,
        "pretrained": official,
        "reason": "official_hash" if official else ("allowlisted" if allowlisted else "not_allowlisted"),
        "url": OFFICIAL_WEIGHTS_URL,
    }
