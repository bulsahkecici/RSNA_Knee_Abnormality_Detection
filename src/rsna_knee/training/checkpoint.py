"""Atomic checkpoints with model/optim/RNG/hashes. CPU map_location restore."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from rsna_knee.hashing import sha256_json


def save_numpy_checkpoint(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".writing.npz")
    arrays = {k: np.asarray(v) if not isinstance(v, np.ndarray) else v for k, v in payload.items() if k != "meta"}
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    tmp.replace(path)
    meta = payload.get("meta", {})
    meta_path = path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return path


def load_numpy_checkpoint(path: Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    out = {k: data[k] for k in data.files}
    meta_path = path.with_suffix(".json")
    if meta_path.exists():
        out["meta"] = json.loads(meta_path.read_text(encoding="utf-8"))
    return out


def torch_save(path: Path, obj: dict[str, Any]) -> Path:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".writing")
    torch.save(obj, tmp)
    tmp.replace(path)
    return path


def torch_load(path: Path, map_location: str = "cpu") -> dict[str, Any]:
    import torch

    return torch.load(path, map_location=map_location, weights_only=False)


def rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {"python": random.getstate(), "numpy": np.random.get_state()}
    try:
        import torch

        state["torch"] = torch.get_rng_state()
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
    except Exception:
        pass
    return state


def restore_rng(state: dict[str, Any]) -> None:
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    try:
        import torch

        if "torch" in state:
            torch.set_rng_state(state["torch"])
    except Exception:
        pass


def checkpoint_meta(**hashes: str) -> dict[str, Any]:
    return {"hashes": hashes, "hashes_digest": sha256_json(hashes)}
