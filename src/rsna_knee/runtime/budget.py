"""CPU/GPU gates on the path that actually starts training."""

from __future__ import annotations

from rsna_knee.config import load_config


def dataset_float_bytes(n_studies: int, img_size: int, *, slots: int = 3, centers: int = 3) -> int:
    return max(0, n_studies) * slots * centers * 3 * img_size * img_size * 4


def execution_allowed(
    *,
    profile: str,
    n_studies: int,
    img_size: int,
    cuda: bool,
    slots: int = 3,
    centers: int = 3,
) -> tuple[bool, str]:
    """Campaign and oversized float32 sets stay off CPU. A small pilot may run."""
    cfg = load_config()
    if profile == "campaign" and not cuda:
        return False, "campaign_requires_gpu"
    if cuda:
        return True, "cuda"
    profiles = (cfg.raw or {}).get("profiles") or {}
    max_studies = (profiles.get(profile) or {}).get("max_studies")
    if max_studies is not None and n_studies > int(max_studies):
        return False, "pilot_study_limit"
    if dataset_float_bytes(n_studies, img_size, slots=slots, centers=centers) > 8 * 1024**3:
        return False, "ram_budget"
    return True, "cpu_pilot"
