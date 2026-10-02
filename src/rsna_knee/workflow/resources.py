"""Local resource budgets. Not a hard guarantee against macOS jetsam."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import psutil


@dataclass
class ResourceSnapshot:
    physical_ram_gb: float
    available_ram_gb: float
    process_rss_gb: float
    cpu_count: int
    notes: str


def snapshot() -> ResourceSnapshot:
    vm = psutil.virtual_memory()
    proc = psutil.Process()
    rss = proc.memory_info().rss / (1024**3)
    return ResourceSnapshot(
        physical_ram_gb=vm.total / (1024**3),
        available_ram_gb=vm.available / (1024**3),
        process_rss_gb=rss,
        cpu_count=os_cpu_count(),
        notes="macOS memory pressure is not a hard lock-prevention guarantee.",
    )


def os_cpu_count() -> int:
    return psutil.cpu_count(logical=True) or 1


def ram_guard(need_gb: float, snap: ResourceSnapshot | None = None) -> dict[str, Any]:
    snap = snap or snapshot()
    ok = snap.available_ram_gb >= need_gb + 2.0
    return {
        "ok": ok,
        "need_gb": need_gb,
        "available_gb": round(snap.available_ram_gb, 2),
        "physical_gb": round(snap.physical_ram_gb, 2),
        "action_tr": None
        if ok
        else "Bellek bütçesi yetmiyor. Tam DICOM arşivini Mac'e yüklemeyin; cache pilotunu küçültün.",
    }
