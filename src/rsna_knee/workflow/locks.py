"""Competing controller lock and fencing tokens."""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

from filelock import FileLock, Timeout

from rsna_knee.paths import STATE_DIR, ensure_runtime_dirs


class ControllerLock:
    def __init__(self, path: Path | None = None, timeout: float = 1.0):
        ensure_runtime_dirs()
        self.path = path or (STATE_DIR / "controller.lock")
        self.timeout = timeout
        self._lock = FileLock(str(self.path), timeout=timeout)

    def acquire(self) -> None:
        try:
            self._lock.acquire()
        except Timeout as exc:
            raise RuntimeError(
                "Başka bir controller çalışıyor. Aynı anda iki pipeline açmayın."
            ) from exc

    def release(self) -> None:
        if self._lock.is_locked:
            self._lock.release()

    def __enter__(self) -> ControllerLock:
        self.acquire()
        return self

    def __exit__(self, *args: object) -> None:
        self.release()


def new_fencing_token(run_id: str, attempt: str | None = None) -> str:
    attempt = attempt or uuid.uuid4().hex[:8]
    return f"{run_id}:{attempt}:{uuid.uuid4().hex}:{int(time.time())}:{os.getpid()}"


def new_lease_id() -> str:
    return f"lease-{uuid.uuid4().hex}"


def new_run_id(prefix: str = "run") -> str:
    return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
