"""Interruptible clock. Tests inject a virtual clock; production uses wall time."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class Clock:
    monotonic_fn: Callable[[], float] = time.monotonic
    sleep_fn: Callable[[float], None] = time.sleep

    def monotonic(self) -> float:
        return float(self.monotonic_fn())

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            self.sleep_fn(seconds)


@dataclass
class VirtualClock:
    now: float = 0.0
    sleeps: list[float] = field(default_factory=list)

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += max(0.0, seconds)

    def as_clock(self) -> Clock:
        return Clock(monotonic_fn=self.monotonic, sleep_fn=self.sleep)


WALL = Clock()
