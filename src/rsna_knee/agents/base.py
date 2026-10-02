"""Runtime Python agents. Markdown skills are contracts, not these functions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class AgentContract:
    name: str
    reads: list[str]
    writes: list[str]
    budget: dict[str, Any]
    acceptance: list[str]
