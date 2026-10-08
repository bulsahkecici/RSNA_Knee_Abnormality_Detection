"""Recover exact source spans when an extractor only changed whitespace."""

from __future__ import annotations

import copy
import re
from typing import Any

EVIDENCE_POLICY_VERSION = "unique-whitespace-source-span.v1"


def recover_source_spans(payload: dict[str, Any], report: str) -> tuple[dict[str, Any], list[str]]:
    """Replace a whitespace-only mismatch with its unique verbatim source span.

    No case, punctuation, or word changes are accepted. Ambiguous matches remain
    untouched so the strict evidence validator can reject them. The raw model
    response remains available in the local provenance record.
    """
    result = copy.deepcopy(payload)
    repaired: list[str] = []
    for name, rec in (result.get("targets") or {}).items():
        if not isinstance(rec, dict):
            continue
        span = rec.get("evidence_span")
        if not isinstance(span, str) or not span or span in report:
            continue
        parts = re.split(r"(\s+)", span)
        pattern = "".join(r"\s+" if part.isspace() else re.escape(part) for part in parts)
        matches = list(re.finditer(pattern, report))
        if len(matches) == 1:
            rec["evidence_span"] = matches[0].group(0)
            repaired.append(name)
    return result, repaired
