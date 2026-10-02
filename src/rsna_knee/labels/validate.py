"""Validate extraction payloads against schema, evidence, and ontology."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema

from rsna_knee.ontology import ONTOLOGY_VERSION, TARGET_COLUMNS, LabelState
from rsna_knee.paths import SCHEMA_DIR

VALID_STATES = {s.value for s in LabelState}


def load_schema(path: Path | None = None) -> dict[str, Any]:
    p = path or (SCHEMA_DIR / "report_labels.json")
    return json.loads(p.read_text(encoding="utf-8"))


def parse_json_content(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid_json: {exc}") from exc
    if not isinstance(data, dict) or data == {}:
        raise ValueError("empty_or_non_object")
    return data


def evidence_in_report(span: str, report: str) -> bool:
    if span == "":
        return True
    return span in report


def validate_extraction(
    payload: dict[str, Any],
    report: str,
    schema: dict[str, Any] | None = None,
    *,
    expected_uid: str | None = None,
) -> list[str]:
    errors: list[str] = []
    if expected_uid is not None and payload.get("StudyInstanceUID") not in {None, expected_uid}:
        errors.append("uid_mismatch")
        return errors
    schema = schema or load_schema()
    try:
        jsonschema.validate(payload, schema)
    except jsonschema.ValidationError as exc:
        errors.append(f"schema:{exc.message}")
        return errors
    targets = payload.get("targets") or {}
    missing = [t for t in TARGET_COLUMNS if t not in targets]
    extra = [t for t in targets if t not in TARGET_COLUMNS]
    if missing:
        errors.append(f"missing_targets:{missing}")
    if extra:
        errors.append(f"unknown_targets:{extra}")
    for name, rec in targets.items():
        if not isinstance(rec, dict):
            errors.append(f"{name}:not_object")
            continue
        state = rec.get("state")
        if state not in VALID_STATES:
            errors.append(f"{name}:bad_state:{state}")
        span = rec.get("evidence_span") or ""
        if state == LabelState.NOT_MENTIONED.value:
            if span:
                # allowed only if empty; leftover span is a mismatch
                if not evidence_in_report(span, report):
                    errors.append(f"{name}:evidence_not_in_report")
        else:
            if not span:
                errors.append(f"{name}:missing_evidence")
            elif not evidence_in_report(span, report):
                errors.append(f"{name}:evidence_not_in_report")
        if state == LabelState.EXPLICIT_NEGATIVE.value and state == LabelState.NOT_MENTIONED.value:
            errors.append(f"{name}:state_collision")
    return errors


def states_to_training_arrays(targets: dict[str, dict[str, Any]]) -> tuple[dict[str, float | None], dict[str, float], dict[str, float]]:
    """Return y, mask, weight. -1/unmentioned/uncertain are masked, never BCE targets."""
    y: dict[str, float | None] = {}
    mask: dict[str, float] = {}
    weight: dict[str, float] = {}
    for name in TARGET_COLUMNS:
        rec = targets.get(name) or {}
        state = rec.get("state", LabelState.NOT_MENTIONED.value)
        if state == LabelState.CRITERION_POSITIVE.value:
            y[name], mask[name], weight[name] = 1.0, 1.0, 0.25
        elif state == LabelState.EXPLICIT_NEGATIVE.value:
            y[name], mask[name], weight[name] = 0.0, 1.0, 0.25
        elif state == LabelState.BORDERLINE_NEGATIVE_HINT.value:
            y[name], mask[name], weight[name] = 0.0, 1.0, 0.15
        else:
            y[name], mask[name], weight[name] = None, 0.0, 0.0
    return y, mask, weight


def gold_to_training(value: Any) -> tuple[float | None, float, float]:
    if value is None:
        return None, 0.0, 0.0
    if isinstance(value, str) and value.strip() in {"", "nan", "NaN", "None"}:
        return None, 0.0, 0.0
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None, 0.0, 0.0
    if v == -1.0:
        return None, 0.0, 0.0
    if v not in (0.0, 1.0):
        return None, 0.0, 0.0
    return v, 1.0, 1.0


def ontology_banner() -> dict[str, str]:
    return {"ontology_version": ONTOLOGY_VERSION}
