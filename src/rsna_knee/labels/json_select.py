"""Pick one JSON object from an LM Studio chat message.

Content wins. reasoning_content is a controlled fallback only when content is
empty, the caller enabled it, and the whole field is one schema-valid object.
Free reasoning text is never scraped.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import jsonschema


class JsonSelectionError(ValueError):
    """The message did not yield one schema-valid JSON object."""


@dataclass(frozen=True)
class JsonSelection:
    data: dict[str, Any]
    field: str
    raw: str


def select_json_message(
    message: dict[str, Any],
    schema: dict[str, Any] | None,
    *,
    allow_reasoning_json: bool,
    finish_reason: str | None,
) -> JsonSelection:
    if finish_reason not in {None, "stop"}:
        raise JsonSelectionError("truncated_response")
    content = message.get("content")
    content_text = content if isinstance(content, str) else ""
    if content_text.strip():
        data = _load_entire_object(content_text, invalid_reason="invalid_json")
        _check_schema(data, schema)
        return JsonSelection(data=data, field="content", raw=content_text)
    if not allow_reasoning_json:
        raise JsonSelectionError("empty_content")
    reasoning = message.get("reasoning_content")
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise JsonSelectionError("empty_content")
    data = _load_entire_object(reasoning, invalid_reason="reasoning_not_json_object")
    _check_schema(data, schema)
    return JsonSelection(data=data, field="reasoning_content", raw=reasoning)


def _load_entire_object(text: str, *, invalid_reason: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise JsonSelectionError(invalid_reason) from exc
    if not isinstance(data, dict) or data == {}:
        raise JsonSelectionError("empty_or_non_object")
    return data


def _check_schema(data: dict[str, Any], schema: dict[str, Any] | None) -> None:
    if schema is None:
        return
    try:
        jsonschema.validate(data, schema)
    except jsonschema.ValidationError as exc:
        raise JsonSelectionError(f"schema:{exc.message}") from exc
