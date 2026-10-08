"""LM Studio structured-output selector. No live server and no patient reports."""

from __future__ import annotations

import json

import httpx
import pytest
from typer.testing import CliRunner

from rsna_knee.cli import app
from rsna_knee.labels.extract import LabelStore, extract_one
from rsna_knee.labels.json_select import JsonSelectionError, select_json_message
from rsna_knee.labels.lmstudio import PROBE_SCHEMA, JsonProbe, LMStudioClient, ModelInfo

# Live probe observed from the local server: JSON is the entire reasoning field.
LIVE_REASONING = '{"ok": true}'
LIVE_MESSAGE = {"content": "", "reasoning_content": LIVE_REASONING}
LIVE_CHOICE = {"finish_reason": "stop", "message": LIVE_MESSAGE}

TARGETS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]


def _select(message: dict, *, allow: bool, finish: str | None = "stop"):
    return select_json_message(message, PROBE_SCHEMA, allow_reasoning_json=allow, finish_reason=finish)


def test_normal_content_beats_reasoning_text():
    selected = _select(
        {"content": '{"ok": true}', "reasoning_content": 'The answer is {"ok": true}'},
        allow=True,
    )
    assert selected.field == "content"
    assert selected.data == {"ok": True}


def test_valid_reasoning_json_is_accepted_only_when_enabled():
    selected = _select(LIVE_MESSAGE, allow=True, finish="stop")
    assert selected.field == "reasoning_content"
    assert selected.raw == LIVE_REASONING
    assert selected.data == {"ok": True}
    with pytest.raises(JsonSelectionError, match="empty_content"):
        _select(LIVE_MESSAGE, allow=False)


def test_reasoning_with_explanation_is_rejected():
    message = {"content": "", "reasoning_content": 'Here is the JSON: {"ok": true}'}
    with pytest.raises(JsonSelectionError, match="reasoning_not_json_object"):
        _select(message, allow=True)


def test_wrong_schema_is_rejected_without_reading_reasoning():
    message = {"content": '{"ok": "yes"}', "reasoning_content": LIVE_REASONING}
    with pytest.raises(JsonSelectionError, match="schema:"):
        _select(message, allow=True)
    extra = {"content": "", "reasoning_content": '{"ok": true, "extra": 1}'}
    with pytest.raises(JsonSelectionError, match="schema:"):
        _select(extra, allow=True)


def test_truncated_response_is_rejected():
    message = {"content": '{"ok": tru', "reasoning_content": LIVE_REASONING}
    with pytest.raises(JsonSelectionError, match="truncated_response"):
        _select(message, allow=True, finish="length")


def test_invalid_content_does_not_fall_back_to_reasoning():
    message = {"content": "not json", "reasoning_content": LIVE_REASONING}
    with pytest.raises(JsonSelectionError, match="invalid_json"):
        _select(message, allow=True)


def _client(handler) -> LMStudioClient:
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://127.0.0.1:1234/v1")
    return LMStudioClient(transport=http)


def test_live_probe_fixture_keeps_json_schema():
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        kind = body["response_format"]["type"]
        calls.append(kind)
        if kind == "json_object":
            return httpx.Response(400, json={"error": {"message": "json_object is not supported"}})
        assert body["messages"][0]["content"] == 'Return {"ok": true}'
        return httpx.Response(200, json={"choices": [LIVE_CHOICE]})

    probe = _client(handler).probe_json_schema("qwen/qwen3.8-27b", allow_reasoning_json=True, thinking="off")
    assert probe.mode == "json_schema"
    assert probe.response_field == "reasoning_content"
    assert probe.http_error is None
    assert probe.empty_reason is None
    assert calls == ["json_schema"]


def test_probe_reports_empty_content_and_http_400():
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        kind = body["response_format"]["type"]
        calls.append(kind)
        if kind == "json_schema":
            return httpx.Response(200, json={"choices": [LIVE_CHOICE]})
        return httpx.Response(400, json={"error": {"message": "json_object is not supported"}})

    probe = _client(handler).probe_json_schema("qwen/qwen3.8-27b", allow_reasoning_json=False)
    assert probe.mode == "none"
    assert probe.empty_reason == "empty_content"
    assert probe.http_error is not None
    assert probe.http_error.startswith("http_400:")
    assert "json_object is not supported" in probe.http_error
    assert calls == ["json_schema", "json_object"]


def _label_payload(uid: str, *, acl_state: str = "not_mentioned", acl_span: str = "") -> dict:
    targets = {
        name: {"state": "not_mentioned", "evidence_span": "", "criterion_mapping": "none"} for name in TARGETS
    }
    targets["ACL"] = {"state": acl_state, "evidence_span": acl_span, "criterion_mapping": "none"}
    return {"StudyInstanceUID": uid, "targets": targets}


def _extract_client(message: dict, calls: list[str], *, object_body: dict | None = None) -> LMStudioClient:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        kind = body["response_format"]["type"]
        calls.append(kind)
        if kind == "json_object":
            payload = object_body if object_body is not None else {"choices": [{"message": {"content": "{}"}}]}
            return httpx.Response(200, json=payload)
        return httpx.Response(200, json={"choices": [{"finish_reason": message.get("finish_reason", "stop"), "message": message}]})

    return _client(handler)


def test_extract_records_reasoning_field_and_keeps_uid_evidence(tmp_path):
    report = "Complete ACL tear."
    calls: list[str] = []
    good = {
        "content": "",
        "reasoning_content": json.dumps(_label_payload("u1", acl_state="criterion_positive", acl_span=report)),
        "finish_reason": "stop",
    }
    store = LabelStore(tmp_path / "ok.sqlite")
    out = extract_one(
        _extract_client(good, calls),
        uid="u1",
        report=report,
        model_id="qwen/qwen3.8-27b",
        model_revision="r",
        store=store,
        json_mode="json_schema",
        max_retries=0,
        allow_reasoning_json=True,
    )
    assert out["status"] == "ok"
    assert out["provenance"]["response_field"] == "reasoning_content"
    assert out["provenance"]["json_mode"] == "json_schema"
    assert "response_field" not in out["payload"]
    assert calls == ["json_schema"]

    bad_uid = {"content": json.dumps(_label_payload("other-uid")), "finish_reason": "stop"}
    uid_out = extract_one(
        _extract_client(bad_uid, []),
        uid="u1",
        report=report,
        model_id="qwen/qwen3.8-27b",
        model_revision="r",
        store=LabelStore(tmp_path / "uid.sqlite"),
        json_mode="json_schema",
        max_retries=0,
        allow_reasoning_json=True,
    )
    assert uid_out["status"] == "quarantine"
    assert "uid_mismatch" in uid_out["reason"]

    bad_evidence = {
        "content": json.dumps(_label_payload("u1", acl_state="criterion_positive", acl_span="not in report")),
        "finish_reason": "stop",
    }
    evidence_out = extract_one(
        _extract_client(bad_evidence, []),
        uid="u1",
        report=report,
        model_id="qwen/qwen3.8-27b",
        model_revision="r",
        store=LabelStore(tmp_path / "evidence.sqlite"),
        json_mode="json_schema",
        max_retries=0,
        allow_reasoning_json=True,
    )
    assert evidence_out["status"] == "quarantine"
    assert "evidence_not_in_report" in evidence_out["reason"]
    assert evidence_out["provenance"]["response_field"] == "content"


def test_extract_does_not_downgrade_when_json_schema_is_rejected(tmp_path):
    calls: list[str] = []
    message = {"content": "", "reasoning_content": 'Here is the JSON: {"ok": true}', "finish_reason": "stop"}
    out = extract_one(
        _extract_client(message, calls, object_body={"choices": [{"message": {"content": json.dumps(_label_payload("u1"))}}]}),
        uid="u1",
        report="Complete ACL tear.",
        model_id="qwen/qwen3.8-27b",
        model_revision="r",
        store=LabelStore(tmp_path / "labels.sqlite"),
        json_mode="json_schema",
        max_retries=0,
        allow_reasoning_json=True,
    )
    assert out["status"] == "quarantine"
    assert out["reason"] == "reasoning_not_json_object"
    assert calls == ["json_schema"]


def test_cli_prints_http_and_empty_reasons(monkeypatch):
    seen: dict[str, object] = {}

    class Fake:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def list_models(self):
            return [ModelInfo(id="qwen/qwen3.8-27b", raw={})]

        def resolve_model(self, preferred, available=None):
            del preferred
            return available[0]

        def probe_json_schema(self, model, allow_reasoning_json=False, thinking=None):
            del model
            seen["allow_reasoning_json"] = allow_reasoning_json
            seen["thinking"] = thinking
            return JsonProbe(mode="none", http_error="http_400: json_object is not supported", empty_reason="empty_content")

        def close(self):
            return None

    monkeypatch.setattr("rsna_knee.pipeline.LMStudioClient", Fake)
    monkeypatch.setattr("rsna_knee.cli.ensure_runtime_dirs", lambda: None)
    result = CliRunner().invoke(app, ["labels", "pilot", "--live", "--limit", "1", "--run-id", "probe-cli"])
    assert result.exit_code == 0, result.stdout
    assert seen["allow_reasoning_json"] is True
    assert seen["thinking"] == "off"
    assert "http_400: json_object is not supported" in result.stdout
    assert "empty_content" in result.stdout
    assert "json_capability_none" in result.stdout
