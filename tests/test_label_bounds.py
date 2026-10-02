from __future__ import annotations

import json

import httpx

from rsna_knee.labels.extract import LabelStore, extract_one, split_report
from rsna_knee.labels.lmstudio import LMStudioClient


def _client(uid: str = "u1") -> LMStudioClient:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        text = body["messages"][-1]["content"]
        returned = uid if "WRONG" not in text else "other-uid"
        payload = {
            "StudyInstanceUID": returned,
            "targets": {
                t: {"state": "not_mentioned", "evidence_span": "", "criterion_mapping": "none"}
                for t in [
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
            },
        }
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(payload)}}]})

    return LMStudioClient(transport=httpx.Client(transport=httpx.MockTransport(handler), base_url="http://127.0.0.1:1234/v1"))


def test_long_report_is_not_silently_sliced(tmp_path):
    assert split_report("x" * 5000, 100) is None
    store = LabelStore(tmp_path / "labels.sqlite")
    out = extract_one(
        _client(),
        uid="u1",
        report="x" * 5000,
        model_id="qwen/qwen3.8-27b",
        model_revision="",
        store=store,
        json_mode="json_object",
        context_tokens=64,
        output_tokens=32,
        max_retries=0,
    )
    assert out["status"] == "quarantine"
    assert out["reason"] == "report_exceeds_context"


def test_uid_mismatch_quarantines(tmp_path):
    store = LabelStore(tmp_path / "labels.sqlite")
    out = extract_one(
        _client("other-uid"),
        uid="u1",
        report="WRONG study text",
        model_id="qwen/qwen3.8-27b",
        model_revision="",
        store=store,
        json_mode="json_object",
        max_retries=0,
    )
    assert out["status"] == "quarantine"
    assert "uid_mismatch" in out["reason"]
