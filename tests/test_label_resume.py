"""Resume is UID+hashes, not row numbers."""

from __future__ import annotations

from rsna_knee.labels.extract import LabelStore, extract_one, resume_key
from rsna_knee.labels.lmstudio import LMStudioClient
import json
import httpx


def _handler(request: httpx.Request) -> httpx.Response:
    payload = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "StudyInstanceUID": "u1",
                            "targets": {
                                t: {
                                    "state": "not_mentioned",
                                    "evidence_span": "",
                                    "criterion_mapping": "none",
                                }
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
                    )
                }
            }
        ]
    }
    return httpx.Response(200, json=payload)


def test_resume_after_interrupt(tmp_path):
    transport = httpx.MockTransport(_handler)
    client = LMStudioClient(transport=httpx.Client(transport=transport, base_url="http://127.0.0.1:1234/v1"))
    store = LabelStore(tmp_path / "labels.sqlite")
    report = "No effusion mentioned."
    a = extract_one(client, uid="u1", report=report, model_id="qwen/qwen3.8-27b", model_revision="r1", store=store, json_mode="json_object")
    assert a["status"] == "ok" and a["cached"] is False
    b = extract_one(client, uid="u1", report=report, model_id="qwen/qwen3.8-27b", model_revision="r1", store=store, json_mode="json_object")
    assert b["cached"] is True
    # different report hash must not reuse
    c = extract_one(client, uid="u1", report=report + " extra", model_id="qwen/qwen3.8-27b", model_revision="r1", store=store, json_mode="json_object")
    assert c["cached"] is False
    k1 = resume_key("u1", report, "qwen/qwen3.8-27b", "r1", "x")
    k2 = resume_key("u1", report, "qwen/qwen3.8-27b", "r2", "x")
    assert k1 != k2
    client.close()
