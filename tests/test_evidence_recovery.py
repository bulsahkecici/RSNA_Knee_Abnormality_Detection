"""Whitespace recovery must retain strict, attributable report evidence."""

import json

import httpx

from rsna_knee.labels.evidence import recover_source_spans
from rsna_knee.labels.extract import LabelStore, extract_one
from rsna_knee.labels.lmstudio import LMStudioClient
from rsna_knee.labels.validate import validate_extraction
from rsna_knee.ontology import TARGET_COLUMNS


def payload(span):
    targets = {t: {"state": "not_mentioned", "evidence_span": "", "criterion_mapping": "unmentioned"} for t in TARGET_COLUMNS}
    targets["ACL"] = {"state": "explicit_negative", "evidence_span": span, "criterion_mapping": "intact"}
    return {"StudyInstanceUID": "u1", "targets": targets}


def test_unique_whitespace_recovery_is_verbatim_and_preserves_original():
    original = payload("ACL is intact.")
    fixed, repaired = recover_source_spans(original, "ACL\n is  intact.")
    assert fixed["targets"]["ACL"]["evidence_span"] == "ACL\n is  intact."
    assert original["targets"]["ACL"]["evidence_span"] == "ACL is intact."
    assert repaired == ["ACL"]
    assert validate_extraction(fixed, "ACL\n is  intact.") == []


def test_ambiguous_or_changed_words_remain_rejected():
    for report in ("ACL\nis intact. ACL  is intact.", "ACL is torn.", "acl is intact.", "ACL is intact!"):
        fixed, repaired = recover_source_spans(payload("ACL is intact."), report)
        assert repaired == []
        assert "ACL:evidence_not_in_report" in validate_extraction(fixed, report)


def test_live_extraction_records_recovery_provenance_and_strict_payload(tmp_path):
    raw = json.dumps(payload("ACL is intact."))
    client = LMStudioClient(transport=httpx.Client(base_url="http://127.0.0.1:1234/v1", transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json={"choices": [{"message": {"content": raw}, "finish_reason": "stop"}]}))))
    store = LabelStore(tmp_path / "labels.sqlite")
    result = extract_one(client, uid="u1", report="ACL\n is intact.", model_id="local", model_revision="r1", store=store, max_retries=0)
    assert result["status"] == "ok"
    assert result["provenance"]["recovered_source_spans"] == ["ACL"]
    assert result["provenance"]["raw_response"] == raw
    assert validate_extraction(result["payload"], "ACL\n is intact.") == []
