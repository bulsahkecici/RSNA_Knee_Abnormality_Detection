"""Gold vs weak semantics, -1 masking, explicit negative vs not_mentioned."""

from __future__ import annotations

from rsna_knee.labels.validate import gold_to_training, parse_json_content, validate_extraction
from rsna_knee.ontology import TARGET_COLUMNS
from rsna_knee.training.losses import masked_bce_with_logits
import numpy as np
import pytest


SCHEMA_MIN = {
    "type": "object",
    "required": ["StudyInstanceUID", "targets"],
    "properties": {
        "StudyInstanceUID": {"type": "string"},
        "targets": {"type": "object"},
    },
}


def _full(targets):
    for t in TARGET_COLUMNS:
        targets.setdefault(t, {"state": "not_mentioned", "evidence_span": "", "criterion_mapping": "x"})
    return {"StudyInstanceUID": "u1", "targets": targets}


def test_invalid_json_rejected():
    with pytest.raises(ValueError):
        parse_json_content("{")
    with pytest.raises(ValueError):
        parse_json_content("{}")


def test_evidence_must_be_in_report():
    report = "ACL is torn. MCL intact."
    payload = _full(
        {
            "ACL": {"state": "criterion_positive", "evidence_span": "not in report", "criterion_mapping": "x"},
        }
    )
    errors = validate_extraction(payload, report)
    assert any("evidence_not_in_report" in e for e in errors)


def test_explicit_negative_vs_not_mentioned():
    report = "ACL is torn. MCL intact."
    payload = _full(
        {
            "ACL": {"state": "criterion_positive", "evidence_span": "ACL is torn", "criterion_mapping": "high-grade"},
            "MCL": {"state": "explicit_negative", "evidence_span": "MCL intact", "criterion_mapping": "intact"},
            "Effusion": {"state": "not_mentioned", "evidence_span": "", "criterion_mapping": "absent text"},
        }
    )
    errors = validate_extraction(payload, report)
    assert errors == []
    assert payload["targets"]["MCL"]["state"] != payload["targets"]["Effusion"]["state"]


def test_minus_one_is_masked_not_bce_target():
    y, m, w = gold_to_training(-1)
    assert y is None and m == 0.0
    logits = np.zeros((1, 12))
    targets = np.zeros((1, 12))
    mask = np.zeros((1, 12))
    # unmasked -1 must raise
    bad = np.full((1, 12), -1.0)
    with pytest.raises(ValueError):
        masked_bce_with_logits(logits, bad, np.ones((1, 12)))
    assert masked_bce_with_logits(logits, targets, mask) == 0.0
