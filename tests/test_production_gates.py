"""Negative tests: the previous pipeline treated fixtures and CSV checks as success."""

from __future__ import annotations

import json

import pytest

from rsna_knee.agents.auditor import audit_run
from rsna_knee.errors import ContractError
from rsna_knee.gates import assert_real_encoder, assert_real_uids
from rsna_knee.models.encoder import load_encoder
from rsna_knee.pipeline import run_pipeline
from rsna_knee.workflow.state import Stage


def test_real_run_rejects_synthetic_fixtures():
    with pytest.raises(ContractError):
        assert_real_uids(["1.2.3", "syn-000"])
    with pytest.raises(ContractError):
        assert_real_encoder("tiny_test_encoder")
    with pytest.raises(FileNotFoundError):
        load_encoder("dinov2_vits14", checkpoint=None)


def test_default_audit_is_not_pass():
    audit = audit_run({}, production=True)
    assert audit["overall"] != "PASS"
    assert audit["gates"]["gold_leakage"] == "BLOCKED"
    assert audit["gates"]["identity"] == "BLOCKED"


def test_csv_alone_does_not_open_submit(tmp_path):
    state = run_pipeline("smoke", synthetic=True, resume=False)
    assert state.stage != Stage.READY_TO_SUBMIT
    assert state.stage == Stage.BLOCKED
    audit = json.loads(open(state.artifacts["audit"], encoding="utf-8").read())
    assert audit["overall"] != "PASS"
