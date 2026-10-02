from __future__ import annotations


import numpy as np
import pytest

from rsna_knee.errors import FakeResultError
from rsna_knee.evaluation.metrics import per_class_auc, reject_fake_production
from rsna_knee.ontology import TARGET_COLUMNS, STUDY_ID_COL
from rsna_knee.submission.infer import write_submission_csv
from rsna_knee.submission.validate import validate_submission


def test_undefined_auc_not_half():
    y = np.ones((10, 12))
    p = np.random.default_rng(0).random((10, 12))
    m = per_class_auc(y, p)
    for name, rec in m["per_class"].items():
        assert rec["undefined"] is True
        assert rec["auc"] is None
    assert m["macro_auc"] is None


def test_fake_result_rejected_in_production():
    with pytest.raises(FakeResultError):
        reject_fake_production({"kind": "synthetic", "macro_auc": 0.81}, production=True)


def test_submission_contract(tmp_path):
    uids = ["a", "b", "c", "d"]
    rows = []
    for u in uids:
        row = {STUDY_ID_COL: u}
        for t in TARGET_COLUMNS:
            row[t] = 0.25
        rows.append(row)
    path = tmp_path / "submission.csv"
    write_submission_csv(rows, path)
    ok = validate_submission(path, uids)
    assert ok["ok"]
    # NaN illegal
    rows[0]["ACL"] = float("nan")
    write_submission_csv(rows, path)
    bad = validate_submission(path, uids)
    assert not bad["ok"]
