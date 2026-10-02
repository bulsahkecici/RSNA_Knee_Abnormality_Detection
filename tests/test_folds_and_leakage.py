from __future__ import annotations

import csv
from pathlib import Path

import pytest

from rsna_knee.data.duplicates import duplicate_report_groups
from rsna_knee.data.folds import (
    assert_group_integrity,
    assert_no_gold_eval_in_training,
    create_folds,
    load_folds,
)
from rsna_knee.errors import LeakageError
from rsna_knee.ontology import TARGET_COLUMNS


def _write_train(path: Path, n: int = 20, n_gold: int = 8) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        cols = ["StudyInstanceUID", "Report", *TARGET_COLUMNS]
        w = csv.DictWriter(handle, fieldnames=cols)
        w.writeheader()
        for i in range(n):
            row = {"StudyInstanceUID": f"s{i}", "Report": f"report {i // 5}"}  # duplicate reports
            if i < n_gold:
                for t in TARGET_COLUMNS:
                    row[t] = str(i % 2)
            else:
                for t in TARGET_COLUMNS:
                    row[t] = ""
            w.writerow(row)


def test_folds_group_and_gold_eval(tmp_path, monkeypatch):
    train = tmp_path / "train.csv"
    _write_train(train)
    dest = tmp_path / "folds.csv"
    info = create_folds(train_csv=train, dest=dest, n_folds=4, seed=2026)
    folds = load_folds(dest)
    assert_group_integrity(folds)
    eval_uids = {r["StudyInstanceUID"] for r in folds if r["gold_split"] == "eval"}
    assert_no_gold_eval_in_training(set(), folds)
    with pytest.raises(LeakageError):
        assert_no_gold_eval_in_training(eval_uids, folds)
    h1 = info["hash"]
    info2 = create_folds(train_csv=train, dest=dest, n_folds=4, seed=2026)
    assert info2["hash"] == h1
    # changed data invalidates hash
    _write_train(train, n=21, n_gold=8)
    info3 = create_folds(train_csv=train, dest=tmp_path / "folds2.csv", n_folds=4, seed=2026)
    assert info3["hash"] != h1
    dups = duplicate_report_groups(train)
    assert dups["n_duplicate_report_groups"] >= 1
    assert dups["patient_id_inferred"] is False
