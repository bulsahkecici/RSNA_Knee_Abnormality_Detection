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


def test_all_gold_holdout_keeps_folds_and_excludes_both_gold_sets(tmp_path):
    import csv as _csv

    from rsna_knee.data.folds import derive_all_gold_holdout, image_training_uids, load_folds

    rows = [
        {"StudyInstanceUID": "g-eval", "fold": "0", "is_gold": "1", "gold_split": "eval", "group_id": "g-eval"},
        {"StudyInstanceUID": "g-train", "fold": "1", "is_gold": "1", "gold_split": "train_image_ok", "group_id": "dup"},
        {"StudyInstanceUID": "dup-weak", "fold": "1", "is_gold": "0", "gold_split": "weak", "group_id": "dup"},
        {"StudyInstanceUID": "w", "fold": "2", "is_gold": "0", "gold_split": "weak", "group_id": "w"},
    ]
    source = tmp_path / "folds.csv"
    with source.open("w", newline="") as handle:
        writer = _csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = derive_all_gold_holdout(source, tmp_path / "folds-v2.csv")
    derived = load_folds(tmp_path / "folds-v2.csv")
    assert [r["fold"] for r in derived] == [r["fold"] for r in rows]
    assert image_training_uids(derived) == ["w"]  # duplicate-group member of former-train gold is dropped too
    assert payload["gold_sets"] == {"A_historic_eval": 1, "B_former_train": 1}
    assert payload["per_fold"]["1"] == {"trainable": 0, "gold": 1, "dropped_group_member": 1}
