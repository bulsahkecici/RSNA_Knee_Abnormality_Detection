from __future__ import annotations

import csv
from pathlib import Path

import pytest

from rsna_knee.data.folds import assert_train_uids, create_folds, image_training_uids, load_folds
from rsna_knee.errors import LeakageError
from rsna_knee.ontology import TARGET_COLUMNS


def _write(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        cols = ["StudyInstanceUID", "Report", *TARGET_COLUMNS]
        writer = csv.DictWriter(handle, fieldnames=cols)
        writer.writeheader()
        for i in range(20):
            row = {"StudyInstanceUID": f"s{i}", "Report": f"same report {i // 5}"}
            for name in TARGET_COLUMNS:
                row[name] = str(i % 2) if i < 8 else ""
            writer.writerow(row)


def test_duplicate_groups_and_gold_counts(tmp_path):
    train = tmp_path / "train.csv"
    _write(train)
    info = create_folds(train_csv=train, dest=tmp_path / "folds.csv", n_folds=4, seed=2026)
    folds = load_folds(tmp_path / "folds.csv")
    groups: dict[str, set[str]] = {}
    for row in folds:
        groups.setdefault(row["group_id"], set()).add(row["fold"])
    assert all(len(folds_) == 1 for folds_ in groups.values())
    assert info["gold_class_counts"]["ACL"]["pos"] > 0
    assert info["gold_class_counts"]["ACL"]["neg"] > 0
    allowed = set(image_training_uids(folds))
    eval_uids = {row["StudyInstanceUID"] for row in folds if row["gold_split"] == "eval"}
    assert not (allowed & eval_uids)
    with pytest.raises(LeakageError):
        assert_train_uids(eval_uids, folds)
