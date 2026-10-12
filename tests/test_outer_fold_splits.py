"""Cross-validation arms hold out their own outer fold; fold-0 arms are unchanged."""
import pytest

from rsna_knee.data.folds import LeakageError
from rsna_knee.runtime.train_payload import assert_split_disjoint, pilot_splits


def _folds():
    rows = []
    for k in range(20):
        gold = k < 3
        rows.append({"StudyInstanceUID": f"u{k}", "fold": str(k % 5), "is_gold": str(int(gold)),
                     "gold_split": "eval" if gold else "weak", "group_id": f"g{k}",
                     "gold_set": "A_historic_eval" if gold else ""})
    return rows


def test_default_holdout_is_fold_zero():
    folds = _folds()
    assert pilot_splits(folds) == pilot_splits(folds, "0")
    _, weak, _ = pilot_splits(folds)
    assert {u for u in weak} == {r["StudyInstanceUID"] for r in folds if r["fold"] == "0" and r["gold_split"] == "weak"}


def test_outer_fold_one_holds_out_fold_one_and_trains_on_fold_zero():
    folds = _folds()
    train, weak, gold = pilot_splits(folds, "1")
    fold = {r["StudyInstanceUID"]: r["fold"] for r in folds}
    assert weak and all(fold[u] == "1" for u in weak)
    assert any(fold[u] == "0" for u in train) and not any(fold[u] == "1" for u in train)
    assert set(gold) == {"u0", "u1", "u2"} and not set(gold) & set(train)
    assert_split_disjoint(train, weak, gold, folds)


def test_fold_zero_holdout_with_outer_fold_one_training_set_is_rejected():
    folds = _folds()
    train_fold1, _, _ = pilot_splits(folds, "1")
    _, weak_fold0, gold = pilot_splits(folds, "0")
    with pytest.raises(LeakageError):
        assert_split_disjoint(train_fold1, weak_fold0, gold, folds)


def test_outer_fold_none_trains_on_every_fold_and_keeps_gold_out():
    folds = _folds()
    train, weak, gold = pilot_splits(folds, "none")
    assert weak == []
    assert set(train) == {r["StudyInstanceUID"] for r in folds if r["gold_split"] == "weak"}
    assert not set(gold) & set(train)
    assert_split_disjoint(train, weak, gold, folds)
