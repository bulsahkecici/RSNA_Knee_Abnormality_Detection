"""Study-grouped folds. Gold eval rows never enter image training."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from rsna_knee.errors import LeakageError
from rsna_knee.hashing import sha256_file, sha256_text
from rsna_knee.ontology import STUDY_ID_COL, TARGET_COLUMNS
from rsna_knee.paths import METADATA_DIR, STATE_DIR, ensure_runtime_dirs


def _nonempty(value: Any) -> bool:
    return value is not None and str(value).strip() not in {"", "nan", "NaN", "None"}


def _gold_flag(row: dict[str, str]) -> int:
    return int(all(_nonempty(row.get(c)) for c in TARGET_COLUMNS))


def create_folds(
    train_csv: Path | None = None,
    n_folds: int = 5,
    seed: int = 2026,
    gold_eval_fraction: float = 0.5,
    dest: Path | None = None,
) -> dict[str, Any]:
    ensure_runtime_dirs()
    train_csv = train_csv or (METADATA_DIR / "train.csv")
    if not train_csv.exists():
        raise FileNotFoundError("train.csv missing; run rsna metadata fetch")
    with train_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    uids = np.array([r[STUDY_ID_COL] for r in rows])
    gold = np.array([_gold_flag(r) for r in rows])
    report_counts: dict[str, int] = defaultdict(int)
    report_keys: list[str] = []
    for row in rows:
        report = (row.get("Report") or "").strip()
        key = sha256_text(report) if report else row[STUDY_ID_COL]
        report_keys.append(key)
        report_counts[key] += 1
    group_ids = [
        key if report_counts[key] > 1 else uid for key, uid in zip(report_keys, uids, strict=True)
    ]
    groups = np.array(group_ids)
    assignments: dict[str, int] = {}
    try:
        if len(set(group_ids)) >= n_folds:
            splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            splits = splitter.split(np.zeros(len(rows)), gold, groups)
        else:
            raise ValueError("too few groups")
        for fold_i, (_, val_idx) in enumerate(splits):
            for uid in uids[val_idx]:
                assignments[str(uid)] = fold_i
    except ValueError:
        splitter = StratifiedKFold(n_splits=min(n_folds, max(2, int(gold.sum() or 1))), shuffle=True, random_state=seed)
        # Keep duplicate-report studies in one fold even if stratification falls back.
        fold_of_group: dict[str, int] = {}
        try:
            raw = {}
            for fold_i, (_, val_idx) in enumerate(splitter.split(uids, gold)):
                for uid in uids[val_idx]:
                    raw[str(uid)] = fold_i
        except ValueError:
            raw = {str(uid): i % n_folds for i, uid in enumerate(uids)}
        for uid, gid in zip(uids, group_ids, strict=True):
            fold_of_group.setdefault(str(gid), raw[str(uid)])
        for uid, gid in zip(uids, group_ids, strict=True):
            assignments[str(uid)] = fold_of_group[str(gid)]

    gold_uids = [str(u) for u, g in zip(uids, gold, strict=True) if g == 1]
    rng = np.random.default_rng(seed)
    rng.shuffle(gold_uids)
    n_eval = max(1, int(round(len(gold_uids) * gold_eval_fraction))) if gold_uids else 0
    gold_eval = set(gold_uids[:n_eval])
    gold_train = set(gold_uids[n_eval:])
    uid_group = {str(uid): str(gid) for uid, gid in zip(uids, group_ids, strict=True)}

    dest = dest or (STATE_DIR / "folds.csv")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["StudyInstanceUID", "fold", "is_gold", "gold_split", "group_id"],
        )
        writer.writeheader()
        for row in rows:
            uid = row[STUDY_ID_COL]
            is_gold = uid in gold_eval or uid in gold_train
            if uid in gold_eval:
                gsplit = "eval"
            elif uid in gold_train:
                gsplit = "train_image_ok"
            else:
                gsplit = "weak"
            writer.writerow(
                {
                    "StudyInstanceUID": uid,
                    "fold": assignments[uid],
                    "is_gold": int(is_gold),
                    "gold_split": gsplit,
                    "group_id": uid_group[uid],
                }
            )
    tmp.replace(dest)
    class_counts = {
        name: {
            "pos": sum(1 for row in rows if _gold_flag(row) and str(row.get(name)) == "1"),
            "neg": sum(1 for row in rows if _gold_flag(row) and str(row.get(name)) == "0"),
        }
        for name in TARGET_COLUMNS
    }
    payload = {
        "path": str(dest),
        "n_studies": len(rows),
        "n_folds": n_folds,
        "seed": seed,
        "n_gold": len(gold_uids),
        "n_gold_eval": len(gold_eval),
        "n_gold_train": len(gold_train),
        "gold_class_counts": class_counts,
        "hash": sha256_file(dest),
        "patient_cv_guaranteed": False,
        "note": "Duplicate reports share a fold. No reliable patient identifier in official tables.",
    }
    (dest.with_suffix(".json")).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_folds(path: Path | None = None) -> list[dict[str, str]]:
    path = path or (STATE_DIR / "folds.csv")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def assert_no_gold_eval_in_training(train_uids: set[str], folds: list[dict[str, str]]) -> None:
    eval_uids = {r["StudyInstanceUID"] for r in folds if r.get("gold_split") == "eval"}
    leak = train_uids & eval_uids
    if leak:
        raise LeakageError(f"gold eval UIDs in training: {sorted(leak)[:8]}")


def image_training_uids(folds: list[dict[str, str]]) -> list[str]:
    """Drop gold-eval studies and every member of a duplicate group that touches them."""
    eval_uids = {r["StudyInstanceUID"] for r in folds if r.get("gold_split") == "eval"}
    eval_groups = {r["group_id"] for r in folds if r["StudyInstanceUID"] in eval_uids}
    out: list[str] = []
    for row in folds:
        if row["StudyInstanceUID"] in eval_uids:
            continue
        if row["group_id"] in eval_groups:
            continue
        out.append(row["StudyInstanceUID"])
    return out


def assert_train_uids(train_uids: set[str], folds: list[dict[str, str]]) -> None:
    assert_no_gold_eval_in_training(train_uids, folds)
    allowed = set(image_training_uids(folds))
    extra = train_uids - allowed
    if extra:
        raise LeakageError(f"train list is outside the image-training UID set: {sorted(extra)[:8]}")


def assert_group_integrity(folds: list[dict[str, str]]) -> None:
    by_uid: dict[str, set[str]] = {}
    for row in folds:
        by_uid.setdefault(row["StudyInstanceUID"], set()).add(row["fold"])
    multi = {u: f for u, f in by_uid.items() if len(f) > 1}
    if multi:
        raise LeakageError(f"study in multiple folds: {list(multi)[:5]}")
    by_group: dict[str, set[str]] = {}
    for row in folds:
        by_group.setdefault(row["group_id"], set()).add(row["fold"])
    split_groups = {g: folds_ for g, folds_ in by_group.items() if len(folds_) > 1}
    if split_groups:
        raise LeakageError(f"duplicate group split across folds: {list(split_groups)[:5]}")
