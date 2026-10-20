"""Numeric-only target validation and leakage-safe soft-label training pools."""
from __future__ import annotations

import json
import math

from rsna_knee.data.folds import assert_group_integrity, assert_train_uids, image_training_uids
from rsna_knee.ontology import TARGET_COLUMNS


def normalize_soft_targets(row) -> dict[str, float | None]:
    targets = {}
    for name in TARGET_COLUMNS:
        value = row.get(name)
        if value is None or value == -1:
            targets[name] = None
        elif not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
            raise ValueError(f'invalid soft target: {name}')
        else:
            targets[name] = float(value)
    return targets


def soft_table(data: bytes) -> dict[str, dict[str, float | None]]:
    table = {}
    for line in data.decode().splitlines():
        row = json.loads(line)
        uid = row['StudyInstanceUID']
        if uid in table:
            raise ValueError('duplicate soft-label UID')
        table[uid] = normalize_soft_targets(row['soft_targets'])
    return table


def soft_training_uids(folds, data: bytes, outer_fold: str, exclude=()) -> list[str]:
    assert_group_integrity(folds)
    table = soft_table(data)
    allowed = set(image_training_uids(folds))
    excluded = {str(x) for x in exclude} | {str(outer_fold)}
    uids = sorted(r['StudyInstanceUID'] for r in folds
                  if r['StudyInstanceUID'] in allowed and str(r['fold']) not in excluded
                  and any(v is not None for v in table.get(r['StudyInstanceUID'], {}).values()))
    if not uids:
        raise ValueError('empty soft-label training pool')
    assert_train_uids(set(uids), folds)
    return uids
