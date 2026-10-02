---
name: leakage-audit
description: Gold eval leakage, duplicate groups, fold hash invalidation. Use before training or submit.
---

# Leakage audit

## When

Before `rsna experiment run` or `rsna audit`.

## Procedure

1. Confirm folds grouped by StudyInstanceUID.
2. `gold_split=eval` UIDs absent from image training lists, including weak-label joins.
3. Duplicate report groups are flagged, not treated as patient ids.
4. Config/label/cache hash change invalidates downstream.

## Entrypoints

- `rsna folds create`
- `rsna audit --run-id`
- tests/test_folds_and_leakage.py
