---
name: kaggle-offline-submit
description: Package offline bundle and submit via kernel version. Use for code-competition submit. CSV-only is insufficient.
---

# Kaggle offline submit

## When

READY_TO_SUBMIT after PASS audit on real metrics.

## Procedure

1. `rsna package --run-id RUN`
2. Confirm MANIFEST: no internet, no LM Studio, no Drive at scoring.
3. Push inference notebook; wait for success.
4. Check `kaggle competitions submission-limits`.
5. `rsna submit --run-id RUN --kernel USER/KERNEL --version N --submit`
6. Persist request id. Crash recovery = status reconcile, not blind repeat.
7. SUBMITTED ≠ SCORED.

## Entrypoints

- `rsna package`
- `rsna submit`
- `kaggle competitions submit -k -v`

Smoke must not set execute=True.
