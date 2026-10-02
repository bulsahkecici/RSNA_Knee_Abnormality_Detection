---
name: experiment-review
description: Champion/submit gates beyond AUC. Use after evaluate. 0.86 alone is not enough.
---

# Experiment review

## When

A run finished with metrics.json.

## Procedure

1. Require `metrics.kind=real` for production.
2. Read audit gates, baseline delta, bootstrap interval, source legitimacy, runtime, finite outputs.
3. REFERENCE vs OWN: unknown overlap → not independent OOF.
4. Write decision to `docs/EXPERIMENTS.md` run log, not into weights.

## Entrypoints

- `rsna evaluate --run-id`
- `src/rsna_knee/evaluation/review.py`
