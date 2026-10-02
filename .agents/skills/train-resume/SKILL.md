---
name: train-resume
description: Checkpoint resume across GPU types with hashes and RNG. Use when training interrupts.
---

# Train resume

## When

Colab/Kaggle disconnect, SIGINT, or OOM microbatch change.

## Procedure

1. Load last durable checkpoint with `map_location=cpu` then to device.
2. Restore model, optimizer, scheduler, scaler, step, sampler/RNG, hashes.
3. If image size/architecture/targets changed, new experiment id — do not silent-mutate.
4. Precision change: record scaler handling; no bitwise equality claim.

## Entrypoints

- `src/rsna_knee/training/checkpoint.py`
- tests/test_training_resume.py
