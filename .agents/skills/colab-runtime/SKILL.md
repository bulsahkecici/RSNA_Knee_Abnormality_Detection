---
name: colab-runtime
description: Honest Colab capabilities and VS Code extension handoff. Use when acquiring GPU. No fake allocate_gpu.
---

# Colab runtime

## When

User needs A100/L4/T4 for training.

## Procedure

1. `rsna runtime capabilities` — expect `automatic_acquisition: false` for Colab.
2. `rsna runtime acquire --gpu-order A100,L4,T4 --wait-seconds 300` writes handoff instructions.
3. In VS Code/Cursor: install `Google.colab`, Select Kernel → Colab → New Colab Server, pick GPU.
4. Complete browser OAuth. Do not copy cookies.
5. Run `notebooks/02_colab_worker.ipynb` until handshake JSON appears.
6. Controller continues jobs after verified handshake. 300s instruction change ≠ automatic GPU switch.

## Entrypoints

- `rsna runtime acquire`
- `src/rsna_knee/runtime/providers/colab.py`

## Recovery

If all GPUs exhausted: QUEUED/NEEDS_RUNTIME. Optionally Kaggle fallback. Never CPU full train.
