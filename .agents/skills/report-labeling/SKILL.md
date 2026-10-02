---
name: report-labeling
description: Local LM Studio ontology extraction with resume keys and quarantine. Use for labels pilot/run. Never cloud OpenAI.
---

# Report labeling

## When

Need weak labels from `train.csv` Report column.

## Procedure

1. `rsna doctor` — confirm local LM Studio `/v1/models`.
2. Set `RSNA_LMSTUDIO_MODEL_ID` to an exact listed id (preferred `qwen/qwen3.8-27b` if present).
3. Probe json_schema; else json_object + local validation.
4. `rsna labels pilot --limit 100 --live` then `rsna labels run --resume --live`.
5. Quarantine invalid JSON / missing evidence. Do not auto-negative not_mentioned.

## Entrypoints

- `rsna labels pilot`
- `src/rsna_knee/labels/extract.py`

## Recovery

Server down → stop, no fake labels. Model change → new label version.
