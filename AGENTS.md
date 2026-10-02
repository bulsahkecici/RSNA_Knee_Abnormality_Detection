# AGENTS.md

This is a medical imaging **code competition** pipeline. Follow these rules in every session.

## Runtime vs Cursor

- Durable work is `rsna` CLI / `src/rsna_knee`. Cursor subagents edit code; they do not run the campaign by existing.
- Max two concurrent build-time subagents. Respect file ownership in `.cursor/agents`.
- Do not send raw radiology reports to cloud context. Use `rsna labels` locally.
- `.cursorignore` excludes `data/metadata/train.csv` and caches.

## Validity

- Never write simulated AUC, empty training, or SUBMITTED without a Kaggle receipt.
- `-1` and unmentioned/uncertain labels are masks, not BCE targets.
- StudyInstanceUID is a study id, not a patient id.
- Gold eval UIDs never enter image training, including as weak labels.
- Tiny test encoders are not DINOv2 results. Synthetic metrics cannot pass production audit.

## Resources

- Do not download the ~570GB archive to this Mac by default.
- CUDA is not expected on macOS. Full training is remote GPU.
- Colab headless GPU allocation is capability-gated: default is official VS Code extension handoff (`Google.colab`). There is no `colab.allocate_gpu()`.
- Local LLM concurrency is 1. One large LM Studio model at a time.

## Commands

`rsna doctor` → `rsna smoke --synthetic` → labels pilot → remote worker → `rsna pipeline run --profile pilot`.
