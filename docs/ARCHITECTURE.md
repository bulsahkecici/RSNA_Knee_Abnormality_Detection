# Architecture

Two layers:

1. **Build-time Cursor agents** in `.cursor/agents` plus skills in `.agents/skills`. Isolated context, ownership, max two concurrent.
2. **Runtime** `rsna` CLI: SQLite registry (`state/registry.sqlite`), optional LangGraph SqliteSaver, stage functions in `src/rsna_knee/pipeline.py`.

LLM is used for report extraction (local LM Studio) and not for rewriting training code each loop. Experiments are YAML files under `configs/experiments/`.

Remote jobs travel as `job.json` / `result.json` with fencing tokens. Controller SQLite is never a shared Drive database.

Colab: manual VS Code extension handoff unless a documented headless API is later verified.
