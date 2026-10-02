# Local LLM (LM Studio)

Backend: `lmstudio` at `http://127.0.0.1:1234/v1` (OpenAI-compatible, **not** OpenAI cloud).

Preferred model on this machine (2026-10-02 `/v1/models`): `qwen/qwen3.8-27b`.
Do not send the Hugging Face id `lmstudio-community/Qwen3.8-27B-MLX-4bit` as the API model field unless that is the listed id.

Setup:

1. Load one large model (36GB unified memory: keep a single 27B 4-bit loaded).
2. Developer tab: start server on 127.0.0.1:1234.
3. `export RSNA_LMSTUDIO_MODEL_ID=qwen/qwen3.8-27b`
4. `rsna doctor` then `rsna labels pilot --limit 5 --live`

Structured output: https://lmstudio.ai/docs/developer/openai-compat/structured-output  
If `json_schema` fails, the client reports a downgrade to `json_object` plus local schema validation. Missing server → quarantine, no fake labels.

Thinking: config `thinking: off` for bulk extraction. Do not invent undocumented sampling keys.
