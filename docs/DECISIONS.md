# Decisions

Snapshot of implementation choices, 2026-10-02.

| Decision | Choice | Why |
| --- | --- | --- |
| Python | 3.12 (`requires-python >=3.12`). Machine also has 3.14; target 3.12. | Spec + available Homebrew 3.12.14 |
| CLI | Typer, entrypoint `rsna` | Nested `metadata fetch` style commands |
| Persistence | Local SQLite registry + optional LangGraph `SqliteSaver` | Durable resume; Drive is not a live DB |
| LM Studio model | Config preferred id `qwen/qwen3.8-27b`; must match `/v1/models` | HF path is not an API id |
| json_schema | Probe; fall back to json_object + validation | LM Studio structured output docs |
| Colab | `automatic_acquisition: false`; Google.colab handoff | No public headless allocate API in colab-vscode User Guide |
| Kaggle submit | `competitions submit -k KERNEL -v VERSION` | Code competition; CSV-only insufficient |
| First vision model | DINOv2-S/14 2.5D; tests use TinyEncoder | Spec |
| Cache host | Kaggle for raw DICOM; local synthetic shards for smoke | 570GB not downloaded |
| Folds | StratifiedKFold on gold vs rest, group = StudyInstanceUID | No patient id |
| Fluid/Fat flags | Independent despite 24371/24371 agreement in this train_series snapshot | Official data note |
| Ontology | Host discussion 733343 confirmed | Criteria copied into `ontology.py` |
| Pin strategy | Lower bounds in pyproject; no CUDA wheels on macOS | extras: dicom, training, langgraph, kaggle, dev |
