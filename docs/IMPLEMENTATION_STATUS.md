# Implementation status

Date: 2026-10-02

| Item | Class |
| --- | --- |
| Package, CLI, configs, SQLite, Cursor agents/skills/rules | implemented + offline verified (after tests in this session) |
| Metadata CSV download | live verified (Kaggle credential present; CSVs on disk) |
| Host ontology 733343 | live verified (CLI topic body) |
| LM Studio reachability | live verified (`/v1/models` includes `qwen/qwen3.8-27b`) |
| Live report extraction of real train.csv | needs explicit `--live` and remaining GPU/RAM for 27B; not run as bulk in install |
| Colab headless allocate | provider capability **unsupported**; handoff implemented |
| Kaggle kernel training job | adapter implemented; not dispatched in smoke |
| DINOv2 training on competition MRI | needs remote runtime + cache; **not done** |
| Code-competition SUBMITTED | not done (0 lifetime submissions) |
| Full 570GB DICOM | not downloaded by design |

Production must not treat simulator or synthetic AUC as a real result.
