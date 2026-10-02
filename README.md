# RSNA Knee pipeline

Durable, modular training and labeling infrastructure for
[RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection).

This repository is the **runtime**. Cursor agents help you edit it; they do not replace `rsna` CLI.

## Shortest first path

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,kaggle]"
rsna doctor
rsna smoke --synthetic
# optional live labels if LM Studio is up with an explicit model id:
rsna labels pilot --limit 5 --live
# then connect a remote worker (Colab handoff or Kaggle notebook) and:
rsna pipeline run --profile pilot
```

`rsna smoke --synthetic` writes a real artifact chain and marks metrics `kind=synthetic`. That AUC is **not** a competition score.

## What is implemented vs live

| Area | Status |
| --- | --- |
| CLI, config, SQLite registry, tests | implemented + offline verified |
| Metadata CSV fetch (not 570GB) | live if Kaggle credentials work |
| LM Studio extraction | live if server + explicit model id |
| Colab GPU allocate from terminal | **unsupported** (VS Code `Google.colab` handoff) |
| Kaggle kernel push/submit | CLI wired; not executed by smoke |
| DINOv2 full training | defined experiments, not run |
| Code-competition SUBMITTED | only after real `kaggle competitions submit -k -v` receipt |

## Sources (snapshot 2026-10-02)

See `docs/COMPETITION_SNAPSHOT.md` and `docs/DECISIONS.md`.
