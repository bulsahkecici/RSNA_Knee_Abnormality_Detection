# Implementation status

Date: 2026-10-02. Review notes for commit `2911e7c` were checked against that same HEAD (the review file was not in the workspace). Later edits in this session close those gates in code. They were not a live GPU campaign.

| Item | Implemented | Tested offline | Live verified | Blocked |
| --- | --- | --- | --- | --- |
| Synthetic smoke isolated from real folds, label cache, and cache index | yes | yes | n/a | no |
| `synthetic=False` refuses TinyEncoder, fixtures, and `syn-*` UIDs | yes | yes | n/a | no |
| Audit default is not PASS; CSV shape alone cannot set READY_TO_SUBMIT | yes | yes | n/a | no |
| Submit requires audit PASS, keeps local attempt id separate from server receipt, blocks repeat kernel/version | yes | yes, provider monkeypatched | no Kaggle submit | live submit not started |
| DICOM plane select, IOP/IPP order, triplet cache, quarantine, manifest hashes | yes | yes, synthetic DICOM files | no full archive | 570GB download not done |
| LM Studio model id, revision-from-record, JSON probe, no silent truncation, concurrency lock | yes | yes, mock HTTP | earlier `/v1/models` probe only | bulk `--live` extraction not run |
| Gold-eval exclusion, duplicate-report folds, per-class gold counts | yes | yes | n/a | no |
| DINOv2 ViT-S/14 module and strict checkpoint load | yes | yes, CPU, img 28 and strict mismatch | official weights file not required for the unit test | pretrained `.pth` not downloaded; no CUDA training |
| Study model, masked BCE, AMP hook, microbatch, resume vs uninterrupted | yes | yes, CPU | no | live GPU training not run |
| Evaluate loads a saved checkpoint | yes for the synthetic numpy checkpoint and the torch loader | yes | no held-out MRI eval | production evaluate stays BLOCKED without a real checkpoint |
| Job bundle, worker reject of no-items/no-checkpoint, token ingest | yes | yes | no remote worker | Colab/Kaggle job not dispatched |
| Resume hash invalidation and pause control file | yes | yes | n/a | no |
| Colab 300s policy only when allocation API exists; cancel-unsupported does not open the next GPU | yes | yes, scripted clock | no | headless Colab allocate remains unsupported |
| Kaggle kernels push/status/output plan | yes, `execute=False` | argv prepared | no | kernel not pushed |
| Offline inference notebook and package hash manifest | yes | package builder unit path via smoke; DICOM inference function covered by cache builder | no hidden test | no competition submit |
| Hidden-test 9h guarantee from 3 visible studies | refused in `project_runtime` | yes | no benchmark | no measured hidden-test timing |

Production smoke ends `BLOCKED`. That is the intended gate, not a leaderboard result.
