# Implementation status

Date: 2026-10-03. Review `RSNA_Repo_Tekrar_Inceleme_833da87.md` was checked against commit `833da87`. This pass wires the worker to real files, the trainer, and held-out reload. It is not a live GPU campaign and not a Kaggle submission.

`state/folds.csv` had been overwritten by an older test (`StudyInstanceUID,fold` / `REAL,0`). It was rebuilt from `data/metadata/train.csv`: 4407 studies, 58 gold, 29 gold-eval. Fixture label files `real-labels.jsonl` and `syn-*-labels.jsonl` were removed from `data/labels`. Tests now redirect real roots with `RSNA_ROOTS`.

| Item | Implemented | Tested offline | Live verified | Blocked |
| --- | --- | --- | --- | --- |
| Pytest does not write real folds, labels, cache, registry, or control | yes | yes, autouse tmp roots plus smoke hash snapshot | n/a | no |
| Job bundle carries folds/labels/cache/weights URIs and hashes; worker loads them | yes | yes, example shards | no MRI cache | official weights file absent |
| Training success requires a new optimizer step, not an existing file | yes | yes | no | no CUDA run |
| TinyEncoder and random state dicts are not pretrained | yes | yes | n/a | published SHA256 not in repo |
| Microbatch slicing, partial accumulation, OOM retry, unfreeze resume | yes | yes, CPU, img 28 | no | no CUDA OOM |
| Held-out gold and weak predictions from the saved checkpoint | yes | yes, example UIDs | no MRI holdout | pilot AUC is not a leaderboard score |
| Cache resume checks shape, centers, slots, source, and content hash | yes | yes | no full archive | 570GB not downloaded |
| Missing shard is not CACHE_READY; scope is `pilot` or `full` | yes | yes | n/a | no |
| Package copies `asset/assets/checkpoint.pt`, verify result blocks READY | yes | yes | no | offline wheels not vendored; kernel not pushed |
| Submit rechecks run, checkpoint, and package; UNKNOWN is not resent | yes | yes, monkeypatched CLI | no | no server reconciliation, no receipt |
| Inference device selection and stopwatch | yes | function covered | no hidden-test timing | 9h is not guaranteed |
| Pause ack/consume and file-hash resume | yes | yes | n/a | no |
| Colab after manual connect runs the worker | yes, no allocate API | yes, missing inputs stay blocked | no | headless allocation still unsupported |

Example-data CPU tests are not the MRI pilot. Production load still refuses a weight file whose SHA256 is not allowlisted.
