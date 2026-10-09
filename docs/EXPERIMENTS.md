# Experiments

Defined experiment queue (live pilot runs are recorded below):

| ID | Family | Variable | Status |
| --- | --- | --- | --- |
| EXP-001-reference | REFERENCE | public reproduction | defined_not_run |
| EXP-002-frozen | OWN | frozen DINOv2-S | defined_not_run |
| EXP-003-local-labels | OWN | label source ablation | defined_not_run |
| EXP-004-finetune | OWN | last-block unfreeze | defined_not_run |

Champion rule: audit + baseline delta + uncertainty + source legitimacy + runtime + finite outputs. `AUC >= 0.86` is not enough.

Synthetic smoke metrics must stay `kind=synthetic`.

## Live GPU pilot — 2026-10-03

Run `pilot-gpu-20261003` completed on Kaggle kernel
[bulsahkecici/rsna-knee-training-pilot, version 1](https://www.kaggle.com/code/bulsahkecici/rsna-knee-training-pilot).
The verified worker handshake reports Tesla T4. Frozen DINOv2 ViT-S/14 at 224 px
trained for two epochs on 26 supervised studies with 8 optimizer updates.
The official Meta download matched the local weight allowlist; no published checksum claim is made.

The cache contains 117 real studies with zero quarantines. Folds, duplicate groups,
and gold-eval exclusions passed before the job was sent. Numeric canonical labels
were uploaded privately; raw radiology reports stayed on the Mac.

Gold holdout: 29 studies, 12 defined class AUCs, macro AUC 0.530851.
This is a small pilot holdout, not a competition score or complete independent OOF result.
Group-bootstrap 95% macro-AUC interval: 0.466964–0.593088 (1909 / 2000 replicates retain all 12 classes).
The fold-0 metric called `weak_metrics` contains 15 cached studies but only five
with observed targets in this snapshot; those five use gold labels. It must not
be reported as validation of weak-label accuracy.

Decision: pilot only; not champion, not submission-ready. Production audit failed.
No Kaggle competition submission was made. Checkpoint, result record, input hashes,
and finite predictions were verified after download. There is no prior baseline delta.
See `artifacts/runs/pilot-gpu-20261003/` for metrics, review, audit, and checkpoint.

Local label expansion retained 4 valid weak rows and
quarantined 3 rows due to non-verbatim evidence. The large batch
was stopped for quality review; its SQLite resume cache remains intact. Resolve
these evidence errors using a non-gold-eval development set before scaling labels
and running a larger training pilot. The GPU training snapshot used three valid
weak rows and all 58 numeric gold rows; the later fourth valid weak row did not
enter this trained checkpoint.

## First real submission — 2026-10-03

The pilot checkpoint was packaged with Linux Python 3.13 offline DICOM decoder
wheels. Kaggle inference kernel `bulsahkecici/rsna-knee-first-infer`, version 2,
completed with internet disabled. Thirty real training studies were decoded
again and matched the existing uint8 cache shard hashes exactly. Decode-inclusive
benchmark: 60.509 s / 30 studies, projecting 2622.063 s for approximately 1300
hidden studies. This is a projection, not a time guarantee.

Submission eligibility audit now passes all ten gates from these downloaded
artifacts. The code/weights package is frozen with SHA256
`733f5caa0338cd49acfea6f6e01fe9e988497503fa236c3f83ef6525ba3cbad4`.
This does not change the pilot-only model-quality decision.

Kaggle server submission ID: **56786806**, kernel version **2**. The CLI omitted
the receipt text; it was reconciled against the unique attempt description in
Kaggle's actual submission history. One submission was sent. Scoring was PENDING
at registration; the durable scorer watcher records its final server result in
`artifacts/runs/pilot-gpu-20261003/scoring-status.json`. No public score is inferred
from the local gold holdout.

### Hidden rerun error and retry

The single-submission API returned COMPLETE with an error description and no public score for 56786806: Notebook Threw Exception. The list-submissions endpoint retained a stale PENDING status. The status poller now reads the exact submission detail and treats a nonempty error description as ERROR, never SCORED. The failed package and receipt remain preserved; this attempt was not successful.

Kaggle does not expose a traceback through the available submission details. Two identified failure paths were addressed without claiming a proven root cause: training-data benchmark execution during hidden reruns/incomplete mounts, and a single invalid DICOM series aborting an otherwise decodable study. Hidden reruns skip the benchmark. Inference masks failed series explicitly and keeps other valid image slots; training remains strict, and an entirely undecodable study is still quarantined instead of silently fabricated. The retry rehearsal checks 117 real DICOM studies and exact cache parity before a new submission.

Retry offline rehearsal: kernel version 4 COMPLETE, internet disabled, 117 actual DICOM studies decoded with zero quarantine or partial-series failures; all 117 uint8 shard hashes matched. Runtime 254.767 s, projection 2830.748 s for 1300 studies. Ten eligibility gates passed. Frozen retry package SHA256: `7296edec3833e199d79e3763590f80e04585c2c1a46a6e8b5d2f7454d5978154`.

Second real submission receipt: **56787970**, inference kernel version **4**. Sent once after the failed v2 attempt. Public scoring still pending at registration.

### First successful scored submission

Kaggle submission **56787970**, kernel version **4**, completed without a server error and received real public score **0.49**. The exact receipt and server record are saved in `artifacts/runs/pilot-gpu-20261003/scoring-status.json`; registry run and submission stages are SCORED. This completes the first real, successfully scored submission objective. Two competition submissions were used: v2 failed, v4 scored. The frozen successful package hash remained unchanged.

Model-quality decision remains pilot only, not champion: public AUC 0.49 is low and below the 0.50 random-ranking reference. The actual local gold holdout AUC 0.53085 is a different measurement and must not replace this public result. Scaling valid local weak labels, expanding image training, and additional held-out review would be required to improve performance.

## Label repair and next campaign — 2026-10-03

The three previously quarantined records contain four whitespace-only evidence
mismatches. A local replay now recovers each unique verbatim source span and
passes the unchanged strict validator. Word, punctuation, case changes and
ambiguous matches remain rejected. The evidence policy and updated prompt
invalidate older resume keys; raw responses for repaired spans are preserved
only in local provenance.

Live development run `labels-dev-20261003-v2`: five reports, five valid, zero
quarantines, zero recovered spans needed. Across 60 targets: 6 positives,
23 explicit negatives, 2 borderline negative hints, 1 uncertain, 28 unmentioned.
Uncertain/unmentioned remain masked. All gold UIDs and gold-eval duplicate groups
are excluded from development extraction. This verifies schema/evidence, not
clinical label accuracy.

The 100-new-report development pilot `labels-dev-20261003-v2-100` is running
locally, reusing the five valid current-version cache entries. Numeric progress
is written to
`state/campaigns/campaign-20261003-v2/local-label-progress.json`.
No new model-quality score is available.

`rsna campaign` prepares a sequential workflow: development evidence gate,
full local extraction, numeric labels with gold reserved for held-out evaluation,
private Kaggle CPU cache, then separate eight-epoch frozen and last-block
fine-tuning jobs on shared inputs. The remote preparation excludes raw reports,
SQLite, extractor envelopes, and credentials. Gold evaluation never enters
image training. Completed remote results require experiment review before
OOF expansion, ensemble, or competition submission.

The automatic approval reviewer rejected the remote workflow launch because
the sensitive payload and destination were not sufficiently authorized/verified.
Read-only follow-up confirmed the configured owner `bulsahkecici` and that the
existing RSNA kernels appear under authenticated `kernels list --mine`.
The exact private destinations, allowed file/field schema, and a 63-row numeric
preview SHA256 are recorded in
`state/campaigns/campaign-20261003-v2/authorization-manifest.json`.
Explicit approval was requested. The remote campaign has **not** started;
no new Kaggle upload, GPU job, or competition submission was made.

Validation: 104 tests passed; three pre-existing DICOM fixture UID warnings.
The new campaign templates compile and their privacy/split gates pass fixture
tests. They have not yet been verified on live remote campaign data.

### Development pilot completion

`labels-dev-20261003-v2-100` completed: 100 new attempts, 99 valid, one quarantine; with five cached current-version successes, 104 valid records and 836 observed training targets. Strict local source-evidence and selection gate passed (1/105 rejected). This is not a clinical accuracy estimate. Full local development extraction `labels-full-20261003-v2` was started with resume and the same gold/duplicate exclusions. Remote upload/GPU approval remains pending; no new remote campaign was launched.

### Explicit remote authorization and controller startup

The user explicitly approved the private Kaggle workflow. The authorization manifest now records `EXPLICIT_USER_APPROVAL_RECEIVED`. Controller `campaign-20261003-v2` was launched with idle-sleep prevention; it adopted the completed development pilot and is `WAITING_FOR_LOCAL_LABELS` on the existing `labels-full-20261003-v2` job. No second LLM job or repeated pilot was started. After the full local evidence gate passes, it will prepare the private CPU cache and run frozen and last-block fine-tuning jobs sequentially. No remote job had been pushed at this startup verification. A new adoption test verifies final registry state, real-run identity, and label hash before reuse.


### rsna-ab-20261008-v2-frozen
Real remote training completed, exit_reason=ok; 3366 train studies, 3368 updates. Gold holdout (29 studies) macro AUC 0.68239079; weak holdout (824 studies) macro AUC 0.75065202. No competition submission or leaderboard score. v3 comparison still running; do not select a winner yet.


### v3 interactive result comparison
User downloaded v3 metrics: kind=real, 3368 updates, same 3366 training UIDs as v2, gold eval excluded. Gold AUC v3=0.73150272 versus v2=0.68239079 (delta +0.04911193); weak AUC v3=0.76245724 versus v2=0.75065202. Ten of twelve gold target AUCs improved, MCL and lateral meniscus declined. Prefer v3 for next development experiment, conditional on collecting interactive checkpoint/provenance. API saved version still ERROR; no leaderboard submission. Only 29 gold studies, no paired uncertainty computed.


### v3 fine-tune launch
Verified user-provided checkpoint SHA e1a26f02dc46a5bc74ee09bf4c1e995e98aa4b3535885309ac55f002c82dfeac (frozen 3368 steps). Private checkpoint dataset ready. Private kernel rsna-v3-finetune-20261008-finetune version 1 pushed with receipt. New model warm-start experiment, 4 epochs, head lr 1e-4, last-block lr 1e-5; optimizer/scheduler reset, same v3 cache/folds/labels and 3366 train studies. No leaderboard submission; await real evaluation.


### v3 fine-tune result
Real run COMPLETE, exit_reason=ok, same training UIDs and gold excluded. Gold holdout AUC 0.75068253 versus frozen 0.73150272; weak AUC 0.77612742 versus 0.76245724. 1684 updates. Only 29 gold studies; no uncertainty computed or submission performed. Detailed local result: state/campaigns/rsna-ab-20261008/finetune-comparison.json.


### v3 offline packaging check
Fine-tuned checkpoint hash verified against metrics; private offline asset includes exact v3 data preprocessing modules, interior centers 0.15/0.50/0.85, strict decoding, local Linux codec wheels and file hashes. Private kernel rsna-v3-offline-check-20261008 version 1 acknowledged and RUNNING, Internet disabled. Measures end-to-end on 30 real train studies and produces visible-test CSV when mounted, with finite/column/UID validation. No competition submission. Results pending.


### v3 offline check version 2 completed
Internet-disabled real 30-study benchmark and visible-test CSV inference completed. Finite probabilities and schema/unique UID checks passed. Results: artifacts/kaggle/v3-offline-check/output-v2. Runtime projection is an estimate based on 30 studies, not a hidden-test guarantee. No competition submission performed.


### v3 competition submission receipt
Final inference kernel version 1 COMPLETE; submission ref 56973673 confirmed in Kaggle submissions API, PENDING, no public score yet. Previous scored pilot 0.490; local fine-tuned gold AUC 0.75068253 is not a leaderboard score.


### Public result and seed 2047 experiment
Kaggle submission 56973673 COMPLETE, public score 0.740 confirmed by API (previous pilot 0.490). New private kernel rsna-v3-seed2047-20261009 version 1 pushed, QUEUED. Same frozen v3 initialization, cache, labels, folds and four-epoch fine-tune recipe; seed changed 2026 to 2047. This is shared-initialization ensemble diversity, not independent pretraining. Compare aligned heldout probabilities and equal-weight average before selecting next submission.


### seed 2047 and ensemble decision
Second seed COMPLETE, real gold AUC 0.74756610; seed 2026 0.75068253. Equal probability average 0.74832980. Weak AUCs 0.77553664 / 0.77612742 / ensemble 0.77576533. Exact prediction UID alignment and finite values checked. Do not replace public 0.740 model or submit ensemble: no validation improvement. Autonomous local poller started (60 seconds, macOS notifications on change and hourly), auto-collection and comparison enabled. No automatic OS reboot startup.


### Autonomous round expanded with structural experiment
Live duration8 kernel confirmed QUEUED, receipt retained. Plane-concat head implemented with function-preserving mean checkpoint conversion; checkpoint/evaluation/offline pooling identities persisted, legacy default mean preserved. 18 combined focused tests passed; agent reports 20 targeted model tests and Ruff passed. Queue: duration8, headlr3e4, plane, strictly sequential within12 observedGPUhours/24wallhours. Reviewed candidates require nonnegative gold delta, weak delta>=0.002, paired gold lower CI>=-0.03, then real internet-disabled30-study runtime/CSV validation before a single submission. Unknown receipts require reconciliation. At quota refresh new jobs stop for accounting. No new score claimed.


### Autonomous candidate gate hardening
All gold and weak probabilities are checked for finite [0,1] values, unique UIDs and disjoint validation splits. Macro AUCs are recomputed with numeric masks and compared to stored metrics; undefined targets fail promotion. Twelve gate/publication tests passed, and actual champion predictions reproduced both reported metrics under the strengthened review. Duration8 still live QUEUED; no remote restart or new metric claimed.


### Six-center coverage preparation
Existing v3 manifest has12657 present series, median30 slices, nine unique decoded slices each with3 centers. Six interior centers cover mean17.71257 distinct slices; uint8 image payload estimate11.431GB versus5.716GB (not actual output yet). Same4219 requested studies, labels/folds/source identity, private CPU kernel rsna-v3-sixcenters-cache-20261009 v1 acknowledged and confirmed RUNNING. Existing GPU duration8 remains QUEUED. Auxiliary local watcher will automatically launch a small-output CPU verification bridge after completion, checking every shard hash, shape [3,6,3,224,224], dtype and masks. No Mac MRI download, no new AUC claimed, no GPU quota charged for CPU cache.


### Versioned inference policy
Live duration8 transitioned QUEUED→RUNNING; six-center CPU cache remains RUNNING. Offline inference now takes a versioned preprocess-config asset and resolves center count from it, rejects conflicting explicit counts, unsupported pixel policy and mismatching cache/preprocess/DICOM source hashes. Candidate packaging includes that configuration. Legacy default3-center path remains supported; six-center future candidates cannot silently score with3 centers. Sixteen policy+pooling tests and Ruff passed. No new AUC or public score available yet.


### Full regression verification
Full pytest suite exit0 after numerical-policy, pooling, autonomous gates and inference-policy changes. Only existing invalid synthetic-fixture DICOM UID warnings. Local controller heartbeats confirmed fresh while both duration8 GPU training and six-center CPU cache were live RUNNING; no restart/reupload on observation delays. No new training metrics available yet.

### Autonomous round: verified results and submission

All three real training jobs completed. Independent review recomputed masked AUCs, checked finite probabilities and split identities, and confirmed gold studies excluded from training. Compared with the four-epoch fine-tuned reference:

| Experiment | Gold AUC (29 studies) | Weak AUC (824 studies) | Gold delta | Weak delta | Offline candidate gate |
| --- | --- | --- | --- | --- | --- |
| duration8 | 0.75701531 | 0.78355471 | +0.00633278 | +0.00742729 | Passed |
| headlr3e4 | 0.74929965 | 0.77691960 | -0.00138288 | +0.00079219 | Failed |
| plane | 0.75133385 | 0.77867027 | +0.00065132 | +0.00254285 | Passed |

Duration8 was selected by weak holdout AUC among qualified candidates. Its paired gold delta 95% bootstrap interval is [-0.00582903, 0.02033794]; the small gold set does not establish a statistically clear improvement. Observed GPU quota use across these jobs was 1.31 hours. Evidence: `state/campaigns/autonomous-20261009/status.json` and downloaded prediction artifacts.

The first duration8 offline kernel failed because its integrity manifest included `.DS_Store`, which Kaggle removed. Packaging now excludes hidden files and Python cache files. Corrected private asset version 2 and offline kernel version 2 completed. The real 30-study benchmark took 44.602 seconds; projecting to 1300 studies gives approximately 32 minutes, an estimate rather than a hidden-test runtime guarantee. Visible-test CSV validation passed. Kaggle submission receipt **56995656** is confirmed PENDING; no new public score is available. The scored reference remains receipt **56973673**, public **0.740**.

### Verified six-center cache and controlled training

The CPU cache and subsequent independent verification kernel completed. All 4219 studies have three planes, uint8 shards of shape [3,6,3,224,224], and zero quarantine; verification checked every shard hash, masks and nonconstant present planes. Cache manifest SHA256: `8a2332466bd15b955eda61acc7e0e46792cc503e8b484f4cb19c5bbe116de87a`. Payload totals 11,431,397,376 bytes. MRI shards remain remote.

Private kernel `bulsahkecici/rsna-auto-20261009-sixcenter-finetune` version 1 is confirmed RUNNING. It uses the same frozen initialization, eight epochs, learning rates, seed, folds and numeric labels as duration8, changing coverage from three to six centers. This is a new warm-start experiment with reset optimizer/scheduler, not a strict checkpoint resume. Its controller reviews against the stronger duration8 baseline and packages the six-center preprocessing policy only if the candidate passes. No six-center metrics or competition submission exist yet.
