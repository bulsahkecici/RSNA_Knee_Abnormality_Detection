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

### 2026-10-09 — Colab six-center seed 2047

Real 8-epoch DINOv2 training finished with `exit_reason=ok`; archive ingested under `artifacts/colab/sixcenter-seed2047/output`. Independent review verified matching training UIDs, exclusion of gold evaluation UIDs, unique aligned finite probabilities, and recomputed masked gold/weak AUCs matching reported metrics. Candidate checkpoint SHA-256: `aeacc0e53563b71d63c9c4060553dc224caac52bd1276c6d652eccabc8ee8fb0`.

| Model | Gold AUC (29 studies) | Weak holdout AUC (824 studies) |
|---|---:|---:|
| Six-center seed 2026 (public 0.767) | 0.781277 | 0.796132 |
| Colab seed 2047 | 0.780467 | 0.796418 |
| Fixed 50/50 probability ensemble | 0.781473 | 0.796324 |

The ensemble improves gold by 0.000196 and weak by 0.000192. Paired 500-resample gold delta interval: [-0.007302, 0.008196]. Neither the seed candidate nor ensemble meets the existing promotion gate (weak delta >= 0.002). Keep the public 0.767 model; no new Kaggle submission. The reused small gold split limits certainty; these are heldout metrics, not independent cross-validation or new leaderboard scores. Detailed evidence: `review.json` and `ensemble-review.json` in the ingested output directory. Next planned controlled experiment remains last-two-block fine-tuning with matched cache, labels and split.

### 2026-10-09 — Two-block Colab result; submissions disabled

Imported the replaced `artifacts/colab/colab-results.zip` into `artifacts/colab/two-block-seed2026/output`. Real DINOv2 training completed; matching train UIDs, gold exclusion, aligned finite predictions and independently recomputed masked AUCs pass. Gold AUC 0.79532030 (+0.01404333), weak holdout AUC 0.80838210 (+0.01224970) versus six-center one-block seed2026. Paired gold delta interval [-0.00882433, 0.03584443]. This is a stronger local baseline, not a measured public score or independent OOF result. User explicitly declined submission; no Kaggle publication was performed. `state/submission-policy.json` disables the autonomous publisher until explicit user reauthorization.

Next prepared local notebook: `07_colab_sixcenter_label_attention.ipynb`. Controlled finding-specific attention on the existing six-center views, same frozen initialization/two blocks/seed2026/8 epochs. Zero-initialized attention reproduces masked mean; head optimizer includes attention parameters. No additional images, labels or public weights are introduced. Attention parity, masked-view invariance, finite empty-mask outputs and gradient/optimizer coverage were tested. Subsequent experiment should broaden coverage/series after measured aggregation results, not infer a 0.9+ leaderboard promise.

#### Attention notebook evaluation packaging correction

The initial Colab handoff trained successfully through the training phase but failed in checkpoint evaluation: its embedded historic `evaluation/heldout.py` constructed mean pooling, rejecting `attention.weight`. Updated the generated notebook to embed and reload the current pooling-aware evaluator before running the worker. Existing Drive checkpoints can resume under the same experiment identity; no new score is claimed until evaluation completes. Added an attention checkpoint round-trip evaluation test, including nonzero attention parameters, to verify that the evaluator preserves the model's outputs.

#### Completed-checkpoint supervision recovery

After fixing attention evaluation, rerunning a completed job returned `no_supervision`: the training loop counted only supervision observed in the current invocation, which is zero when all epochs have already finished. The loop now persists/restores cumulative supervision. Legacy checkpoints without that counter qualify for evaluation-only recovery only when completed epochs/steps, nonempty optimizer state, exact input/config identity and current nonzero masked supervision are verified. No new training or synthetic score is implied. Updated the Colab attention handoff and added completed-resume/legacy-identity regression tests.

#### AMP-aware completed-resume correction and recovery handoff

The previous legacy check incorrectly required successful update count to reach the planned scheduler budget. Completed epochs can have fewer successful updates because AMP skips nonfinite steps or masked-only groups produce no update. Legacy completion now requires completed epochs, positive successful updates, consistent attempted/successful/skipped counters, nonempty optimizer state, no partial group and exact input/config identity. It does not waive gold exclusion or current label supervision. A regression fixture covers skipped updates and rejects inconsistent counters. `08_recover_colab_attention_evaluation.ipynb` provides a dedicated existing-runtime evaluation handoff: it prints nonsecret counters, refuses incomplete epochs, reloads pinned implementations, asserts no new successful steps, saves real metrics/archive and never submits. Colab execution still awaits user-run evidence.

### 2026-10-09 — Verified attention result and fixed ensemble

Attention archive ingested under `artifacts/colab/label-attention-seed2026/output`. Independent review passed real training, train-UID equality, gold-image exclusion, unique aligned finite predictions, and masked AUC recomputation. Checkpoint SHA-256 `bca3fb160434fbfadc220006379dfc9bbba955ffc638be3cb84bbb6c8d50c7dc`. Completed epochs 8, successful steps 3361, attempted updates 3368, AMP skips 7; recovery evaluated without new optimizer updates.

| Model | Gold AUC (29 studies) | Weak AUC (824 studies) |
|---|---:|---:|
| Two-block mean baseline | 0.79532030 | 0.80838210 |
| Finding-specific attention | 0.78327653 | 0.81028034 |
| Fixed 50/50 probability average | 0.79040248 | 0.81103080 |

Attention gold delta -0.01204377, weak +0.00189824; paired gold delta interval [-0.03376324, 0.00553083]. Equal ensemble gold delta -0.00491783, weak +0.00264871; paired interval [-0.01954442, 0.00812109]. Neither passes the current promotion gate due to gold regression; do not tune blend weights on the reused 29-study gold set. Keep two-block mean as local reference, while retaining attention as an exploratory candidate for broader image bags. No Kaggle submission/public-score claim. Submission policy remains disabled. Next work is denser physical coverage and additional MRI series with bounded cache size, sampled training windows and real runtime/QC evidence.

### 2026-10-09 — P0: dense cache pilot, all-gold folds, numeric label audit

**Cache code consolidation.** The v3 builder that produced the six-center cache existed only in `scripts/cache_v3_source`, and a v4 draft forked it again; `src/rsna_knee/data/cache_build.py` was an older, unused builder. `src` now holds one builder (commit 484ca3a, parallel decoding 6e3deec): NPZ shards with a running byte budget, configurable fluid/structural slots, 12 windows per series inside one series, recorded through-plane spacing/extent; legacy v3 NPY manifests remain readable. Kaggle kernels are generated from a committed git ref by `scripts/make_kaggle_kernel.py` (deterministic source archive, no labels or reports embedded) instead of string-patching earlier payloads.

**Real dense-cache pilot** — private CPU kernel `bulsahkecici/rsna-v4-dense-pilot-20261009` v1, 64 random training studies (seed 2026), 4 CPUs, source `c00ba26f…`. Both policies: 64/64 built, 0 quarantine, all IOP×IPP ordered, full manifest verification passed.

| Policy | s/study (1 process) | MB/study mean / p95 | Compression | NPZ read s/study | Projection 4219 studies |
|---|---:|---:|---:|---:|---|
| 12 windows × 3 fluid slots | 2.53 | 3.61 / 4.23 | 1.50× | 0.025 | 14.2 GiB, 3.0 h single process |
| 12 windows × 5 slots (+ sagittal/coronal structural) | 2.39 | 5.47 / 6.70 | 1.65× | 0.039 | 21.5 GiB — exceeds 18 GiB budget |

Median slice spacing 3.3 mm sagittal, 3.6 coronal, 3.85 axial; 12 sagittal windows decode ~60 % of available sagittal slices, coronal ~97 %. Structural coverage in the sample: sagittal 59/64, coronal 48/64. In the 5-slot policy 7 sagittal and 5 coronal fluid slots lose their v3 non-fluid fallback to the structural slot, which would confound a D1→D2 comparison. Decision: build D1 (3 fluid slots, 12 windows) for all 4407 training studies now; redesign D2 as D1 views plus a separate structural cache rather than a single 5-slot shard. Edge windows (5 %/95 %) often show little anatomy; masking/attention handles them, kept for now. Projections are sample estimates, not guarantees.

**Full D1 cache** — private CPU kernel `bulsahkecici/rsna-v4-dense12-cache-20261009` v1 pushed (git 80542e2, source `12e34aab…`, 4 workers, 18 GiB hard budget, every shard re-verified in-kernel). Status tracked in `state/campaigns/p0-20261009/`.

**Folds v2 (all-gold holdout)** — `state/folds-v2-allgold.csv`, sha256 `e27530d2…`, derived from folds v1 (`f13720c5…`) with identical fold numbers and duplicate groups. All 58 gold studies are held out of image training: set A = 29 historic eval (repeatedly used for selection), set B = 29 formerly trained on. 4348 trainable studies; one duplicate-group member of a gold study is also dropped. Neither gold set is an unseen test set for earlier models.

**Numeric label audit** (`scripts/label_audit.py`, local, numeric fields only; aggregates in `state/label-audit/20261009.json`). Public CC0 label sets (stevenleehans full/v2/v4_blend, yunusgmsoy; hashes recorded) vs the 58 gold: macro AUC 0.878 / 0.887 / 0.893 / 0.880 — our measurement; those authors may also have tuned on the same gold. Weakest targets for every source: Synovitis (0.68–0.79), Lateral OA (~0.83), Fracture (0.79–0.88). Against our local extraction, explicit states agree closely (AUC of public score vs our explicit positive/negative 0.976–0.997). Public sources score our `not_mentioned` low (mean ≈0.18–0.27), but score `borderline_negative_hint` high for OA, Effusion, Baker's and Synovitis (mean 0.79–0.90), i.e. the main ontology disagreement is the host-borderline rule, not extraction. Largest raw disagreements: Lateral/Medial Meniscus (ours negative, public ≥0.8: 122 / 103), Contusion and ACL (ours positive, public ≤0.2: 81 / 74). Our extraction was never run on gold reports, so our own gold agreement is unknown; a gold-only local extraction (`rsna labels pilot --live --gold-measurement`, run `labels-goldmeasure-20261009`) is running for measurement only — gold labels override these rows and gold images never enter image training.

### 2026-10-10 — Our extractor vs gold; D1 cache complete; D0/D1/L1 prepared

**Our local extraction measured on gold reports for the first time** (`rsna labels pilot --live --gold-measurement`, run `labels-goldmeasure-20261009`, Qwen 27B local, 57/58 valid, 1 truncated response quarantined; measurement only — these rows never become training targets). Macro AUC against gold under state→score policies (`scripts/label_audit.py --tag 20261010`, `state/label-audit/20261010.json`):

| Label source | Macro AUC vs gold (57–58 studies) |
|---|---:|
| Ours as used in training (masked targets dropped) | 0.804 |
| Ours, unmentioned scored 0.25, uncertain 0.5 | 0.838 |
| Ours, additionally borderline-negative scored 0.5 | 0.841 |
| Public stevenleehans v4_blend / yunusgmsoy (CC0) | 0.893 / 0.880 |

Main error is positive precision against the host's high-grade criteria: of our `criterion_positive` calls, gold is positive for ACL 0.77, MCL 0.56, Medial OA 0.62, Lateral OA 0.53, Effusion 0.74, Contusion 0.58, Synovitis 0.79. Hard binary labels also discard severity ranking that the public probability scales retain. The previous image model's gold AUC (0.795) is close to our training labels' own gold agreement (0.804), consistent with — not proof of — labels being the binding constraint. Small sample; the same gold studies were used for earlier selection.

**Decision:** promote L1 ahead of P3. Pre-registered soft table `state/soft-labels/public-mean-v1.jsonl` (sha `7d045805…`): equal mean of one CC0 table per author (stevenleehans v4_blend, yunusgmsoy), all 58 gold studies removed, fixed before any image result. Evaluation targets stay the fixed hard masked labels; weak AUC therefore measures agreement with our labels and is not decisive for L1.

**D1 cache complete** — `bulsahkecici/rsna-v4-dense12-cache-20261009` v1: 4407/4407 studies, 0 quarantine, all three fluid planes present in every study (fallbacks: sagittal 257, coronal 159), all series IOP×IPP ordered, 15.0 GiB, build 70 min on 4 workers, every shard re-hashed in-kernel (508 s). Manifest sha256 `171ef4ff…`, preprocess hash `acdd5baa…`. The shard images stay on Kaggle; only summaries were downloaded (Kaggle API returned HTTP 429 when listing the full output with default 20-item pages; `--page-size 200` with a file pattern worked).

**High-resolution policy pilot** — `bulsahkecici/rsna-policy-pilot-hires-20261009` v1, 64 studies, 5 slots × 12 windows, 154 mm crop, series intensity window, 0 quarantine both configs:

| Policy | s/study (4 workers) | NPZ MB | PNG MB | JPEG q95 MB (max err 9–10) | NPZ for 4219 |
|---|---:|---:|---:|---:|---:|
| 384 px (0.40 mm/px) | 2.54 | 15.4 | 10.1 | 5.7 | 60 GiB |
| 256 px (0.60 mm/px) | 1.42 | 7.1 | 4.9 | 2.9 | 27.8 GiB |

No lossless option fits one 18 GiB kernel output. Next coverage cache: 256 px / 154 mm / 5 slots / series window / NPZ in two shards merged by `merge_cache_parts` (commit ae4809e); 384 px only if 256 px shows a gain.

**Training code** (commit d90acbf, model agent; session ended during lint, all tests pass): window sampling, unfreeze depth incl. full backbone with layer decay, official-pretrained init, ConvNeXt-T / CoAtNet-0 via sha-verified timm checkpoints, soft targets for training only, window-token transformer, evaluation-only anatomical TTA, run identity and timing/memory metrics. Experiment configs D0–D5, C1, C2, L1 pass `validate_arm` (declared changes equal actual differences vs control). A local end-to-end rehearsal of D1 and L1 (synthetic DICOM → v4 cache → build_job/run_job with official DINOv2 weights) finished `ok` with the expected label source in metrics.

**Prepared Colab notebooks** (gitignored, embed private numeric labels): `notebooks/generated/D0.ipynb` (v3 six-center cache, official init), `D1.ipynb` (dense 12-window cache), `L1.ipynb` (D1 + public soft labels). All pin git `963c8b5`, folds-v2 `e27530d2…`, 3341 train UIDs, official DINOv2 `b938bf1b…`; outputs and checkpoints on Drive with resume. Results are ingested with `scripts/review_experiment.py <ID> --zip <colab-results.zip>`, which re-verifies every pinned hash and the folds-v2 leakage check and reports gold-all / gold-A / gold-B / weak plus paired deltas vs the control arm. No result exists yet for any of these arms.

**v5 coverage cache complete (2026-10-10)** — `bulsahkecici/rsna-v5-five256-shard{0,1}-20261010` v1: 2204 + 2203 = 4407 studies, 0 quarantine, identical preprocess hash `8dd9d9d8…` (5 slots, 12 windows, 256 px, 154 mm crop, series intensity window), part manifests `e0803730…` / `4b308824…`, 15.4 + 15.3 GiB, ~85 min each on 4 workers, every shard re-verified in-kernel. Slot presence: all 5 in 3082 studies, 4 in 1092, 3 in 233; coronal structural absent in ~1000 studies (masked). D2 now targets this cache with DINOv2 at 252 px (resized from 256, recorded) — a declared package change vs D1; ablate resolution vs slots only if D2 wins. Local rehearsal (two synthetic shards → `merge_cache_parts` → `manifest_ready` → `run_job`) finished `ok`. Notebooks D0, D1, L1, D2 regenerated from git `61d7432`.

### 2026-10-11 — P1 results: D0, D1, D2, L1 (local validation, not leaderboard)

All four ran on Colab A100 80 GB, outer fold 0, folds-v2 (58 gold excluded from training), 3341 train UIDs, official DINOv2 ViT-S/14, seed 2026, one run each. Every archive passed `scripts/review_experiment.py`: pinned hashes, exit `ok`, `kind: real`, train-UID hash and folds-v2 leakage check. D1 and L1 ran from notebook commit `9ca062c`; the notebooks were later regenerated (`8bde598`) for download-cell fixes only, and the review accepted the drift because only `scripts/make_colab_notebook.py` changed and its payload/train/collect cells are identical (recorded as `git_commit_drift` in `review.json`). v4/v5 caches cover 32 fold-0 weak studies with no numeric label; they are skipped in weak AUC.

| Arm | gold all (58) | gold A (29) | gold B (29) | weak (n) | wall |
|---|---:|---:|---:|---:|---:|
| D0 v3 sparse | 0.740 | 0.784 | 0.683 | 0.807 (819) | 26 min |
| D1 v4 dense 12 | 0.752 | 0.783 | 0.718 | 0.806 (837) | 62 min |
| D2 v5 five-series 252 px | 0.748 | 0.769 | 0.732 | 0.794 (837) | 105 min |
| L1 D1 + public-mean-v1 | 0.783 | 0.816 | 0.733 | 0.796 (837) | 62 min |

Paired bootstrap deltas (2000 resamples) against each arm's pre-registered control:

| Arm vs control | gold all Δ [95% CI] | weak Δ [95% CI] | Verdict under its decision criterion |
|---|---|---|---|
| D1 − D0 | +0.012 [−0.004, +0.027] | −0.001 [−0.006, +0.003] (n 819) | not supported: gold CI includes 0, weak sign disagrees |
| D2 − D1 | −0.004 [−0.027, +0.023] | −0.012 [−0.019, −0.006] | not supported; series-selection audit never implemented; 1.7× slower |
| L1 − D1 | **+0.031 [+0.009, +0.052]** | −0.010 [−0.018, −0.004] | supported (weak drop non-decisive by pre-registration) |

L1's per-class gold gains are broad (Medial OA +0.06, Lateral OA +0.08, Baker's +0.07, Contusion +0.08, ACL +0.06; MCL and Effusion flat), consistent with the 2026-10-10 label audit that put our hard labels' own gold agreement (0.804) near the previous image-model ceiling. Caveats: one seed, 58 gold studies, gold A was used for earlier selection, and L1 is the only CI-excluding-zero result among three comparisons.

**Decision:** before D3–D5 change their base labels, replicate L1 at a second seed against a same-seed control. Pre-registered pair `D1S2` (D1, seed 2027) and `L1S2` (L1, seed 2027, control D1S2); the success rule is in `configs/experiments/P1-L1S2-soft-public-labels-seed2027.yaml` and was fixed before either run. D2's v5 cache is parked; 384 px is not pursued.

### 2026-10-12 — L1 replicated at seed 2027; D3–D5, C1, C2 rebased onto L1

D1S2 and L1S2 (D1 and L1 at seed 2027, otherwise identical; notebook `589d8b1`, Colab A100, ~61 min each) both passed `scripts/review_experiment.py`.

| Arm | gold all | gold A | gold B | weak (837) |
|---|---:|---:|---:|---:|
| D1 (seed 2026) | 0.752 | 0.783 | 0.718 | 0.806 |
| D1S2 (seed 2027) | 0.748 | 0.774 | 0.708 | 0.806 |
| L1 (seed 2026) | 0.783 | 0.816 | 0.733 | 0.796 |
| L1S2 (seed 2027) | 0.774 | 0.805 | 0.732 | 0.796 |

| Comparison | gold all Δ [95% CI] | weak Δ [95% CI] |
|---|---|---|
| D1S2 − D1 (seed noise) | −0.004 [−0.018, +0.010] | −0.000 [−0.004, +0.004] |
| L1S2 − D1S2 | **+0.025 [+0.005, +0.047]** | −0.011 [−0.019, −0.003] |
| L1 − D1 (from 2026-10-11) | +0.031 [+0.009, +0.052] | −0.010 [−0.018, −0.004] |

**Verdict under the pre-registered L1S2 rule: replicated** (CI lower bound > 0). The seed-to-seed gap of the hard-label control (−0.004) is small next to the label effect. Per-class gold deltas repeat across seeds: Lateral OA +0.08/+0.07, Contusion +0.08/+0.07, Baker's +0.07/+0.07, Medial OA +0.06/+0.05, ACL +0.06/+0.03, Fracture +0.03/+0.02. **Lateral Meniscus drops in both seeds (−0.04/−0.05)**, the one consistent loss — a candidate for a per-target label check, not acted on here (58 gold studies). The weak drop repeats too and is expected: weak is scored against the hard labels L1 no longer trains on.

**Action taken (rule fixed before the seed-2027 runs):** D3, D4, D5, C1 and C2 now use control L1 and public-mean-v1 soft labels; coverage stays v4 with 6 train centers, because no D0–D2 coverage arm won and L1 is the best measured recipe on that cache. Each arm's declared change set is unchanged and passes `validate_arm`. Suggested order: D3, D5, C1, C2, then D4 (its memory/time pilot is still open; ViT-S peaked at 0.45 GB on A100 so far).

Notebooks D3, D5 and D4 generated at `62a7f8d` (control L1, v4 cache, public-mean-v1) and rehearsed `ok`. C1 and C2 cannot be generated yet: their timm weight allowlists are empty and the notebook only downloads and pins the official DINOv2 file; the generator now refuses other encoders instead of silently fetching DINOv2.

### 2026-10-12 — D3 and D5 on the L1 base

Both ran from notebook `62a7f8d` (control L1: v4 cache, public-mean-v1 soft labels, seed 2026, Colab A100) and passed `scripts/review_experiment.py`. The folder handed over as D4 contained the L1S2 archive (identical sha256); D4 is not reviewed yet.

| Arm | gold all | gold A | gold B | weak (837) | wall |
|---|---:|---:|---:|---:|---:|
| L1 (2 blocks, mean) | 0.783 | 0.816 | 0.733 | 0.796 | 62 min |
| D3 (4 blocks) | **0.820** | 0.854 | 0.784 | 0.823 | 66 min |
| D5 (label attention) | 0.790 | 0.821 | 0.739 | 0.802 | 64 min |

| Arm vs L1 | gold all Δ [95% CI] | weak Δ [95% CI] | Verdict |
|---|---|---|---|
| D3 | **+0.038 [+0.020, +0.059]** | +0.027 [+0.020, +0.035] | supported: gold CI > 0 and weak agrees in sign |
| D5 | +0.007 [−0.007, +0.022] | +0.006 [+0.000, +0.011] | not supported: gold CI includes 0 |

D3 is the best arm so far and the first above our hard labels' own gold agreement (0.804); gold A and gold B both rise (+0.038, +0.052), so the gain is not confined to the subset reused for selection. Largest per-class gains: Fracture +0.13, Contusion +0.06, Baker's +0.06, ACL +0.06, MCL +0.06, Medial OA +0.06; Lateral OA (−0.02) and Synovitis (−0.02) dip. Peak memory 0.74 GB, 7% slower than L1. Single seed. Trainable depth is now the strongest lever measured; D4 (all blocks, layer decay 0.75, control L1 as pre-registered) is the direct follow-up and will also be reported descriptively against D3.

### 2026-10-12 — D4 (full DINO adaptation) on the L1 base

Notebook `62a7f8d`, Colab **A100 40 GB** (others ran on 80 GB), passed review. Gold all 0.817, gold A 0.837, gold B 0.801, weak 0.824; 84 min wall, peak 1.88 GB.

| Comparison | gold all Δ [95% CI] | weak Δ [95% CI] |
|---|---|---|
| D4 − L1 (pre-registered) | +0.035 [+0.014, +0.059] | +0.028 [+0.019, +0.037] |
| D4 − D3 (descriptive, not pre-registered) | −0.003 [−0.018, +0.010] | +0.000 [−0.004, +0.005] |

Supported against L1, but indistinguishable from D3 while costing ~1.3× wall time and 2.5× memory; per-class differences vs D3 are all within ±0.02. **D3 (last four blocks) is the working recipe.** Before it becomes the base, D3S2 repeats D3 at seed 2027 against the existing L1S2 run (one extra run instead of two); its rule is fixed in `configs/experiments/P1-D3S2-four-blocks-seed2027.yaml`.

### 2026-10-12 — D3 replicated at seed 2027; D3 is the base recipe

D3S2 (D3 at seed 2027, notebook `1fb673a`, A100 40 GB, 66 min, peak 0.74 GB) passed review. Gold all 0.814, gold A 0.841, gold B 0.788, weak 0.825.

| Comparison | gold all Δ [95% CI] | weak Δ [95% CI] |
|---|---|---|
| D3 − L1 (seed 2026) | +0.038 [+0.020, +0.059] | +0.027 [+0.020, +0.035] |
| **D3S2 − L1S2 (seed 2027)** | **+0.040 [+0.022, +0.060]** | +0.030 [+0.023, +0.036] |

**Verdict under the pre-registered D3S2 rule: replicated** (gold CI lower bound > 0, weak positive). Gold A and gold B rise in both seeds. Per-class gains repeat for MCL, Medial OA, Contusion, Fracture (+0.13/+0.15), Medial Meniscus, ACL and PF OA; Lateral OA dips in both (−0.02/−0.02). Combined path from our hard labels: D1 0.752 → L1 0.783 → D3 0.820 (seed 2026); D1S2 0.748 → L1S2 0.774 → D3S2 0.814 (seed 2027).

Descriptive only (not pre-registered): averaging D3 and D3S2 predictions gives gold 0.820 (Δ vs D3 −0.001 [−0.008, +0.007]) and weak 0.826 (+0.003); adding D4 gives gold 0.821, weak 0.828. Seed averaging is not a lever on this gold set.

**D3 (v4 cache, public-mean-v1 soft labels, last four DINOv2 blocks trained) is the base recipe** for later arms. Open next steps: pin and wire ConvNeXt-T / CoAtNet-0 weights for C1/C2 on this base; the three-outer-fold CV row; a Kaggle inference path for the competition test set.

### 2026-10-12 — Next arms prepared: C1/C2 on the D3 base, three-fold CV, first D3 submission

**C1 / C2.** ConvNeXt-T (`convnext_tiny.fb_in1k`, sha `08b9dc9c…`) and CoAtNet-0 (`coatnet_0_rw_224.sw_in1k`, sha `efe01c8c…`) `model.safetensors` were downloaded from huggingface.co/timm (Apache-2.0), hashed locally, verified to load strictly into the registered architectures, and allowlisted in `configs/models/*.yaml`. The notebook generator now pins each encoder's own weights. Both arms use control **D3** (rebased from L1) with trainable depth scaled to D3's third of encoder units: ConvNeXt-T 6 of 18 blocks; CoAtNet-0 both transformer stages (2 of 4; no whole-stage count equals 1/3, and D3/D4 showed deeper adaptation helps then saturates). Notebooks `C1.ipynb`, `C2.ipynb` (`f2d9cfc`) rehearsed `ok` with the real pretrained files.

**Cross-validation.** Outer fold is now a config section (absent = 0); the worker holds `train.outer_fold` out as the weak split (before this, any outer fold other than 0 would have stopped with `train_weak_overlap`). Arms L1F1/D3F1 (outer fold 1, 3326 train studies) and L1F2/D3F2 (outer fold 2, 3324) repeat the L1→D3 comparison. The rule, fixed before any run (in the D3F1/D3F2 configs): D3 is consistent across folds if D3Fk − L1Fk has a positive gold_all delta and a weak ci95 lower bound > 0 in **both** folds 1 and 2. Gold is the same 58 studies in every fold, scored after training on a different study set. Notebooks rehearsed `ok`.

**Submission.** `scripts/make_submission_kernel.py` builds an internet-off kernel that preprocesses test DICOMs with the cache kernel's exact embedded source (policy hash must equal training's `acdd5baa…` once pixel-library versions are set to training's), predicts with the notebook's exact training source, and, outside the competition rerun, first requires the 58 gold studies re-preprocessed from DICOM to reproduce the reviewed Colab probabilities. Private dataset `bulsahkecici/rsna-d3-submit-20261012-asset` and kernel `bulsahkecici/rsna-d3-submit-20261012` v1 (D3 checkpoint `803f0e70…`) were pushed; the user authorized this submission on 2026-10-12. No receipt or score yet.

### 2026-10-12 — First D3 submission: public 0.835; CV confirms D3; C1/C2 do not beat D3

**Kaggle submission 57114448: public score 0.835** (previous best 0.767, receipt 57003846). Submitted by the user from the Kaggle UI on kernel `bulsahkecici/rsna-d3-submit-20261012` v1 (D3 checkpoint `803f0e70…`); the API row has no kernel reference, so that link is the user's statement. In the kernel's validation run the 58 gold studies, re-preprocessed from DICOM on Kaggle, reproduced the reviewed Colab probabilities (max |Δ| 0.0014, mean 0.00013); numpy/Pillow/pydicom matched the training cache exactly; 3/3 visible test studies decoded strictly, 5.8 s/study end to end. Record: `artifacts/kaggle/rsna-d3-submit-20261012/submission-record.json`. The one-time authorization is used; `state/submission-policy.json` is back to `allow_submission: false`.

**Cross-validation (rule fixed before the runs): consistent.** All six Colab archives passed review (notebook `f2d9cfc`).

| Outer fold | L1 gold | D3 gold | D3 − L1 gold [95% CI] | D3 − L1 weak [95% CI] (n) |
|---|---:|---:|---|---|
| 0 (2026-10-12) | 0.783 | 0.820 | +0.038 [+0.020, +0.059] | +0.027 [+0.020, +0.035] (837) |
| 1 | 0.785 | 0.828 | +0.043 [+0.026, +0.065] | +0.031 [+0.024, +0.038] (846) |
| 2 | 0.781 | 0.825 | +0.044 [+0.025, +0.067] | +0.031 [+0.024, +0.038] (849) |

Both folds 1 and 2 meet the rule (gold point delta > 0, weak CI lower bound > 0). With seed 2027 (+0.040) the D3 gain over L1 now holds in four independent comparisons. Gold A and gold B rise in every fold.

**C1 / C2 (control D3, seed 2026, fold 0): not supported.**

| Arm | gold all | Δ vs D3 [95% CI] | weak Δ vs D3 | wall (A100 40 GB) |
|---|---:|---|---|---:|
| C1 ConvNeXt-T, 6/18 blocks | 0.740 | −0.080 [−0.110, −0.050] | −0.045 [−0.058, −0.031] | 75 min |
| C2 CoAtNet-0, 2/4 stages | 0.818 | −0.003 [−0.024, +0.019] | −0.003 [−0.012, +0.007] | 88 min |

ConvNeXt-T is clearly worse on this recipe. CoAtNet-0 is statistically indistinguishable from D3 but slower and not better.

Descriptive, not pre-registered: averaging D3 and C2 leaves gold unchanged (−0.000) with weak +0.003 [−0.002, +0.009]; D3 + D3S2 + C2 gives gold +0.001, weak +0.005 [+0.001, +0.010]. Averaging the three fold models D3, D3F1, D3F2 (none saw gold) gives gold 0.824, +0.004 [−0.003, +0.010] vs D3 — not separable from D3 on 58 gold studies; weak cannot score fold ensembles (each model trained on the others' weak folds). No ensemble meets a promotion bar; any future submission of one should be a pre-registered decision, not a choice made from these numbers.

### 2026-10-12 — Prepared D3ALL / D3E12; ensemble rule fixed in advance; Lateral Meniscus check

**D3ALL** (outer_fold `none`: all 4160 non-gold studies with an observed target, +25 % over D3's 3341; no weak split) and **D3E12** (D3 with 12 epochs) — configs `P3-D3ALL-*`, `P3-D3E12-*`, notebooks at `8040f46`, rehearsed `ok`. Rules fixed in the configs before running: D3ALL is the next submission candidate if its paired gold_all delta vs D3 is ≥ −0.01 (non-inferiority; gold only); D3E12 is supported if the gold_all delta vs D3 is positive and the weak ci95 lower bound is > 0. (`4b559f1` repairs a review regression introduced and caught in the same session; past reviews re-checked unchanged.)

**Ensemble rule (fixed now, before D3ALL / D3E12 exist).** The next submission is a single model: D3ALL if it passes, otherwise D3E12 if supported, otherwise no new submission. A probability average of several models is submitted instead only if, on gold_all, it beats that single candidate with a paired ci95 lower bound > 0. The averages measured so far do not come close (fold models +0.004 [−0.003, +0.010], D3+D3S2+C2 +0.001 vs D3), and the submission kernel builder currently packages one checkpoint.

**Lateral Meniscus (descriptive; gold not used to choose anything).** On the 4225 non-gold studies with both label sources, our hard labels and public-mean-v1 agree closely for LM: mean soft value 0.82 where ours is positive, 0.05 where ours is negative; only 8 of 3025 our-negatives have soft ≥ 0.5, while 91 of 680 our-positives (13 %) have soft < 0.5, and 520 studies we mask get soft supervision (mean 0.21). Medial Meniscus behaves the same way (0.86 / 0.09). Gold has 23 LM positives.

| LM AUC | D1 | L1 | D3 | D1S2 | L1S2 | D3S2 | L1F1 | D3F1 | L1F2 | D3F2 |
|---|---|---|---|---|---|---|---|---|---|---|
| gold | 0.820 | 0.781 | 0.794 | 0.840 | 0.790 | 0.826 | 0.798 | 0.783 | 0.795 | 0.812 |
| weak (our labels) | 0.722 | 0.721 | 0.747 | 0.720 | 0.721 | 0.751 | 0.675 | 0.705 | 0.685 | 0.731 |

Switching to soft labels costs LM 0.04–0.05 on gold but nothing on weak (scored against our own labels), and D3 recovers about half of the gold loss while raising weak LM. LM is the hardest finding on weak in every arm (~0.70–0.75). The two label sources barely disagree on LM, so this looks like an image-difficulty problem rather than a label-source choice; no label change is proposed. Re-picking per-target label sources by gold AUC remains off the table.

### 2026-10-13 — D3ALL passes, D3E12 supported; D3ALL is the pre-registered submission candidate

Both archives (notebook `8040f46`, Colab A100 40 GB) passed review.

| Arm | gold all | gold A | gold B | weak (837) | Δ gold vs D3 [95% CI] | Δ weak vs D3 [95% CI] | wall |
|---|---:|---:|---:|---:|---|---|---:|
| D3 | 0.820 | 0.854 | 0.784 | 0.823 | — | — | 66 min |
| D3ALL (all 4160 non-gold, no weak split) | 0.832 | 0.863 | 0.798 | — | +0.012 [+0.002, +0.021] | — | 80 min |
| D3E12 (12 epochs) | **0.839** | 0.876 | 0.804 | 0.839 | +0.019 [+0.005, +0.035] | +0.016 [+0.010, +0.022] | 98 min |

D3ALL meets its non-inferiority rule (≥ −0.01) and in fact improves gold. D3E12 meets its rule (gold > 0, weak CI > 0). Per-class: both gain ACL (+0.03) and Medial Meniscus (+0.03/+0.04); D3E12 also MCL +0.05, Contusion +0.06, Baker's +0.04, Synovitis +0.03, with Lateral Meniscus −0.01.

Pre-registered ensemble rule applied: mean(D3ALL, D3E12) vs D3ALL on gold +0.005 [−0.003, +0.016] — CI includes 0, so the candidate stays the single model **D3ALL** (descriptively D3E12 − D3ALL is +0.007 [−0.005, +0.021]). Submission kernel `rsna-d3all-submit-20261013` built (D3ALL checkpoint `e0851666…`, 58-row gold parity file) and its private dataset uploaded; the kernel push and any submission await the user's authorization.

Next arm, rule fixed before running: **D3ALLE12** (D3ALL with 12 epochs, control D3ALL) replaces D3ALL as the candidate if its gold delta vs D3ALL is > 0 with ci95 lower bound > −0.01.

**D3ALL submission (2026-10-13): Kaggle 57133827, public 0.845** (D3 57114448: 0.835). The user pushed kernel `rsna-d3all-submit-20261013` v1 and submitted from the Kaggle UI. Validation run: 58-gold parity max |Δ| 0.0016, mean 0.00015; pixel-library versions equal training; 3/3 visible test studies decoded strictly, 4.9 s/study. The +0.010 public gain matches the +0.012 local gold gain of D3ALL over D3. Record: `artifacts/kaggle/rsna-d3all-submit-20261013/submission-record.json`.

### 2026-10-13 — Three parallel candidates on the D3ALL base; joint selection rule fixed first

To use three concurrent Colab GPUs, two arms join the already prepared D3ALLE12, each one variable away from D3ALL (all non-gold studies, gold-only evaluation): **D3ALLE16** (16 epochs; with D3ALL and D3ALLE12 this gives an 8/12/16 schedule curve) and **D3ALLLR2** (backbone learning rate 2e-5 instead of 1e-5 at 8 epochs; the cheap alternative answer to "four blocks are under-trained").

Joint selection rule, fixed before any of the three has run (written in the D3ALLE16 and D3ALLLR2 configs; D3ALLE12's own rule is the same qualification test): a candidate qualifies if its paired gold_all delta vs D3ALL is > 0 with ci95 lower bound > −0.01; the qualifying candidate with the highest gold_all point estimate becomes the submission candidate, the shorter training winning ties within 0.002; if none qualifies, D3ALL (public 0.845) stays. Picking the best of three on 58 gold studies favours a lucky arm, so the winner's delta will be reported as optimistic. Any submission still needs the user's explicit authorization.

### 2026-10-13 — D3ALLE12, D3ALLE16, D3ALLLR2: all qualify; D3ALLLR2 is the candidate by the joint rule

All three archives passed review (gold only; GPUs differed: A100 40 GB, L4, T4).

| Arm | gold all | gold A | gold B | Δ gold vs D3ALL [95% CI] | wall |
|---|---:|---:|---:|---|---:|
| D3ALL (8 ep, lr_bb 1e-5) | 0.832 | 0.863 | 0.798 | — | 80 min |
| D3ALLE12 (12 ep) | 0.847 | 0.881 | 0.813 | +0.015 [−0.000, +0.031] | 119 min |
| D3ALLE16 (16 ep) | 0.849 | 0.889 | 0.808 | +0.017 [−0.003, +0.038] | 167 min |
| D3ALLLR2 (8 ep, lr_bb 2e-5) | 0.848 | 0.875 | 0.811 | +0.016 [−0.001, +0.034] | 96 min (T4) |

Joint rule (fixed before the runs): all three qualify (Δ > 0, CI lower > −0.01). Highest gold is D3ALLE16 (0.8491); D3ALLLR2 (0.8476) is within 0.002, so the shorter training wins: **D3ALLLR2 is the submission candidate**. D3ALLE12 (0.8467) is 0.0024 behind the top. Ensemble rule: mean(E12, E16, LR2) +0.003 [−0.007, +0.012] and mean(E16, LR2) +0.003 [−0.006, +0.012] vs D3ALLLR2 — CIs include 0, so a single model. Best-of-three on 58 gold studies: the +0.016 is optimistic.

Descriptive: the 8→12→16 epoch curve flattens (+0.015, +0.017), and 16 epochs lose Lateral Meniscus (−0.077) and Effusion (−0.042) while gaining MCL, Medial Meniscus, ACL — a sign of over-fitting some findings. Doubling the backbone LR gets the same overall gain in 8 epochs without large per-class losses (largest −0.010 Fracture). Submission kernel `rsna-d3alllr2-submit-20261013` (checkpoint `9008ec19…`) built and its private dataset uploaded; the kernel push and submission are the user's.

### 2026-10-13 — Resolution/series test on the LR2 recipe prepared

Following `docs/research/PUBLIC_UPDATE_20261013.md` (our 0.71 mm/px and 3 fluid slots vs 0.36–0.49 mm/px and 5–6 series in every eligible strong recipe), three arms, notebooks at `494d130`, rehearsed `ok`:

| Arm | Cache | Fold | Control | Rule (fixed before running) |
|---|---|---|---|---|
| D3LR2 | v4 (3 fluid, 224 px) | 0 | D3 | supported if weak Δ ci95 lower > 0 and gold Δ > 0 |
| V5LR2 | v5 (5 slots, 154 mm → 256 px, ≈0.60 mm/px, model 252 px) | 0 | D3LR2 | supported if weak Δ ci95 lower > 0 and gold Δ > 0 (weak first, per thread 748011) |
| V5ALLLR2 | v5 | none | D3ALLLR2 | replaces D3ALLLR2 as submission candidate only if V5LR2 passes and gold Δ > 0 with ci95 lower > −0.01 |

All three use public-mean-v1 soft labels, four trainable blocks and lr_backbone 2e-5. The v5 cache (2 parts, ~31 GiB) has no Drive copy yet; Drive partials are now per-runtime, so parallel first downloads cannot corrupt each other.

### 2026-10-13 — Pseudo-label and 5-fold plan, fixed before any teacher runs

User-approved plan: pseudo labels (the most cited lever behind ≥ 0.95 single models, no external data) plus a 5-fold ensemble. Teacher recipe = D3LR2 (v4 cache, public-mean-v1, four trainable blocks, lr_backbone 2e-5, 8 epochs, seed 2026).

| Group | Arms | Trains on | Purpose |
|---|---|---|---|
| Outer 5-fold teachers | D3LR2 (fold 0) + LR2F1–LR2F4 | all non-gold except fold k | OOF predictions for every non-gold study → `pseudo-v1-all`; 5-model ensemble candidate |
| Inner teachers | LR2I1–LR2I4 | folds 1–4 except k (never fold 0) | fold-0-blind OOF for folds 1–4 → `pseudo-v1-nofold0` |
| Students (configs written once the tables exist) | PL0 (fold 0, control D3LR2), PLALL (all folds, control D3ALLLR2) | pseudo tables | weak-scored test; submission candidate |

Why inner teachers: an outer teacher that labels fold 1 was trained on fold-0 labels, so a fold-0 student taught by it would carry fold-0 information into its own weak score. Gold is unaffected either way (no model ever trains on it).

Fixed now (`scripts/build_pseudo_labels.py`): pseudo target = 0.5 × public-mean-v1 + 0.5 × teacher plain probability, per finding; teachers must have passed review, hold out pairwise different folds, never label their own training studies; the fold-0-blind table must come from teachers that excluded fold 0 and contains no fold-0 rows; gold rows are impossible. Dry run on D3/D3F1/D3F2: 2608 rows, 0 gold, guards reject a fold-0-seeing teacher and duplicate folds.

Rules fixed now:
- **PL0** is supported if its weak Δ vs D3LR2 has ci95 lower bound > 0 and its gold_all Δ is positive.
- **PLALL** replaces the current candidate (D3ALLLR2, or V5ALLLR2 if it passes) only if PL0 is supported and PLALL's gold_all Δ vs that candidate is > 0 with ci95 lower bound > −0.01.
- **5-fold ensemble** (mean of D3LR2 + LR2F1–4, or of the student folds if a second round happens) is submitted instead of the single candidate only if it beats that candidate on gold_all with ci95 lower bound > 0 — the ensemble rule of 2026-10-12, unchanged.

**D3ALLLR2 submission (2026-10-13): Kaggle 57143034, public 0.862** (D3ALL 57133827: 0.845). Validation run parity 0.0 on 58 gold (Colab T4 and Kaggle GPU produced identical predictions). The +0.017 public gain matches the local gold gain (+0.016) despite the best-of-three selection; local gold has tracked public in all three submissions so far (D3 +0.068 / D3ALL +0.010 / D3ALLLR2 +0.017 public vs +0.012 / +0.016 local where comparable).

### 2026-10-14 — v5 does not help on the LR2 recipe; teachers done; pseudo tables built; students prepared

All eleven archives passed review (D3LR2, V5LR2, V5ALLLR2 at `494d130`; teachers at `702261e`).

| Arm | gold all | weak | Δ gold [95% CI] | Δ weak [95% CI] | Rule |
|---|---:|---:|---|---|---|
| D3LR2 vs D3 | 0.835 | 0.839 | +0.015 [+0.001, +0.030] | +0.015 [+0.009, +0.022] | **supported** — the LR2 gain now holds on weak too |
| V5LR2 vs D3LR2 | 0.829 | 0.833 | −0.007 [−0.028, +0.014] | −0.005 [−0.011, +0.001] | not supported |
| V5ALLLR2 vs D3ALLLR2 | 0.850 | — | +0.002 [−0.018, +0.024] | — | not a candidate (V5LR2 failed) |

v5 (0.60 mm/px, 5 slots incl. structural sagittal/coronal, 252 px model) gives nothing on the current recipe, as on the old one (D2). Finer pixels at this step size, or the structural series, are not the missing piece; a 0.4 mm/px cache is not justified by this evidence.

Teachers (gold / weak on their own held-out fold): D3LR2 0.835/0.839, LR2F1 0.849/0.846, LR2F2 0.837/0.861, LR2F3 0.842/0.861, LR2F4 0.844/0.845; fold-0-blind LR2I1–4 0.823–0.830 / 0.835–0.850 (trained on ~2,500 studies).

Ensemble rule applied: mean of the 5 outer teachers on gold 0.844, Δ vs D3ALLLR2 −0.003 [−0.014, +0.008] — no ensemble submission.

Pseudo tables (`scripts/build_pseudo_labels.py`, α = 0.5): `pseudo-v1-all` 4348 rows, folds 0–4, sha `660a4b57…`; `pseudo-v1-nofold0` 3479 rows, folds 1–4 from fold-0-blind teachers, sha `1838f379…`; no gold rows, every row has a public label. Students: **PL0** (D3LR2 + pseudo-v1-nofold0, control D3LR2) and **PLALL** (D3ALLLR2 + pseudo-v1-all, control D3ALLLR2), with the rules fixed on 2026-10-13 copied into their configs.

**Ensembles of our own fold-0 models (2026-10-14, descriptive, weak-scored).** Versus D3LR2 alone: D3LR2 + D3E12 gold +0.004 [−0.002, +0.010], weak +0.0015 [−0.001, +0.004]; adding D3, D3S2 or C2 lowers both (to −0.002 weak). Our members share data, labels, backbone and cache, so their errors are correlated; the best two average for at most ~+0.002–0.004. An ensemble worth its cost needs a structurally different member (e.g. an eligible ConvNeXt reader), not more of our DINOv2 runs.

### 2026-10-14 — Eligible external reader for an ensemble; blend rule fixed before scoring it

Our members are too similar to gain from averaging (weak +0.002 at best). A structurally different member is needed. **Eligibility check of `goodpjw2008/rsna-knee-2-5d-convnext-reader`** (read-only; code read, nothing executed):
- License Apache-2.0. Three fold checkpoints (`cnxt_v0_fold{0,1,2}.pt`, EMA weights), plus `train.py`, `make_labels.py`, `knee.py`, `preprocess.py`, `infer.py`.
- Labels `v0` = mean of four public CC0 report-label tables (stevenleehans v4_blend, pilkwang v2, lixin73 sol56, laymond); all four are CC0-1.0 and derived from competition reports only.
- Gold-58 rows get fold −1 and are never trained on; the saved checkpoint is the final EMA, not selected on gold.
- Backbone `timm convnext_tiny`, ImageNet pretrained. Preprocessing reads competition DICOMs only. No OAI, fastMRI, OrthoFoundation or private data.
- Verdict: **eligible under our policy** (claims are the author's; the training run itself cannot be re-verified). The author's 0.944 stack also uses OAI-trained components; only the reader is considered here. Claimed public score of the reader alone: 0.929.

**Rule, fixed before any reader prediction on gold is seen.** R = reader (mean of its 3 folds, its own `infer.py`); O = our current candidate (D3ALLLR2, or PLALL if it passes its rule); B = per-finding rank average 0.5·R + 0.5·O (one fixed weight, no tuning on gold). On gold_all: the candidate is B if B ≥ R − 0.005, else R; it replaces O only if its paired gold_all delta vs O has ci95 lower bound > 0. Gold only (the reader's predictions on our weak studies are in-sample for it). Scoring kernel: `scripts/make_reader_eval_kernel.py` (runs the reader on gold-58, records the sha256 of every reader file for pinning).

### 2026-10-14 — Pseudo labels and lr 4e-5: no supported gain

| Arm vs control | gold all | Δ gold [95% CI] | Δ weak [95% CI] | Rule |
|---|---:|---|---|---|
| PL0 vs D3LR2 (fold-0-blind pseudo labels) | 0.835 | +0.000 [−0.009, +0.009] | −0.002 [−0.005, +0.002] | not supported |
| PLALL vs D3ALLLR2 (5-fold pseudo labels) | 0.848 | +0.000 [−0.008, +0.009] | — | not a candidate (PL0 failed) |
| D3LR4 vs D3LR2 (lr_backbone 4e-5) | 0.841 | +0.006 [−0.009, +0.022] | +0.004 [−0.001, +0.009] | not supported (weak CI includes 0) |

Pseudo labels from teachers of the same recipe (α = 0.5) change nothing on gold or weak: the teachers agree with the public labels they were trained on and add little new information — the same family as the NTejas OOF self-distillation negative (2026-10-09 notes). The pseudo-label gains reported by ≥ 0.95 teams presumably come from stronger, more diverse teachers. lr 4e-5 trends up (+0.004 weak) but is not separable from noise; no all-folds LR4 arm is pre-registered. The candidate stays **D3ALLLR2** (public 0.862), which is also O in the reader blend rule.

### 2026-10-14 — Reader scored on gold: reader alone becomes the candidate by the pre-registered rule

Kernel `rsna-reader-gold-eval-20261014` ran the reader's own `infer.py` (3 fold checkpoints, mean) on the 58 gold studies from DICOM; file hashes recorded in `artifacts/kaggle/rsna-reader-gold-eval-20261014/output/reader-files.json`.

| | gold all |
|---|---:|
| R = reader | **0.911** |
| O = D3ALLLR2 (ours, public 0.862) | 0.848 |
| B = per-finding rank average 0.5·R + 0.5·O | 0.895 |

Rule (fixed before scoring): B − R = −0.016 [−0.033, +0.001] < −0.005, so the candidate is R; R − O = **+0.063 [+0.031, +0.096]**, CI lower bound > 0, so **R replaces O as the submission candidate**. The reader wins 10 of 12 findings; ours is better on Effusion (0.943 vs 0.878) and slightly on Synovitis (0.775 vs 0.757), so an equal-weight blend dilutes the reader. Any other weighting or per-finding routing would now be chosen on gold and is not pre-registered. Submission kernel `rsna-reader-submit-20261014` (reader_only, reader files pinned, our D3ALLLR2 still parity-checked as a pipeline health check) built; its private dataset uploaded; push and submission are the user's.

**Reader 0.8 + ours 0.2 (2026-10-14, LB check, not a gold decision).** On gold, adding D3ALLLR2 to the reader changes nothing at weight 0.2 (0.9100 vs 0.9105) or 0.1, and hurts from 0.3 down. To test whether our model adds anything on the larger public split (~390 studies vs 58 gold), kernel `rsna-reader80-ours20-20261014` (per-finding rank average 0.8 reader / 0.2 D3ALLLR2) is built next to the reader-only submission 57178710; the higher public score of the two is kept for the final selection. Picking between two public scores slightly overfits the public split; noted.

**Per-finding check, reader vs ours on gold (2026-10-14).** Reader significantly better on ACL (+0.12), MCL (+0.19), Medial Meniscus (+0.13) and Fracture (+0.09) — fine structural findings, exactly where our DINOv2 recipe is weak (K5 targets this). Ours is ahead only on Effusion (0.943 vs 0.878; Δ −0.065 [−0.193, +0.046], 35 positives) and our Effusion is stable across 35 reviewed runs (median 0.943, min 0.868), while the reader's is a single measurement. Rank correlation between the two models per finding 0.71–0.89. Because the Effusion pick was seen on gold, it is tested on the public LB instead of being adopted from gold: kernel `rsna-reader-effusion-ours-20261014` (route mode: Effusion from D3ALLLR2, the other 11 findings from the reader). Its gold score (≈0.916) is optimistic by construction.

**Reader-only submission (2026-10-14): Kaggle 57178710, public 0.929** — the reader's own claimed score, and +0.067 over our best model (D3ALLLR2, 0.862); the gold gap was +0.063. This is the vetted public reader's predictions, not a model we trained.

**K5ALL — KneeNet on our v5 cache, all non-gold studies (2026-10-14).** A100, bf16, 14 epochs uninterrupted, 7280 updates, 0 skipped (so the fp16-skip and resume-RNG issues found later in the notebook did not touch this run). Gold 0.892 (A 0.922, B 0.869); vs D3ALLLR2 **+0.045 [+0.014, +0.080]** — the first recipe of ours with a clear gold gain since D3. Against the reader it is still below: K5ALL − R = −0.018 [−0.035, −0.002]. The fixed 0.5/0.5 rank blend B5 = 0.907; B5 − R = −0.004 [−0.012, +0.004].

| Finding | Reader | K5ALL | D3ALLLR2 | B5 |
|---|---|---|---|---|
| ACL | 0.966 | 0.978 | 0.849 | 0.975 |
| MCL | 0.948 | 0.959 | 0.762 | 0.955 |
| Medial Meniscus | 0.959 | 0.904 | 0.831 | 0.936 |
| Lateral Meniscus | 0.891 | 0.867 | 0.810 | 0.896 |
| Medial OA | 0.966 | 0.964 | 0.943 | 0.969 |
| Lateral OA | 0.865 | 0.851 | 0.787 | 0.863 |
| PF OA | 0.844 | 0.833 | 0.798 | 0.841 |
| Effusion | 0.878 | 0.877 | 0.943 | 0.886 |
| Synovitis | 0.757 | 0.688 | 0.775 | 0.726 |
| Baker's | 0.976 | 0.976 | 0.942 | 0.980 |
| Contusion | 0.962 | 0.918 | 0.906 | 0.945 |
| Fracture | 0.914 | 0.890 | 0.826 | 0.908 |

Rule (fixed before the run): K5ALL alone needs a CI lower bound > 0 vs R — fails; B5 needs delta > 0 with lower bound > −0.01 — fails (delta negative). Both branches fail whatever K5F0 shows, so **reader-only stays the candidate**. The KneeNet recipe closes most of the gap (ACL and MCL now match or beat the reader), but with the same architecture its Effusion (0.877) is the reader's (0.878), not D3's 0.943: the Effusion edge belongs to the DINOv2 model, which supports the route kernel rather than a KneeNet route.

**KneeNet path hardened, seed ensemble prepared (2026-10-14).** Following an outside code review of a117207: (1) an fp16-overflow-skipped step no longer advances the scheduler, step count or EMA (`applied + skipped = attempted`, as in `loop.py`); (2) resume restores CPU, CUDA, NumPy and Python RNG, and DataLoader workers are no longer persistent, so they re-seed from the restored torch RNG each epoch; (3) the notebook rehearsal now stops a run after epoch 0, resumes it and requires EMA weights and predictions bit-identical to an uninterrupted run — passed for K5F0/K5ALL, and fails as it should when the RNG restore is removed. K5ALL itself (A100, bf16, uninterrupted, 0 skipped) is unaffected. Notebooks now also score gold in strict fp32 (`metrics.json` `gold_fp32`). (5) The submission kernel takes KneeNet members on the v5 shard cache and predicts them in strict fp32; `scripts/rehearse_submission_kernel.py` checks kernel predictions against the notebook's own V5Study evaluation on the kernel's test cache (K5ALL: max abs 3e-8), then parity and submission.csv. K5ALL's only parity reference is its bf16-autocast gold predictions; whether that fits the 0.02/0.002 parity limits is measured on K5ALLS2, which records both. Seed arms K5ALLS2 (2027) and K5ALLS3 (2028) with the E3 rule in their configs; notebooks built at `dff35b4`, rehearsed `ok` (resume bit-identical, fp32 gold present).

**Reader + ours on the public LB (2026-10-14).** Kaggle 57184259 = route (Effusion from D3ALLLR2, kernel `rsna-reader-effusion-ours-20261014` v358343544) public **0.927**; 57184253 = reader 0.8 / D3ALLLR2 0.2 (`rsna-reader80-ours20-20261014` v358343537) public **0.926** (mapping from the submissions API `url` field) — both below reader-only 0.929 and far from the 0.932 bar fixed before scoring. Neither replaces reader-only; D3ALLLR2's gold Effusion edge does not show on public.

**K5ALLS2 / K5ALLS3 and the three-seed rule (2026-10-14).** S2 (seed 2027, L4) gold 0.895, S3 (2028, A100) 0.898, K5ALL 0.892; all bf16, 14 epochs, 0 skipped. E3 = mean of the three: gold **0.902**, E3 − K5ALL +0.010 [+0.001, +0.018] (seeds help), but **E3 − R = −0.009 [−0.024, +0.007]** and BE (0.5 rank R + 0.5 rank E3) 0.909, **BE − R = −0.001 [−0.010, +0.007]**. Rule fixed in the configs: E3 needs a lower bound > 0 — fails; BE needs delta > 0 — fails. **Reader-only stays.** Per finding E3 beats the reader on ACL, MCL, Medial OA, PF OA and Effusion and trails on Synovitis (0.668 vs 0.757), Medial Meniscus, Contusion and Lateral OA.

Precision check from the two fp32 gold records: bf16-autocast vs strict fp32 gold probabilities differ by max 0.065 / mean 0.005 (S2) and 0.062 / 0.0047 (S3) — beyond the kernel's 0.02 / 0.002 parity limits. So a bf16 reference cannot serve as parity for the fp32 kernel; K5ALL would need an fp32 re-score before any kernel containing it. Not needed now, since no K5 candidate passed.

**Final selection decision (2026-10-14).** Model development stops here: the fixed rules left no K5 candidate, and more seeds of the same recipe are the diminishing-returns area. Final selections (made by the user in the Kaggle UI; the CLI cannot select): **1. reader-only 57178710 (public 0.929)**; **2. route 57184259 (public 0.927)** — within noise of reader-only, and the one entry that carries a model we trained, as a small private-split hedge. K5F0 is still recorded when it arrives but cannot change this.

### 2026-10-15 — K0/K1 implementation prepared at the user's request; not trained

P5 K0BASE/K0/K1 introduce a matched fp32 measurement baseline, soft-source UID coverage,
and canonical full-series sampling with KneeNet's forward resolution held at 256 px.
Fold-0 training coverage changes 3341 → 3479 (+138); the all-fold eligible soft pool is
4348, excluding every gold/group member. Full-cache accessibility has not been verified
locally. K5F0 results are absent locally, not assumed complete. New identities preserve
all historic K5 records. Fixed weak delta ≥ +0.004 and paired bootstrap lower bound > 0;
gold/class regressions are reported, and a second outer fold must confirm direction.
No new real AUC, remote job, submission, promotion or final selection change.
Implementation and handoff: [K0/K1](K0_K1_IMPLEMENTATION_20261015.md).

### 2026-10-15 — K0BASE / K0 completed: coverage expansion does not pass the pre-run gain rule

Real Colab outputs ingested and reviewed. Both runs used A100-SXM4-40GB, 14 epochs,
strict fp32 evaluation, the same pinned source/training cell, soft labels, fixed numeric
evaluation targets, folds, v5 cache and pretrained weights. Checkpoint hashes and
train UID hashes verified; gold and duplicate-group exclusions passed. K0BASE trained
3341 studies / 5838 updates; K0 trained 3479 / 6076 updates, with zero skipped updates.
The added 138 studies also increase optimizer updates at the unchanged epoch schedule;
this comparison does not isolate example quality from that exposure difference.

| Split | K0BASE | K0 | K0 − baseline, paired 95% CI |
|---|---:|---:|---|
| Weak hard labels (837 scored studies) | 0.87415 | 0.86955 | −0.00461 [−0.01338, +0.00371] |
| Gold all (58, reused development set) | 0.89344 | 0.88588 | −0.00756 [−0.02582, +0.00880] |
| Gold A (29) | 0.91599 | 0.91505 | −0.00093 [−0.01933, +0.01914] |
| Gold B (29) | 0.87016 | 0.85708 | −0.01308 [−0.05200, +0.03265] |

All 869 expected weak studies have predictions; 32 lack a local numeric label row and
are explicitly excluded from weak scoring. Bootstrap: 2000 paired study draws, seed
2026. The pre-registered weak requirement (delta ≥ +0.004 and CI lower bound > 0)
fails: **K0 not supported; no promotion**. Both macro CIs include zero, so these runs
do not establish a statistically clear overall loss either. The corrected soft-source
pool remains a data-coverage policy, not a demonstrated performance improvement.

Descriptive class regressions: weak Synovitis −0.07373 and Baker's −0.02612;
gold Fracture −0.06806, Lateral OA −0.05222, Baker's −0.02717 and Contusion −0.02699.
Gold Lateral Meniscus improves +0.08199. These are exploratory class deltas, not
classwise significance or a routing rule. Runtime was 1941.9 s vs 2037.0 s; peak
allocation about 27.1 GB in each run.

K1's representation hypothesis is still untested. Keep K0 as its fixed same-pool
control; require the real DICOM geometry/decode pilot and verified full-volume cache
before K1 training. A K1-vs-K0 pass would require second-fold confirmation and must
also be inspected against K0BASE; beating a lower control alone is not evidence of
beating the legacy baseline. Reader-only and the existing submission choices remain
unchanged. No new public score, submission or GPU job was created during review.

Evidence: `artifacts/colab/p1-K0BASE/output/review.json`,
`artifacts/colab/p1-K0/output/review.json`,
`artifacts/research/k0-run-verification.json`. Original configurations/build identities
were not edited to mark completion.

### 2026-10-15 — K1 real-DICOM pilot pushed through Kaggle CLI

At the user's request, the private CPU notebook `bulsahkecici/rsna-k1-volume-pilot-20261015`
was pushed as **version 1**, with competition data attached, Internet enabled, GPU disabled
and a 3600-second run limit. Kaggle returned a successful push receipt and status
`KernelWorkerStatus.RUNNING`. Scope: 64-study canonical-volume header/decode/QC pilot;
no model training or competition submission. The notebook exports JSON-only QC results;
MRI volumes stay remote. Success and cache suitability remain unverified until outputs
are collected. Receipt: `artifacts/kaggle/rsna-k1-volume-pilot-20261015/push-receipt-v1.json`.

### 2026-10-15 — Five K1 full-cache CPU jobs explicitly authorized and started

The user explicitly approved five private Kaggle CPU cache jobs after automatic review
required a scope-specific approval. All five kernels
`bulsahkecici/rsna-k1-vol-shard{0,1,2,3,4}-20261015` were successfully pushed as
**version 1** and independently observed as `KernelWorkerStatus.RUNNING`.
Each has an 18 GiB compressed payload guard and a 10800-second runtime limit.
The disjoint sorted-UID shard policy is index::5. The real 64-study pilot measured
981742812 payload bytes and 370.78 seconds, projecting approximately 62.96 GiB
for 4407 studies (sample projection, not a guaranteed full-build size or duration).
No MRI payload is downloaded to this Mac, no GPU training has started, and no
competition submission occurred. Next prerequisite: completed jobs with complete,
verified manifests and payloads; K1 training remains dependent on those outputs.
Job receipt/status records: `artifacts/research/k1-cache-execution-plan.json` and each
`artifacts/kaggle/rsna-k1-vol-shard*-20261015/push-receipt-v1.json`.

### 2026-10-15 — K1 full-cache audit passed; actual GPU training job pushed

All five cache kernels completed with status ok, covering 4407/4407 studies,
65.8654 GiB and 24370 usable series. One unsupported-in-plane-geometry series
was explicitly excluded; no study was quarantined. 1936 studies have alternatives.
Receipt/source/manifest hashes, disjoint part membership and complete study universe
were verified from downloaded JSON only. All MRI NPZ payloads remain on Kaggle.
K1 has 3479 training / 58 gold / 869 weak studies; the training/gold overlap is zero
and duplicate-group exclusions passed.

A Kaggle-native K1 notebook now reads the five pinned kernel outputs through a
symlink view in /tmp, avoiding a 65.9 GiB copy or re-export in the working folder.
Full payload hashes and shape/pixel QC are rechecked remotely before training.
The recipe remains 14 epochs, seed 2026, 256 px forward resolution and strict fp32
evaluation, matched to K0's targets/schedule. Actual GPU and AMP dtype will be
recorded by the runtime (K0 used A100/bf16); this is not claimed to be a pure
single-component causal ablation.

Kaggle initially rejected the 7.9 MiB notebook because source must be <1 MB.
Lossless, SHA256-verified LZMA packaging reduced API source to 885044 bytes;
the decoded payload is byte-identical to the successful CPU rehearsal. No label,
model or training-code content was altered by packing. 61 targeted tests and
Ruff passed; the mounted K1 rehearsal resumed with bit-identical EMA and predictions,
with all collected rehearsal outputs explicitly synthetic.

Private kernel `bulsahkecici/rsna-k1-kneenet-fold0-20261015` was accepted as
**version 1**, with requested accelerator `NvidiaTeslaT4` and a 43200-second runtime
limit. GPU quota check before launch: 24.79 hours remaining. Status was independently
observed as `KernelWorkerStatus.RUNNING`; GPU handshake/optimizer progress had not
yet appeared in the initial logs. **No K1 training result or new AUC yet.** No model
promotion or competition submission. Receipt:
`artifacts/kaggle/rsna-k1-kneenet-fold0-20261015/push-receipt-v1.json`;
cache audit: `artifacts/research/k1-full-cache-audit.json`.

### 2026-10-15 — K1 completed: passes the pre-registered fold-0 rule vs K0

Kaggle `rsna-k1-kneenet-fold0-20261015` v1 completed successfully. Real metrics:
14 epochs, 3479 training studies, 6076 attempted / 6073 applied updates and 3
fp16 overflow skips (scheduler/EMA skipped with the optimizer). Actual hardware:
Tesla T4, microbatch 2 with accumulation to 8; K0 was A100/bf16. Evaluation is
strict fp32 in both arms. Runtime recorded by the training/evaluation cell:
11573.9 s (~3.21 h), peak CUDA allocation 8.708 GB. Initial measured throughput
0.554 studies/s includes the early training startup and is not a steady-state ETA.

| Split | K0 | K1 | K1 − K0, paired 95% CI |
|---|---:|---:|---|
| Weak hard labels (837 scored) | 0.86955 | 0.87816 | +0.00861 [+0.00133, +0.01670] |
| Gold all (58, reused development set) | 0.88588 | 0.90311 | +0.01724 [+0.00265, +0.03383] |
| Gold A (29) | 0.91505 | 0.91723 | +0.00218 [−0.01997, +0.02422] |
| Gold B (29) | 0.85708 | 0.89119 | +0.03411 [−0.00360, +0.07602] |

The fixed weak requirement delta ≥ +0.004 and CI lower bound > 0 passes.
**Decision: passes_this_fold_only; no automatic promotion.** Bootstrap uses 2000
paired study draws, seed 2026. All 58 gold and 869 weak predictions are present,
unique and finite; 32 weak studies lack a local numeric-label row and are
explicitly excluded from weak scoring. Gold remains repeatedly used development
data; neither gold subset is called an unseen test set.

Source/config/labels/folds/cache/training/platform hashes matched the pinned build.
Training UID hash, gold/duplicate-group exclusions, expected update count, checkpoint
SHA256 and model metadata passed. Saved weights are finite and nonempty. The
strict-fp32 gold reference equals the gold prediction file exactly (max abs delta 0).
The original results ZIP download failed with a connection reset; exact individual
metrics/prediction/provenance files and checkpoint were successfully fetched instead.
No MRI volumes were downloaded. No raw reports were read or uploaded.

Class regressions vs K0 are descriptive: gold Effusion −0.02609, Lateral Meniscus
−0.02236, Medial OA −0.01550 and Medial Meniscus −0.00841; weak MCL −0.00795,
Medial Meniscus −0.00349 and Fracture −0.00239. Gold gains include Contusion +0.05263,
Synovitis +0.05257, Fracture +0.05139 and MCL +0.04762. No class routing is selected.
This is a representation-package experiment with a runtime precision/hardware
change; it does not identify one preprocessing component as the cause.

Supplementary checks against the stronger legacy K0BASE: weak +0.00401
[−0.00473, +0.01149], gold +0.00968 [−0.00555, +0.02615]. Those CIs include zero,
so improvement over the legacy baseline is not established. Reader gold remains
0.91053; K1 − reader = −0.00742 [−0.02223, +0.00627]. Reader weak predictions are
not an independent holdout and were not scored. These checks do not revise the
pre-registered K0 comparison or justify replacing reader-only.

Next required confirmation is the same-direction K1F1-vs-K0F1 comparison on a
second outer fold. Neither second-fold job, K2 resolution experiment, submission,
blend/routing search nor model promotion was started during this review. The
reader-only candidate and existing final submission choices remain unchanged.

Evidence: `artifacts/colab/p1-K1/output/review.json`,
`artifacts/research/k1-run-verification.json`,
`artifacts/research/k1-supplementary-review.json`, and the actual Kaggle v1 receipt.
Historic YAML/build identities and model weights were not edited by the review.


## 2026-10-15 — Explicit K1 exploratory submission authorization

After the review above, the user explicitly requested doing the necessary work
and submitting K1 (then requested continuation). This authorizes an exploratory
code-competition submission after real offline integrity/parity checks; it does
not supply the missing second-fold confirmation or promote K1 over reader-only.
The dated one-time authorization in state/submission-policy.json was renewed for
this K1 checkpoint only. Final selection settings are outside this action.

The offline generator now supports the exact frozen canonical-volume preprocessing
source and VolumeStudy evaluation (12 windows, deterministic first candidate,
actual slice positions, strict fp32). It verifies packed-notebook hashes and all
five actual cache manifests. It bundles Linux codecs and pinned OpenCV wheels,
disables Internet, and processes hidden test studies in batches of 32 while deleting
completed NPZ caches. A 65-study unit check verifies chunk boundaries, prediction
aggregation and cache release. 52 targeted tests passed.

The local synthetic engineering rehearsal passed both competition-rerun and
interactive paths, with all 15 studies predicted and zero fallback. Notebook vs
kernel maximum probability difference was 2.9802322387695312e-08; the three-study
synthetic parity check also passed. These checks are not real-data validity
evidence; 58-study real gold parity on Kaggle is still required before submission.
Evidence: artifacts/research/k1-offline-rehearsal.json.


### K1 real offline PASS and competition submission — 2026-10-15

Private inference kernel `bulsahkecici/rsna-k1-submit-20261015`, version 1,
completed with Internet disabled. Re-decoded all 58 real gold studies using
the frozen cache source: no quarantine; preprocessing hash and numpy/pydicom/
OpenCV versions matched training. Strict-fp32 CUDA predictions matched the
reviewed training reference exactly (maximum and mean absolute difference 0).
Gold preprocessing plus inference took 307.3 seconds. All three live preview
test studies were predicted with zero fallback; the CSV contract passed.
Total preflight wall time was 710.7 seconds, including offline dependency setup.
The conservative projection for approximately 1300 hidden studies was 25,370.2
seconds (2x the larger measured per-study time plus 600 seconds), below nine
hours; this is an estimate, not a measured hidden-test runtime.

The actual checkpoint passed strict hash-pinned KneeNet architecture/finite-
weight validation. The shared submit checkpoint validator now supports the
vetted KneeNet source alongside the existing DINO implementation, requiring
positive optimizer steps, the exact adjacent reader code hash, supported model
config, target order, finite tensors and strict state loading. Submission/contract/
status tests passed, including rejection of an untrained/unvetted KneeNet.

`audit_k1_submission.py --register` produced a real operational PASS bound to
the completed kernel v1, checkpoint/package SHA256, config/folds, gold-group
exclusion, exact parity, valid CSV, runtime estimate and explicit user authorization.
It registered `K1-submit-20261015` as READY_TO_SUBMIT. The guarded `rsna submit`
then submitted kernel v1 and reconciled the numeric Kaggle server receipt:
**57232441** (local attempt `5bc4deba-b50a-4ee7-b513-41fa69e6761d`).
Kaggle accepted it; four daily submissions remained immediately after this request.
At 2026-10-15 13:41:57 UTC, the actual competition status was
**PENDING**, public/private scores blank, no reported error. Local stage is
SUBMITTED, not SCORED. The hidden-test rerun is pending.

This remains a user-authorized exploratory first-fold K1 submission. It does
not supply second-fold confirmation, make reused gold independent, promote K1
over reader-only, or change final submission selection.

Evidence: `artifacts/kaggle/rsna-k1-submit-20261015/{submission-audit.json,
submission-receipt.json,competition-status.json,push-receipt-v1.json,
output/offline-result.json,output/submission.csv}`; private asset upload receipt
and SHA256-bound MANIFEST; local registry records the actual server receipt.


### K1 scored — 2026-10-15

Kaggle receipt **57232441**, kernel v1: official status COMPLETE, public score
**0.928**, confirmed by API at 2026-10-15T17:57:12.778316+00:00.
Local stage is SCORED. Reader-only public best remains 0.929; its candidate/final
selection was not changed. The user authorized proceeding with the paired
K0F1/K1F1 second outer fold. This public score does not supply that confirmation.


## 2026-10-15 — K0F1/K1F1 second outer fold launched

Following explicit user authorization, both private Kaggle T4 training notebooks
were accepted as version 1 at 18:07 UTC (21:07 Istanbul):
`bulsahkecici/rsna-k0f1-kneenet-fold1-20261015` and
`bulsahkecici/rsna-k1f1-kneenet-fold1-20261015`. Each has a 21,600-second
(6-hour) upper runtime limit. GPU quota checked immediately before launch:
19.99 hours remaining. Expected total use approximately 6–8 hours, based on
previous training; queue/preprocessing time and actual duration can differ.

Both use 3478 identical train UIDs, 871 identical weak holdout UIDs and the same
58 excluded gold studies. Outer fold=1 and duplicate-group/gold exclusions passed.
Weak fold1 UIDs are disjoint from fold0. Source archive, training cell, platform
recipe, labels, seed, optimizer, 14-epoch schedule, initial weights, evaluation
precision and 256px model forward resolution match across the pair. The shared
source SHA256 is 5717075b5862ccc760ed94bd7f1557dd96b8dc741e50389c5a891c3450a62d66.
Representations retain their original difference: K0 fixed v5 windows; K1 canonical
full-series volumes and fresh sampling. No fold0 build/config/result was overwritten.

Both caches cover all 4407 studies, from exact real receipts/pinned manifests.
For v5, the campaign receipts' complete shard UID hashes were checked against
the frozen folds. Attempts to fetch old v5 manifests hit Kaggle output API 429;
they were not treated as successful downloads. Runtime pins the actual manifests
and performs full payload SHA256/shape/mask QC before any optimizer updates.
The new v5 symlink view preserves the original merge metadata and sidecars while
keeping all MRI payloads on Kaggle read-only inputs. Canonical preprocessing code
was not modified. 41 targeted cache/leakage tests passed. Both generated notebooks
passed a real-model synthetic CPU training/interruption/resume rehearsal; resumed
EMA weights and predictions were bit-identical to uninterrupted runs. Synthetic
metrics remain explicitly marked synthetic and cannot authorize production results.

`monitor_second_fold.py` is prepared to poll only these two accepted jobs, fetch
numeric provenance/predictions/checkpoints after completion, verify checkpoint
identity/finite weights/full update counts, review K0F1 then K1F1, and write the
paired bootstrap/class review plus the cross-fold direction check. It never starts
additional training, submits, or promotes a model. Reused gold is not pooled as
116 independent samples. Reader-only candidate/final selections remain unchanged.

Evidence: artifacts/research/second-fold-campaign.json, second-fold-rehearsal.json,
and each private kernel folder's push-receipt-v1.json. No training AUC is claimed
for either F1 run before actual output verification.


### Second-fold remote startup verified

Both jobs reported RUNNING and actual Tesla T4 CUDA handshakes: 15,636,037,632
bytes VRAM, torch 2.11.0+cu128, numpy 2.1.3, timm 1.0.29, safetensors 0.8.0,
pydicom 3.0.1. Runtime package versions match across the pair. All pinned cache
mounts were located, and embedded source/model-code/numeric-label verification
passed. Full cache QC precedes optimizer updates; no F1 AUC is claimed yet.
The detached local monitor was started and observed to produce a second
heartbeat after 60 seconds; PID/log details are in second-fold-monitor-process.json.
It can collect/review while this Mac remains available. No raw MRI is fetched.
Startup evidence: artifacts/research/second-fold-startup-verification.json.


## K0F1/K1F1 completed real comparison — 2026-10-15T22:27:27.863752+00:00

Both real 14-epoch trainings and checkpoint/prediction identity checks passed. Matched runtime: {'gpu': 'Tesla T4', 'torch': '2.11.0+cu128', 'packages': {'numpy': '2.1.3', 'timm': '1.0.29', 'safetensors': '0.8.0', 'pydicom': '3.0.1'}, 'amp_dtype': 'torch.float16', 'micro_batch': 2, 'eval_precision': 'fp32'}. Fold1 weak K1F1−K0F1 delta +0.005438, paired bootstrap 95% CI [-0.0013074341110825478, 0.011849874605919174]; 847 scored pairs. Same positive direction across folds: True. Fold1 strict threshold pass: False. Gold remains reused development data and was not pooled. Class regressions and per-class AUCs are in the individual reviews. Reader-only/final selections unchanged; no automatic promotion/submission. Evidence: artifacts/research/second-fold-review.json and each p1-*/output/review.json.


## 2026-10-16 — Post-confirmation path: fixed ensemble screen, bounded K2R320

The second-fold campaign is COMPLETE_REVIEWED, both checkpoints/hash/finite/update
checks passed. Fold0 K1−K0 weak delta +0.00861166, CI [0.00133200, 0.01670475]
(837 scored pairs); fold1 K1F1−K0F1 +0.00543772, CI [-0.00130743, 0.01184987]
(847 pairs). Fold1 K0F1=0.89295, K1F1=0.89839. The fixed same-direction
confirmation criterion passes; the stricter per-fold CI criterion does not pass
on fold1. Descriptive pooled two-fold weak delta +0.00613937, CI
[0.00092433, 0.01115831], 1684 unique scored studies. Gold was not duplicated
or pooled; it remains repeatedly used development data, not an independent test.

Before computing new blend scores, fixed the primary development screen to
rank50(reader, mean probabilities(K1,K1F1)). No class routing/weight search.
On the same 58 reused gold studies: reader 0.91053309; K1 0.90311329; K1F1
0.88463467; two-fold K1 probability mean 0.89814861; primary rank50 0.90758218
(delta reader -0.00295092). The descriptive single-K1 rank50 was 0.91048789
(delta -0.00004521), effectively unchanged at this precision/sample size. The
primary fails the positive-point-delta gate regardless of its CI, so no bootstrap
is needed to reject promotion by that gate. This does not prove every ensemble
is inferior. Reader overlap is unknown; this is not independent reader OOF.
No blend submission or further weight tuning. Reader-only public 0.929 remains
the candidate; actual K1 public score is 0.928. Evidence: k1-next-step-screen*.json.

K1's same-direction confirmation permits the next planned resolution hypothesis.
K2R320 changes only model forward resolution 256→320 at fixed K1 fold0 train/
weak/gold UIDs, labels, masks, cache, sampling, backbone/initial weights, seed,
14-epoch schedule, effective batch8 and fp32 evaluation. Exact pixel, volume
dataset, soft-pool and mount source bytes were checked against K1's embedded
archive and match. The 384px canonical cache is reused read-only; no rebuild
and no MRI download to Mac. K1 checkpoints/configs/builds remain immutable.

Before results: require weak delta vs K1 >= +0.004 and 2000-draw paired
bootstrap lower bound >0; report every gold/class regression; only then consider
a fold1 confirmation. Cap this hypothesis at one full fold0 run. No automatic
promotion or competition submission.

Measured resource stop: discard the initial 16 cold micro-batches, measure the
next64, stop if projected total training exceeds 21600 seconds or peak allocated
VRAM exceeds90% of device memory. Kernel hard limit25200 seconds(7h). The
initial16-batch throughput was unsuitable for budgeting (K1 cold .55 studies/s
versus much faster whole-run throughput), so it is not used as steady-state
estimate. Resource-gate stops emit no completed-training AUC.

46 targeted tests passed. A synthetic real-model CPU rehearsal actually exercised
the guard with shortened warmup, interrupted/resumed the new training cell, and
required EMA weights and predictions bit-identical to uninterrupted execution.
Synthetic results remain explicitly labeled and are not production evidence.

Private T4 job `bulsahkecici/rsna-k2r320-kneenet-fold0-20261016` v1 accepted
at 2026-10-16 07:16:05 UTC (10:16 Istanbul); observed RUNNING. GPU quota
checked before launch:12.77h. Only this one job was started. A separate detached
local monitor collects resource gate/provenance/checkpoint/predictions and reviews
K2 against K1; no restarts or submissions. Its host must remain available.
Evidence: artifacts/research/k2-campaign.json, k2-rehearsal.json, k2-monitor.log,
and the kernel folder's push-receipt-v1.json. K2 AUC/public score is pending.


## K2R320 completed — 2026-10-16T14:33:05.123101+00:00

Real K2R320 versus K1: weak delta -0.000199, 95% paired bootstrap CI [-0.006139648637193846, 0.005595426230275528]. Decision: not_supported. Checkpoint/coverage/finite-output checks passed. Reused gold/class regressions in review.json. No automatic submission/promotion; confirm fold1 only if fold0 passes.


### K2R320 resolution branch closed — 2026-10-16T16:40:14.657589+00:00

Weak delta -0.00019905, CI [-0.00613965, 0.00559543], fails the pre-set gain rule. Gold development delta -0.00392689; reused gold is not independent. Run completed and integrity checks passed, but increased resolution is not supported. No K2 fold1, resolution scan or submission. Preserve K1/256 as own-model baseline and reader-only as candidate. No new GPU job launched.

## K3EV2 — one complementary encoder package, 2026-10-16

K2R320 is closed: weak delta -0.000199 versus K1, extra runtime and memory. K1 submission receipt 57232441 remains public 0.928; the external reader reference receipt 57178710 is 0.929. No new competition submission is claimed.

K3EV2 pre-registers one EfficientNetV2-S (`tf_efficientnetv2_s.in21k_ft_in1k`) encoder package against K1 on fold 0: identical canonical volumes, numeric soft labels, gold/group exclusions, 256 px forward, k=8, 14 epochs, seed 2026, effective batch 8. The architecture, ImageNet21k→1k pretraining and required mean/std 0.5 normalization change together; this is not a pure architecture causal ablation. Micro batch 1 / accumulation 8 is fixed for T4 memory. Head remains the pinned reader's 384-dimensional two-layer study model. The vetted reader source bytes are unchanged.

Official pretrained source: https://huggingface.co/timm/tf_efficientnetv2_s.in21k_ft_in1k (Apache-2.0), revision ea9abc143ea2b9d8e1ec1de277bce02149b9cf0e, safetensors SHA-256 6f1933fb6c0d760eae03863ad0110570393e16c0610f8fe94dc9f978e52ec59c. Model download hash verified locally; MRI stays on Kaggle.

A separate K3EV2PILOT uses 192 train-only studies, most present slots first with deterministic UID-hash ties, without looking at target values. Its derived cache retains the pinned original manifests, full payload hashes and pixel QC. It runs 16 warmup + 64 measured micro batches with real masked loss, AdamW, scaler and EMA, then checks strict-fp32 12-window evaluation memory on training images only and stops. It produces no AUC, validation predictions or completed-model checkpoint. Its full-training projection uses all 3479 studies ×14 epochs, not the smaller pilot cohort. Pilot hard limit 1 hour, projected full training <=8 hours, peak train/eval VRAM <=90%.

The controller `scripts/monitor_k3.py` uploads the pilot once after engineering checks. A passed pilot and API-observed GPU quota >=10.5h are both required for one full K3EV2 upload (10h hard kernel limit); quota refresh timestamp has no assumed timezone. A push attempt is persisted before network mutation; uncertain uploads cannot be retried automatically. Full results require exact checkpoint/UID/hash verification and paired review against K1. Fixed fold-0 requirement: weak delta >=0.004, paired CI lower bound >0; report all gold/class regressions. No automatic competition submission, second fold or additional model family. Controller state/receipts: artifacts/research/k3-campaign.json. Local controller requires this Mac to remain awake; an already launched Kaggle job continues remotely.

K3 local checks completed: 55 tests passed; real pretrained encoder loaded in the synthetic CPU rehearsal; interrupted/resumed EMA weights and predictions were bit-identical to an uninterrupted run. A separate synthetic rehearsal passed the pilot-only stop path without validation/model artifacts. These are engineering checks, not medical validation or GPU speed results. Evidence: artifacts/research/k3-local-checks.json.

K3EV2PILOT v1 accepted by Kaggle at 2026-10-16T16:58:00Z, timeout 3600 seconds, NvidiaTeslaT4; server status RUNNING observed at 16:58:22Z. This receipt is a notebook execution, not a competition submission. The bounded local controller was started; process/log evidence in artifacts/research/k3-monitor-process.json and k3-monitor.log. Full K3EV2 has not been uploaded yet; it remains gated on successful real calibration and sufficient observed quota.

## K3 Colab handoff — 2026-10-16

User requested Colab execution. The verified pilot passed at 4.1101 studies/s, 11850.3 seconds projected full training (3.29h, excludes transfer/QC/evaluation), peak train/eval 5,461,640,192 bytes. GPU quota remained 2.50h. The waiting local Kaggle launcher was terminated after verifying its PID command and confirming that no full-job push attempt existed. Campaign state is COLAB_HANDOFF; Kaggle auto-start is held to prevent two full runs.

Prepared `notebooks/generated/colab/K3EV2.ipynb` and a separate `K3EV2.build.json`; the receipt-bound Kaggle notebook/build were preserved. Colab's decoded training cell is byte-identical to the verified K3 cell. All source/config/weights/UID pins remain equal. Setup, download and collection are separately pinned for Colab, and GPU/AMP differences are logged. Numeric review must use the Colab-specific build rather than the Kaggle platform record.

Colab downloads only pinned `cache/` outputs directly to its ephemeral disk (65.865 GiB); no MRI is copied to Mac or Drive. Setup requires >=75 GiB free ephemeral disk. Checkpoints/optimizer resume state and numeric results go to `/content/drive/MyDrive/RSNA/colab/K3EV2-colab`. Kaggle credentials are requested with getpass and are not embedded. The notebook verifies GPU, source, labels, manifests and every pixel payload before real training; final archive includes the resource gate and handshake.

`rsna runtime capabilities` confirmed automatic_acquisition=false. `rsna runtime acquire --gpu-order A100,L4,T4 --wait-seconds 300` produced a manual handoff; no GPU lease or training is claimed. User must open a Colab GPU runtime, upload this notebook and Run all, completing Google Drive OAuth and hidden Kaggle authentication. Colab availability and storage are runtime-dependent. Evidence: artifacts/research/k3-colab-handoff.json. The four controller tests, including the terminal Colab hold, pass.

K3 Colab cache correction — 2026-10-17: User confirmed previous notebook downloads had been retained on Drive. The handoff now checks `MyDrive/RSNA/cache` first, including the established `<kernel-slug>-<manifest-sha-prefix>.tar` format and extracted cache directories. It reuses only exact pinned K1 canonical-volume manifests with complete payload sizes; the unchanged training cell still performs full SHA/pixel QC before training. Runtime caches are reused first, then matching Drive data is copied/extracted to ephemeral Colab disk. Kaggle authentication/download is deferred until a missing or incompatible part is found. Existing Drive files are never modified or deleted; incompatible K0/v5 archives are preserved. Safe archive extraction, exact version binding, incomplete-payload recovery and archive preservation passed three tests. Notebook/build regenerated only in the separate Colab directory; Kaggle identities and training bytes remain unchanged. No Colab execution claimed.

## K3EV2 completed and rejected — 2026-10-17

Imported the user's `artifacts/colab/K3EV2-colab/colab-results.zip` (SHA-256 e7464331614d98b6d368c660b4a25ef153d518376f0fa254198e0d57ca6c2087). Verified against the separate, exactly matching Colab build, preserving the receipt-bound Kaggle build. Strict checkpoint load, approved reader source, required EfficientNet normalization, finite weights, source/config/labels/folds hashes, train/gold/group exclusion and exact fp32 prediction coverage passed. Real training: 3479 studies, 14 epochs, 6076 attempted/applied updates, zero skipped; 58 gold +869 weak predictions, 837 weak studies scoreable (32 without numeric labels excluded). Checkpoint SHA-256 61680fa0e3cab067ddfc3924068233e5052f45887f9946af2d18f746771c9002.

K3EV2 weak AUC 0.84775642 versus K1 0.87815868, paired delta -0.03040226, bootstrap 95% CI [-0.04443553, -0.01764142], 2000 draws, seed 2026, 837 pairs. All twelve weak per-class deltas are negative. Reused-gold AUC 0.87471739 versus K1 0.90311329, delta -0.02839589, 95% CI [-0.05544599, -0.00065092]. All gold subgroup/class outcomes are retained in review.json. This does not pass the fixed fold-0 gate.

One fixed complementarity screen (no weight/class search) on the same reused 58 gold: reader 0.91053309, rank50(reader,K3EV2) 0.90593633, delta -0.00459676. Negative point estimate rejects this blend; reader training overlap is unknown and these reused gold results are not independent OOF or leaderboard evidence.

Runtime: NVIDIA A100-SXM4-40GB, torch2.11.0+cu130, timm1.0.30, bf16 training, micro1/accum8, fp32 evaluation. Reported training+evaluation 5909.4s (1.64h), peak 5.50GB; cache download/QC are excluded. All five cache-sources entries report Kaggle downloads rather than Drive restoration. The previous K1 used T4/fp16/timm1.0.29/micro2; changes in hardware/AMP and encoder/pretraining/normalization prevent isolating an architecture-only causal effect.

Decision: close the tested K3 recipe; no second fold, retraining, promotion or submission. Existing K1 public receipt remains 0.928; external reader reference receipt remains 0.929. Kaggle auto-launch stays disabled and the campaign is COMPLETE_REVIEWED. Evidence: artifacts/research/k3-decision.json, artifacts/research/k3-reader-screen.json, artifacts/colab/p1-K3EV2/output/{verification.json,review.json}. Nine relevant platform-build/controller/cache/leakage tests passed. Review CLI now accepts an explicit --build-dir to validate platform identity without overwriting older builds or waiving any hashes.


## 2026-10-18 — Public CoAtNet + R6 reference reproduction

User authorized execution and submission after the open-source review. Private offline kernel: `bulsahkecici/rsna-r6-coatnet-public-reference-20261018`, version 1. Source: goodpjw2008 v27 recipe, source notebook SHA256 `a906df9328b6a53d7e0fe4e18962c655867c8617c54e6745a8af647033b2399c`. All original cells retained unchanged; only attribution, timing and final CSV validation added. R6 manifest verified locally against `6508ce6921eda986c885b2226fa55dd4d2aa8e5fc570b4c007c9a60c05bfc648`; remote original notebook verifies weights.

Prepared by `scripts/prepare_public_r6.py`; remote audit by `scripts/check_public_r6.py`. Evidence: `artifacts/kaggle/rsna-r6-coatnet-reference-20261018/`. Initial state RUNNING, no submission receipt or measured score yet. Upstream 0.958 is not our score. This is a public-reference inference reproduction, no new training or independent-holdout claim. R6 gold exclusion is reported by the hash-pinned manifest and matching trainer code; complete training UID lists were not independently reconstructed. Kaggle changed the requested id to the title-derived canonical slug above; do not push a duplicate under the initial id.

The authorized watcher `scripts/run_public_r6_submission.py --submit` is active (exec session 37037). It waits for v1 COMPLETE, validates actual output hashes/CSV and zero R6/CoAtNet fallback, checks reader completion in execution logs, checks daily capacity, then submits that kernel version once. Exclusive locking and durable pre-request attempt records prevent blind duplicate submissions. A Kaggle receipt is required before a submitted state is recorded; scores are read from submission history. A failed check stops the controller. Current status remains RUNNING; no receipt or score claimed.


### 2026-10-19 — R6/CoAtNet reference scored

Kaggle submission history independently refreshed: receipt **57367731**, `SubmissionStatus.COMPLETE`, public **0.958**, private score unavailable. Kernel `bulsahkecici/rsna-r6-coatnet-public-reference-20261018` v1. Automatic controller completed the output audit and submitted successfully. Public improvement: +0.029 over previous reader 0.929 and +0.030 over K1 0.928. This reproduces the public reference score; it is not a new independently trained model or evidence of private leaderboard rank. Final-selection UI was not inspected or changed. Evidence: `artifacts/kaggle/rsna-r6-coatnet-reference-20261018/submission-record.json`, `operational-audit.json`, and `submit-response.json`.


## 2026-10-19 — Fixed V2 blend weights candidate launched

User authorized running the prepared candidate. Private offline GPU kernel `bulsahkecici/rsna-r6-v2weights-20261019` v1 successfully pushed; observed QUEUED. Baseline receipt 57367731, public 0.958. Only one recipe cell changed: Raptor share lateral OA .20→.15, PF OA .60→.65, synovitis .80→.85, Baker .35→.30. Attribution updated; all other source cells unchanged. Exact diff and syntax checks passed. No local gold fitting or new training.

Watcher session 30672: `scripts/run_public_r6_submission.py --run-folder artifacts/kaggle/rsna-r6-v2weights-20261019 --submit`. Requires completed kernel, real CSV/UID/hash checks, zero model fallbacks, reader completion, exact expected blend weights and daily capacity before a single submission. Durable attempt guard prevents blind retries. Evidence folder `artifacts/kaggle/rsna-r6-v2weights-20261019/`. No score or submission receipt yet; baseline/final selections unchanged.


### 2026-10-19 — V2 weights submitted after watcher recovery

Notebook v1 COMPLETE. Initial watcher stopped on a failed read-only status query before any submission attempt. Added bounded retries for read-only Kaggle operations; mocked regression check confirms recovery and no submit retry. Restarted watcher session 76863. Operational audit PASS: three visible studies, zero R6/CoAtNet fallbacks, exact fixed weights, 505.14 seconds. Submission receipt **57394395**, status PENDING, no public score yet. Baseline remains 57367731 / 0.958.


### 2026-10-20 — V2 weights rejected: 0.957

Live Kaggle history confirms receipt 57394395 COMPLETE, public 0.957. Baseline 57367731 remains 0.958. Reject the fixed four-class weight variant; retain the reference. Displayed difference -0.001 is not a significance estimate. The predeclared single-variant experiment is closed; no further gold/public weight sweep. Private scores unavailable. Final-selection UI has not been changed. Decision: `artifacts/kaggle/rsna-r6-v2weights-20261019/decision.json`. Next useful investigation is diagnostic component errors and complementary anatomical evidence, not another blind weight change.


## 2026-10-20 — Meniscus sequence-head pilot launched

Private GPU kernel `bulsahkecici/rsna-meniscus-sequence-pilot-20261020` v1 accepted, 3600-second run limit. 96 train-only studies selected by stable UID hash after numeric target confidence and verified sagittal/coronal geometry availability filters. Fold0 and gold/duplicate groups excluded using existing folds. Existing K1 canonical 384px cache reused remotely; selected payload SHA256 checked remotely.

Frozen ImageNet ConvNeXt-tiny, native central 320px view, 16 ordered adjacent triplets in each available sagittal/coronal FS/non-FS slot, trainable temporal convolution plus finding-specific pooling; three head epochs. This is NOT a localized meniscus ROI model and does NOT use exact per-slice mm coordinates (cache retains order/median spacing). Numeric target confidence <=.1 or >=.9 is a proxy, not new report adjudication. No validation AUC, promotion, second fold, or submission claimed. Local head checks passed for masking absent series, order sensitivity, gradient flow and two-target output. Artifacts: `artifacts/kaggle/rsna-meniscus-sequence-pilot-20261020/`. Entry code: `scripts/meniscus_sequence_pilot.py`; builder: `scripts/prepare_meniscus_pilot.py`.


### Meniscus sequence pilot — COMPLETE verified

Kaggle v1 COMPLETE; remote experiment matches local configuration and training source hash. Log confirms 96/96 feature extractions. Real pilot PASS: 36 applied optimizer updates over 3 epochs; first/last minibatch training loss 0.68522/0.42530 (different batches, not validation improvement). Slice reversal changes logits by max 0.04469, establishing order sensitivity only. Pilot-code timer 56.50 seconds, not total notebook wall time. No AUC, ROI localization, checkpoint quality or ensemble gain established. Next: evaluate a sequence head against an order-agnostic head on identical features under two held-out folds; do not submit the pilot. Evidence: `artifacts/kaggle/rsna-meniscus-sequence-pilot-20261020/review.json`.


## 2026-10-20 — Meniscus two-fold evaluation launched

User requested real AUC and submission. Private GPU kernel `bulsahkecici/rsna-meniscus-twofold-20261020` v1 launched with 6 h upper bound. Frozen ImageNet ConvNeXt-tiny features extracted for 4113 studies, ordered and order-invariant heads trained separately for 15 fixed epochs on folds 0/1. Train counts 3095/3074, weak eval counts 869/870, gold58 excluded from every training pool and its duplicate groups. Feature extraction may include holdout/gold because encoder is fixed ImageNet, with no fit or adaptation. Gold remains reused development data. No local report text is uploaded. Four head correctness tests passed.

Predeclared review: ordered-minus-invariant two-target weak AUC positive on both folds, paired group-bootstrap mean delta CI lower >0. Gold AUC reported but not used for epoch/weight search. This alone does not prove complementarity against 0.958 R6/CoAtNet. Final submit remains authorized but requires a supported candidate plus offline checks; failed head hypothesis is not submitted. Watcher `scripts/watch_meniscus_twofold.py` (session 64361) downloads numeric metrics/predictions only and writes `review.json`. No AUC or submission receipt exists yet for this new model.


### Meniscus two-fold outcome — no supported sequence gain

Kaggle v1 COMPLETE; real metrics show 1455 applied updates per head/fold. Two-target weak AUC: fold0 invariant 0.744344 vs ordered 0.728078 (delta -0.016266); fold1 invariant 0.721401 vs ordered 0.719535 (delta -0.001866). Paired duplicate-group bootstrap mean delta 95% CI [-0.024144, 0.005934]. Gold58 two-target AUC is inconsistent: fold0 ordered 0.815942 vs control 0.773915; fold1 ordered 0.780683 vs control 0.829066. Gold was reused development data, not independent confirmation.

Decision: close this fixed-ImageNet-feature sequence-head experiment; do not submit or add it to the 0.958 reference. No supported gain on either outer fold, no claim of significant harm or of a universal failure of sequence models. This experiment did not test anatomical ROI localization, MRI-specific encoder fine-tuning, or exact-mm positions. A new GPU experiment would require a distinct hypothesis addressing those limitations rather than another head-weight sweep. Evidence: `artifacts/kaggle/rsna-meniscus-twofold-20261020/review.json`.


## MRI encoder follow-up — authorized and launched

User explicitly approved transferring the numeric medical label package to a private Kaggle notebook, GPU training, and committing/pushing code. Kernel `bulsahkecici/rsna-meniscus-mri-twofold-20261020` v1 successfully pushed; observed QUEUED. This replaces ImageNet weights with our fold-matched K1/K1F1 encoders while keeping the previous input/head/training recipe fixed. Original encoder training UIDs, checkpoint hashes, fold hashes and exclusion of validation/gold duplicate groups were checked. Both encoder exports strictly load into the inference architecture; four head tests pass.

No new weight dataset was uploaded: checkpoints are mounted read-only from existing private Kaggle training outputs and verified by SHA256. The earlier proposed dataset upload was rejected by automatic review; the subsequent notebook push was initially blocked for explicit numeric-label-transfer approval, which the user then supplied. The successful push contains code and numeric targets, no MRI or raw report text.

Watcher: `scripts/watch_meniscus_twofold.py --run-folder artifacts/kaggle/rsna-meniscus-mri-twofold-20261020`, session 34791. Remote result not yet available; no AUC or submission claimed. Evidence: `artifacts/kaggle/rsna-meniscus-mri-twofold-20261020/`.

## Lateral OA and synovitis diagnostics — completed

Numeric-only component audit on the existing 58 gold studies; no new GPU training or competition submission. Downloaded only component prediction CSVs from completed complementarity kernel v1. Reference/R6 AUC: lateral OA 0.847195/0.864603, synovitis 0.836320/0.859020. CoAtNet training overlap and reused gold prohibit independent promotion claims; no blend search. Gold lateral OA has only 11 positives. All four components share 52 lateral-OA and 53 synovitis misordered pairs. No missing anatomical planes or series failures in our cache manifests; that does not prove public decoder correctness.

Verified existing K1/K1F1 weak outer-fold UID/group exclusions and label/fold hashes. Lateral OA in medial-OA-positive subgroup: 0.756018/0.797436 vs overall 0.847277/0.882754. Synovitis has only 10/20 labeled negatives overall and 4/4 in effusion-positive strata, so subgroup causation or dependable improvement cannot be inferred. Report and two future controlled protocols: docs/experiments/oa-synovitis-20261020.md. OA protocol prepared, GPU implementation not yet built; synovitis requires a stronger independent evaluation design first. No raw reports read, label edits, or clinical error-cause claims. Three numeric diagnostic tests passed; full real diagnostic completed with exact prediction alignment. Artifacts: artifacts/research/oa-synovitis-diagnostic-20261020/.

## OA-COND-01 implemented and launched

User requested implementation. Private GPU kernel `bulsahkecici/rsna-oa-conditional-twofold-20261020` v1 accepted with 21600-second limit. Fold-matched existing K1/K1F1 encoders remain frozen; canonical read-only 384px caches produce central320 views, 16 adjacent triplets across all six slots. Only geometry-verified series with >=3 slices are eligible. No anatomical ROI or exact-mm localization claim.

Both arms use identical six-slot order-invariant two-target heads, initial weights, masked binary numeric medial/lateral OA labels, AdamW settings, 15 epochs, 32-study batches and 2301/2292 draws per epoch. Both sample with replacement; natural arm is uniform and conditional arm changes only train-only probabilities. Known joint groups [00,01,10,11] counts fold0 [895,74,226,523], fold1 [873,76,233,517]. Inverse-square-root group frequencies are normalized over known training rows and capped at3; missing companion remains weight1. Conditional maximum weights 2.6133/2.5554. No soft pseudo-label confidence selection for this experiment. Unknown/-1 targets remain masked, not negative BCE targets.

Encoder checkpoint hashes, original encoder training UID/group exclusion, new head train/gold/holdout group exclusion, cache manifests and numeric labels were verified locally. Remote binds the same hashes. Weak eval869/870; reused gold58 excluded from training. Both arms must complete1080 updates/fold with matching initial-weight hashes. Seven head/sampling tests passed. CPU synthetic-feature rehearsal of the actual training code passed paired init/equal updates/checkpoint/prediction paths; no synthetic AUC promoted or reported as quality. Packed notebook syntax and target/slot/sampling routing checks passed.

Reviewer scripts/review_oa_conditional.py downloads numeric results only. Primary gate: lateral OA delta >0 on each weak fold and paired duplicate-group bootstrap mean delta CI lower >0 (2000 draws). Secondary: lateral OA delta within medial-OA-positive subgroup >=0 on both folds. Medial OA and reused gold outcomes reported. Passing only permits separate reference complementarity evaluation, not automatic submit. Sinovitis remains evaluation-limited (4 effusion-positive negatives per fold); no unsupported second training run launched. Source: scripts/prepare_oa_conditional.py, scripts/oa_conditional_train.py, src/rsna_knee/models/conditional_sampling.py. Existing four-slot head behavior retained by default. No training AUC or new submission receipt exists yet.


## 2026-10-21 — OA-COND-01 completed: reject conditional sampling

Live Kaggle v1 COMPLETE. Downloaded numeric experiment/metrics/predictions and handshake only; local reviewer passed configuration, notebook/label/fold hashes, paired initialization, 1080 applied updates per arm/fold, finite probabilities, exact UID coverage and train/eval group exclusions. Lateral OA natural -> conditional AUC: fold0 0.82710471 -> 0.80422570 (delta -0.02287901), fold1 0.85928363 -> 0.83000731 (delta -0.02927632). Medial-OA-positive subgroup deltas -0.02985075 / -0.03012821. Paired duplicate-group bootstrap primary delta CI95 [-0.04131975, -0.00853297], 2000 draws. Medial OA deltas +0.01524709 / -0.01141043. Runtime 4012.16 seconds within training cell, not complete hidden inference runtime.

Decision REJECT_CONDITIONAL_SAMPLING. No reference blend, new training, or competition submission. Evidence: artifacts/kaggle/rsna-oa-conditional-twofold-20261020/review.json. The result rejects this frozen-feature sampling experiment, not all anatomical compartment models. Research refresh and next proposed experiment: docs/research/STRATEGY_20261021.md.
