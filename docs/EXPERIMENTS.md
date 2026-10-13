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
