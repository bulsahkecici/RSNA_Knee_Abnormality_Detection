# K0/K1 implementation — 15 October 2026

Latest result: K1 completed 14 epochs and passed the fold-0 rule vs K0 (weak +0.00861, 95% CI [+0.00133, +0.01670]). This is a first-fold pass only; second-fold confirmation remains required, and reader-only is unchanged. Full results are recorded in [EXPERIMENTS.md](EXPERIMENTS.md).

Prepared code and local verification; no real training result, GPU allocation,
upload, submission or promotion is implied. Reader-only remains the candidate.
The existing K5 configurations and result files retain their identities.

## Matched arms

| Arm | Training pool | Representation | Control |
|---|---|---|---|
| K0BASE | Legacy hard-observed UID filter | Existing v5 | Measurement baseline |
| K0 | Observed targets in pinned mean4 soft source | Same v5 | K0BASE |
| K1 | Same pool as K0 | Canonical series volumes | K0 |
| K0F1 / K1F1 | Same selection, outer fold 1 | Matched v5 / volumes | K1F1 vs K0F1 |

All use the K5F0 label file/hash, seed 2026, 14 epochs, batch 8, pretrained
weights and KneeNet head, and **256 px forward resolution**. K1 stores native
384 px volumes to preserve detail for later work; this does not enable K2.
Evaluation uses 12 windows and strict fp32 on both gold and weak studies.
The immutable numeric evaluation targets and folds are embedded and saved by
the notebook, with their hashes. No reports are embedded.

The local K5F0 result is absent. K0BASE provides a fresh fp32 baseline instead
of declaring the missing job complete. If a genuine K5F0 result arrives, collect
and verify its identity separately; do not rename it K0BASE.

Numeric pool audit: 4348 non-gold/group-safe studies have soft targets; the old
all-fold pool was 4160. On outer fold 0 the pool changes **3341 → 3479 (+138)**.
The train/gold overlap is zero. Local full v5 payloads are unavailable, so cache
coverage is **unverified**, not presumed complete. Every training and evaluation
UID must exist in the verified merged cache before the optimizer is created.

## Local preparation

```sh
.venv/bin/python scripts/prepare_k01.py --ref WORKTREE
```

This writes `notebooks/generated/K0BASE.ipynb`, `K0.ipynb` and the private numeric
audit `artifacts/research/k01-preparation.json`, using the existing K5F0 cache
pins. To verify locally downloaded cache payloads, repeat `--cache-manifest PATH`
for every part. No cache download is performed by preparation. The notebook
repeats payload validation remotely and fails on missing train/gold/weak UIDs.

`--ref WORKTREE` explicitly freezes the current source/config bytes in a
deterministic, hashed archive. `git_commit` identifies its parent commit;
`source_is_worktree=true` distinguishes it from committed source. The generator
and training cell have separate hashes; resume identity includes code, config,
labels, folds, cache parts/merged cache, weights, model code and batching.
For subsequent committed releases, use that commit as `--ref`.

## K1 real-data pilot and full cache

The notebook handoff is now available:

```sh
.venv/bin/python scripts/prepare_k1_notebooks.py --ref WORKTREE
```

Open **`notebooks/generated/K1-PILOT.ipynb`** in Kaggle, attach the competition
data, enable Internet, select CPU and Run All. Download **`K1-PILOT-results.zip`**
for review; it contains JSON QC/timing/manifest metadata, without MRI arrays.
The notebook exports an error report and fails explicitly if the data is not
mounted or the pilot fails. No GPU or training is needed for this pilot.

**`notebooks/generated/K1.ipynb`** is also supplied with the full training recipe,
explicitly marked `awaiting_real_volume_cache`. Its first setup step stops before
GPU/Drive access. After the pilot and complete cache build, regenerate it using
the actual cache kernel/manifest SHA256 pairs. No placeholder cache hashes are
used. Preparing these files does not change existing K0/K0BASE notebooks or builds.

Prepare the remote CPU pilot without pushing it:

```sh
.venv/bin/python scripts/make_kaggle_kernel.py volume_cache \
  --slug rsna-k1-volume-pilot-20261015 --ref WORKTREE
```

The private kernel mounts competition data, samples 64 studies deterministically,
and writes `cache/manifest.json`, per-study volume NPZs, build timing and QC.
Inspect each candidate's transforms, padding, retained indices, slice gaps,
series failures and explicit fallback order. Count orientation/fallback patterns
on these **real** headers before proceeding. The kernel fails if any requested
study has no usable series. The default compressed payload limit is 18 GiB.

The equivalent CLI on an existing remote worker is:

```sh
rsna cache volume-build --series-csv /mounted/train_series.csv \
  --dicom-root /mounted/train_series --dest /worker/k1-pilot --limit 64
```

K1 uses anatomical LPS transpose/flips and signed IOP×IPP ordering, fixed
153.6 mm FOV with physical padding, six plane/fat-suppression slots, every
usable candidate series, and up to 64 evenly retained slices per series.
The series window is [1, 99.5] on the resampled volume sampled every fourth
pixel, matching the audited reader. Training randomly chooses a candidate per
slot and new stratified centers; evaluation takes the largest candidate with a
stable series-ID tie break. Positions are actual rounded center/(n−1).

Differences from the permissive reader are deliberate and recorded: missing
IOP/spacing, incoherent geometry, plane mismatches, duplicate slices, constant
volumes or decode failures quarantine that series. Multi-frame input is rejected
by the shared decoder. Missing IPP may use InstanceNumber, explicitly marked
`anatomical_order_verified=false`; it is not evidence of anatomical direction.
Inversion follows rescale, including negative slopes. No zero-volume success is
allowed. Review fallback/quarantine patterns in the real pilot before a full build.

After that pilot passes, prepare bounded full-cache parts explicitly, for example:

```sh
.venv/bin/python scripts/make_kaggle_kernel.py volume_cache \
  --slug rsna-k1-volume-shard0-20261015 --ref WORKTREE \
  --define 'LIMIT=None' --define "SHARD={'index':0,'count':4}"
```

Choose the shard count from the measured pilot size/time; four is an example,
not a capacity claim. Repeat with disjoint indices. Cache policy changes require
a new destination and invalidate downstream identity. The loader rechecks each
part and the merged payload; copied merges also require disk space for both
source and destination. Do not use the 64-study pilot cache for full training.

Only after genuine full manifests exist, generate a runnable training K1 with their actual file
SHA256 values using `scripts/make_kneenet_notebook.py` and
`configs/experiments/P5-K1-kneenet-fold0.yaml`. The supplied pending-cache K1 must
be regenerated before training. A generated CPU pilot kernel is also available in
`artifacts/kaggle/rsna-k1-volume-pilot-20261015/`.

## Decision and validation

The pre-run K0/K1 rule is in the P5 YAMLs: weak hard-label macro AUC delta
**≥ +0.004**, paired study-bootstrap 95% lower bound **> 0** (2000 draws,
seed 2026), complete matched weak predictions and identical numeric target hash.
The reviewer uses unrounded deltas/CI for decisions, reports gold delta/CI and
class regressions, and never promotes automatically. A pass on fold 0 still
requires K1F1 vs K0F1 in the same direction. K0F1 is a measurement baseline, not
a comparison against another fold. Gold remains a repeatedly used development
set. At most K0BASE/K0/K1 are planned as full fold-0 pilots; close unsupported
branches before another family or higher model resolution.

The first 16 microbatches of actual training write measured GPU throughput and
peak allocation; decode throughput is recorded separately. Neither is estimated
from this Mac's CPU rehearsal.

Local tests cover leakage, missing/-1/invalid soft targets, signed anatomical
permutations, anisotropic spacing, small-FOV padding, order independent of normal
sign, intensity/rescale, capped volumes, all candidate series, explicit masks,
cache hashing/resume/byte budget/merging and fresh window sampling. The separate
reader check used its verified preprocess SHA256
`68601ad84ec3a0f030b5c2d37cc0d657cd3e12b17f7130e7385fba9648d324b6`:
24 synthetic plane/transpose/flip cases had maximum uint8 delta **0**. This is
pixel parity on those valid cases, not a real-data pilot or an AUC result.

CPU notebook rehearsal uses private scratch artifacts and synthetic inputs,
actual allowlisted ConvNeXt weights, two small epochs, an epoch-boundary
interruption, and exact EMA/prediction equality after resume. It cannot qualify
as competition evidence. Run against a generated notebook with:

```sh
.venv/bin/python scripts/rehearse_kneenet_notebook.py K0 --scratch /tmp/k01-rehearsal
```

The next required external step is the real mounted-data pilot, then complete
cache verification and remote GPU training. No raw MRI archive is downloaded
to this Mac; no raw report is sent to a cloud model.

Final local verification: **295 tests passed**, Ruff and `git diff --check` passed.
K0 and K1 CPU notebook rehearsals both resumed with bit-identical EMA weights
and predictions. Their metrics and collected ZIPs are explicitly synthetic and
cannot pass the production reviewer. The generated cache-kernel entry also
passed a 15-study synthetic DICOM smoke run. Private verification records are
under `artifacts/research/k01-local-validation.json` and
`artifacts/research/k01-reader-parity.json`; none contains a new model AUC.

## Completed full cache and Kaggle training handoff

The five explicitly authorized CPU cache jobs completed as version 1. Their
real manifests cover **4407/4407** studies with no study quarantine, **65.8654 GiB**
of compressed payload and **24370** usable series. One series was explicitly
rejected for unsupported in-plane geometry; its study still has usable series.
1936 studies have alternative series candidates. All successful series use
IOP/IPP ordering. Downloaded artifacts contain only JSON metadata and logs;
the MRI volumes remain on Kaggle.

```sh
.venv/bin/python scripts/prepare_k1_kaggle_training.py --ref WORKTREE
```

This binds the five real manifest hashes, verifies source/metadata identities
and complete disjoint study coverage, and writes a Kaggle-native **K1.ipynb**
plus `artifacts/kaggle/rsna-k1-kneenet-fold0-20261015/`. Gold/group exclusions
and the 3479/58/869 train/gold/weak counts are verified before packaging.
The private training job mounts only these five cache outputs; raw competition
reports are not mounted. A symlink view is created in `/tmp`, outside working
outputs, so the 65.9 GiB payload is never copied to a 20 GiB working directory
or re-exported as training output. Every mounted payload is hashed and decoded
by the existing QC gate before training. The source cache policy/hash is unchanged.

The Kaggle setup requires actual CUDA and records its handshake; no full CPU
fallback exists. The GPU quota check returned 24.79 hours remaining at preparation.
The recipe retains K0's labels, seed, epoch schedule and 256 px forward resolution;
the actual GPU/AMP dtype is recorded and may differ from K0's A100/bf16 runtime.
Only predictions, metrics, model checkpoint and numeric run provenance enter
`colab-results.zip`, preserving the established review interface. Platform and
training recipe hashes are checked during result review. No new model result is
claimed merely because the cache completed.


## K1 submission status — 15 October 2026

K1 fold0 training and review completed (weak macro AUC 0.87816, reused-gold
macro AUC 0.90311). After explicit user authorization, the offline K1 kernel
passed exact real-data parity on all 58 gold studies (max abs difference 0),
produced all three visible test rows with zero fallback, and was submitted via
the guarded rsna CLI. Kaggle receipt **57232441**, kernel
`bulsahkecici/rsna-k1-submit-20261015` **v1**; observed competition status
**PENDING**, score not yet available at 13:41:57 UTC. Submission evidence is
under `artifacts/kaggle/rsna-k1-submit-20261015/`. This does not replace the
missing second-fold experiment or change final selections.


## Second outer fold — launched 15 October 2026, 21:07 Istanbul

K0F1 and K1F1 private Kaggle v1 jobs are running on matched Tesla T4 environments.
Both use 3478 train/871 weak holdout/58 excluded reused-gold UIDs, the same seed,
labels, model resolution and 14-epoch schedule. All local leakage/cache tests and
training/interruption/resume rehearsals passed. Real F1 results are pending.
The local monitor collects verified checkpoints/predictions and writes
`artifacts/research/second-fold-review.json` after both jobs finish. Live state:
`artifacts/research/second-fold-campaign.json`; monitor log:
`artifacts/research/second-fold-monitor.log`. Reader-only remains the candidate.
K1 fold0 official public score is 0.928, receipt 57232441, status SCORED.
