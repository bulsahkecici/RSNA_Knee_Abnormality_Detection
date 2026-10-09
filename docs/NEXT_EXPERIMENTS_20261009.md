# Next experiment priorities — 2026-10-09

## Current plan after completed attention review

The attention job and notebook 08 recovery are COMPLETE and independently reviewed; do not re-open that solved task. Local baseline: two-block mean gold 0.79532030 / weak 0.80838210. Attention: gold 0.78327653 / weak 0.81028034; equal blend: 0.79040248 / 0.81103080. Public baseline remains 0.767, with submissions explicitly disabled.

The active research plan is [LIDERLIK_STRATEJISI_20261009.md](research/LIDERLIK_STRATEJISI_20261009.md), supported by the Kaggle/open-source source reports and [experiment queue](research/EXPERIMENT_QUEUE_20261009.csv). It prioritizes dense coverage, validation repair, independently initialized model families and conditional medical encoder probes. Approximately 18 full fits are a conditional upper scope, not queued remote jobs. Later sections below are historical planning notes, superseded where they conflict with that report.


Verified public reference: six-center submission 57003846 COMPLETE, 0.767. Previous duration8 submission 56995656: 0.748. Current GPU quota: 19.13 hours, observed via Kaggle CLI; refresh timestamp timezone unspecified.

The six-center model achieved gold macro AUC 0.78127697 and weak macro AUC 0.79613239. Relative to the three-center duration8 baseline, gold delta +0.02426167 and weak delta +0.01257769; paired gold bootstrap interval [0.00117294, 0.05138211]. Only 29 gold studies: do not use target-specific gold tuning or public score as training labels.

## Priorities

1. **Controlled model adaptation.** Compare last-block tuning with last two blocks, retaining six centers, eight epochs, seed, folds and masked labels. Track numerical skips, memory and wall time. This isolates backbone adaptation before increasing encoder size.
2. **Preserve spatial and slice information.** Current encoder returns only CLS and study model averages all valid centers. Evaluate CLS plus patch summaries, then learned masked slice aggregation as separate experiments. Version feature/checkpoint identities and prove offline parity before submission.
3. **Label quality and supervision coverage.** Reports have already been processed across the available development pool; repeat extraction alone adds no new studies. Audit class-specific numeric masks and extraction criteria locally. Synovitis weak validation has only 12 observed negatives versus 121 positives; MCL gold has three positives. Raw reports must remain local, uncertain/unmentioned labels remain masked. Any revised label set gets new hashes and unchanged heldout exclusions.
4. **Coverage beyond six centers.** Six-center gains support testing more anatomical coverage, but not an assumption of monotonic gains. Compare additional sequences or more centers with explicit slot policy and realistic memory/inference measurements. Higher resolution and bigger encoders require separate resource validation.
5. **Validation and ensembles.** Preserve the current holdout for controlled comparisons; consider additional development folds with per-fold training from official image-pretrained weights and explicit gold exclusions; the current supervised frozen warm start already saw folds 1–4 and cannot initialize honest heldout evaluation on those folds. Shared warm starts are not independent OOF. Only ensemble candidates with measured aligned holdout improvements and verified runtime.

## Resource policy

Prepare two focused training candidates first. Reserve nine GPU hours for offline validation; allocate at most eight observed GPU hours to the next training round, with the remaining margin for accounting. These are spending bounds, not training duration predictions. No new remote run is authorized by a plan file alone: the existing user authorization permits execution only after leakage, initialization, identity and quota checks are satisfied. Unknown upload/submission receipts require reconciliation before retry.

## Promotion

Keep the 0.767 artifact reproducible. Review new candidates against sixcenter, recompute masked AUCs, check split identity and paired uncertainty, then run real internet-disabled inference with the exact candidate preprocessing and feature policy. Require a Kaggle receipt for submission and a completed score before reporting a public result.


## Colab two-block handoff and public-code review (2026-10-09)

Seed 2047 replication and its fixed 50/50 ensemble failed the existing promotion threshold; retain public 0.767. `notebooks/06_colab_sixcenter_two_blocks.ipynb` is a local generated handoff: seed 2026, same frozen initialization/cache/folds/labels/8 epochs, last two blocks trained. Worker and optimizer now support an explicit block count and checkpoints persist/validate it. Two-block optimizer coverage and interrupted-vs-uninterrupted equivalence tests pass; full suite passes. No Colab runtime was started automatically and this experiment has no measured score yet.

Public discussion/code research points to denser 2.5D bags, more selected series and finding-specific attention as larger likely levers than seed averaging. Refer to `KAGGLE_DISCUSSION_RESEARCH_20261009.md` and `KAGGLE_CODE_RESEARCH_20261009.md` for primary links and provenance caveats. Prioritize controlled coverage and attention experiments after the two-block comparison; reported public scores are not reproduced proof. Preserve gold exclusion and uncertain-label masks. Any public pretrained supervised weights require overlap/provenance review before being used for validation.


## Revised objective after two-block review

User requires stronger-model preparation and explicitly declines Kaggle submission. Publisher reads `state/submission-policy.json`; `allow_submission: false` blocks publication before asset creation and again immediately before submit. Continue experiments without leaderboard-based fitting.

1. Run generated `07_colab_sixcenter_label_attention.ipynb` on Colab, using masked class-specific attention and the matched two-block training recipe. Compare independently against two-block gold 0.79532030 / weak 0.80838210; report bootstrap uncertainty, per-class changes and real GPU duration.
2. Design dense-window coverage as a new cache contract, retaining physical order, triplet adjacency, slot identities and gold exclusion. Do not just multiply uncompressed cache centers: six centers already use 11.43 GB; twelve would exceed 22 GB, and 94 windows across five slots multiplies image encoder cost. Profile compressed shards or sampled training windows plus bounded deterministic inference windows before committing to full conversion.
3. Broaden structural and fluid-sensitive series as a separate experiment, with cache QC and sequence-selection evidence. Native higher resolution follows a real throughput/memory benchmark.
4. Train a complementary ConvNeXt reader on the same valid splits; image-teacher soft targets require truly held-out training groups. Unknown public-weight overlap is REFERENCE-only and cannot validate our own candidate.
5. Resume Kaggle submission only after explicit user reauthorization and offline end-to-end runtime/integrity gates. No current local score guarantees public 0.9 or a final rank.
