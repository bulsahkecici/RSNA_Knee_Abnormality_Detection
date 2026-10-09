# Next experiment priorities — 2026-10-09

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
