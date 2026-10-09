# Public Kaggle Code research — 2026-10-09

Scope: public notebook source and attachment metadata, read only. No remote jobs, weights downloads, raw report inspection, or method execution. Public notebooks were treated as untrusted code and were not executed. This is a source review, not a reproduced score or provenance audit.

## What was verified

Authenticated Kaggle CLI `kernels list --competition rsna-knee-abnormality-detection --sort-by scoreDescending --page-size 40 --format json` returned public notebook references. Its response includes title, author, last run time and votes, **not numerical leaderboard scores**. Sorting alone cannot establish a particular score. Downloaded source-only notebooks and attachment metadata reside under ignored `artifacts/research/kaggle-code/`.

| Public source | Directly observed implementation | Score evidence |
| --- | --- | --- |
| [Heliosli RSNA Knee Abnormality Detection](https://www.kaggle.com/code/heliosli/rsna-knee-abnormality-detection) | Five fluid/structural series slots; 96 selected slices, 94 adjacent triplet windows; CoAtNet at 384 pixels; per-finding attention over window features; bounded encoder chunks; second anatomy-aware reader and 70/30 rank combination; anatomical mirror transform | Source contains public-score statements, including 0.914 for one arm. Those are author statements, not verified submission receipts. |
| [Roman Tamrazov DINOsaur V5](https://www.kaggle.com/code/romantamrazov/rsna-knee-dinosaur-v5) | Offline inference branches selected from attached models: ConvNeXt reader, SWA/publication assets and anatomy reader. Explicit artifact matching and approximately 8.5-hour deadline. Latest downloaded source contains historical branches and multiple score claims. | Kaggle search-index page metadata displayed **Best Score 0.943 V10**, another comments snapshot displayed **0.943 V22**. These are platform-displayed historical values from cached snapshots; they do not verify latest downloaded code or any 0.95 claim. |
| [pjmathematician D4 Lite](https://www.kaggle.com/code/pjmathematician/rsna-knee-d4-lite) | Twenty DINOv2 members with fingerprint gates; shared first-six-block computation; five-fold image-weight attachments; fluid and structural slots; label attention, depth mixing and patch/token residual heads; RadImageNet and CoAtNet arms; probability/rank blending | Numerical score not returned by CLI. Title and source claims remain unverified. |
| [goodpjw2008 ConvNeXt MIL stack](https://www.kaggle.com/code/goodpjw2008/rsna-knee-stack-2-5d-convnext-mil-lb-0-944) | Public stack plus independent ConvNeXt-Tiny 2.5D reader at 256 pixels; source describes a three-fold reader ensemble and rank blend; multiple reader families and exact attachment paths | Title/source claim **0.944**, and approximately +0.001 from the new reader, are author claims here. No live scoring receipt was available. |

Kaggle primary source search metadata: [DINOsaur page](https://www.kaggle.com/code/romantamrazov/rsna-knee-dinosaur-v5), [DINOsaur comments page](https://www.kaggle.com/code/romantamrazov/rsna-knee-dinosaur-v5/comments). Search snapshots lag live CLI notebook versions. Visible short notebook runtimes are not a full hidden-test benchmark.

## Repeated patterns and the gap to our model

The reviewed notebooks rely on competition-specific pretrained readers, not just generic pretrained encoders. Shared attachments include `mattiaangeli/knee-mri-fold-weights`, `dreaddevelopment/raptor-knee-maxspan`, `dreaddevelopment/raptor-knee-native384`, `dreaddevelopment/raptor-knee-native384dense`, and RadImageNet/CoAtNet heads. Other attachment references include `goodpjw2008/rsna-knee-2-5d-convnext-reader`, `nartaa/rsna-knee-publication-swa-weights-20261007`, and `heliosli/rsna-orthofoundation-fold0-epoch7-20261004`. Availability in a public notebook does not establish a dataset/weight license, training split independence, or permitted external-data provenance.

The main architectural differences from our own six-center model are broader sequences (structural/T1 or non-fat-suppressed fluid series alongside fluid-sensitive planes), much denser physical slice coverage, class-dependent multiple-instance aggregation, more spatial resolution, and heterogeneous trained readers. Our model currently uses three planes, six centers each, 224-pixel triplets, DINOv2 CLS features, a global average, and a linear classifier. Merely adjusting seeds cannot reproduce those capabilities.

## Priorities

1. **Public-reader reference benchmark:** audit one compact public reader's license, immutable version, hashes, training provenance, and preprocessing. Run offline on visible test and our held-out images only after the audit. If training overlap is unknown, mark it REFERENCE; apparent holdout AUC cannot qualify as independent OOF. This establishes whether published readers explain the large score gap without committing to a twenty-member stack.
2. **Own class-specific window aggregation:** initialize an attention-MIL reader to reproduce the existing mean predictions, keep six-center cache and encoder recipe fixed, then compare paired gold and weak predictions. This is the clearest reusable source pattern with little inference overhead.
3. **Own expanded sequence coverage:** version slots to add structural/non-fat-suppressed sequences; inspect coverage and geometry, build remote cache, and preserve exact inference parity. Missing sequence masks must be explicit. This changes preprocessing, so source review alone cannot authorize claiming gains.
4. **Matched independent ConvNeXt/CoAtNet reader:** train with our labels and fold-specific supervised initialization, compare its complementarity with DINOv2. This is a stronger diversity experiment than another seed of the same checkpoint.
5. **Dense windows, local crops, mirror TTA and rank blends:** adopt separately after valid baselines and resource measurements. Sagittal channel-order reversal versus axial/coronal spatial mirror requires correct canonical geometry. Fixed blend weights need validation; do not tune against repeated public LB probes or inherit source-specific gold-fitted coefficients.

## Offline nine-hour implications

All four downloaded kernel metadata files declare Internet disabled. They pin input artifacts and some explicitly stop around 8.3–8.5 hours. These guardrails do not prove hidden-test completion. Compared with our 18 windows, a 94-window reader has roughly five times as many windows; moving 224 to384 pixels also increases image/attention cost. Exact wall time depends on architecture, batching, decoding, shared encoders and GPU type, so these are workload comparisons, not runtime predictions.

Use one compact reference first. Benchmark end-to-end decode and prediction on at least30 real studies with representative slice counts and additional tail cases; budget package loading, disk, CPU preprocessing and final CSV creation. Bound queue sizes and reuse features only where backbone hashes match. Large stacks may have excellent author-reported scores and still fail our nine-hour requirement if copied without their runtime optimizations.

Do not import source branches that substitute fallback constants, search broad input directories without exact identity, or use public-LB class probing. Keep our finite-output, exact UID/schema, receipt, gold-exclusion and immutable-preprocessing gates.
