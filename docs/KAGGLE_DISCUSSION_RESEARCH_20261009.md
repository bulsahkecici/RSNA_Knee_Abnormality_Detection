# Kaggle Code and Discussion research — 2026-10-09

Read-only authenticated Kaggle CLI scan of competition topics and score-sorted public notebooks. No training, submissions or raw reports accessed. Public scores below are authors' claims unless stated otherwise; notebook list sorting does not expose a verified submission receipt or score.

## Strongest practical evidence

- [Detailed single CoAtNet recipe, discussion 746792](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/746792): author reports 0.949 public AUC. Uses 96 slices in five series slots, 94 three-slice windows, finding-specific attention pooling, 384-pixel training, native 320-pixel inference crop, and anatomical mirror averaging. Reported recipe starts at 0.939 without external OAI; external-data variants introduce confounded changes. 4,349 report-labeled competition studies; 58 expert studies excluded from training but reused for selection. 16 epochs; training 12 windows/study, evaluation 24. Not a controlled proof that CoAtNet itself causes the score.
- Same author reports reducing inference windows 94 to 24 hurt 0.940 to 0.934; generic width flip hurt 0.940 to 0.939. Anatomical mirroring reverses sagittal slice channels and width-flips coronal/axial images. Reusing gold for checkpoint selection is explicitly acknowledged to be non-independent validation.
- [Original public training source](https://www.kaggle.com/code/dreaddevelopment/knee-mri-training-the-twelve-finding-model) was pulled locally and inspected without executing it or reading outputs. Verified code has `StudyWindows`, `RaptorClassifier`, finding-specific softmax attention and `einsum("bkn,bkf->bnf", a, h)`. Supports backbone selection, `--k 12`, `--k_eval 24`, `--bb_lr 3e-5`, `--head_lr 1e-3`, folds, gradient checkpointing and ROI options. This is actionable implementation evidence, not an independently reproduced score. [Original inference notebook](https://www.kaggle.com/code/dreaddevelopment/knee-mri-twelve-findings-from-a-single-model).
- [Best single-model thread 735304](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304): recent authors report DINOv2 ViT-S 2.5D 0.951, ConvNeXt-T 2.5D 0.953, blend 0.955; they explicitly use pseudo labels. Other posts mention OOF predictions, mixing report targets with teacher predictions and larger slice bags. These are self-reports, not reproduced results. Crucially, our existing backbone class is not inherently capped at 0.767.

## Labels and validation

[Public label benchmark 747540](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/747540) reports six-source consensus gold-58 AUC 0.8887 versus steven v4 0.8927, with overlapping uncertainty for almost all findings. It warns gold-58 is small and some source rules were informed by gold. [Benchmark notebook](https://www.kaggle.com/code/karttikjangid05/rsna-knee-which-llm-labels-can-you-trust), [dataset](https://www.kaggle.com/datasets/karttikjangid05/rsna-knee-public-llm-label-sets-compared). This supports disagreement audit and OOF teacher experiments; it does not justify choosing per-target source winners on our 29 gold studies. No benchmark artifacts were downloaded or reproduced in this scan.

Preserve project masks for missing/uncertain report evidence. Image-teacher probabilities are a distinct soft supervision source, not repaired expert labels. Teachers must predict held-out groups; gold-eval images stay excluded from teacher/student training. Never validate on labels replaced with the same predictions being scored.

## External-data uncertainty

[External datasets thread 743416](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743416) and [0.949 weights discussion 747660](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/747660) show unresolved questions about OAI access and OAI-derived weights. A participant's claim that OAI is forbidden is challenged in the latter thread; neither establishes host clearance. The 0.949 recipe declares 2,399 OAI knees. Do not adopt these released weights without source/access/license review. KneeCoT prohibition is reported in the broader thread; original host ruling should be retrieved before decisions about that resource.

## Runnable next experiments, ranked

1. Native denser image bags across selected fluid-sensitive series, preserving plane and adjacency; establish a real cache QC and timing baseline.
2. Finding-specific attention MIL on the current DINOv2 features, with randomly sampled training windows and deterministic broader inference windows. Isolate aggregation from label and backbone changes.
3. Grouped OOF image teacher predictions and conservative mixing with observed report labels; explicit provenance and masks, no gold training. Avoid whole-dataset teacher predictions falsely described as OOF.
4. ConvNeXt-T or original non-OAI CoAtNet recipe as one controlled alternate backbone, then complementary ensemble if validation supports it.
5. Anatomical mirror TTA only after window-plane geometry is verified; generic plane-agnostic flips are not interchangeable.

With 19.13 GPU hours reported by the controller and final deadline October 22 23:59 UTC, prioritize information coverage and attention before many seed runs or a costly external-image integration.

## Discovery caveat

Score-sorted code listing included many forks and titles claiming 0.95+ (DINOsaur, Apex, Versia). Such titles, votes and sort position do not verify reproducibility, licensing, independent leaderboard score or scientific benefit. Trace each to its original source, exact inputs and scored version before reuse; no high-score notebook was automatically trusted or launched.
