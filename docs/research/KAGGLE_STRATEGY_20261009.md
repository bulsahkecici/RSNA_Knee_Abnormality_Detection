# Kaggle strategy refresh — 2026-10-09

## Evidence and scope

Authenticated Kaggle CLI refreshed discussion index, recent external-data questions, original host rulings and current leaderboard. Earlier same-day source inspection is recorded in `docs/KAGGLE_DISCUSSION_RESEARCH_20261009.md`. No models launched, submissions made, reports processed or labels changed. Topic pagination returned the same first 20 topics with both page and token flags; this is a targeted substantial scan, not a claim to have read every forum topic. Public posts can contain report examples; these are not reproduced or used here.

**Verified leaderboard:** [public leaderboard](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/leaderboard) currently shows CentralyDS, Sergey Bryansky and Ilya Kolb at **0.964**, with updates on October 9. Our confirmed 0.767 remains **0.197 below** that public leader; no evidence guarantees eventual first place or private-board outcome.

## Why our gap is likely large

1. **Image coverage and aggregation are limited.** Our local improvement from three to six centers is real: gold 0.7813/weak 0.7961 and public 0.767. Public recipes instead describe 80–96 slices, multiple series slots and finding-specific attention across 3-slice windows. A subtle tear outside six centers cannot be recovered by seed averaging. This is a mechanism-based hypothesis supported by our own ablation, not proof of the entire gap.
2. **Supervision is sparse and report-biased.** Current training Synovitis has 443 positive, 75 negative and 2,848 masked cells (84.6%). Other highly masked targets are Contusion 49.7%, Baker 47.0%, Lateral OA 43.1%, Fracture 42.5%. Correct report extraction cannot invent findings omitted from reports. Image-teacher training is a plausible separate source of supervision.
3. **Current encoder class is not demonstrably the bottleneck.** [Single-model thread 735304](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304) includes author claims of DINOv2 ViT-S 2.5D 0.951 and ConvNeXt-T 0.953, both using pseudo labels. These are unverified author scores, but disprove any evidence-based assertion that only a huge encoder is worth testing.
4. **Validation is a small ruler.** Gold 29 studies, with just 3 MCL positives, is repeatedly reused. Current MCL gold 0.4487 versus weak 0.7901 deserves investigation but is not enough to determine label error or select per-class blends. Weak holdout is 824 studies; labels measure report agreement rather than independent image truth. Gold image evaluation must remain isolated from every teacher/student training path.

## Public recipes: claims versus source evidence

| Resource | Evidence level | Transferable idea |
|---|---|---|
| [Detailed CoAtNet recipe 746792](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/746792) | Author reports 0.949; comparative rows change multiple factors | 96 slices, five series slots, 94 adjacent 3-slice windows, finding-specific MIL; sample 12 windows for training and 24 for evaluation. |
| [Original training notebook](https://www.kaggle.com/code/dreaddevelopment/knee-mri-training-the-twelve-finding-model) | Downloaded source inspected, not executed/reproduced | `StudyWindows`, target-specific softmax attention, weighted feature aggregation; configurable backbones/windows/folds/gradient checkpointing. |
| [Original inference notebook](https://www.kaggle.com/code/dreaddevelopment/knee-mri-twelve-findings-from-a-single-model) | Public original reference, not independently scored here | Trace original inputs and preprocessing before borrowing fork recipes. |
| [Single-model scores 735304](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304) | Author self-reports | Grouped OOF teachers, mixed report/image targets, complementary backbones; avoid equating fold ensemble with a single checkpoint. |
| [Label-set benchmark 747540](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/747540) | Author reports uncertainty analysis, not locally reproduced | Six-set consensus gold-58 0.8887, Steven v4 0.8927, broadly overlapping confidence intervals. Public source choice alone may not explain the gap. |
| [Fresh extraction benchmark 747658](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/747658) | Work in progress; comments highlight report/image disagreement | Distinguish extraction fidelity from clinical image GT agreement. |

The 0.949 recipe reports excluding gold-58 from training but reusing it for checkpoint selection, so it is not independent validation. External OAI variants change data size, updates and serving precision. Generic width-flip TTA is reported worse; anatomical mirroring reverses sagittal slice channels and flips coronal/axial image width. Mirroring requires verified plane and slice ordering. Native crop gains must be checked in our geometry rather than copied as universal pixel constants.

## Host decisions and external data

- **Direct host ruling:** [734109](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734109), August 25: KneeCoT is not allowed because its required institutional agreement creates access concerns. Do not use that dataset for training or submission assets.
- **Direct host statement:** [733965](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733965): commercially hosted LLM processing may be permitted under competition accessibility/cost conditions. Our project explicitly requires local report processing; this statement does not override that instruction.
- [747455](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/747455), freshly refreshed: OAI-trained public weights question still has no host answer in the CLI-visible thread. [737950](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737950) likewise leaves institutional-agreement generalization unresolved. Participant opinions are not permission or prohibition. Do not automatically use OAI-trained 0.949 weights.
- A public label consensus [dataset](https://www.kaggle.com/datasets/karttikjangid05/rsna-knee-public-llm-label-sets-compared) may be useful for local disagreement auditing after schema/license/provenance checks; its numeric content has not been validated here. Keep absent evidence masked.

## Concrete experiment ladder

1. **Freeze validation/provenance:** identify fixed grouped folds, encoder/input/label hashes, seed and measured time. Preserve incumbent 0.767 assets. Add paired resampling and per-target uncertainty; never classify a tiny gold change as certain progress.
2. **Dense cache experiment:** native volumes with selected fluid-sensitive series and adjacent 2.5D windows; validate geometry, coverage and masks remotely. No full archive on Mac. Sample bags during training, broader deterministic bags at inference. Compare 24 versus 48/96 slices under matched training updates first.
3. **Attention experiment:** finding-specific MIL with current DINOv2; isolate pooling change from label and backbone changes. A random- or mean-pooling control establishes whether attention helps.
4. **OOF teacher experiment:** produce held-out predictions for development groups; train a student with conservative mixing on observed labels and a separately versioned soft-supervision policy. Do not silently unmask uncertain cells under the old report policy. No gold training and no self-scoring repaired validation targets.
5. **Backbone diversity:** one ConvNeXt-T/CoAtNet pilot with identical bags and labels; compare against DINOv2. Promote complementary models based on paired validation before multiseed or fivefold expansion.
6. **Final serving:** anatomical TTA/epoch averaging only after geometry verification and runtime check; select by validation, then a limited measured leaderboard confirmation. Keep inference internet-disabled and under 9 hours.

Each rung gets one causal question, matched data/split, explicit numeric validity checks and a bounded pilot. More available models should mean a larger disciplined search, not concurrent unreviewed jobs. Review quota before launches; reserve GPU time for confirmation and offline packaging. Official final deadline is **October 22 23:59 UTC / October 23 02:59 Istanbul**; [timeline](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/timeline).
