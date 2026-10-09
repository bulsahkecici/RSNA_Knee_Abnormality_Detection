# Public recipes — 2026-10-09

Read-only extraction, 2026-10-09 ~19:35–20:05 UTC. Sources:
- Discussion threads, read with the Kaggle CLI `topics show` / `topic-messages`.
- The GitHub raw README and code of two repos.
- Nine public notebooks (8 inference + 1 training), pulled with `kaggle kernels pull` into a scratch directory outside the repo. Only code and markdown were read; outputs were stripped. Nothing was executed.

All scores below are **CLAIMED** by their authors unless stated otherwise. The CLI `kernels list` does not expose notebook scores, so "top by score" means the order returned by `--sort-by scoreDescending`. No report text is reproduced. Eligibility verdicts come from `P0_SOURCE_AUDIT_20261009.md`.

## 0. Key finding: the top public notebooks are not reproducible for us

The top `scoreDescending` notebooks are `heliosli/rsna-knee-abnormality-detection`, `evgendvorkin/rsna-versia-7`, `heliosli/rsna-knee-blend-gold` and `romantamrazov/rsna-knee-dinosaur-v5`. All four are **rank blends that depend on two things**:
- `nartaa/rsna-knee-publication-swa-weights-20261007`: the OAI-trained 0.949 CoAtNet checkpoint, `raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt`, sha256 `7e5315da…44ef`.
- For three of the four, `heliosli/rsna-orthofoundation-fold0-epoch7-20261004` plus `rsna-knee-ofl-ssl-weights`. This is a frozen **OrthoFoundation-L** trunk (`vit_large_patch16_dinov3_qkvb`) with trained tail blocks, and it uses **SKM-TEA**-trained knee-anchor geometry.

Both dependencies are in our UNKNOWN_NEEDS_HOST / no-OAI category, so these blends are REFERENCE_ONLY for us. The transferable parts are:
- the **Raptor geometry and MIL recipe**, which predates the OAI weights;
- the **ConvNeXt-T 2.5D reader** (goodpjw2008, trained only on competition data and public CC0 labels);
- **anatomical mirror TTA**;
- **label handling**.

## 1. Per-source recipe tables

### S1 — Discussion 746792 and `nartaa/rsna-knee-0949-anatomical-mirror` (Danial Zakaria, CoAtNet)
| Field | Value |
|---|---|
| Public LB (CLAIMED) | **0.949** with OAI. Best **non-OAI** row: **0.942** (96 slices, native 384, blend+JEV labels). 0.940 for the 224-crop model with epoch averaging. 0.939 for the 80-slice / 78-window starting point. |
| Series | 5 slots: Sag fluid, Sag non-fluid, Cor fluid, Cor other, Axial. Counts are **26/22/18/12/18 = 96 slices** (constants in the heliosli copy of the scored source). |
| Slice choice | Even spread over the geometrically ordered stack (projection on the slice normal). The Raptor lineage uses a 2–98% span. |
| Dimensionality | 2.5D: 3 adjacent slices become the RGB channels of one window. Windows may cross slot boundaries. |
| Image size | Native 384 cache from a 140 mm crop. Trained at 384. Inference uses a **native 320 crop `[32:352]`, no resize**. The efficiency model uses a 307 crop resized to 224. |
| Backbone | `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k` (timm, Apache-2.0), fully fine-tuned |
| Pooling | Finding-specific softmax attention (12 queries) over the whole window bag |
| Labels | Own report-derived soft labels blended with stevenleehans, pilkwang and riadmohamed42 (JEV) public labels. Unavailable cells are **masked** (stated for OAI rows). The pos_weight comes from the RSNA labels. |
| Loss | BCE with pos_weight (Raptor lineage); exact form not restated |
| Epochs / LR / batch | 16 epochs. Backbone LR 4.01e-5, head LR 1.95e-3, wd 0.0588. Batch 1 study with grad accumulation 16. Seed 42. |
| Train / eval windows | **12 windows** per study during training, 24 for gold eval, **94 (all positions) at inference**, in chunks of 16 |
| Augmentation | Random window sampling. Raptor lineage: ±10% intensity gain, **no flips** |
| CV | None. All-data training. Gold-58 used for selection. |
| TTA | Anatomical mirror: sagittal reverses the 3 slice channels, coronal/axial flip the width. Applied before normalisation. Probabilities averaged equally. Claimed +0.001. |
| Ensemble | 1 checkpoint, an average of epochs 14/11/12 ("SWA") |
| Runtime | 26.1 min submit-to-score (includes queueing; not an official runtime) |
| Gold-58 excluded from training | **Yes**, but reused for selection, so it is not independent |
| Negative results (CLAIMED) | 94→24 windows: −0.006. Width flip on all planes: −0.001. 320→288 crop: −0.002. Single epoch instead of averaged: −0.001. Training at 320 crop: −0.003. |

### S2 — Raptor original (`dreaddevelopment/knee-mri-twelve-findings-from-a-single-model` and `knee-mri-training-the-twelve-finding-model`)
| Field | Value |
|---|---|
| Public LB (CLAIMED) | **0.924** single model, no TTA (widedense). The `maxspan` checkpoint is 0.928 according to mathischmp. Gold-58 0.9167. |
| Series / slices | 5 slots, **18/14/12/8/12 = 64**, evenly spread over 6–94% (inference) or 2–98% (maxspan). The training corpus is 44×336×336 uint8 per study. |
| Image size | 140 mm crop → 336 (INTER_AREA) → bilinear to **384** |
| Backbone | `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`. The script default is `vit_small_patch16_224`. Has a `--freeze_blocks` option, default 0 (full fine-tune). |
| Pooling | Per-finding softmax attention over windows |
| Labels | Own LLM soft probabilities (hedged findings around 0.8). No masking: soft targets throughout. |
| Loss | `BCEWithLogitsLoss(pos_weight=pw)` on soft targets |
| Optimisation | AdamW, backbone 3e-5, head 1e-3, wd 0.02, OneCycle (pct_start 0.15). Typical run: 16 epochs at 384, batch 8, grad checkpointing. About 3 h on a 4090, 23.7 GB. |
| Windows | k=12 during training, k_eval=24, **42** at inference |
| Augmentation | Random windows plus intensity gain ±10%. **No flips.** |
| CV / selection | All 4,349 non-gold studies used for training. Gold-58 used for validation and top-3 epoch averaging. |
| Gold-58 excluded | **Yes** (selection only) |

### S3 — github.com/mathischmp/challenge_rsna (MIT; last push 2026-09-14)
| Field | Value |
|---|---|
| Public LB (CLAIMED) | **0.920** CoAtNet with window bags (gold 0.903, holdout 0.890). Previous generation: DINOv2-S, 6 views, 18 slices, **0.853**. |
| Series / slices | Raptor maxspan 5 slots, 64 slices, 140 mm crop, 336 px corpus, percentile 2/98 window per series |
| Image size / backbone | **384** / `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`, full fine-tune |
| Labels / loss | LLM soft targets with per-cell confidence weights (one run used `--no-conf-weights`). BCE with `pos_weight = clip((1−p)/p, 1, 10)`. |
| Optimisation | AdamW, backbone 3e-5, head 1e-3, wd 0.02, OneCycle, 12 epochs, grad clip 3. Batch 4 per GPU on 2×T4 with DDP, fp16, grad checkpointing. About 32 min per epoch, about 7 h per run. |
| Windows | Train k=12, eval 24, final **62** (all positions) |
| CV / selection | Derived holdout of 652 studies, split by report-text hash. Gold is **never** used for selection. Measured noise floor: 0.0015 on holdout, 0.0058 on gold. Adoption threshold: ≥0.004 on holdout. |
| TTA | **No flips.** Experimented with canonical left-knee mapping. |
| Inference | 0.95 s per study, about 21 min for 1,322 studies |
| Gold-58 excluded | **Yes**, and not used for selection |
| Negatives | DINOv2-B with registers: worse on the full run. Fewer training slices than at inference: worse. Blend with the public 0.928 model: worthless (rank correlation 0.92). Equal fusion of 4 label readers: worse. |

### S4 — github.com/NTejas-1/RSNA-Knee-Abnormality-Detection (MIT; last push 2026-09-27)
| Field | Value |
|---|---|
| Public LB (CLAIMED) | 0.941, which is the 0.940 public stack plus two arms. The arms add +0.001, judged indistinguishable from noise. |
| Series / slices | 6 slots (3 planes × fluid yes/no), deepest series per slot, 16 slices cached, **8 sampled per slot** during training. About 81% of slots filled. |
| Image size | 140 mm crop → 256 cache → **224** crop. Laterality normalised to the left knee. |
| Backbone | Arm A: RadImageNet ResNet50 (input scaled to [−1, 1]). Arm B: torchvision ImageNet ResNet50. |
| Pooling | Gated attention (Ilse) over slices, then over slots, with absent slots masked. Claimed about +0.014 over mean pooling. |
| Labels / loss | Own LLM labels via Snowflake Cortex (`claude-opus-4-5`), 3-state output plus a fractional probability. Soft BCE on that probability. Rounding to hard labels hurt. |
| Optimisation | 8 epochs, LR 3e-4, wd 1e-2, warmup 10%, batch 2 studies with accumulation 4, 5 folds |
| Gold-58 | **Not excluded.** Gold studies sit in the 5-fold training pool with LLM labels; only their OOF predictions are scored. |
| Results (OOF gold) | Arm A 0.8955, Arm B 0.8892, 50/50 blend 0.9078 |
| Negatives | OOF self-distillation of silent cells: gold 0.8955 → 0.8891 (Synovitis −0.038). CoAtNet was 8.5× slower than ResNet50 on T4 and hit OOM. |

### S5 — `goodpjw2008/rsna-knee-stack-2-5d-convnext-mil-lb-0-944` (ConvNeXt-T reader)
| Field | Value |
|---|---|
| Public LB (CLAIMED) | Reader alone, 3 folds: **0.929**. Stack 70% + reader 30%: 0.944. |
| Series / slices | Up to **6 series** (3 planes × fat-suppressed yes/no), **12 windows of 3 adjacent slices per series** |
| Geometry | Canonical volume at **0.4 mm/px, 154 mm FOV** (about 385 px native), plane orientation from IOP, slices sorted along the normal, one intensity window per series |
| Image size | **256** px training |
| Backbone | Shared **ConvNeXt-Tiny** (timm; exact pretrained tag UNKNOWN) |
| Aggregation | Each window becomes a token tagged with series type and slice position. A **2-layer transformer** runs over all study tokens, then one attention pool per finding. |
| Labels / loss | **Mean of four public CC0 label tables** (stevenleehans, pilkwang, lixin73, laymond), used as soft targets with plain BCE |
| Epochs / CV | 14 epochs, **5 folds grouped by report text** |
| Gold-58 excluded | **Yes** (check only) |
| Runtime | Reader alone under 90 min scored. The full stack takes 6.5–8 h of the 9 h limit. |

### S6 — Other top-score / top-vote notebooks (brief)
- **heliosli/rsna-knee-abnormality-detection and -blend-gold:**
  - Rank blend: Raptor-OAI (S1) weighted 0.7, plus their own `convnext_small` 4-fold model weighted 0.3.
  - Their model has an OrthoFoundation-L (DINOv3 ViT-L) frozen-prefix arm and SKM-TEA knee anchors.
  - Score: UNKNOWN via the CLI. Another notebook cites "heliosli — 0.954"; that is a third-party CLAIM.
  - **Not eligible for us** (OAI + OrthoFoundation).
- **evgendvorkin/rsna-versia-7:** same OAI and OrthoFoundation dependencies. Claims 0.950 → 0.951 from per-finding routing and weight tweaks. Not eligible.
- **romantamrazov/rsna-knee-dinosaur-v5:** Raptor-OAI plus the S5 ConvNeXt reader, with per-finding weights (Lateral OA reader weight raised 0.18 → 0.52). V14 CLAIMED 0.950. Weights tuned on LB and gold. Depends on OAI.
- **pilkwang/rsna-knee-baseline-v1** (most votes, 695):
  - DINOv2-S with the **last 6 of 12 blocks unfrozen**. Backbone LR 8e-6, head LR 1e-3. 10 epochs, batch 8 studies.
  - 6 slots, each a single 3-slice group, combined by slot attention with a presence mask.
  - 130 mm crop, 336 cache, explicit **224-vs-336 run pair**. Argument given: at 130 mm, 224 px is 0.58 mm/px, which fails Nyquist for 1 mm tears; 336 px is 0.39 mm/px.
  - Augmentation: rotation 8°, scale 0.08, shift 0.05, intensity 0.10, no flips.
  - Laterality from the image-centre x coordinate.
  - LB not stated in the cells read (UNKNOWN).

### S7 — Discussion 735304 single-model claims (all CLAIMED, participant self-reports)
- **colum2131:**
  - ConvNeXt-T 2.5D single model **0.953**, DINOv2 ViT-S 2.5D **0.951**, ensemble of the two 0.955.
  - Both use **pseudo labels** and no external data.
  - Same preprocessing, head and augmentation for both; the ViT needed its own LR and epoch count.
- **azamat1ch** (5-fold, same pipeline): ConvNeXt-T 0.959, EfficientNetV2-S 0.956, CoAtNet 0.952.
- Archit Konde: CoAtNet 2.5D at **224**, single fold, 0.950. Says labels are the bottleneck. OOF gold 0.930.
- Musa Tur Farazi: 224 px, 0.952.
- CoolinLai: 5-fold ResNet at 224, 0.954.
- Tanuki_boosting: 5-fold single model, 0.955.
- Scott Willis: 0.949 single fold, own local Gemma labels around 0.89 gold, about 5 min scoring.
- Raymond Yuen: CoAtNet at 288, 0.938 → 0.949 from label work alone (pseudo labels and blends).
- tennogh: 288 px, 0.942, or 0.943 with TTA. OOF pseudo labels correlated with LB.
- Tom Aindow: DINOv2 at 392 px, 150 mm crop (0.383 mm/px), bag of 32 slices, 0.915.
- Tucker Arrants: ResNet/EfficientNet, says 224–288 is the sweet spot, 0.943 at 224.
- NguyenThanhNhan: Qwen 3.5 at 384, single fold, 0.950.

**Pattern across these claims:** ≥0.950 single models exist at **224 px** with ConvNeXt-T, DINOv2-S, CoAtNet and ResNet. The shared factor named is **labels / pseudo labels**, not resolution or backbone size.

### S8 — Discussion 747540 (label benchmark, karttikjangid05)
- Method: 8 public label sets scored against gold-58, with bootstrap 95% CIs (2,000 resamples).
- **11 of 12 findings show no clear best set.** MCL has too few positives to rank.
- Macro AUC: consensus of 6 CC0 sets 0.8887; steven_v4 0.8927; steven_v2 0.8873.
- Lowest ceilings: Synovitis (best 0.790), Lateral OA (0.835), Contusion (0.860), Effusion (0.877).
- steven_v2's Synovitis rule was informed by gold-58.
- Per-finding blending fitted on gold overfits (credited to JEV).
- Consensus file: `karttikjangid05/rsna-knee-public-llm-label-sets-compared` (CC0).

## 2. Geometry comparison against our current setup (224 px, 12 windows)
| Recipe | Crop (mm) → px | mm/px | Slices per study | Train windows | Infer windows |
|---|---|---|---|---|---|
| S1 0.949 / 0.942 | 140 → 384 native, 320 crop | ~0.36 | 96 (5 slots) | 12 | 94 |
| S2 Raptor 0.924 | 140 → 336 → 384 | ~0.42 | 64 (5 slots) | 12 | 42 |
| S3 mathischmp 0.920 | 140 → 336 → 384 | ~0.42 | 64 | 12 | 62 |
| S5 ConvNeXt-T 0.929 | 154 mm FOV at 0.4 mm/px → 256 px (crop vs resize UNKNOWN) | 0.4–0.6 | 6 series × 12 windows (~72 windows) | UNKNOWN | ~72 |
| S1 efficiency 0.945 (OAI) / 0.940 (non-OAI) | 307 crop → 224 | ~0.49 | 96 | 12 | 94 |
| pilkwang | 130 → 336 / 224 | 0.39 / 0.58 | 6 groups | 6 | 6 |
| **Ours** | per our cache | — | 3 fluid slots (D1) | 12 per slot (D1: 36) | same |

Takeaways:
- **Train k=12 is universal.** The public lever is **inference coverage**: 42–94 windows across 64–96 slices and 5 series slots.
- Resolution claims conflict. Several 0.95 claims use 224; the CoAtNet lineage uses 384 with a native crop. Any resolution gain is not isolated anywhere.

## 3. What our controlled experiments should copy vs test
| Queue id | Copy | Test |
|---|---|---|
| **D1** dense, 3 fluid slots × 12 windows | Geometric slice order, 140 mm physical crop, per-series 2/98 percentile window, even 2–98% slice span, masked empty slots | More inference windows than training windows (S1: 94 vs 24 = +0.006 CLAIMED). Keep per-window geometry identical between train and inference; mathischmp lost when slice sampling changed. |
| **D2** 5 slots | Raptor slots Sag fluid / Sag non-fluid / Cor fluid / Cor other / Axial, 18/14/12/8/12 (64) or 26/22/18/12/18 (96). Every top CoAtNet recipe includes structural sagittal and axial; our 3-slot design omits them. | D1 views plus a separate structural cache (cheaper than one 21.5 GiB shard), as planned |
| **D3 / D4** unfreeze depth | pilkwang: last **6/12** DINOv2-S blocks at backbone LR 8e-6. Raptor/mathischmp: full fine-tune, backbone 3e-5–4e-5, head 1e-3–2e-3, OneCycle pct 0.15, top-3 epoch weight averaging | D3 at 4 blocks vs D4 full fine-tune, with matched optimizer updates |
| **D5** attention | Per-finding attention (12 queries) is in every strong recipe. S5 adds series-type/position token embeddings and a 2-layer transformer before pooling. NTejas: gated attention ≈ +0.014 over mean (CLAIMED). | Keep a mean-pool control |
| **C1** ConvNeXt-T | Eligible template S5: 256 px, 0.4 mm/px, 6 series × 12 windows, report-text-grouped 5 folds, mean-of-public-labels soft BCE, 14 epochs; 0.929 alone at 3 folds (CLAIMED). colum2131: ConvNeXt-T 0.953 vs DINOv2-S 0.951 on a matched pipeline (CLAIMED). | Our identical windows and labels first, then 256 px |
| **C2** CoAtNet | The lineage uses `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k` (295.6 MB, Apache-2.0, not yet in our registry) at 384, not coatnet_0/1 at 224. NTejas: CoAtNet 8.5× slower than ResNet50 on T4. | coatnet_0/1 at 224 is a different hypothesis; only Archit Konde claims 0.950 at 224 |
| **L1** labels | 747540: public label sets are statistically indistinguishable on gold-58. Gains are attributed to pseudo labels / OOF teachers and blending. Use mean-of-CC0-sets soft targets (S5) as one arm, drop gold-58 UIDs, mask unmentioned cells. | Avoid OOF self-distillation on silent cells (NTejas: Synovitis −0.038) |
| **TTA** (after D-series) | Anatomical mirror: sagittal reverses slice channels, coronal/axial flip width; applied before normalisation, only with verified plane and laterality | Never a plain horizontal flip in training |
| **Selection** | mathischmp: report-hash-grouped holdout, measured noise floor, ≥0.004 adoption rule | S1 and S2 select on gold-58, so their gold numbers are optimistic |

## 4. Unknown / unverified
- Exact ConvNeXt-T pretrained tag in S5.
- The pseudo-label procedure behind the 0.95x claims from colum2131 and azamat1ch (not published).
- Exact LB scores of the heliosli and evgendvorkin notebooks (not exposed by the CLI).
- The pilkwang baseline LB.
