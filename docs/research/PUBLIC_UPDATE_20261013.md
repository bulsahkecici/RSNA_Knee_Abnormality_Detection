# Public update — 2026-10-13

Read-only. Sources: the public leaderboard CSV (downloaded 12:56 UTC), `kaggle kernels list` sorted by score and votes, and discussion threads 748111, 747613, 745861, 748011 and 748480 read with `kaggle competitions topics show` / `topic-messages`. All scores below are **CLAIMED** by their authors unless they come from the leaderboard. Earlier extraction: `PUBLIC_RECIPES_20261009.md`; eligibility verdicts: `P0_SOURCE_AUDIT_20261009.md`.

## Where we stand

| | Public score |
|---|---:|
| Leaderboard top (CentralyDS, Sergey Bryansky) | 0.966 |
| 10th percentile of 5913 teams | 0.955 |
| Median | 0.938 |
| Teams at ≥ 0.950 | 1552 |
| **Us (D3ALL, 57133827), rank 4562 of 5913** | **0.845** |

The mass at 0.95–0.957 matches the one-click public blends (`heliosli/0-957-latest-dataset-score-in-one-click`, `jenillangaliya/...-0-956-...`, `goodpjw2008/...-lb-0-954`). Thread 748181 ("Kaggle could we please start doing something about the public notebooks?", 92 votes) complains about exactly this.

## What changed since 2026-10-09

- **Top public notebooks are still blends of other people's checkpoints** (heliosli, Raptor-OAI, DINOsaur, Versia, Matterhorn). The ones read on 2026-10-09 depend on OAI-trained or OrthoFoundation weights; the new forks reuse them. Not re-audited one by one here.
- **External data is still not settled for OAI.** 748480 (2026-10-12) asks whether OAI via the NIMH Data Archive (account + terms) and fastMRI (institutional application) are allowed and whether models trained on them are compatible with publishing winning weights; no host answer yet. The earlier host statement (741819) stays conditional: allowed if accessible without institutional sign-off and free. Participants disagree on whether top scorers use it (747613).
- **Validation signal (748011, colum2131, 15 own submissions, 5-fold):** AUC on ~4,300 LLM-label studies vs public LB r = 0.87; AUC on gold-58 vs public LB r = 0.66. A reply simulates the public LB itself as noisy (~390 scored studies). Another participant: public 0.961 with OOF 0.937 and gold-58 0.938.
- **0.960 single submission** (748111) with no method disclosed; a comment says the author described it as basically a single model.
- **Label work remains the lever named most often** (745861, hengck23): stronger LLM labelers (Claude Opus 4.5 labels, Qwen JEV), using the strongest labeler only for the most uncertain cells, and segmentation/ROI labels from image + report.

## How our pipeline differs from the eligible ≥ 0.92 single-model recipes

| | Ours (D3ALLLR2) | Raptor / mathischmp / S5 ConvNeXt reader (2026-10-09 doc) |
|---|---|---|
| In-plane resolution | 160 mm crop → 224 px ≈ **0.71 mm/px** | 0.36–0.49 mm/px (140 mm → 307–384 px; S5 0.4 mm/px) |
| Series | 3 fluid slots | 5–6 slots incl. structural sagittal/coronal and axial |
| Windows at inference | 3 × 12 = 36 | 42–94 |
| Backbone training | last 4 DINOv2-S blocks, lr 2e-5 | full fine-tune, backbone lr 3–4e-5, head 1–2e-3, OneCycle |
| Pooling | masked mean | per-finding attention (all strong recipes) |
| Labels | mean of 2 CC0 tables | blends of 3–6 CC0 tables, own labels, pseudo labels |
| Gold-58 local | 0.848 | 0.903–0.917 (Raptor/mathischmp), LB 0.920–0.928 |

Our local gold has tracked our public score so far (D3 0.820 / 0.835, D3ALL 0.832 / 0.845), so the remaining gap is a modelling gap, not a validation artefact.

## Implications for our next decisions

1. **Resolution is the largest untested difference.** Our crop gives ~0.71 mm/px; every eligible strong recipe works at 0.36–0.49 mm/px. D2 (v5: 5 slots, 154 mm → 256 px ≈ 0.60 mm/px) was tested only on the old base (hard labels, 2 blocks) and did not help there; it has not been tried on the D3ALL/LR2 recipe. The v5 cache already exists on Kaggle.
2. **Prefer weak-label AUC over gold-58 for adoption decisions** where a weak split exists (748011 evidence). The all-folds arms (D3ALL*) have none; pair future all-folds candidates with a fold-0 twin that does.
3. **Training intensity:** our learning-rate result (LR2 ≈ E16 at half the time) points the same way as the public recipes (full fine-tune at 3–4e-5).
4. **Public blends / OAI weights:** a public-blend submission would likely score ~0.95–0.957 immediately, but it rests on checkpoints whose external-data eligibility is unresolved (748480 open). That is a user decision, not an engineering one; our policy remains no OAI-trained weights without explicit host confirmation.
