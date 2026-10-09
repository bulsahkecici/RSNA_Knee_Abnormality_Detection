# P0 source / eligibility audit — 2026-10-09

Scope: read-only checks on 2026-10-09, 19:17–19:30 UTC. Tools used: authenticated Kaggle CLI 2.2.4 (`competitions pages/leaderboard/submission-limits/topics`, `quota`, `datasets metadata/files`), the GitHub REST API, raw.githubusercontent, the Hugging Face model API (LFS sha256), HTTP HEAD requests and two web lookups. No weights, MRI data or report text were downloaded or read. Nothing was submitted, pushed or posted. The machine-readable registry is `docs/research/EXTERNAL_SOURCE_REGISTRY_20261009.csv`.

Note on sha256 values. Each "sha256" comes from the Hugging Face or GitHub LFS pointer unless marked otherwise. The DINOv2 hash was computed from our local `artifacts/weights/dinov2_vits14_pretrain.pth`. Its size matches the official URL (88,283,115 bytes) and the hash matches `configs/dinov2_vits14_allowlist.json`, but Meta does not publish a sha256.

## 1. Summary verdict table

| Candidate | Verdict | One-line reason |
|---|---|---|
| DINOv2 ViT-S/14 official (current baseline) | **ELIGIBLE_OWN** | Code and weights are Apache-2.0. Trained on natural images. Free and ungated. |
| ConvNeXt-T official (`convnext_tiny_1k_224_ema.pth`) and timm `convnext_tiny.fb_in1k` / `.fb_in22k_ft_in1k` | **ELIGIBLE_OWN** | Official repo is MIT, timm cards are Apache-2.0. ImageNet only. |
| CoAtNet via timm (`coatnet_0_rw_224.sw_in1k`, `coatnet_1_rw_224.sw_in1k`, `coatnet_rmlp_1_rw_224.sw_in1k`, `coatnet_bn_0_rw_224.sw_in1k`, `coatnet_2_rw_224.sw_in12k_ft_in1k`) | **ELIGIBLE_OWN** | Apache-2.0, ImageNet. These are the "rw" variants trained in timm; Google released no official CoAtNet weights. |
| OrthoFoundation-L | **UNKNOWN_NEEDS_HOST** (treat as not usable) | OAI + fastMRI + a private hospital cohort, on a DINOv3-L base. The weights' license conflicts with the DINOv3 License. No host ruling covers checkpoints trained on private or OAI data. |
| RadImageNet (ResNet50/DenseNet121/InceptionV3) | **UNKNOWN_NEEDS_HOST** | Weights are freely downloadable. The pretraining data is a private single-center set, available by request only. The weights license is not separately stated. Thread 745283 asks exactly this question and has no answer. |
| BiomedCLIP ViT-B/16 | **ELIGIBLE_OWN** (low priority) | MIT, trained on public PubMed Central figures. The model card says research use only, which is a statement of intent, not a license term. |
| OrthoDiffusion | **UNKNOWN_NEEDS_HOST** | MIT card, but the abstract does not name its data sources. Same author group as OrthoFoundation. 3 × 553 MB. |
| Public LLM label sets (`stevenleehans`, `yunusgmsoy`) | **ELIGIBLE_OWN** (with gold-UID filtering) | Both CC0-1.0 and derived only from competition reports. The host allows LLM label extraction (733965). No host comment on these specific datasets. |
| KneeCoT | **BLOCKED** | Explicit host ruling (734109). |

## 2. Competition facts (primary source: Kaggle CLI `competitions pages`, `list`, `leaderboard`, `quota`)

**Timeline** ([timeline](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/timeline)):
- Start: 2026-07-30.
- Entry deadline and team merger deadline: **2026-10-15 23:59 UTC**.
- Final submission deadline: **2026-10-22 23:59 UTC** (2026-10-23 02:59 Istanbul).
- Winners' requirements deadline (code, video, method description): 2026-11-05.
- `competitions list` shows the deadline as `2026-10-22 23:59:00`. Teams: 5,610. Our rank: 4,643.

**Code Requirements**, verbatim:
- "CPU Notebook <= 9 hours run-time"
- "GPU Notebook <= 9 hours run-time"
- "Internet access disabled"
- "Freely & publicly available external data is allowed, including pre-trained models"
- File must be named `submission.csv`.
- The hidden test set has about 1,300 studies. No `Report` field is given at test time.

**Submission limits** (rules §2.x): a maximum of 5 submissions per day and a maximum team size of 5. The CLI `submission-limits` shows: **2 used today, 3 remaining, 5 lifetime**.

**Prizes**: main leaderboard places 1–10 ($9k down to $5k), plus a separate **Efficiency Track** ($7k / $6k / $5k).

**Winner obligations (Prizes page)**: a video, plus public code and weights posted on the forum, plus a final model released openly.

**License conflict in the rules (unresolved)**:
- The competition-specific header says "WINNER LICENSE TYPE CC-BY-NC 4.0".
- §5.a.2 calls CC-BY-NC 4.0 "an Open Source Initiative-approved license … that in no event limits commercial use". That description is self-contradictory.
- §5.a.4 says that if "input data or pretrained models with an incompatible license are used … you do not need to grant an open source license … for that data and/or model(s)."
- General rule 6.c: open-source *code* used must be OSI-approved and must not limit commercial use. Thread 739905 asks about this and has no host answer.

**External data rule text** (Competition rules §6 "External Data and Tools"):
- (a) "you will ensure the External Data is either publicly available and equally accessible to use by all Participants of the Competition for purposes of the competition at no cost to the other Participants, or satisfies the Reasonableness criteria…"
- (b) "The use of external data and models is acceptable unless specifically prohibited by the Host. … their use must be "reasonably accessible to all" and of "minimal cost"."
- The Reasonableness Standard "will be determined and applied by the Host in light of things like cost thresholds and accessibility". Example given in the rules: "Purchasing a license to use a proprietary dataset that exceeds the cost of a prize … would not be considered reasonable."

**Public leaderboard (CLI, 2026-10-09 ~19:18 UTC)**:
- Top score **0.964**, shared by CentralyDS, Sergey Bryansky and Ilya Kolb.
- 0.963 held by Pa3฿aJluHа, joinT fusion, Zhengru Li, Wassim Dobbi and rhoskeri.
- 0.962 held by YOKKOISHO, dalab and one other team.

**Our GPU quota (`kaggle quota`)**:
- GPU: 10.87 h used, **19.13 h remaining of 30 h**, refreshes at 2026-10-10T00:00:00.
- TPU: 20 h / 20 h unused.

## 3. Host rulings

Method: the CLI listed **all 175 discussion topics** (10 pages, sort=new) and fetched every topic with author names. Host and staff authors found: **Po-Hao "Howard" Chen** (Challenge Organizing Team lead, Cleveland Clinic, per the Acknowledgements page; participants address him as @javaduke95) and **María Cruz** (Kaggle; Discord/getting-started post only). Their posts are dated 2026-07-30 through **2026-09-21**. **No host or staff post after 2026-09-21 exists in any topic.** That includes 747455, 747660, 746792, 735304, 747540, 743416, 737950, 744056, 743420, 739905, 745283, 745666 and 746443.

Host statements, quoted:
- [734109](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734109) (2026-08-25): "For now, the KneeCoT dataset is not allowed in this competition." The stated reason is "the required data agreement with dataset institution". The host may change its stance "if the data use policy changes on HF".
- [733965](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733965), topic post (2026-08-09): "Use of commercially hosted LLMs and other external inference services is permitted, provided that…" the rules on accessibility and cost are met. Comment (2026-08-27): "You can use LLM API, such as those from OpenAI, to read the reports to generate the labels."
- 733965, comment 3517696 (2026-08-27):
  - "These datasets would not be excluded by the competition rules solely because their licenses restrict use to non-commercial purposes."
  - "Datasets that can be accessed through a straightforward registration or click-through data-use agreement would generally satisfy this requirement."
  - "datasets requiring institution-specific approvals, negotiated written agreements, IRB approval, lengthy credentialing, or other substantial administrative steps may present a meaningful accessibility barrier"
  - The host also says RSNA and Kaggle "cannot interpret or waive" third-party data-use agreements, and that teams remain responsible.
- **[741819](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/741819) (2026-09-21), the newest host ruling:**
  - "Using OAI as an external dataset does not violate the RSNA Knee Abnormality Detection challenge rules if it is generally accessible to all participants without an escalated (i.e. institutional) review and sign-off or access to a legal team, and is free of charge."
  - "We do not own or control the OAI dataset and cannot comment on whether this challenge complies OAI data-use terms…"

Thread status. The statements below are **participant claims, not host statements**:
- [747455](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/747455) (OAI-trained public weights): one participant reply advises avoiding them. No host reply.
- [747660](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/747660) (0.949 weights): participants say the organizers "haven't given a clear response". No host reply.
- [743416](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743416): no host reply. Participants claim that NIMH Data Archive staff described OAI as openly accessible. One participant says their individual OAI application was approved immediately. Another says access appears blocked from China. All UNVERIFIED.
- [737950](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737950): asks whether public checkpoints pretrained on inaccessible private clinical data are allowed. No host reply.
- [744056](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/744056): asks about prize eligibility when using public weights trained on competition data by others. No host reply.
- [745283](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/745283) (RadImageNet), [745666](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/745666) (torchvision ConvNeXt delivery) and [746443](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/746443) (models from HF): no replies of any kind.
- [746792](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/746792), [735304](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304) and [747540](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/747540): no host posts. Scores in them are author self-reports.

## 4. Per-candidate evidence

**DINOv2 ViT-S/14.**
- Artifact: `https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth`. HEAD returned 200, 88,283,115 bytes, last-modified 2023-04-13. Local sha256 `b938bf1b…0cd9`.
- README: "DINOv2 code and model weights are released under the Apache License 2.0." The FAIR-NC license in the same README applies only to XRay-DINO, which we do not use.
- Pretraining data: LVD-142M, curated natural images. No medical or overlap concern.
- HF mirrors: `facebook/dinov2-small` (safetensors `ae1e99fc…4be1`) and `timm/vit_small_patch14_dinov2.lvd142m` (`04d27f34…0081`), both Apache-2.0.

**ConvNeXt-T.**
- The official repo is MIT and **archived** (last push 2023-01). README: "This project is released under the MIT license."
- `convnext_tiny_1k_224_ema.pth`: HEAD 200, 114,414,741 bytes. sha256 is not published (UNVERIFIED). There are also 22k and 22k→1k variants.
- timm `convnext_tiny.fb_in1k`: 114,374,272 bytes, sha256 `08b9dc9c…9276`, Apache-2.0. `convnext_tiny.fb_in22k_ft_in1k` has sha256 `77b619e1…868a`.
- ImageNet's own terms of access are research-oriented. This is standard practice and the host has never objected.

**CoAtNet (timm).**
- All of these exist with pretrained weights, all Apache-2.0 per the HF cards, all `datasets: imagenet-1k` (12k for coatnet_2):
  - `coatnet_0_rw_224.sw_in1k`: 109.8 MB, `efe01c8c…366c`
  - `coatnet_1_rw_224.sw_in1k`: 167.0 MB, `ef617b9c…5fbc`
  - `coatnet_rmlp_1_rw_224.sw_in1k`: 166.9 MB
  - `coatnet_bn_0_rw_224.sw_in1k`: 109.9 MB
  - `coatnet_rmlp_2_rw_224.sw_in1k`
  - `coatnet_2_rw_224.sw_in12k_ft_in1k`: 295.6 MB
  - `coatnet_nano_rw_224.sw_in1k` and `coatnet_rmlp_nano_rw_224.sw_in1k`
- Thread 746792 says only "CoAtNet". The exact variant behind the 0.949 claim is UNVERIFIED.

**OrthoFoundation-L.**
- LFS pointer at `raw.githubusercontent.com/ytrsk/OrthoFoundation/main/OrthoFoudation-L.pth`: `oid sha256:385a775822107b68eaa486336feb982e1ce7bd6d4e8c03ceb482a0bf546f2ff9`, `size 1213056638`. The media URL returned HEAD 200, with an ETag equal to the oid.
- Commits: checkpoint released 2026-06-18. The MIT `LICENSE` was added 2026-08-25.
- Pretraining data, from the README:
  - 1,251,655 knee images.
  - "OAI contributed approximately 100,000 knee X-ray images and 200,000 knee MRI slices, while fastMRI contributed 300,000 knee MRI slices."
  - A private cohort from "four centers of Peking University Third Hospital Groups" (130,567 patients), "available under restricted access".
  - The backbone is "initialized with DINOv3-L weights".
- Obligations:
  - The DINOv3 License (§1.b.i) requires that derivative works be distributed "only … under the terms of this Agreement", with a copy of the Agreement. It also contains trade-control and ITAR clauses. It is not OSI-approved, so the repo's MIT label for the weights is doubtful.
  - fastMRI access requires an application with "the investigator's institutional affiliation", plus review, plus a research/education-only DSA (fastmri.med.nyu.edu). Under the host's 733965 criteria this may count as an accessibility barrier.
  - OAI is allowed only conditionally (741819).
  - The private cohort is plainly inaccessible to other participants (the 737950 question, unanswered).
- Overlap: Peking University Third Hospital, NYU (fastMRI) and the OAI clinical sites do not appear among the 22 listed competition data contributors (Acknowledgements page). Overlap is low but cannot be verified.
- Runtime: ViT-L, 1.2 GB. Inference cost against the 9 h limit is unmeasured.

**RadImageNet.**
- The repo is MIT. Weights are on Google Drive:
  - TF/Keras: `drive.google.com/drive/folders/1Es7cK1hv7zNHJoUW0tI0e6nLFVYTqPqK`, plus a zip `1UgYviv2K6QPM1SCexqqab5-yTgwoAFEc`.
  - PyTorch: `RadImageNet_pytorch.zip`, `drive.google.com/file/d/1RHt2GnuOYlc_gcoTETtBDSW73mFyRAtR`.
  - Size and sha256 of the Drive files are UNVERIFIED (Drive does not expose them without a download).
- An unofficial HF mirror, `Lab-Rasool/RadImageNet`, has `ResNet50.pt` at 94,345,733 bytes, sha256 `08629f7e…0734`. Its provenance against the official weights is unverified.
- Pretraining data ([PMC9530758](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9530758/)): 1.35 M CT/MRI/US images from 131,872 patients at a single outpatient facility (NYC per the preprint), 2005–2020. The dataset is "available by request".
- A weights-specific license is not stated.

**BiomedCLIP.**
- `open_clip_pytorch_model.bin`: 783,705,670 bytes, sha256 `52cc993c…76be`. MIT (LICENSE.md, © Microsoft). Ungated.
- Data: PMC-15M figure-caption pairs from PubMed Central.
- The card says intended "solely for (I) future research … and (II) reproducibility" and that "Any deployed use case … is currently out of scope". This is not a license restriction.

**OrthoDiffusion.**
- HF `lanstat0123/orthodiffusion` (MIT card) has three `*_model.pt` files of 552,919,566 bytes each. Their sha256 values are in the CSV.
- The arXiv 2602.20752 abstract says "pre-trained … on 15,948 unlabeled knee MRI scans" but names no sources (UNVERIFIED). Several authors overlap with OrthoFoundation, so the data may come from the same private cohort (UNVERIFIED).

**Public label sets** (`state/public-labels/` holds only `stevenleehans/` and `yunusgmsoy/`, each containing a `dataset-metadata.json`; no CSVs are stored there):
- [stevenleehans/rsna-knee-llm-report-labels](https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels): CC0-1.0, 62 votes. Files `llm_labels_full.csv`, `llm_labels_v2.csv`, `llm_labels_v4_blend.csv`. v4 blends in lixin73's label set (also CC0, per the card).
- [yunusgmsoy/rsna-knee-llm-report-labels](https://www.kaggle.com/datasets/yunusgmsoy/rsna-knee-llm-report-labels): CC0-1.0, file `yunus_llm_labels.csv`.
- Both cards say the labels come only from the competition `Report` column and cover **all 4,407 training studies, including gold-58**. Under AGENTS.md, gold eval UIDs must be dropped before any image training.
- The host has made no comment on either dataset. The general permission for LLM label extraction is 733965.

## 5. Open questions needing a user or host decision

1. **OrthoFoundation / OrthoDiffusion / RadImageNet.** Are public checkpoints pretrained partly on private or institution-gated data allowed for prize-eligible submissions? This is unanswered in 737950, 745283 and 747455. Recommendation: do not use them in submissions unless the host answers. If explored at all, keep it to offline analysis that never touches submission assets, and get a user decision first.
2. **The OAI ruling is conditional.** It depends on access needing no institutional sign-off. Participant claims on that point conflict. Our policy remains: no OAI-trained weights (including the 0.949 public CoAtNet) without explicit host confirmation.
3. **Winner license conflict.** CC-BY-NC 4.0 versus "OSI, no commercial limit" (739905, unanswered). This matters only if we place in the top 10. Apache/MIT backbones are compatible with both readings.
4. **fastMRI accessibility** under the 733965 criteria: an affiliation-plus-review application. UNVERIFIED whether this counts as "straightforward registration".
5. **Team/entry deadline 2026-10-15.** Any merge decision must be made by then.
6. **Unverified items:** the exact CoAtNet variant in 746792, RadImageNet Drive sizes and hashes, the official ConvNeXt sha256, and OrthoDiffusion's data sources.
