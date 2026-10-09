# Competition snapshot

**Retrieved:** 2026-10-03 via Kaggle CLI (`competitions pages list --content`, `topics show 733343`, `submission-limits`, `quota`, metadata CSVs).

## Identity

- URL: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection
- Host: Radiological Society of North America
- Metric: macro-average ROC-AUC over 12 targets (Evaluation page).
- Code competition: notebooks, internet disabled, ≤9 hours, `submission.csv`.
- External data: freely & publicly available, including pretrained models (Code Requirements).
- Team size max 5; 5 submissions/day; 2 final selections (rules + `submission-limits` this day: remaining 5; 0 submissions before this run).
- Winner license CC-BY-NC 4.0; data MIRA (rules). Unread subsections of the long rules page are **not guessed**.

## Timeline (UTC 23:59)

- Start 2026-07-30
- Entry and team merger 2026-10-15
- Final submission 2026-10-22
- Winners' materials 2026-11-05

## Files (measured locally after CSV-only download)

- `train.csv`: 4407 studies, columns StudyInstanceUID, Report, 12 targets. 58 rows with all targets 0/1; 4349 empty labels. Reports present (not printed here).
- `train_series.csv`: 24371 series, 4407 studies. Planes Sagittal/Coronal/Axial. Fluid_Sensitive == Fat_Suppression on every row **in this file**; official text still says they are not necessarily equivalent.
- `test.csv`: 3 StudyInstanceUIDs, no Report. Official data-description: about 1300 test studies; preview is replaced at scoring.
- Sample columns: `StudyInstanceUID,ACL,MCL,Medial Meniscus,Lateral Meniscus,Medial OA,Lateral OA,PF OA,Effusion,Synovitis,Baker's,Contusion,Fracture`

## Host criteria (discussion 733343, Po-Hao Chen, 2026-08-06)

Confirmed: high-grade ACL/MCL; meniscal surface-contacting tear vs intrasubstance degeneration; OA as ≥~1cm high-grade (>50%) cartilage loss per compartment; moderate/large effusion and Baker cyst; synovitis as synovial thickening; contusion without fracture line; acute fracture. Borderline graded negative for specificity. Image labels are consensus of MSK readers, not the report text.

## Hidden test

~1300 studies (data-description). Visible 3 UIDs are not the final test.

## Efficiency prize

Documented on overview (formula with Benchmark, maxAUC, RuntimeSeconds/32400). Not used as a training target in this repo yet.

## Refresh evidence — 2026-10-03

Official pages, host discussion, and live limits were refreshed into `state/competition-intel-20261003/`. Data description explicitly uses `train_series/` and `test_series/`; inference was corrected to those directories. Evaluation column order, offline code requirements, and final deadline match the prior snapshot. No complete training-data conversion is required merely to submit a valid trained pilot.


## Official refresh — 2026-10-09

Read-only competition-researcher agent verified Evaluation, Timeline, Code Requirements, Rules, Data and host discussion 733343 through authenticated Kaggle CLI. Public leaders 0.964, top 20 at least 0.960; our confirmed submission 56973673 is 0.740. Exact user rank is unmeasured, public ranking does not establish private final standing.

GPU quota: 22.62h remaining /30h; API refresh timestamp 2026-10-10T00:00:00 has unspecified timezone. Submission API: 1 today, 4 remaining, daily limit 5 and two final selections. Final deadline 2026-10-22 23:59 UTC (2026-10-23 02:59 Europe/Istanbul). Entry/team deadline 2026-10-15 23:59 UTC. Runtime remains <=9h, Internet disabled, submission.csv, macro AUC across all12 targets. About1300 hidden studies is approximate official data-page information.

Sources: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/evaluation ; https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/timeline ; https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/code-requirements ; https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules ; https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/leaderboard ; https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343
