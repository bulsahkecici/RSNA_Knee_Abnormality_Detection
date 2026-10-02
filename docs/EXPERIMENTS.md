# Experiments

Bounded first queue (none trained on real MRI in this install):

| ID | Family | Variable | Status |
| --- | --- | --- | --- |
| EXP-001-reference | REFERENCE | public reproduction | defined_not_run |
| EXP-002-frozen | OWN | frozen DINOv2-S | defined_not_run |
| EXP-003-local-labels | OWN | label source ablation | defined_not_run |
| EXP-004-finetune | OWN | last-block unfreeze | defined_not_run |

Champion rule: audit + baseline delta + uncertainty + source legitimacy + runtime + finite outputs. `AUC >= 0.86` is not enough.

Synthetic smoke metrics must stay `kind=synthetic`.
