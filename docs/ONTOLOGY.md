# Ontology

Version `2026-08-06.host-733343.v1` from https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343

See `src/rsna_knee/ontology.py` HOST_CRITERIA.

Report extraction states:

- `criterion_positive`
- `explicit_negative`
- `uncertain`
- `not_mentioned` (not auto-negative)
- `borderline_negative_hint` (host graded borderline negative; distinct from missing text)

Gold image labels in `train.csv` are 0/1 or empty. Empty and `-1` are masked from BCE.
