---
name: competition-intel
description: Refresh dated competition snapshot from official Kaggle pages and host discussion 733343. Use before changing rules, schema, or ontology.
---

# Competition intel

## When

Rules, deadlines, column order, or host label criteria might have changed.

## Procedure

1. `kaggle competitions pages list --content --page-name Evaluation rsna-knee-abnormality-detection`
2. Same for Timeline, data-description, rules, Code Requirements.
3. `kaggle competitions topics show 733343`
4. `kaggle competitions submission-limits -c rsna-knee-abnormality-detection`
5. Update `docs/COMPETITION_SNAPSHOT.md` with date and URLs. Mark unread fields unknown.

## Entrypoints

- `rsna metadata audit`
- `rsna doctor`

## Inputs / outputs

Inputs: Kaggle CLI auth. Outputs: snapshot markdown. Acceptance: no guessed quotas.

## Recovery

If 403, write NEEDS_AUTH handoff; continue with last snapshot marked stale.
