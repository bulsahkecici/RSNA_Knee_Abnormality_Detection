"""Labeler agent wraps extract_one; it does not invent labels when LM Studio is down."""

from __future__ import annotations

from rsna_knee.labels.extract import LabelStore, extract_one
from rsna_knee.labels.lmstudio import LMStudioClient


def run_labeler(
    client: LMStudioClient,
    rows: list[dict],
    model_id: str,
    model_revision: str,
    store: LabelStore,
    json_mode: str = "json_schema",
    allow_reasoning_json: bool = False,
):
    out = []
    for row in rows:
        out.append(
            extract_one(
                client,
                uid=row["StudyInstanceUID"],
                report=row["Report"],
                model_id=model_id,
                model_revision=model_revision,
                store=store,
                json_mode=json_mode,
                allow_reasoning_json=allow_reasoning_json,
            )
        )
    return out
