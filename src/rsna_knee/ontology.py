"""Versioned competition ontology.

Host criteria are taken from discussion 733343 (Po-Hao Chen, 2026-08-06),
the pinned Knee Abnormality Detection AI Challenge Overview. Borderline
findings were graded negative to favor specificity. Report extraction
states are distinct from those image-based gold labels.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

ONTOLOGY_VERSION = "2026-08-06.host-733343.v1"
ONTOLOGY_SOURCE = (
    "https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343"
)

# Official sample_submission.csv column order. Do not reorder.
TARGET_COLUMNS: tuple[str, ...] = (
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
)
TARGETS = TARGET_COLUMNS

STUDY_ID_COL = "StudyInstanceUID"
REPORT_COL = "Report"
SERIES_ID_COL = "SeriesInstanceUID"


class LabelState(StrEnum):
    CRITERION_POSITIVE = "criterion_positive"
    EXPLICIT_NEGATIVE = "explicit_negative"
    UNCERTAIN = "uncertain"
    NOT_MENTIONED = "not_mentioned"
    BORDERLINE_NEGATIVE_HINT = "borderline_negative_hint"


class Laterality(StrEnum):
    LEFT = "left"
    RIGHT = "right"
    BILATERAL = "bilateral"
    UNSTATED = "unstated"
    NA = "na"


class Acuity(StrEnum):
    ACUTE = "acute"
    CHRONIC = "chronic"
    UNSPECIFIED = "unspecified"
    NA = "na"


# Host image-label criteria. Confirmed from discussion 733343.
HOST_CRITERIA: dict[str, dict[str, Any]] = {
    "ACL": {
        "positive": (
            "High-grade partial or full-thickness tear of the ACL: complete discontinuity "
            "or more than 50 percent of fibers disrupted, with or without secondary signs "
            "such as characteristic pivot-shift bone contusions."
        ),
        "negative": "Mild signal change, degeneration, or thickening without discontinuity is graded negative.",
        "notes": "Ambiguous/borderline findings were graded negative to favor specificity.",
        "source_confirmed": True,
    },
    "MCL": {
        "positive": (
            "High-grade partial or complete acute tear of the MCL, with disrupted fibers "
            "and edema within and adjacent to the ligament."
        ),
        "negative": "Low-grade sprains and chronic or remote stress changes are graded negative.",
        "source_confirmed": True,
    },
    "Medial Meniscus": {
        "positive": (
            "Abnormal signal that definitely contacts the meniscal surface on at least two "
            "images, or a morphologic abnormality such as a truncated, diminutive, or "
            "displaced fragment, involving the medial meniscus."
        ),
        "negative": "Intrasubstance degeneration that does not reach the surface is negative.",
        "source_confirmed": True,
    },
    "Lateral Meniscus": {
        "positive": "The same criteria applied to the lateral meniscus.",
        "negative": "Intrasubstance degeneration that does not reach the surface is negative.",
        "source_confirmed": True,
    },
    "Medial OA": {
        "positive": (
            "Moderate or large area (roughly 1 cm or greater) of high-grade cartilage loss, "
            "defined as greater than 50 percent of cartilage thickness, in the medial "
            "compartment, with or without underlying subchondral marrow changes."
        ),
        "negative": "Smaller or lower-grade cartilage loss is negative under host criteria.",
        "source_confirmed": True,
    },
    "Lateral OA": {
        "positive": "The same criteria applied to the lateral compartment.",
        "source_confirmed": True,
    },
    "PF OA": {
        "positive": "The same criteria applied to the patellofemoral compartment.",
        "source_confirmed": True,
    },
    "Effusion": {
        "positive": "A moderate or large amount of fluid distending the joint.",
        "negative": "Physiologic or small fluid is not a host-positive effusion.",
        "source_confirmed": True,
    },
    "Synovitis": {
        "positive": "Inflammation and thickening of the synovial lining of the joint.",
        "source_confirmed": True,
    },
    "Baker's": {
        "positive": "A moderate or large fluid collection in the characteristic location behind the knee.",
        "source_confirmed": True,
    },
    "Contusion": {
        "positive": "Bone marrow edema-like signal from impact, without a discrete fracture line.",
        "source_confirmed": True,
    },
    "Fracture": {
        "positive": "An acute cortical break or fracture line.",
        "notes": "Host wording: acute fracture. Chronic/remote fracture is not automatically positive.",
        "source_confirmed": True,
    },
}


PILOT_PLANES: tuple[str, ...] = ("Sagittal", "Coronal", "Axial")
ADVANCED_SLOTS: tuple[str, ...] = (
    "SAG_FLUID_FS",
    "COR_FLUID_FS",
    "AX_FLUID_FS",
    "SAG_FLUID_NOFS",
    "COR_T1",
    "SAG_T1",
)


def target_index(name: str) -> int:
    return TARGET_COLUMNS.index(name)


def empty_target_map(fill: Any = None) -> dict[str, Any]:
    return {name: fill for name in TARGET_COLUMNS}
