"""Competition metadata discovery, audit, and local CSV fetch."""

from __future__ import annotations

import csv
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from rsna_knee.hashing import sha256_file, sha256_json
from rsna_knee.ontology import REPORT_COL, STUDY_ID_COL, TARGET_COLUMNS
from rsna_knee.paths import METADATA_DIR, ensure_runtime_dirs

SLUG = "rsna-knee-abnormality-detection"
CSV_FILES = ("sample_submission.csv", "test.csv", "test_series.csv", "train.csv", "train_series.csv")

KAGGLE_LAYOUTS = (
    Path("/kaggle/input") / SLUG,
    Path("/kaggle/input/competitions") / SLUG,
)


def discover_input_root(explicit: Path | None = None) -> Path | None:
    if explicit and explicit.exists():
        return explicit
    for layout in KAGGLE_LAYOUTS:
        if layout.exists():
            return layout
    if (METADATA_DIR / "train.csv").exists():
        return METADATA_DIR
    return None


def _kaggle_cmd(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["kaggle", *args], capture_output=True, text=True, check=False)


def fetch_metadata_csvs(dest: Path | None = None, dry_run: bool = False) -> dict[str, Any]:
    ensure_runtime_dirs()
    dest = dest or METADATA_DIR
    dest.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {"dest": str(dest), "files": {}, "blocked": None}
    for name in CSV_FILES:
        target = dest / name
        if target.exists() and target.stat().st_size > 0:
            results["files"][name] = {"path": str(target), "bytes": target.stat().st_size, "sha256": sha256_file(target), "skipped": True}
            continue
        if dry_run:
            results["files"][name] = {"would_download": True}
            continue
        proc = _kaggle_cmd(["competitions", "download", "-c", SLUG, "-f", name, "-p", str(dest), "--quiet"])
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            results["blocked"] = {
                "file": name,
                "error": err,
                "action_tr": "Kaggle kurallarını kabul edin ve kaggle.json ile oturum açın. Tam 570GB arşivi indirmeyin.",
            }
            return results
        path = dest / name
        if not path.exists():
            zipped = dest / f"{name}.zip"
            if zipped.exists():
                import zipfile

                with zipfile.ZipFile(zipped) as zf:
                    zf.extractall(dest)
            path = dest / name
        if path.exists():
            results["files"][name] = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return results


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _nonempty(value: str | None) -> bool:
    return value is not None and str(value).strip() not in {"", "nan", "NaN", "None"}


def audit_metadata(meta_dir: Path | None = None) -> dict[str, Any]:
    meta_dir = meta_dir or METADATA_DIR
    train_path = meta_dir / "train.csv"
    series_path = meta_dir / "train_series.csv"
    sample_path = meta_dir / "sample_submission.csv"
    test_path = meta_dir / "test.csv"
    if not train_path.exists():
        return {"status": "blocked", "reason": "train.csv missing", "action_tr": "rsna metadata fetch çalıştırın."}

    train = _read_csv(train_path)
    sample_cols = []
    if sample_path.exists():
        with sample_path.open(newline="", encoding="utf-8") as handle:
            sample_cols = list(csv.DictReader(handle).fieldnames or [])
    gold_rows = [row for row in train if all(_nonempty(row.get(c)) for c in TARGET_COLUMNS)]
    series_rows = _read_csv(series_path) if series_path.exists() else []
    planes = Counter(r.get("Anatomical_Plane") for r in series_rows)
    fluid = Counter(r.get("Fluid_Sensitive") for r in series_rows)
    fatsat = Counter(r.get("Fat_Suppression") for r in series_rows)
    fluid_eq = sum(1 for r in series_rows if r.get("Fluid_Sensitive") == r.get("Fat_Suppression"))
    test_n = len(_read_csv(test_path)) if test_path.exists() else 0
    report = {
        "status": "ok",
        "n_train_studies": len(train),
        "n_unique_study_uid": len({r[STUDY_ID_COL] for r in train}),
        "n_gold_all_targets": len(gold_rows),
        "n_missing_report": sum(1 for r in train if not _nonempty(r.get(REPORT_COL))),
        "n_train_series": len(series_rows),
        "n_series_studies": len({r[STUDY_ID_COL] for r in series_rows}) if series_rows else 0,
        "planes": dict(planes),
        "fluid_sensitive": dict(fluid),
        "fat_suppression": dict(fatsat),
        "fluid_equals_fatsat_count": fluid_eq,
        "fluid_equals_fatsat_all": bool(series_rows) and fluid_eq == len(series_rows),
        "visible_test_studies": test_n,
        "visible_test_is_not_hidden_test": True,
        "sample_submission_columns": sample_cols,
        "target_columns_match_sample": sample_cols[1:] == list(TARGET_COLUMNS) if sample_cols else False,
        "patient_id_present": False,
        "patient_cv_guaranteed": False,
        "notes": [
            "StudyInstanceUID is a study id, not a patient id.",
            "Do not treat published 4407/58 figures as code constants; this audit is a file measurement.",
            "Fluid_Sensitive and Fat_Suppression agreed on every training series in this snapshot; still treat them as independent flags.",
        ],
        "source_hashes": {
            "train.csv": sha256_file(train_path) if train_path.exists() else None,
            "train_series.csv": sha256_file(series_path) if series_path.exists() else None,
        },
    }
    out = meta_dir / "audit.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["audit_path"] = str(out)
    report["audit_hash"] = sha256_json(report)
    return report
