"""Turn extractor envelopes into the training-label rows the worker reads."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from rsna_knee.labels.validate import gold_to_training, states_to_training_arrays
from rsna_knee.ontology import ONTOLOGY_VERSION, TARGET_COLUMNS

_EMPTY = {"", "nan", "NaN", "None"}


def _uid_of(row: dict[str, Any]) -> tuple[str | None, str | None]:
    payload = row.get("payload") if isinstance(row.get("payload"), dict) else None
    outer = row.get("StudyInstanceUID") or row.get("uid")
    inner = payload.get("StudyInstanceUID") if payload else None
    outer_s = str(outer) if outer else None
    inner_s = str(inner) if inner else None
    if outer_s and inner_s and outer_s != inner_s:
        return None, "uid_mismatch"
    uid = inner_s or outer_s
    if not uid:
        return None, "uid_missing"
    return uid, None


def _state_targets(targets: dict[str, Any]) -> bool:
    for value in targets.values():
        if isinstance(value, dict) and "state" in value:
            return True
    return False


def normalize_record(row: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """Accept a raw extractor envelope or an already numeric training row."""
    uid, error = _uid_of(row)
    if error or uid is None:
        return None, error
    payload = row.get("payload") if isinstance(row.get("payload"), dict) else None
    targets = (payload or {}).get("targets") if payload else None
    if not isinstance(targets, dict):
        targets = row.get("targets") if isinstance(row.get("targets"), dict) else row.get("y")
    if not isinstance(targets, dict):
        return None, "targets_missing"
    if _state_targets(targets):
        if row.get("status") not in {None, "ok"}:
            return None, f"status:{row.get('status')}"
        y, mask, weight = states_to_training_arrays(targets)
        source = row.get("source") or "weak"
    else:
        masks = row.get("y_mask") if isinstance(row.get("y_mask"), dict) else {}
        weights = row.get("y_weight") if isinstance(row.get("y_weight"), dict) else {}
        y, mask, weight = {}, {}, {}
        for name in TARGET_COLUMNS:
            raw = targets.get(name)
            if name in masks:
                present = float(masks[name]) > 0 and raw is not None
            else:
                present = raw is not None and str(raw) not in _EMPTY
            if not present:
                y[name], mask[name], weight[name] = None, 0.0, 0.0
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                y[name], mask[name], weight[name] = None, 0.0, 0.0
                continue
            if value not in (0.0, 1.0):
                y[name], mask[name], weight[name] = None, 0.0, 0.0
                continue
            y[name] = value
            mask[name] = 1.0
            weight[name] = float(weights.get(name, 1.0))
        source = row.get("source") or "weak"
    return {
        "StudyInstanceUID": uid,
        "source": source,
        "targets": y,
        "y_mask": mask,
        "y_weight": weight,
        "ontology_version": (payload or row).get("ontology_version") or ONTOLOGY_VERSION,
    }, None


def load_gold_table(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.is_file():
        return {}
    table: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            uid = row.get("StudyInstanceUID")
            if uid:
                table[str(uid)] = row
    return table


def apply_gold(record: dict[str, Any], gold_row: dict[str, str] | None) -> dict[str, Any]:
    """Gold 0/1 overrides the extractor. -1 clears the target and stays a mask."""
    if not gold_row:
        return record
    applied = False
    for name in TARGET_COLUMNS:
        if name not in gold_row:
            continue
        raw = gold_row.get(name)
        if raw is None or str(raw).strip() in _EMPTY:
            continue
        y, mask, weight = gold_to_training(raw)
        record["targets"][name] = y
        record["y_mask"][name] = mask
        record["y_weight"][name] = weight
        applied = True
    if applied:
        record["source"] = "gold"
    return record


def publish_training_labels(
    rows: list[dict[str, Any]],
    dest: Path,
    *,
    train_csv: Path | None = None,
) -> dict[str, Any]:
    """Write canonical JSONL. Drop UID mismatches. Gold from train metadata wins."""
    gold = load_gold_table(train_csv)
    dest.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    rejected: list[dict[str, str]] = []
    supervised = 0
    with dest.open("w", encoding="utf-8") as handle:
        for row in rows:
            record, error = normalize_record(row)
            if record is None:
                rejected.append({"reason": error or "rejected"})
                continue
            apply_gold(record, gold.get(record["StudyInstanceUID"]))
            supervised += sum(1 for value in record["y_mask"].values() if float(value) > 0)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1
    return {"path": str(dest), "rows": kept, "rejected": rejected, "supervised_targets": supervised}
