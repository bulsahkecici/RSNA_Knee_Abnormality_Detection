"""Sequential local labels and remote training, with durable status and locks."""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

from filelock import FileLock

from rsna_knee.config import load_config
from rsna_knee.data.folds import image_training_uids, load_folds
from rsna_knee.hashing import sha256_file
from rsna_knee.labels.canonical import apply_gold, normalize_record
from rsna_knee.labels.extract import prompt_hash
from rsna_knee.labels.validate import validate_extraction
from rsna_knee.ontology import TARGET_COLUMNS
from rsna_knee.paths import ROOT
from rsna_knee.pipeline import stage_labels
from rsna_knee.runtime.campaign import prepare_campaign_kernel
from rsna_knee.workflow.graph import new_state
from rsna_knee.workflow.registry import Registry


def write_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps({**payload, "updated_ts": time.time()}, indent=2))
    temp.replace(path)


def label_quality(raw: Path, *, minimum: int) -> dict[str, Any]:
    """Check source evidence locally. This is not a clinical accuracy estimate."""
    folds = load_folds()
    gold = {r["StudyInstanceUID"] for r in folds if r.get("is_gold") == "1"}
    allowed = set(image_training_uids(folds)) - gold
    reports = {r["StudyInstanceUID"]: r.get("Report") or "" for r in csv.DictReader((ROOT / "data/metadata/train.csv").open())}
    valid = 0
    attempted = 0
    reasons: Counter[str] = Counter()
    states: dict[str, Counter[str]] = {t: Counter() for t in TARGET_COLUMNS}
    seen: set[str] = set()
    for line in raw.read_text().splitlines():
        row = json.loads(line)
        attempted += 1
        if row.get("status") != "ok":
            reasons["quarantined"] += 1
            continue
        payload = row.get("payload") or {}
        uid = payload.get("StudyInstanceUID")
        if uid not in allowed or uid in seen:
            reasons["selection_or_duplicate_error"] += 1
            continue
        seen.add(uid)
        errors = validate_extraction(payload, reports.get(uid, ""), expected_uid=uid)
        if errors:
            reasons["strict_evidence_or_schema_error"] += 1
            continue
        valid += 1
        for name, rec in payload["targets"].items():
            states[name][rec["state"]] += 1
    error_rate = (attempted - valid) / max(1, attempted)
    return {
        "pass": valid >= minimum and error_rate <= 0.05 and not reasons.get("selection_or_duplicate_error"),
        "attempted": attempted, "valid": valid, "error_rate": error_rate,
        "reasons": dict(reasons), "per_target_states": {k: dict(v) for k, v in states.items()},
        "clinical_accuracy_verified": False, "raw_reports_exported": False,
    }


def numeric_training_labels(weak_path: Path, dest: Path) -> dict[str, Any]:
    """Gold eval labels are retained for evaluation; the worker excludes their images."""
    table = {}
    for line in weak_path.read_text().splitlines():
        row, error = normalize_record(json.loads(line))
        if error or row is None:
            raise ValueError("invalid_weak_label_row")
        if row["StudyInstanceUID"] in table:
            raise ValueError("duplicate_weak_label_row")
        table[row["StudyInstanceUID"]] = row
    with (ROOT / "data/metadata/train.csv").open() as handle:
        for row in csv.DictReader(handle):
            if not any(row.get(t, "").strip() for t in TARGET_COLUMNS):
                continue
            base = {"StudyInstanceUID": row["StudyInstanceUID"], "targets": {}, "y_mask": {}, "y_weight": {}}
            rec, error = normalize_record(base)
            if error or rec is None:
                raise ValueError("invalid_gold_row")
            table[row["StudyInstanceUID"]] = apply_gold(rec, row)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("".join(json.dumps(table[u]) + "\n" for u in sorted(table)))
    return {"rows": len(table), "sha256": sha256_file(dest), "raw_reports_uploaded": False}


def completed_label_run(run_id: str, minimum: int) -> tuple[Path, dict[str, Any]] | None:
    """Adopt an existing local job only after its final registry transaction."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("invalid_label_run_identity")
    registry = Registry()
    try:
        record = registry.get_run(run_id)
    finally:
        registry.close()
    if record is None:
        return None
    if record.get("synthetic") or record.get("stage") != "LABELS_READY" or record.get("blocked_reason"):
        raise ValueError("existing_label_run_not_ready")
    raw = ROOT / "data/labels" / f"{run_id}-labels.jsonl"
    numeric = ROOT / "data/labels" / f"{run_id}-training.jsonl"
    if not raw.is_file() or not numeric.is_file():
        raise ValueError("existing_label_artifacts_missing")
    if sha256_file(numeric) != (record.get("hashes") or {}).get("labels"):
        raise ValueError("existing_label_hash_mismatch")
    quality = label_quality(raw, minimum=minimum)
    if not quality["pass"]:
        raise ValueError("existing_label_quality_gate_failed")
    return numeric, quality


def run_campaign(run_id: str, owner: str, *, pilot_run_id: str | None = None, labels_run_id: str | None = None) -> dict[str, Any]:
    if not re.fullmatch(r"[a-z0-9-]+", run_id) or not re.fullmatch(r"[A-Za-z0-9_-]+", owner):
        raise ValueError("invalid_campaign_identity")
    folder = ROOT / "state/campaigns" / run_id
    folder.mkdir(parents=True, exist_ok=True)
    status_path = folder / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {"run_id": run_id, "owner": owner, "completed": [], "remote": {}}
    cfg = load_config()
    identity = {"prompt": prompt_hash(), "folds": sha256_file(ROOT / "state/folds.csv"),
        "metadata": sha256_file(ROOT / "data/metadata/train.csv"), "lmstudio": cfg.lmstudio.model_dump(),
        "pilot_run_id": pilot_run_id, "labels_run_id": labels_run_id}
    if status.get("identity", identity) != identity or status.get("owner") != owner:
        raise ValueError("campaign_inputs_changed_use_new_run_id")
    status["identity"] = identity

    def save(phase: str, **extra: Any) -> None:
        status.update({"phase": phase, "pid": os.getpid(), **extra})
        write_status(status_path, status)

    def local_labels(phase: str, limit: int, minimum: int) -> Path:
        existing_id = pilot_run_id if phase == "labels-pilot" else labels_run_id
        if existing_id:
            while True:
                ready = completed_label_run(existing_id, minimum)
                if ready is not None:
                    numeric, quality = ready
                    (folder / f"{phase}-quality.json").write_text(json.dumps(quality, indent=2))
                    if phase not in status["completed"]:
                        status["completed"].append(phase)
                    save(phase.upper() + "_COMPLETE", adopted_label_run=existing_id, quality=quality)
                    return numeric
                save("WAITING_FOR_LOCAL_LABELS", waiting_label_run=existing_id)
                time.sleep(45)
        save(phase)
        raw_path = ROOT / "data/labels" / f"{run_id}-{phase}-labels.jsonl"
        if phase not in status["completed"]:
            state = new_state("campaign", synthetic=False, run_id=f"{run_id}-{phase}")
            registry = Registry()
            state = stage_labels(state, registry, limit=limit, live=True, resume=True, development_only=True,
                progress=lambda counts: save(phase, label_progress=counts))
            registry.upsert_run(state.run_id, state.profile, state.stage, state.model_dump(), False)
            quality = label_quality(raw_path, minimum=minimum)
            (folder / f"{phase}-quality.json").write_text(json.dumps(quality, indent=2))
            if state.blocked_reason or not quality["pass"]:
                save("NEEDS_LABEL_REVIEW", blocked_phase=phase, quality=quality)
                raise ValueError("label_quality_gate_failed")
            status["completed"].append(phase)
            save(phase, quality=quality)
        return ROOT / "data/labels" / f"{run_id}-{phase}-training.jsonl"

    def kaggle(args: list[str]) -> str:
        for attempt in range(4):
            result = subprocess.run(["kaggle", *args], capture_output=True, text=True, timeout=300)
            with (folder / "kaggle.log").open("a") as handle:
                handle.write(result.stdout + result.stderr + "\n")
            if not result.returncode:
                return result.stdout
            # Retry only read operations: never resend an ambiguous upload.
            if args[:2] in (["kernels", "status"], ["kernels", "output"]) and "429" in result.stderr + result.stdout and attempt < 3:
                time.sleep(60 * (attempt + 1))
                continue
            raise RuntimeError("kaggle_command_failed_see_local_log")
        raise RuntimeError("kaggle_command_failed_see_local_log")

    def remote(phase: str, labels: Path, cache_ref: str | None = None, cache_hash: str | None = None) -> Path:
        destination = ROOT / "artifacts/kaggle" / f"{run_id}-{phase}"
        output = destination / "output"
        if phase not in status["remote"]:
            if phase != "cache":
                save("CHECKING_GPU_QUOTA", gpu_quota=kaggle(["quota"]).strip())
            prepared = prepare_campaign_kernel(labels, run_id=run_id, owner=owner, dest=destination,
                phase=phase, cache_kernel=cache_ref, cache_sha256=cache_hash)
            # Do not automatically resend an ambiguous network outcome.
            status["remote"][phase] = {"kernel": prepared["kernel"], "push": "UNKNOWN"}
            save(phase.upper() + "_PUSHING")
            receipt = kaggle(["kernels", "push", "-p", str(destination)])
            status["remote"][phase].update({"push": "ACKNOWLEDGED", "receipt": receipt})
            save(phase.upper() + "_RUNNING")
        elif status["remote"][phase]["push"] == "UNKNOWN":
            raise RuntimeError("kernel_push_unknown_reconcile_before_retry")
        reference = status["remote"][phase]["kernel"]
        if phase not in status["completed"]:
            while True:
                observed = kaggle(["kernels", "status", reference])
                save(phase.upper() + "_RUNNING", remote_status=observed.strip())
                if "complete" in observed.lower():
                    break
                if "error" in observed.lower() or "cancel" in observed.lower():
                    raise RuntimeError("remote_kernel_failed")
                time.sleep(45)
            pattern = r"campaign-result\.json$|handshake\.json$|metrics\.json$|predictions\.jsonl$|checkpoint\.pt$|(^|/)manifest\.json$"
            kaggle(["kernels", "output", reference, "-p", str(output), "--file-pattern", pattern])
            record = json.loads((output / "campaign-result.json").read_text())
            if phase == "cache":
                if record.get("n_quarantine") or not record.get("manifest_sha256"):
                    raise ValueError("remote_cache_not_verified")
                manifests = list(output.rglob("manifest.json"))
                if not any(sha256_file(p) == record["manifest_sha256"] for p in manifests):
                    raise ValueError("downloaded_manifest_hash_mismatch")
            elif record.get("exit_reason") != "ok":
                raise ValueError("remote_training_not_successful")
            elif (record.get("metrics") or {}).get("kind") != "real":
                raise ValueError("remote_metrics_not_real")
            status["completed"].append(phase)
            save(phase.upper() + "_COMPLETE")
        return output

    with FileLock(str(folder / "controller.lock"), timeout=0):
        try:
            # Existing successes are cached; this examines at least 100 new/cached
            # non-gold reports before bulk extraction can begin.
            local_labels("labels-pilot", 100, 95)
            weak = local_labels("labels-full", 10**9, 1000)
            labels = folder / "numeric-training.jsonl"
            status["numeric_labels"] = numeric_training_labels(weak, labels)
            cached = remote("cache", labels)
            cache_summary = json.loads((cached / "campaign-result.json").read_text())
            cache_ref = status["remote"]["cache"]["kernel"]
            remote("frozen", labels, cache_ref, cache_summary["manifest_sha256"])
            remote("finetune", labels, cache_ref, cache_summary["manifest_sha256"])
            save("NEEDS_EXPERIMENT_REVIEW", next_action="Review real metrics, per-target coverage, uncertainty, and runtime before OOF expansion or ensemble.",
                competition_submitted=False)
        except Exception as exc:
            if status.get("phase") != "NEEDS_LABEL_REVIEW":
                save("NEEDS_ATTENTION", error_type=type(exc).__name__, reason=str(exc) if isinstance(exc, (ValueError, RuntimeError)) else "local_runtime_error")
            raise
    return status
