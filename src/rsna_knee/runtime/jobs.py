"""Job bundle build and validation. A job does not start with a hash or token mismatch."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rsna_knee.hashing import sha256_file, sha256_json
from rsna_knee.workflow.locks import new_fencing_token, new_lease_id
from rsna_knee.workflow.state import SCHEMA_VERSION


def _resource(path: str | Path, input_root: Path | None) -> dict[str, Any]:
    file_path = Path(path)
    relative = file_path.name
    if input_root is not None:
        try:
            relative = str(file_path.resolve().relative_to(input_root.resolve()))
        except ValueError:
            relative = file_path.name
    return {
        "uri": str(file_path),
        "relative": relative,
        "sha256": sha256_file(file_path) if file_path.is_file() else None,
        "input_root": None if input_root is None else str(input_root),
    }


SPEC_KEYS = (
    "schema_version",
    "job_id",
    "run_id",
    "attempt_id",
    "stage",
    "bundle_sha256",
    "resources",
    "train",
    "train_uids",
    "checkpoint_out",
    "input_root",
    "fencing_token",
    "synthetic",
    "encoder",
)


def job_spec(job: dict[str, Any]) -> dict[str, Any]:
    return {key: job.get(key) for key in SPEC_KEYS}


def refresh_spec(job: dict[str, Any]) -> dict[str, Any]:
    job["spec_sha256"] = sha256_json(job_spec(job))
    return job


def _checkpoint_name(checkpoint_out: str | None) -> str:
    if not checkpoint_out:
        return "checkpoint.pt"
    return Path(checkpoint_out).name


def resolve_bundle_path(job: dict[str, Any], job_json: Path | None = None) -> Path | None:
    """Find the manifest after the bundle directory has moved."""
    import os

    raw = job.get("bundle_path")
    if raw and Path(str(raw)).is_file():
        return Path(str(raw))
    if job_json is not None:
        sibling = job_json.parent / "input_manifest.json"
        if sibling.is_file():
            return sibling
    env = os.environ.get("RSNA_INPUT_ROOT")
    if env:
        for candidate in (Path(env) / "input_manifest.json", Path(env) / "job" / "input_manifest.json"):
            if candidate.is_file():
                return candidate
    return Path(str(raw)) if raw else None


def resolve_checkpoint_out(job: dict[str, Any], job_dir: Path) -> Path:
    """Write next to the relocated bundle. A Mac absolute path is not the Colab output."""
    import os

    name = _checkpoint_name(job.get("checkpoint_out"))
    env = os.environ.get("RSNA_INPUT_ROOT")
    if env:
        root = Path(env)
        root.mkdir(parents=True, exist_ok=True)
        return root / name
    raw = Path(str(job.get("checkpoint_out") or name))
    if raw.is_absolute() and raw.parent.resolve() == job_dir.resolve():
        return raw
    return job_dir / name


def build_job(
    *,
    run_id: str,
    stage: str,
    inputs: dict[str, str],
    dest: Path,
    synthetic: bool = False,
    fencing_token: str | None = None,
    attempt_id: str | None = None,
    resources: dict[str, str] | None = None,
    train: dict[str, Any] | None = None,
    train_uids: list[str] | None = None,
    input_root: Path | None = None,
    checkpoint_out: str | None = None,
) -> dict[str, Any]:
    dest.mkdir(parents=True, exist_ok=True)
    resolved = {key: _resource(path, input_root) for key, path in (resources or {}).items()}
    checkpoint_name = _checkpoint_name(checkpoint_out)
    train_spec = dict(train or {})
    manifest = {
        "run_id": run_id,
        "stage": stage,
        "synthetic": synthetic,
        "inputs": inputs,
        "resources": resolved,
        "train": train_spec,
        "train_uids": train_uids,
        "checkpoint_out": checkpoint_name,
        "encoder": "tiny_test_encoder" if synthetic else "dinov2_vits14",
    }
    man_path = dest / "input_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    digest = sha256_file(man_path)
    attempt = attempt_id or new_lease_id()
    token = fencing_token or new_fencing_token(run_id, attempt)
    job = {
        "schema_version": SCHEMA_VERSION,
        "job_id": f"job-{run_id}-{stage}",
        "run_id": run_id,
        "attempt_id": attempt,
        "stage": stage,
        "bundle_sha256": digest,
        "bundle_path": str(man_path),
        "config_hash": (resolved.get("config") or {}).get("sha256") or inputs.get("config", ""),
        "fold_hash": (resolved.get("folds") or {}).get("sha256") or inputs.get("folds", ""),
        "labels_hash": (resolved.get("labels") or {}).get("sha256") or inputs.get("labels", ""),
        "cache_hash": (resolved.get("cache") or {}).get("sha256") or inputs.get("cache", ""),
        "input_uris": [str(man_path), *[item["uri"] for item in resolved.values()]],
        "resources": resolved,
        "train": train_spec,
        "train_uids": train_uids,
        "checkpoint_out": checkpoint_name,
        "input_root": None if input_root is None else str(input_root),
        "lease_id": new_lease_id(),
        "fencing_token": token,
        "synthetic": synthetic,
        "encoder": "tiny_test_encoder" if synthetic else "dinov2_vits14",
    }
    refresh_spec(job)
    job_path = dest / "job.json"
    job_path.write_text(json.dumps(job, indent=2), encoding="utf-8")
    job["job_sha256"] = sha256_json({key: value for key, value in job.items() if key != "job_sha256"})
    return job


def validate_job(job: dict[str, Any], *, expected_token: str | None = None) -> list[str]:
    errors: list[str] = []
    if job.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema")
    for key in ("job_id", "run_id", "attempt_id", "fencing_token", "bundle_sha256"):
        if not job.get(key):
            errors.append(f"missing:{key}")
    if expected_token is not None and job.get("fencing_token") != expected_token:
        errors.append("fencing_token")
    bundle = resolve_bundle_path(job)
    if job.get("bundle_path"):
        if bundle is None or not bundle.is_file():
            errors.append("bundle_missing")
        elif sha256_file(bundle) != job.get("bundle_sha256"):
            errors.append("bundle_hash")
        else:
            manifest = json.loads(bundle.read_text(encoding="utf-8"))
            if "train" in manifest and manifest.get("train") != job.get("train"):
                errors.append("spec_mismatch:train")
    if job.get("spec_sha256") or not job.get("synthetic"):
        if sha256_json(job_spec(job)) != job.get("spec_sha256"):
            errors.append("spec_hash")
    return errors
