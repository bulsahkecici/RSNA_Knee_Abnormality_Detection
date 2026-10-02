"""Job bundle build and validation. A job does not start with a hash or token mismatch."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rsna_knee.hashing import sha256_file, sha256_json
from rsna_knee.workflow.locks import new_fencing_token, new_lease_id
from rsna_knee.workflow.state import SCHEMA_VERSION


def build_job(
    *,
    run_id: str,
    stage: str,
    inputs: dict[str, str],
    dest: Path,
    synthetic: bool = False,
    fencing_token: str | None = None,
    attempt_id: str | None = None,
) -> dict[str, Any]:
    dest.mkdir(parents=True, exist_ok=True)
    manifest = {"run_id": run_id, "stage": stage, "synthetic": synthetic, "inputs": inputs}
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
        "config_hash": inputs.get("config", ""),
        "fold_hash": inputs.get("folds", ""),
        "labels_hash": inputs.get("labels", ""),
        "cache_hash": inputs.get("cache", ""),
        "input_uris": [str(man_path)],
        "lease_id": new_lease_id(),
        "fencing_token": token,
        "synthetic": synthetic,
        "encoder": "tiny_test_encoder" if synthetic else "dinov2_vits14",
    }
    job_path = dest / "job.json"
    job_path.write_text(json.dumps(job, indent=2), encoding="utf-8")
    job["job_sha256"] = sha256_json(job)
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
    bundle = job.get("bundle_path")
    if bundle:
        path = Path(bundle)
        if not path.is_file():
            errors.append("bundle_missing")
        elif sha256_file(path) != job.get("bundle_sha256"):
            errors.append("bundle_hash")
    return errors
