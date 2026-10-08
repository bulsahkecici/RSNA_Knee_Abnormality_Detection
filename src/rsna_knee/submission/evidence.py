"""Submission eligibility from downloaded worker and offline runtime evidence.

Eligibility confirms a real, reproducible code submission. It does not promote a
small pilot to a champion or claim competition performance from its holdout AUC.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from rsna_knee.agents.auditor import audit_run
from rsna_knee.config import load_config
from rsna_knee.data.folds import assert_group_integrity, assert_train_uids, load_folds
from rsna_knee.hashing import sha256_file
from rsna_knee.ontology import ONTOLOGY_VERSION, TARGET_COLUMNS
from rsna_knee.paths import METADATA_DIR
from rsna_knee.runtime.train_payload import assert_split_disjoint, pilot_splits
from rsna_knee.submission.package import verify_package
from rsna_knee.submission.validate import validate_submission
from rsna_knee.workflow.state import PipelineState


def audit_candidate(state: PipelineState) -> dict[str, Any]:
    artifacts = state.artifacts
    files: dict[str, str] = {}

    def file(key: str) -> Path:
        path = Path(artifacts.get(key) or "__missing__")
        if not path.is_file():
            raise ValueError(f"missing:{key}")
        files[str(path)] = sha256_file(path)
        return path

    def read(key: str) -> dict[str, Any]:
        data = json.loads(file(key).read_text())
        if not isinstance(data, dict):
            raise ValueError(f"not_object:{key}")
        return data

    try:
        if state.synthetic:
            raise ValueError("synthetic_namespace")
        metrics = read("metrics")
        result = read("worker_result")
        job = read("job")
        proof = read("offline_proof")
        inference = read("inference_report")
        remote_meta = read("inference_metadata")
        package_tar = file("package_tar")
        checkpoint = file("checkpoint")
        package = Path(artifacts.get("package") or "__missing__")
        manifest_path = package / "MANIFEST.json"
        manifest = json.loads(manifest_path.read_text())
        files[str(manifest_path)] = sha256_file(manifest_path)
        identity = verify_package(package)
        if manifest.get("run_id") != state.run_id:
            identity.append("package_run_id")
        checkpoint_hash = sha256_file(checkpoint)
        if checkpoint_hash != proof.get("checkpoint_sha256"):
            identity.append("offline_checkpoint_hash")
        if checkpoint_hash != metrics.get("oof_provenance", {}).get("checkpoint_sha256"):
            identity.append("evaluated_checkpoint_hash")
        packaged_checkpoint = package / str(manifest.get("checkpoint"))
        if not packaged_checkpoint.is_file() or sha256_file(packaged_checkpoint) != checkpoint_hash:
            identity.append("packaged_checkpoint_hash")
        status = file("inference_status").read_text()
        kernel = artifacts.get("inference_kernel")
        version = artifacts.get("inference_version")
        if remote_meta.get("id") != kernel or "COMPLETE" not in status or str(kernel) not in status:
            identity.append("inference_kernel_not_complete")
        # Confirm the submitted notebook source, not just its execution summary.
        local_notebook = json.loads((package / "notebooks/04_kaggle_inference.ipynb").read_text())
        remote_notebook_path = file("inference_notebook")
        remote_notebook = json.loads(remote_notebook_path.read_text())
        local_code = ["".join(c["source"]) for c in local_notebook["cells"] if c["cell_type"] == "code"]
        remote_code = ["".join(c["source"]) for c in remote_notebook["cells"] if c["cell_type"] == "code"]
        if local_code != remote_code:
            identity.append("remote_notebook_source")

        folds_path = file("folds")
        labels_path = file("labels")
        folds = load_folds(folds_path)
        assert_group_integrity(folds)
        trained = metrics.get("train_uids") or []
        _, weak, gold = pilot_splits(folds)
        assert_train_uids(set(trained), folds)
        assert_split_disjoint(trained, weak, gold, folds)
        for resource, path in (("folds", folds_path), ("labels", labels_path)):
            if sha256_file(path) != job.get("resources", {}).get(resource, {}).get("sha256"):
                raise ValueError(f"training_{resource}_changed")
        if set(trained) != set(job.get("train_uids") or []):
            raise ValueError("trained_uids_differ_from_job")
        if result.get("run_id") != state.run_id or result.get("exit_reason") != "ok":
            raise ValueError("training_worker_not_successful")
        if result.get("job_id") != job.get("job_id") or result.get("fencing_token") != job.get("fencing_token"):
            raise ValueError("training_worker_identity")
        if result.get("metrics", {}).get("steps") != metrics.get("steps"):
            raise ValueError("training_worker_steps")
        if metrics.get("kind") != "real" or int(metrics.get("steps") or 0) <= 0:
            raise ValueError("training_metrics_not_real")
        if int(metrics.get("gold_metrics", {}).get("n_studies") or 0) != len(gold):
            raise ValueError("gold_evaluation_incomplete")
        for line in labels_path.read_text().splitlines():
            row = json.loads(line)
            if row.get("ontology_version") != ONTOLOGY_VERSION:
                raise ValueError("label_ontology_changed")
            for name in TARGET_COLUMNS:
                mask = float(row["y_mask"][name])
                target = row["targets"][name]
                if mask not in (0.0, 1.0) or (mask > 0 and target not in (0, 1)):
                    raise ValueError("invalid_training_mask_or_target")

        cache_path = file("cache_manifest")
        cache = json.loads(cache_path.read_text())
        if sha256_file(cache_path) != job.get("resources", {}).get("cache", {}).get("sha256"):
            raise ValueError("training_cache_changed")
        weights_path = file("weights")
        if sha256_file(weights_path) != job.get("resources", {}).get("weights", {}).get("sha256"):
            raise ValueError("pretraining_weights_changed")
        parity = proof.get("parity_shard_hashes") or {}
        cache_parity = len(parity) >= 30 and proof.get("preprocess_hash") == cache.get("preprocess_hash")
        for uid, digest in parity.items():
            shard = cache_path.parent / f"{uid}.npy"
            cache_parity = cache_parity and shard.is_file() and sha256_file(shard) == digest
            cache_parity = cache_parity and cache.get("studies", {}).get(uid, {}).get("sha256") == digest
            files[str(shard)] = sha256_file(shard)
        runtime = proof.get("benchmark", {}).get("runtime", {})
        elapsed = float(runtime.get("measured_seconds") or 0)
        n_measured = int(runtime.get("n_measured") or 0)
        projected = float(runtime.get("projected_seconds") or math.inf)
        resources_ok = (
            n_measured >= 30 and elapsed > 0 and runtime.get("stopwatch") is True
            and runtime.get("includes_decode_and_load") is True and projected < 9 * 3600
            and math.isclose(projected, elapsed / n_measured * 1300)
            and not proof.get("benchmark", {}).get("quarantine")
        )
        deps = json.loads((package / "asset/offline-deps.json").read_text())
        source_hashes = {
            str(p.relative_to(package / "asset/src")): sha256_file(p)
            for p in sorted((package / "asset/src").rglob("*.py"))
        }
        offline_ok = (
            remote_meta.get("enable_internet") is False
            and proof.get("source_hashes") == source_hashes
            and bool(deps.get("wheel_hashes"))
            and proof.get("wheel_hashes") == deps.get("wheel_hashes")
            and proof.get("internet_required") is False
            and proof.get("lmstudio_required") is False
            and proof.get("drive_required") is False
            and all(proof.get("runtime_packages", {}).get(name) for name in deps.get("preinstalled_runtime", []))
        )
        submission = file("submission")
        test_csv = METADATA_DIR / "test.csv"
        files[str(test_csv)] = sha256_file(test_csv)
        test_series = METADATA_DIR / "test_series.csv"
        files[str(test_series)] = sha256_file(test_series)
        with test_csv.open(newline="") as handle:
            test_uids = [r["StudyInstanceUID"] for r in csv.DictReader(handle)]
        submission_ok = (
            validate_submission(submission, test_uids)["ok"]
            and sha256_file(submission) == proof.get("test_submission_sha256")
            and sha256_file(test_csv) == proof.get("test_metadata_sha256")
            and sha256_file(test_series) == proof.get("test_series_sha256")
            and inference.get("n_studies") == len(test_uids) and not inference.get("quarantine")
        )
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
        return {"overall": "BLOCKED", "run_id": state.run_id, "reason": str(exc), "evidence_files": files}

    # Pages have a list topology; read it independently of object artifacts.
    return _finish_audit(state, files, metrics, identity, checkpoint_hash, package_tar, kernel, version,
                         cache_parity, resources_ok, offline_ok, submission_ok, submission)


def _finish_audit(state, files, metrics, identity, checkpoint_hash, package_tar, kernel, version,
                  cache_parity, resources_ok, offline_ok, submission_ok, submission):
    try:
        pages_path = Path(state.artifacts["competition_pages"])
        pages = json.loads(pages_path.read_text())
        requirements = next(p["content"] for p in pages if p["name"] == "Code Requirements")
        host_path = Path(state.artifacts["host_criteria"])
        host = host_path.read_text()
        files[str(pages_path)] = sha256_file(pages_path)
        files[str(host_path)] = sha256_file(host_path)
        ontology_ok = (((load_config().raw or {}).get("labels") or {}).get("source_confirmed") is True and "Topic #733343" in host
                       and "ACL tear: A high-grade partial" in host and "borderline" in host)
        external_ok = ("publicly available external data is allowed" in requirements
                       and "including pre-trained models" in requirements
                       and metrics.get("weights_reason") in {"allowlisted", "official_hash"})
        metadata_digest = sha256_file(METADATA_DIR / "train.csv")
        files[str(METADATA_DIR / "train.csv")] = metadata_digest
        with (METADATA_DIR / "train.csv").open(newline="") as handle:
            columns = next(csv.reader(handle))
        if not {"StudyInstanceUID", "Report", *TARGET_COLUMNS}.issubset(columns):
            raise ValueError("metadata_schema_missing_columns")
    except (OSError, KeyError, StopIteration, TypeError, ValueError) as exc:
        return {"overall": "BLOCKED", "run_id": state.run_id, "reason": str(exc), "evidence_files": files}
    payload = {
        "run_id": state.run_id, "synthetic": False, "metrics": metrics,
        "hashes": {**state.hashes, "metadata": metadata_digest},
        "ontology_confirmed": ontology_ok, "leakage": "PASS",
        "cache_parity": "PASS" if cache_parity else "FAIL",
        "resources": "PASS" if resources_ok else "FAIL",
        "external_data": "PASS" if external_ok else "BLOCKED",
        "offline": "PASS" if offline_ok else "FAIL", "identity_errors": identity,
        "submission_path": str(submission) if submission_ok else "__invalid__",
        "checkpoint_sha256": checkpoint_hash, "package_sha256": sha256_file(package_tar),
        "kernel": kernel, "version": version,
    }
    audited = audit_run(payload, production=True)
    audited.update(evidence_files=files, submission_eligible=audited["overall"] == "PASS",
                   champion=False, scope="first real pilot submission; no performance threshold asserted")
    return audited
