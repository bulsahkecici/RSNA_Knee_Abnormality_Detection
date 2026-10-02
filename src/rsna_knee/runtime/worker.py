"""Remote worker: handshake, job consume, heartbeat, fenced results."""

from __future__ import annotations

import json
import os
import platform
from pathlib import Path
from typing import Any

import torch

from rsna_knee.gates import assert_real_encoder
from rsna_knee.hashing import sha256_file, sha256_json
from rsna_knee.models.dinov2 import load_dinov2_vits14
from rsna_knee.models.study import StudyModel
from rsna_knee.models.weights import weight_trust
from rsna_knee.runtime.budget import execution_allowed
from rsna_knee.runtime.jobs import resolve_checkpoint_out, validate_job
from rsna_knee.runtime.train_payload import heldout_batches, prepare_training
from rsna_knee.runtime.transport import FileTransport
from rsna_knee.training.loop import train_study_model
from rsna_knee.training.train import train_tiny
from rsna_knee.workflow.state import SCHEMA_VERSION, ResultMetrics, ResultRecord


def handshake() -> dict[str, Any]:
    gpu = "cpu"
    vram = None
    dtype = {"fp16": False, "bf16": False}
    try:
        import torch

        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory
            dtype["fp16"] = True
            dtype["bf16"] = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
        elif torch.backends.mps.is_available():
            gpu = "mps"
    except Exception:
        pass
    return {
        "gpu": gpu,
        "vram": vram,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dtype": dtype,
        "disk_free": _disk_free(),
        "pid": os.getpid(),
    }


def _disk_free() -> int | None:
    try:
        st = os.statvfs(".")
        return st.f_bavail * st.f_frsize
    except Exception:
        return None


def accept_job(job: dict[str, Any], expected_token: str | None) -> dict[str, Any] | None:
    if expected_token and job.get("fencing_token") != expected_token:
        return None
    return job


def _finish_training(job: dict[str, Any], job_dir: Path) -> tuple[str, dict[str, Any], list[str]]:
    payload, reason = prepare_training(job, job_dir)
    if payload is None:
        if reason == "no_items":
            return "no_items", {"kind": "unavailable", "macro_auc": None, "notes": "worker has no studies"}, []
        if reason and reason.startswith("leakage"):
            return "leakage", {"kind": "unavailable", "macro_auc": None, "notes": reason}, []
        return reason or "no_items", {"kind": "unavailable", "macro_auc": None, "notes": reason or "inputs missing"}, []
    trust = weight_trust(payload["weights"])
    if not trust["allowlisted"] and not trust["official"]:
        return (
            "unverified_weights",
            {
                "kind": "unavailable",
                "macro_auc": None,
                "notes": "random or unverified weights are not pretrained DINOv2",
                "weights_sha256": trust["sha256"],
            },
            [],
        )
    train_cfg = dict(job.get("train") or {})
    allowed, budget_reason = execution_allowed(
        profile=str(train_cfg.get("profile") or "pilot"),
        n_studies=len(payload["train_uids"]),
        img_size=int(payload["img_size"]),
        cuda=bool(torch.cuda.is_available()),
        slots=int(payload.get("slots") or 3),
        centers=int(payload.get("centers") or 3),
    )
    if not allowed:
        return budget_reason, {"kind": "unavailable", "macro_auc": None, "notes": budget_reason}, []
    supervised = 0.0
    for batch in payload["batches"]:
        supervised += float(batch["y_mask"].sum())
    if supervised <= 0:
        return "no_supervision", {"kind": "unavailable", "macro_auc": None, "notes": "mask and weight sum is zero", "supervised_targets": 0}, []
    encoder = load_dinov2_vits14(payload["weights"], img_size=payload["img_size"])
    if encoder.provenance.get("pretrained") and not trust["official"]:
        return "unverified_weights", {"kind": "unavailable", "macro_auc": None, "notes": "pretrained flag without official hash"}, []
    out_path = resolve_checkpoint_out(job, job_dir)
    before = sha256_file(out_path) if out_path.is_file() else None
    model = StudyModel(encoder, freeze_encoder=True)
    trained = train_study_model(
        model,
        payload["batches"],
        epochs=int(train_cfg.get("epochs", 1)),
        effective_batch=int(train_cfg.get("effective_batch", 1)),
        microbatch=int(train_cfg.get("microbatch", 1)),
        seed=int(train_cfg.get("seed", 0)),
        unfreeze_after_steps=train_cfg.get("unfreeze_after_steps"),
        interrupt_after_steps=train_cfg.get("interrupt_after_steps"),
        ckpt_path=out_path,
        resume=bool(train_cfg.get("resume")) and out_path.is_file(),
        input_id=payload["input_id"],
        config_id=payload["config_id"],
    )
    after = sha256_file(out_path) if out_path.is_file() else None
    if float(trained.get("supervised_weight") or 0) <= 0 or trained["step"] < 1 or after is None or (after == before and not train_cfg.get("resume")):
        reason = "no_supervision" if float(trained.get("supervised_weight") or 0) <= 0 else "no_checkpoint"
        return reason, {"kind": "unavailable", "macro_auc": None, "notes": reason, "supervised_weight": trained.get("supervised_weight")}, []
    from rsna_knee.evaluation.heldout import evaluate_checkpoint

    metrics_path = job_dir / "metrics.json"
    eval_metrics = evaluate_checkpoint(
        out_path,
        train_batches=None,
        weak_batches=heldout_batches(payload, payload["weak_uids"]),
        gold_batches=heldout_batches(payload, payload["gold_uids"]),
        dest=metrics_path,
        img_size=payload["img_size"],
        scope=payload["scope"],
    )
    eval_metrics["steps"] = trained["step"]
    eval_metrics["supervised_weight"] = trained.get("supervised_weight")
    eval_metrics["phase"] = train_cfg.get("phase")
    eval_metrics["resumed"] = bool(train_cfg.get("resume"))
    eval_metrics["weights_official"] = bool(trust["official"])
    eval_metrics["weights_reason"] = trust["reason"]
    eval_metrics["train_uids"] = payload["train_uids"]
    eval_metrics["notes"] = (
        "optimizer update from allowlisted weights; not an official-hash claim"
        if not trust["official"]
        else "optimizer update from official DINOv2 hash"
    )
    eval_metrics["kind"] = "real"
    metrics_path.write_text(json.dumps(eval_metrics), encoding="utf-8")
    return "ok", eval_metrics, [str(out_path), str(metrics_path)]


def run_job(
    job: dict[str, Any],
    transport: FileTransport,
    items: list[dict[str, Any]] | None = None,
    *,
    expected_token: str | None = None,
) -> ResultRecord:
    errors = validate_job(job, expected_token=expected_token)
    synthetic = bool(job.get("synthetic"))
    encoder = job.get("encoder")
    hs = handshake()
    exit_reason = "ok"
    checkpoint_uris: list[str] = []
    metrics: dict[str, Any]
    job_dir = Path(job["bundle_path"]).parent if job.get("bundle_path") else Path(".")
    if errors:
        exit_reason = "rejected"
        metrics = {"kind": "unavailable", "macro_auc": None, "notes": ",".join(errors)}
    elif not synthetic:
        try:
            assert_real_encoder(encoder)
        except Exception as exc:  # noqa: BLE001
            exit_reason = "rejected_encoder"
            metrics = {"kind": "unavailable", "macro_auc": None, "notes": str(exc)}
        else:
            if items is not None and not job.get("resources"):
                exit_reason = "rejected"
                metrics = {
                    "kind": "unavailable",
                    "macro_auc": None,
                    "notes": "existing checkpoint or raw items are not a training run",
                }
            else:
                exit_reason, metrics, checkpoint_uris = _finish_training(job, job_dir)
    elif items is None and not (job.get("resources") or {}).get("cache"):
        exit_reason = "no_items"
        metrics = {"kind": "unavailable", "macro_auc": None, "notes": "worker has no studies"}
    else:
        out = train_tiny(items or [], epochs=1, ckpt_dir=None)
        if out.get("step", 0) <= 0:
            exit_reason = "no_checkpoint"
            metrics = {"kind": "unavailable", "macro_auc": None, "notes": "training produced no step"}
        else:
            metrics = {
                "kind": "synthetic",
                "macro_auc": None,
                "notes": "tiny test step only; not a DINOv2 result",
                "steps": out["step"],
            }
    if exit_reason == "ok" and not synthetic and not checkpoint_uris:
        exit_reason = "no_checkpoint"
        metrics = {"kind": "unavailable", "macro_auc": None, "notes": "production run produced no checkpoint"}
    transport.write_heartbeat(job["job_id"], job["attempt_id"], {"stage": exit_reason, "handshake": hs})
    rec = ResultRecord(
        schema_version=SCHEMA_VERSION,
        job_id=job["job_id"],
        run_id=job["run_id"],
        attempt_id=job["attempt_id"],
        fencing_token=job["fencing_token"],
        handshake=hs,
        metrics=ResultMetrics.model_validate(metrics),
        checkpoint_uris=checkpoint_uris,
        artifact_hashes={"job": sha256_json(job)},
        exit_reason=exit_reason,
    )
    transport.write_result(rec.model_dump())
    return rec


def load_job_bundle(path: Path) -> dict[str, Any]:
    from rsna_knee.runtime.jobs import resolve_bundle_path

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    resolved = resolve_bundle_path(data, Path(path))
    if resolved is not None and resolved.is_file():
        data["bundle_path"] = str(resolved)
    data["_bundle_sha256"] = sha256_file(Path(path))
    return data
