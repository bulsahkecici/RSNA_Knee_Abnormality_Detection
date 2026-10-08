"""Stage functions used by CLI and LangGraph."""

from __future__ import annotations

import csv
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from rsna_knee.agents.auditor import audit_run
from rsna_knee.config import load_config
from rsna_knee.data.cache import CacheIndex, synthetic_study_cache
from rsna_knee.data.dataset import StudyDataset
from rsna_knee.data.folds import (
    assert_group_integrity,
    assert_train_uids,
    create_folds,
    image_training_uids,
    load_folds,
)
from rsna_knee.data.metadata import audit_metadata, fetch_metadata_csvs
from rsna_knee.errors import ContractError
from rsna_knee.evaluation.metrics import per_class_auc
from rsna_knee.gates import assert_real_encoder, assert_real_uids, looks_synthetic_uid
from rsna_knee.hashing import sha256_file, sha256_json
from rsna_knee.labels.extract import LabelStore, extract_one
from rsna_knee.labels.lmstudio import LMStudioClient
from rsna_knee.models.classifier import LinearHead, StudyClassifier
from rsna_knee.models.encoder import TinyEncoder
from rsna_knee.ontology import TARGET_COLUMNS
from rsna_knee.paths import METADATA_DIR
from rsna_knee.roots import roots_for
from rsna_knee.runtime.gpu_policy import GpuAllocator
from rsna_knee.runtime.handoff import write_handoff
from rsna_knee.runtime.jobs import build_job
from rsna_knee.runtime.preflight import doctor
from rsna_knee.runtime.providers.colab import ColabProvider
from rsna_knee.runtime.providers.kaggle import KaggleProvider
from rsna_knee.runtime.providers.local import LocalProvider
from rsna_knee.submission.infer import infer_study, write_submission_csv
from rsna_knee.submission.package import package_run
from rsna_knee.submission.validate import validate_submission
from rsna_knee.training.checkpoint import load_numpy_checkpoint
from rsna_knee.training.losses import sigmoid
from rsna_knee.training.train import train_tiny
from rsna_knee.workflow.graph import sequential_run
from rsna_knee.workflow.locks import ControllerLock, new_fencing_token, new_lease_id
from rsna_knee.workflow.registry import Registry
from rsna_knee.workflow.state import PipelineState, Stage

PROVIDERS = {"local": LocalProvider, "colab": ColabProvider, "kaggle": KaggleProvider}


def _roots(state: PipelineState):
    return roots_for(state.synthetic)


def _reject_fixtures(state: PipelineState, uids: list[str], encoder: str | None) -> str | None:
    if state.synthetic:
        return None
    try:
        assert_real_uids(uids)
        if encoder is not None:
            assert_real_encoder(encoder)
    except ContractError as exc:
        state.stage = Stage.BLOCKED
        state.blocked_reason = str(exc)
        state.next_action_tr = "Sentetik UID veya TinyEncoder gerçek çalışmada kullanılamaz."
        return str(exc)
    return None


def stage_preflight(state: PipelineState, registry: Registry) -> PipelineState:
    report = doctor(probe_remote=not state.synthetic)
    path = _roots(state).state / f"{state.run_id}-doctor.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    state.stage = Stage.PREFLIGHT
    state.artifacts["doctor"] = str(path)
    state.hashes["preflight"] = sha256_file(path)
    state.next_action_tr = "Metadata CSV'lerini alın (tam arşiv değil)."
    return state


def stage_metadata(state: PipelineState, registry: Registry) -> PipelineState:
    if state.synthetic:
        state.hashes["metadata"] = "synthetic"
        state.stage = Stage.METADATA_READY
        state.next_action_tr = "Sentetik metadata. Gerçek train.csv'ye dokunulmadı."
        return state
    fetched = fetch_metadata_csvs()
    if fetched.get("blocked"):
        state.stage = Stage.NEEDS_AUTH
        state.blocked_reason = json.dumps(fetched["blocked"])
        state.next_action_tr = fetched["blocked"].get("action_tr", "Kaggle oturumu açın.")
        return state
    audit = audit_metadata()
    digest = audit.get("source_hashes", {}).get("train.csv")
    if not digest:
        state.stage = Stage.BLOCKED
        state.blocked_reason = "metadata_hash_missing"
        return state
    state.hashes["metadata"] = digest
    state.artifacts["metadata_audit"] = audit.get("audit_path", "")
    state.stage = Stage.METADATA_READY
    return state


def stage_folds(state: PipelineState, registry: Registry) -> PipelineState:
    roots = _roots(state)
    if state.synthetic:
        dest = roots.folds
        dest.parent.mkdir(parents=True, exist_ok=True)
        uids = [f"syn-{i:03d}" for i in range(8)]
        with dest.open("w", newline="", encoding="utf-8") as handle:
            w = csv.DictWriter(handle, fieldnames=["StudyInstanceUID", "fold", "is_gold", "gold_split", "group_id"])
            w.writeheader()
            for i, uid in enumerate(uids):
                w.writerow(
                    {
                        "StudyInstanceUID": uid,
                        "fold": i % 4,
                        "is_gold": int(i < 4),
                        "gold_split": "eval" if i < 2 else ("train_image_ok" if i < 4 else "weak"),
                        "group_id": uid,
                    }
                )
        info = {"path": str(dest), "hash": sha256_file(dest), "synthetic": True}
    else:
        if not (METADATA_DIR / "train.csv").exists():
            state.stage = Stage.NEEDS_AUTH
            state.blocked_reason = "train_csv_missing"
            return state
        info = create_folds(dest=roots.folds)
        folds = load_folds(roots.folds)
        assert_group_integrity(folds)
        train_uids = set(image_training_uids(folds))
        assert_train_uids(train_uids, folds)
        if any(looks_synthetic_uid(uid) for uid in train_uids):
            state.stage = Stage.BLOCKED
            state.blocked_reason = "synthetic_uid_in_real_folds"
            return state
    state.hashes["folds"] = info.get("hash") or ""
    state.artifacts["folds"] = info.get("path") or ""
    state.stage = Stage.SPLIT_FROZEN
    return state


def _synthetic_reports() -> list[dict[str, str]]:
    return [
        {
            "StudyInstanceUID": "syn-000",
            "Report": "Complete ACL tear with pivot shift bone contusion. MCL intact. No fracture.",
        },
        {
            "StudyInstanceUID": "syn-001",
            "Report": "Medial meniscus posterior horn tear contacting the articular surface. Lateral meniscus normal.",
        },
        {
            "StudyInstanceUID": "syn-002",
            "Report": "Moderate joint effusion. Small Baker cyst. Cartilage otherwise not described.",
        },
        {
            "StudyInstanceUID": "syn-003",
            "Report": "Acute tibial plateau fracture. No mention of synovitis.",
        },
    ]


def resolve_train_config(profile: str, experiment: dict[str, Any] | None = None) -> dict[str, Any]:
    """Profile and experiment YAML supply epochs, batch, and seed. Callers do not hardcode them."""
    cfg = load_config()
    profiles = (cfg.raw or {}).get("profiles") or {}
    chosen = profiles.get(profile) or {}
    data = (cfg.raw or {}).get("data") or {}
    settings: dict[str, Any] = {
        "epochs": int(chosen.get("epochs") or 1),
        "effective_batch": int(cfg.runtime.training.effective_study_batch),
        "microbatch": 1,
        "seed": int(data.get("seed") or (cfg.raw or {}).get("seed") or 0),
        "profile": profile,
        "img_size": int(data.get("image_size") or 224),
    }
    for key in ("epochs", "effective_batch", "microbatch", "seed", "unfreeze_after_steps", "img_size"):
        if experiment and key in (experiment.get("train") or {}) and experiment["train"][key] is not None:
            settings[key] = experiment["train"][key]
    return settings


def _reusable_labels(state: PipelineState) -> Path | None:
    roots = _roots(state)
    candidates = []
    current = state.artifacts.get("labels")
    if current:
        candidates.append(Path(current))
    candidates.append(roots.labels / f"{state.run_id}-training.jsonl")
    for path in candidates:
        if path.is_file() and path.stat().st_size > 0:
            return path
    return None


def _ensure_train_job(state: PipelineState, exp_path: str | None = None) -> dict[str, Any]:
    """Write the hashed job before a runtime handoff so the worker has a bundle to load."""
    from rsna_knee.paths import CONFIG_DIR

    roots = _roots(state)
    ckpt = roots.runs / state.run_id
    ckpt.mkdir(parents=True, exist_ok=True)
    experiment = None
    if exp_path:
        from rsna_knee.config import load_experiment

        experiment = load_experiment(exp_path)
    elif state.artifacts.get("experiment"):
        from rsna_knee.config import load_experiment

        exp_file = Path(state.artifacts["experiment"])
        if exp_file.is_file():
            experiment = load_experiment(exp_file)
    resources: dict[str, str] = {}
    config_path = CONFIG_DIR / "default.yaml"
    if config_path.is_file():
        resources["config"] = str(config_path)
    for key, artifact in (("folds", "folds"), ("labels", "labels"), ("cache", "cache_manifest"), ("weights", "weights")):
        path = state.artifacts.get(artifact)
        if path and Path(path).is_file():
            resources[key] = path
    weights = state.artifacts.get("weights") or os.environ.get("RSNA_DINOV2_WEIGHTS")
    if not weights:
        default_weights = CONFIG_DIR.parent / "artifacts" / "weights" / "dinov2_vits14_pretrain.pth"
        if default_weights.is_file():
            weights = str(default_weights)
    if weights and Path(weights).is_file():
        resources["weights"] = str(weights)
        state.artifacts["weights"] = str(weights)
    train_csv = state.artifacts.get("train_csv")
    if train_csv and Path(train_csv).is_file():
        resources["train_csv"] = str(train_csv)
    elif (METADATA_DIR / "train.csv").is_file():
        try:
            sha256_file(METADATA_DIR / "train.csv")
        except OSError:
            pass
        else:
            resources["train_csv"] = str(METADATA_DIR / "train.csv")
    train = resolve_train_config(state.profile, experiment)
    job = build_job(
        run_id=state.run_id,
        stage="TRAINING",
        inputs={
            "config": state.hashes.get("config", ""),
            "folds": state.hashes.get("folds", ""),
            "labels": state.hashes.get("labels", ""),
            "cache": state.hashes.get("cache", ""),
        },
        dest=ckpt / "job",
        synthetic=False,
        fencing_token=state.fencing_token,
        resources=resources,
        input_root=config_path.parent.parent if config_path.is_file() else None,
        checkpoint_out="checkpoint.pt",
        train=train,
    )
    job_path = ckpt / "job" / "job.json"
    state.artifacts["job"] = str(job_path)
    if exp_path:
        state.artifacts["experiment"] = exp_path
    state.hashes["job"] = job["spec_sha256"]
    return job


def stage_labels(
    state: PipelineState,
    registry: Registry,
    limit: int = 100,
    live: bool = False,
    resume: bool = True,
    development_only: bool = False,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> PipelineState:
    roots = _roots(state)
    rows: list[dict[str, Any]] = []
    if state.synthetic:
        if live:
            state.stage = Stage.BLOCKED
            state.blocked_reason = "synthetic_run_cannot_call_live_llm"
            return state
        fixtures = _synthetic_reports()
        for row in fixtures[:limit]:
            report = row["Report"]
            payload: dict[str, Any] = {"StudyInstanceUID": row["StudyInstanceUID"], "targets": {}}
            for t in TARGET_COLUMNS:
                if t == "ACL" and "ACL tear" in report:
                    payload["targets"][t] = {
                        "state": "criterion_positive",
                        "evidence_span": "Complete ACL tear",
                        "criterion_mapping": "high-grade ACL",
                    }
                elif t == "MCL" and "MCL intact" in report:
                    payload["targets"][t] = {
                        "state": "explicit_negative",
                        "evidence_span": "MCL intact",
                        "criterion_mapping": "explicit intact",
                    }
                elif t == "Fracture" and "No fracture" in report:
                    payload["targets"][t] = {
                        "state": "explicit_negative",
                        "evidence_span": "No fracture",
                        "criterion_mapping": "explicit negation",
                    }
                elif t == "Fracture" and "fracture" in report.lower() and "No fracture" not in report:
                    payload["targets"][t] = {
                        "state": "criterion_positive",
                        "evidence_span": "Acute tibial plateau fracture",
                        "criterion_mapping": "acute cortical break",
                    }
                else:
                    payload["targets"][t] = {
                        "state": "not_mentioned",
                        "evidence_span": "",
                        "criterion_mapping": "unmentioned",
                    }
            rows.append({"uid": row["StudyInstanceUID"], "payload": payload, "status": "ok", "kind": "synthetic_fixture"})
        state.stage = Stage.LABELS_READY
    elif not live:
        existing = _reusable_labels(state)
        if existing is not None:
            state.artifacts["labels"] = str(existing)
            state.hashes["labels"] = sha256_file(existing)
            state.stage = Stage.LABELS_READY
            state.next_action_tr = "Aynı run_id etiket dosyası yeniden kullanıldı. Ham raporlar yeniden gönderilmedi."
            return state
        state.stage = Stage.NEEDS_RUNTIME
        state.blocked_reason = "labels_require_live_lmstudio"
        state.next_action_tr = "rsna labels pilot --live yerel LM Studio ile çalışır. Fixture gerçek etiket dizinine yazılmaz."
        return state
    else:
        cfg = load_config()
        client = LMStudioClient(base_url=cfg.lmstudio.base_url, timeout=cfg.lmstudio.timeout_seconds)
        models = client.list_models()
        info = client.resolve_model(cfg.lmstudio.preferred_model_id, models)
        revision = None
        for key in ("revision", "sha", "version", "modified_at"):
            if info.raw.get(key):
                revision = str(info.raw[key])
                break
        probe = client.probe_json_schema(
            info.id,
            allow_reasoning_json=cfg.lmstudio.accept_reasoning_json,
            thinking=str(cfg.lmstudio.thinking) if cfg.lmstudio.thinking else None,
        )
        state.artifacts["json_probe"] = json.dumps(probe.as_dict(), ensure_ascii=False)
        if probe.mode == "none":
            client.close()
            state.stage = Stage.BLOCKED
            state.blocked_reason = "json_capability_none"
            state.next_action_tr = (
                "LM Studio JSON üretemiyor. "
                f"HTTP: {probe.http_error or 'yok'}. "
                f"Boş yanıt: {probe.empty_reason or 'yok'}. "
                "Sahte etiket yazılmadı."
            )
            return state
        store = LabelStore(roots.labels / "label_cache.sqlite")
        train = METADATA_DIR / "train.csv"
        with train.open(newline="", encoding="utf-8") as handle:
            reader = list(csv.DictReader(handle))
        if development_only:
            folds = load_folds(roots.state / "folds.csv")
            assert_group_integrity(folds)
            allowed = set(image_training_uids(folds))
            gold = {r["StudyInstanceUID"] for r in folds if r.get("is_gold") == "1"}
            reader = [r for r in reader if r["StudyInstanceUID"] in allowed - gold]
            state.artifacts["label_selection"] = "non_gold_development_only"
        done = 0
        attempted = 0
        rejected = 0
        for row in reader:
            if attempted >= limit:
                break
            uid = row["StudyInstanceUID"]
            if looks_synthetic_uid(uid):
                client.close()
                state.stage = Stage.BLOCKED
                state.blocked_reason = "synthetic_uid_in_real_labels"
                return state
            out = extract_one(
                client,
                uid=uid,
                report=row.get("Report") or "",
                model_id=info.id,
                model_revision=revision or "",
                store=store,
                json_mode=probe.mode,
                temperature=cfg.lmstudio.temperature,
                max_retries=cfg.lmstudio.max_retries,
                context_tokens=cfg.lmstudio.context_tokens,
                output_tokens=cfg.lmstudio.output_tokens,
                thinking=str(cfg.lmstudio.thinking),
                use_cache=resume,
                allow_reasoning_json=cfg.lmstudio.accept_reasoning_json,
            )
            rows.append(out)
            if not out.get("cached"):
                attempted += 1
                if out.get("status") == "ok":
                    done += 1
                else:
                    rejected += 1
                if progress:
                    progress({"attempted": attempted, "valid": done, "quarantined": rejected})
                # Stop a failing extractor before it processes thousands of reports.
                if attempted >= 10 and rejected / attempted > 0.2:
                    state.blocked_reason = "label_quality_failure_rate"
                    break
        client.close()
        state.artifacts["label_counts"] = json.dumps({"attempted": attempted, "valid": done, "quarantined": rejected})
        label_errors = [row.get("reason") for row in rows if row.get("status") == "quarantine" and row.get("reason")]
        if label_errors:
            state.artifacts["label_errors"] = json.dumps(label_errors[:8], ensure_ascii=False)
        response_fields = sorted(
            {
                (row.get("provenance") or {}).get("response_field")
                for row in rows
                if isinstance((row.get("provenance") or {}).get("response_field"), str)
            }
        )
        if response_fields:
            state.artifacts["response_fields"] = json.dumps(response_fields)
        state.stage = Stage.LABELS_READY if any(r.get("status") == "ok" for r in rows) else Stage.LABEL_PILOT
        if state.blocked_reason == "label_quality_failure_rate":
            state.stage = Stage.BLOCKED
    dest = roots.labels / f"{state.run_id}-labels.jsonl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    from rsna_knee.labels.canonical import publish_training_labels

    train_csv = None if state.synthetic else METADATA_DIR / "train.csv"
    if train_csv is not None and not train_csv.is_file():
        train_csv = None
    canonical = publish_training_labels(rows, roots.labels / f"{state.run_id}-training.jsonl", train_csv=train_csv)
    state.artifacts["labels_raw"] = str(dest)
    state.artifacts["labels"] = canonical["path"]
    state.hashes["labels"] = sha256_file(Path(canonical["path"]))
    state.next_action_tr = (
        "Etiket hata oranı %20'yi geçti; yerel inceleme gerekir."
        if state.blocked_reason == "label_quality_failure_rate"
        else "Cache pilotunu çalıştırın. Ham raporları cloud agent'a yapıştırmayın."
    )
    return state


def stage_cache(state: PipelineState, registry: Registry, limit: int = 20) -> PipelineState:
    roots = _roots(state)
    index = CacheIndex(roots.cache)
    if state.synthetic:
        for i in range(min(limit, 8)):
            uid = f"syn-{i:03d}"
            path = synthetic_study_cache(uid, n_slots=3, n_slices=3, size=32, seed=i, root=roots.cache)
            index.record_ok(uid, path, {"synthetic": True})
        state.stage = Stage.CACHE_READY
        state.hashes["cache"] = sha256_json(index.data)
        state.artifacts["cache_index"] = str(index.path)
        return state
    manifest = roots.cache / "manifest.json"
    if not manifest.is_file():
        state.stage = Stage.BLOCKED
        state.blocked_reason = "cache_manifest_missing"
        state.next_action_tr = (
            "rsna cache build --dicom-root <mounted-series> gerçek decode çalıştırır. "
            "Mac'e 570GB arşiv indirmeyin. Var olan npy dosyası cache hazır sayılmaz."
        )
        return state
    from rsna_knee.data.cache_build import manifest_ready

    ready, reason = manifest_ready(manifest)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    studies = payload.get("studies") or {}
    if not ready:
        state.stage = Stage.BLOCKED
        state.blocked_reason = reason
        state.artifacts["cache_manifest"] = str(manifest)
        state.next_action_tr = "Manifest shard, hash veya pilot kapsamı doğrulanmadan CACHE_READY olmaz."
        return state
    if any(looks_synthetic_uid(uid) for uid in studies):
        state.stage = Stage.BLOCKED
        state.blocked_reason = "cache_not_real"
        return state
    state.stage = Stage.CACHE_READY
    state.hashes["cache"] = sha256_file(manifest)
    state.artifacts["cache_manifest"] = str(manifest)
    state.artifacts["cache_scope"] = str(payload.get("scope") or "")
    if payload.get("scope") != "full":
        state.next_action_tr = "Cache pilot kapsamındadır. Bu manifest tam dataset cache'i değildir."
    return state


def stage_runtime(state: PipelineState, registry: Registry, gpu_order: list[str] | None = None) -> PipelineState:
    if not state.synthetic:
        _ensure_train_job(state)
    if state.synthetic:
        state.stage = Stage.WAITING_RUNTIME
        state.lease_id = new_lease_id()
        state.gpu_actual = "synthetic-local"
        state.hashes["runtime"] = "synthetic"
        state.next_action_tr = "Sentetik runtime; Colab/Kaggle tahsisi yok."
        return state
    cfg = load_config()
    name = cfg.runtime.provider
    provider = PROVIDERS.get(name, ColabProvider)()
    alloc = GpuAllocator(provider, cfg.runtime.allocation, fencing_token=state.fencing_token or new_fencing_token(state.run_id))
    result = alloc.acquire(gpu_order)
    state.gpu_desired = (gpu_order or cfg.runtime.allocation.preferred_gpu_order)[0]
    state.gpu_actual = result.gpu
    state.fencing_token = result.fencing_token
    state.hashes["runtime"] = result.status
    if result.status == "handoff":
        path = write_handoff(result, state.run_id)
        state.stage = Stage.NEEDS_RUNTIME
        state.artifacts["handoff"] = str(path)
        state.next_action_tr = result.next_action_tr
        return state
    if result.status in {"exhausted", "duplicate_prevented", "ambiguous_pending"}:
        state.stage = Stage.QUEUED if result.status != "ambiguous_pending" else Stage.NEEDS_RUNTIME
        state.blocked_reason = result.status
        state.next_action_tr = result.next_action_tr
        return state
    if result.status == "auth_failure":
        state.stage = Stage.NEEDS_AUTH
        state.next_action_tr = result.next_action_tr
        return state
    if result.status == "allocated":
        state.stage = Stage.WAITING_RUNTIME
        state.lease_id = new_lease_id()
        return state
    state.stage = Stage.NEEDS_RUNTIME
    state.next_action_tr = result.next_action_tr or "Runtime hazır değil."
    return state


def _items_from_synthetic(cache_root: Path) -> list[dict[str, Any]]:
    labels: dict[str, dict] = {}
    masks: dict[str, dict] = {}
    weights: dict[str, dict] = {}
    uids = [f"syn-{i:03d}" for i in range(8)]
    for i, uid in enumerate(uids):
        y = {t: float((i + j) % 2) for j, t in enumerate(TARGET_COLUMNS)}
        labels[uid] = y
        masks[uid] = {t: 1.0 for t in TARGET_COLUMNS}
        weights[uid] = {t: 1.0 for t in TARGET_COLUMNS}
        synthetic_study_cache(uid, size=16, seed=i, root=cache_root)
    ds = StudyDataset(uids, labels, masks, weights, cache_root=cache_root, n_slots=3, n_slices=3, size=16)
    items = []
    for i in range(len(ds)):
        item = ds[i]
        item["synthetic"] = True
        items.append(item)
    return items


def stage_train(state: PipelineState, registry: Registry, exp_path: str | None = None) -> PipelineState:
    roots = _roots(state)
    ckpt = roots.runs / state.run_id
    ckpt.mkdir(parents=True, exist_ok=True)
    if state.synthetic:
        items = _items_from_synthetic(roots.cache)
        out = train_tiny(items, epochs=1, ckpt_dir=ckpt, resume=False)
        (ckpt / "train.json").write_text(
            json.dumps({k: out[k] for k in ("step", "interrupted", "encoder_name", "kind", "metrics_kind")}, indent=2),
            encoding="utf-8",
        )
        state.stage = Stage.TRAINING
        state.artifacts["checkpoint"] = str(ckpt / "last.npz")
        state.hashes["train"] = sha256_file(ckpt / "last.npz") if (ckpt / "last.npz").exists() else ""
        state.artifacts["encoder"] = "tiny_test_encoder"
        return state
    if _reject_fixtures(state, [], "dinov2_vits14"):
        return state
    from rsna_knee.runtime.budget import execution_allowed
    from rsna_knee.runtime.transport import FileTransport
    from rsna_knee.runtime.worker import run_job

    job = _ensure_train_job(state, exp_path=exp_path)
    resources = set(job.get("resources") or {})
    if not {"folds", "labels", "cache", "weights"} <= resources:
        state.stage = Stage.NEEDS_RUNTIME
        state.blocked_reason = "training_inputs_missing"
        state.next_action_tr = (
            "Job bundle yazıldı. Folds, labels, cache ve allowlist'teki DINOv2 ağırlığı gerekir. "
            "Colab eklentisi bağlandıktan sonra rsna worker --job-bundle bu dosyayı yükler."
        )
        return state
    import torch

    train_cfg = job.get("train") or {}
    n_studies = 0
    img_size = int(train_cfg.get("img_size") or 224)
    cache_path = state.artifacts.get("cache_manifest")
    if cache_path and Path(cache_path).is_file():
        manifest = json.loads(Path(cache_path).read_text(encoding="utf-8"))
        n_studies = int(manifest.get("n_studies") or len(manifest.get("studies") or {}))
        shapes = [list((meta or {}).get("shape") or []) for meta in (manifest.get("studies") or {}).values()]
        if shapes and shapes[0]:
            img_size = int(shapes[0][-1])
    allowed, reason = execution_allowed(
        profile=str(train_cfg.get("profile") or state.profile),
        n_studies=n_studies,
        img_size=img_size,
        cuda=bool(torch.cuda.is_available()),
    )
    if not allowed:
        state.stage = Stage.NEEDS_RUNTIME
        state.blocked_reason = reason
        state.next_action_tr = "Bu profil yerel CPU'da çalıştırılmadı. Job bundle duruyor; GPU worker aynı run_id ile devam eder."
        return state
    record = run_job(job, FileTransport(ckpt / "transport"), expected_token=job["fencing_token"])
    state.artifacts["worker_result"] = str(ckpt / "transport" / "inbox")
    if record.exit_reason != "ok":
        state.stage = Stage.NEEDS_RUNTIME if record.exit_reason == "unverified_weights" else Stage.BLOCKED
        state.blocked_reason = record.exit_reason
        state.next_action_tr = record.metrics.notes or "Worker eğitimi tamamlamadı."
        return state
    state.stage = Stage.TRAINING
    state.artifacts["checkpoint"] = record.checkpoint_uris[0]
    state.hashes["train"] = sha256_file(Path(record.checkpoint_uris[0]))
    if len(record.checkpoint_uris) > 1:
        state.artifacts["metrics"] = record.checkpoint_uris[1]
        state.hashes["evaluate"] = sha256_file(Path(record.checkpoint_uris[1]))
    state.artifacts["encoder"] = "dinov2_vits14"
    return state


def stage_evaluate(state: PipelineState, registry: Registry) -> PipelineState:
    roots = _roots(state)
    if not state.synthetic:
        ckpt = state.artifacts.get("checkpoint")
        if not ckpt or not Path(ckpt).is_file():
            state.stage = Stage.BLOCKED
            state.blocked_reason = "evaluate_requires_saved_checkpoint"
            state.next_action_tr = "Evaluate yeni bir model kurmaz. Kayıtlı DINOv2 checkpoint gerekir."
            return state
        from rsna_knee.evaluation.heldout import evaluate_checkpoint
        from rsna_knee.runtime.train_payload import heldout_batches, prepare_training

        job_path = state.artifacts.get("job")
        if not job_path or not Path(job_path).is_file():
            state.stage = Stage.BLOCKED
            state.blocked_reason = "held_out_inputs_missing"
            state.next_action_tr = "Held-out için job bundle içindeki folds, labels ve cache gerekir."
            return state
        from rsna_knee.runtime.worker import load_job_bundle

        job = load_job_bundle(Path(job_path))
        payload, reason = prepare_training(job, Path(job_path).parent)
        if payload is None:
            state.stage = Stage.BLOCKED
            state.blocked_reason = reason or "held_out_inputs_missing"
            return state
        metrics_path = roots.runs / state.run_id / "metrics.json"
        try:
            metrics = evaluate_checkpoint(
                Path(ckpt),
                train_batches=None,
                weak_batches=heldout_batches(payload, payload["weak_uids"]),
                gold_batches=heldout_batches(payload, payload["gold_uids"]),
                dest=metrics_path,
                img_size=payload["img_size"],
                scope=payload["scope"],
            )
        except RuntimeError as exc:
            state.stage = Stage.BLOCKED
            state.blocked_reason = "checkpoint_unreadable"
            state.next_action_tr = str(exc)
            return state
        state.stage = Stage.EVALUATING
        state.artifacts["metrics"] = str(metrics_path)
        state.artifacts["predictions"] = str(metrics.get("predictions") or "")
        state.hashes["evaluate"] = sha256_file(metrics_path)
        return state
    ckpt_path = Path(state.artifacts.get("checkpoint") or "")
    if not ckpt_path.is_file():
        state.stage = Stage.BLOCKED
        state.blocked_reason = "synthetic_checkpoint_missing"
        return state
    saved = load_numpy_checkpoint(ckpt_path)
    items = _items_from_synthetic(roots.cache)
    head = LinearHead(weight=np.asarray(saved["weight"]), bias=np.asarray(saved["bias"]))
    clf = StudyClassifier(TinyEncoder(seed=0), head)
    ys, ps = [], []
    for item in items:
        logits = clf.forward_numpy(item["images"][None], item["slot_mask"][None], item["slice_mask"][None])
        ys.append(item["y"])
        ps.append(sigmoid(logits[0]))
    metrics = per_class_auc(np.stack(ys), np.stack(ps))
    metrics["kind"] = "synthetic"
    metrics["gold_metrics"] = None
    metrics["weak_metrics"] = None
    metrics["oof_provenance"] = {"checkpoint": str(ckpt_path), "encoder": "tiny_test_encoder", "namespace": "synthetic"}
    metrics["note"] = "synthetic AUC is not a competition result"
    path = roots.runs / state.run_id / "metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    state.stage = Stage.EVALUATING
    state.artifacts["metrics"] = str(path)
    state.hashes["evaluate"] = sha256_file(path)
    return state


def stage_audit(state: PipelineState, registry: Registry) -> PipelineState:
    roots = _roots(state)
    if not state.synthetic and state.artifacts.get("offline_proof"):
        from rsna_knee.submission.evidence import audit_candidate

        result = audit_candidate(state)
        path = roots.runs / state.run_id / "audit.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        state.artifacts["audit"] = str(path)
        state.hashes["audit"] = sha256_file(path)
        state.stage = Stage.AUDITING
        state.blocked_reason = None if result.get("overall") == "PASS" else result.get("reason", "audit_blocks_submit")
        return state
    metrics: dict[str, Any] = {}
    mp = state.artifacts.get("metrics")
    if mp and Path(mp).exists():
        metrics = json.loads(Path(mp).read_text(encoding="utf-8"))
    payload = {
        "synthetic": state.synthetic,
        "run_id": state.run_id,
        "hashes": state.hashes,
        "metrics": metrics,
        "ontology_confirmed": True if state.synthetic else None,
        "leakage": "BLOCKED",
        "cache_parity": "BLOCKED",
        "resources": "BLOCKED",
        "external_data": "BLOCKED",
        "offline": "BLOCKED",
        "identity_errors": ["synthetic_namespace"] if state.synthetic else ["not_live_verified"],
        "checkpoint_sha256": state.hashes.get("train"),
        "package_sha256": state.hashes.get("package"),
        "kernel": state.artifacts.get("kernel"),
        "version": state.artifacts.get("kernel_version"),
    }
    result = audit_run(payload, production=True)
    path = roots.runs / state.run_id / "audit.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    state.artifacts["audit"] = str(path)
    state.hashes["audit"] = sha256_file(path)
    state.stage = Stage.AUDITING
    if state.synthetic and result.get("overall") == "PASS":
        state.stage = Stage.BLOCKED
        state.blocked_reason = "synthetic_audit_cannot_pass"
    return state


def stage_package(state: PipelineState, registry: Registry) -> PipelineState:
    roots = _roots(state)
    previous_stage = state.stage
    existing_package = state.artifacts.get("package")
    ckpt_path = Path(state.artifacts["checkpoint"]) if state.artifacts.get("checkpoint") else None
    pkg = package_run(
        state.run_id,
        dest_root=Path(existing_package).parent if existing_package else roots.submissions,
        synthetic=state.synthetic,
        checkpoint=ckpt_path if ckpt_path and ckpt_path.is_file() else None,
        offline_wheels=Path(state.artifacts["offline_wheels"]) if state.artifacts.get("offline_wheels") else None,
        dataset_slug=state.artifacts.get("dataset_slug"),
        kernel_slug=state.artifacts.get("inference_kernel"),
    )
    if state.synthetic:
        items = _items_from_synthetic(roots.cache)
        saved = load_numpy_checkpoint(Path(state.artifacts["checkpoint"]))
        clf = StudyClassifier(TinyEncoder(0), LinearHead(weight=np.asarray(saved["weight"]), bias=np.asarray(saved["bias"])))
        rows = []
        for item in items:
            probs = infer_study(clf, item["images"], item["slot_mask"], item["slice_mask"])
            rows.append({"StudyInstanceUID": item["uid"], **probs})
        sub = Path(pkg["dir"]) / "submission.csv"
        write_submission_csv(rows, sub)
        val = validate_submission(sub, [r["StudyInstanceUID"] for r in rows])
        (Path(pkg["dir"]) / "submission_validate.json").write_text(json.dumps(val, indent=2), encoding="utf-8")
        state.artifacts["submission"] = str(sub)
    state.artifacts["package"] = pkg["dir"]
    state.artifacts["package_tar"] = pkg["tar"]
    state.hashes["package"] = pkg["sha256"]
    copied = (pkg.get("manifest") or {}).get("checkpoint")
    copied_hash = None
    if copied:
        copied_file = Path(pkg["dir"]) / copied
        if copied_file.is_file():
            copied_hash = sha256_file(copied_file)
            state.hashes["checkpoint"] = copied_hash
    identity_errors = list(pkg.get("identity_errors") or [])
    if state.synthetic and "synthetic_namespace" not in identity_errors:
        identity_errors.append("synthetic_namespace")
    elif not identity_errors and not state.artifacts.get("offline_proof"):
        identity_errors.append("not_live_verified")
    kernel = None
    kernel_meta = Path(pkg["dir"]) / "kernel-metadata.json"
    if kernel_meta.is_file():
        kernel = json.loads(kernel_meta.read_text(encoding="utf-8")).get("id")
    metrics: dict[str, Any] = {}
    metrics_path = state.artifacts.get("metrics")
    if metrics_path and Path(metrics_path).is_file():
        metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    payload = {
        "synthetic": state.synthetic,
        "run_id": state.run_id,
        "hashes": state.hashes,
        "metrics": metrics,
        "ontology_confirmed": True if state.synthetic else None,
        "leakage": "BLOCKED",
        "cache_parity": "BLOCKED",
        "resources": "BLOCKED",
        "external_data": "BLOCKED",
        "offline": "BLOCKED",
        "identity_errors": identity_errors,
        "checkpoint_sha256": state.hashes.get("train") or copied_hash,
        "package_sha256": pkg["sha256"],
        "kernel": kernel,
        "version": state.artifacts.get("kernel_version"),
    }
    if not state.synthetic and state.artifacts.get("offline_proof"):
        from rsna_knee.submission.evidence import audit_candidate

        result = audit_candidate(state)
    else:
        result = audit_run(payload, production=True)
    audit_file = roots.runs / state.run_id / "audit.json"
    audit_file.parent.mkdir(parents=True, exist_ok=True)
    audit_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    state.artifacts["audit"] = str(audit_file)
    state.artifacts["kernel"] = kernel
    state.hashes["audit"] = sha256_file(audit_file)
    approved = result.get("overall") == "PASS" and not identity_errors and copied_hash is not None
    if approved:
        (Path(pkg["dir"]) / "FROZEN").write_text(pkg["sha256"], encoding="utf-8")
        state.stage = previous_stage if previous_stage in {Stage.SUBMITTED, Stage.SCORED} else Stage.READY_TO_SUBMIT
    else:
        state.stage = Stage.BLOCKED
        state.blocked_reason = identity_errors[0] if identity_errors else "audit_blocks_submit"
    state.next_action_tr = "Aday paket doğrulandı ve audit yazıldı. Onaydan sonra paket yeniden üretilmez."
    return state


def run_pipeline(profile: str, synthetic: bool, resume: bool, run_id: str | None = None) -> PipelineState:
    roots = roots_for(synthetic)
    roots.ensure()
    registry = Registry(roots.registry if synthetic else None)
    if resume and run_id:
        existing = registry.get_run(run_id)
        if existing:
            state = PipelineState.model_validate(existing)
        else:
            state = PipelineState(run_id=run_id, profile=profile, synthetic=synthetic, resume=True)
    else:
        from rsna_knee.workflow.graph import new_state

        state = new_state(profile, synthetic=synthetic, run_id=run_id)
    state.resume = resume
    state.synthetic = synthetic
    fns = [
        ("preflight", stage_preflight),
        ("metadata", stage_metadata),
        ("folds", stage_folds),
        ("labels", stage_labels),
        ("cache", stage_cache),
        ("runtime", stage_runtime),
        ("train", stage_train),
        ("evaluate", stage_evaluate),
        ("audit", stage_audit),
        ("package", stage_package),
    ]
    lock = ControllerLock(roots.lock)
    return sequential_run(state, registry, fns, lock=lock)
