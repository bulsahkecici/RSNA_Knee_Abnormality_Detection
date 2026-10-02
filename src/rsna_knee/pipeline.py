"""Stage functions used by CLI and LangGraph."""

from __future__ import annotations

import csv
import json
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


def stage_labels(
    state: PipelineState,
    registry: Registry,
    limit: int = 100,
    live: bool = False,
    resume: bool = True,
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
        mode = client.probe_json_schema(info.id)
        if mode == "none":
            client.close()
            state.stage = Stage.BLOCKED
            state.blocked_reason = "json_capability_none"
            state.next_action_tr = "LM Studio JSON üretemiyor. Sahte etiket yazılmadı."
            return state
        store = LabelStore(roots.labels / "label_cache.sqlite")
        train = METADATA_DIR / "train.csv"
        with train.open(newline="", encoding="utf-8") as handle:
            reader = list(csv.DictReader(handle))
        done = 0
        for row in reader:
            if done >= limit:
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
                json_mode=mode,
                temperature=cfg.lmstudio.temperature,
                max_retries=cfg.lmstudio.max_retries,
                context_tokens=cfg.lmstudio.context_tokens,
                output_tokens=cfg.lmstudio.output_tokens,
                thinking=str(cfg.lmstudio.thinking),
                use_cache=resume,
            )
            rows.append(out)
            if out.get("status") == "ok" and not out.get("cached"):
                done += 1
        client.close()
        state.stage = Stage.LABELS_READY if done or store.successful_uids() else Stage.LABEL_PILOT
    dest = roots.labels / f"{state.run_id}-labels.jsonl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    state.artifacts["labels"] = str(dest)
    state.hashes["labels"] = sha256_file(dest)
    state.next_action_tr = "Cache pilotunu çalıştırın. Ham raporları cloud agent'a yapıştırmayın."
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
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    studies = payload.get("studies") or {}
    if not studies or any(looks_synthetic_uid(uid) for uid in studies):
        state.stage = Stage.BLOCKED
        state.blocked_reason = "cache_not_real"
        return state
    state.stage = Stage.CACHE_READY
    state.hashes["cache"] = sha256_file(manifest)
    state.artifacts["cache_manifest"] = str(manifest)
    return state


def stage_runtime(state: PipelineState, registry: Registry, gpu_order: list[str] | None = None) -> PipelineState:
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
    if _reject_fixtures(state, [], "dinov2_vits14") :
        return state
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
    )
    state.artifacts["job"] = str(ckpt / "job" / "job.json")
    state.artifacts["experiment"] = exp_path or ""
    state.hashes["train"] = job["bundle_sha256"]
    cfg = load_config()
    cuda = False
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
    except Exception:
        cuda = False
    if not cuda or not cfg.runtime.fallback_training.cpu_full_training_enabled:
        state.stage = Stage.NEEDS_RUNTIME
        state.blocked_reason = "cuda_required_for_dinov2"
        state.next_action_tr = (
            "Job bundle yazıldı. Colab eklentisinde kernel bağlandıktan sonra "
            "rsna worker --job-bundle çalışır. TinyEncoder ve sahte AUC yazılmadı."
        )
        return state
    state.stage = Stage.NEEDS_RUNTIME
    state.blocked_reason = "remote_dispatch_not_started"
    state.next_action_tr = "CUDA görünüyor ama bu oturum remote eğitim başlatmaz."
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
        state.stage = Stage.BLOCKED
        state.blocked_reason = "held_out_eval_not_run"
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
    audit_path = state.artifacts.get("audit")
    audit = json.loads(Path(audit_path).read_text(encoding="utf-8")) if audit_path and Path(audit_path).is_file() else {"overall": "BLOCKED"}
    pkg = package_run(state.run_id, dest_root=roots.submissions, synthetic=state.synthetic)
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
    state.stage = Stage.READY_TO_SUBMIT if audit.get("overall") == "PASS" else Stage.BLOCKED
    if state.stage == Stage.BLOCKED:
        state.blocked_reason = "audit_blocks_submit"
    state.next_action_tr = "CSV biçimi submit açmaz. Audit PASS ve kimlik doğrulaması gerekir."
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
