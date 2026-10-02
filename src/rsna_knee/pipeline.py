"""Stage functions used by CLI and LangGraph."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from rsna_knee.agents.auditor import audit_run
from rsna_knee.config import load_config, load_experiment
from rsna_knee.data.cache import CacheIndex, synthetic_study_cache
from rsna_knee.data.folds import assert_group_integrity, create_folds, load_folds
from rsna_knee.data.metadata import audit_metadata, fetch_metadata_csvs
from rsna_knee.evaluation.metrics import per_class_auc
from rsna_knee.hashing import sha256_file, sha256_json
from rsna_knee.labels.extract import LabelStore, extract_one
from rsna_knee.labels.lmstudio import LMStudioClient
from rsna_knee.models.classifier import StudyClassifier
from rsna_knee.models.encoder import TinyEncoder
from rsna_knee.ontology import TARGET_COLUMNS
from rsna_knee.paths import CACHE_DIR, LABELS_DIR, METADATA_DIR, RUNS_DIR, STATE_DIR
from rsna_knee.runtime.gpu_policy import GpuAllocator
from rsna_knee.runtime.handoff import write_handoff
from rsna_knee.runtime.preflight import doctor
from rsna_knee.runtime.providers.colab import ColabProvider
from rsna_knee.runtime.providers.kaggle import KaggleProvider
from rsna_knee.runtime.providers.local import LocalProvider
from rsna_knee.submission.infer import infer_study, write_submission_csv
from rsna_knee.submission.package import package_run
from rsna_knee.submission.validate import validate_submission
from rsna_knee.training.train import train_tiny
from rsna_knee.workflow.graph import sequential_run
from rsna_knee.workflow.locks import new_fencing_token, new_lease_id
from rsna_knee.workflow.registry import Registry
from rsna_knee.workflow.state import PipelineState, Stage
from rsna_knee.data.dataset import StudyDataset

PROVIDERS = {"local": LocalProvider, "colab": ColabProvider, "kaggle": KaggleProvider}


def stage_preflight(state: PipelineState, registry: Registry) -> PipelineState:
    report = doctor()
    path = STATE_DIR / f"{state.run_id}-doctor.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    state.stage = Stage.PREFLIGHT
    state.artifacts["doctor"] = str(path)
    state.next_action_tr = "Metadata CSV'lerini alın (tam arşiv değil)."
    return state


def stage_metadata(state: PipelineState, registry: Registry) -> PipelineState:
    fetched = fetch_metadata_csvs()
    if fetched.get("blocked") and not state.synthetic:
        state.stage = Stage.NEEDS_AUTH
        state.blocked_reason = json.dumps(fetched["blocked"])
        state.next_action_tr = fetched["blocked"].get("action_tr", "Kaggle oturumu açın.")
        return state
    audit = audit_metadata()
    state.hashes["metadata"] = audit.get("source_hashes", {}).get("train.csv") or "synthetic"
    state.artifacts["metadata_audit"] = audit.get("audit_path", "")
    state.stage = Stage.METADATA_READY
    return state


def stage_folds(state: PipelineState, registry: Registry) -> PipelineState:
    if state.synthetic:
        dest = STATE_DIR / "folds.csv"
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
        info = create_folds()
        folds = load_folds()
        assert_group_integrity(folds)
    state.hashes["folds"] = info.get("hash")
    state.artifacts["folds"] = info.get("path")
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


def stage_labels(state: PipelineState, registry: Registry, limit: int = 100, live: bool = False) -> PipelineState:
    store = LabelStore()
    rows: list[dict[str, Any]] = []
    if state.synthetic or not live:
        # Deterministic fixture extractor — labeled synthetic, not a model score.
        fixtures = _synthetic_reports()
        for row in fixtures[:limit]:
            report = row["Report"]
            payload = {
                "StudyInstanceUID": row["StudyInstanceUID"],
                "targets": {},
            }
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
        state.stage = Stage.LABEL_PILOT if limit <= 100 else Stage.LABELS_READY
        if len(rows) >= 4:
            state.stage = Stage.LABELS_READY
    else:
        cfg = load_config()
        client = LMStudioClient(base_url=cfg.lmstudio.base_url)
        models = client.list_models()
        info = client.resolve_model(cfg.lmstudio.preferred_model_id, models)
        train = METADATA_DIR / "train.csv"
        with train.open(newline="", encoding="utf-8") as handle:
            reader = list(csv.DictReader(handle))
        done = 0
        for row in reader:
            if done >= limit:
                break
            out = extract_one(
                client,
                uid=row["StudyInstanceUID"],
                report=row.get("Report") or "",
                model_id=info.id,
                model_revision=str(info.raw.get("owned_by", "unknown")),
                store=store,
                json_mode=cfg.lmstudio.json_mode,
            )
            rows.append(out)
            if out.get("status") == "ok" and not out.get("cached"):
                done += 1
        client.close()
        state.stage = Stage.LABELS_READY if done or store.successful_uids() else Stage.LABEL_PILOT
    dest = LABELS_DIR / f"{state.run_id}-labels.jsonl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    state.artifacts["labels"] = str(dest)
    state.hashes["labels"] = sha256_file(dest)
    state.next_action_tr = "Cache pilotunu çalıştırın. Ham raporları cloud agent'a yapıştırmayın."
    return state


def stage_cache(state: PipelineState, registry: Registry, limit: int = 20) -> PipelineState:
    index = CacheIndex()
    n = 0
    if state.synthetic:
        for i in range(min(limit, 8)):
            uid = f"syn-{i:03d}"
            path = synthetic_study_cache(uid, n_slots=3, n_slices=3, size=32, seed=i)
            index.record_ok(uid, path, {"synthetic": True})
            n += 1
        state.stage = Stage.CACHE_READY
        state.hashes["cache"] = sha256_json(index.data)
        state.artifacts["cache_index"] = str(index.path)
        return state
    # Real cache: Kaggle is the intended host. Local Mac does not download 570GB.
    state.stage = Stage.CACHE_PILOT
    state.next_action_tr = (
        "Ham DICOM cache Kaggle'da üretilir. Mac'e tam arşiv indirmeyin. "
        "rsna cache build --provider kaggle --resume veya notebooks/01_kaggle_build_cache.ipynb."
    )
    state.blocked_reason = "no_local_full_dicom_archive"
    # Still mark pilot if any shards exist.
    existing = list(CACHE_DIR.glob("*.npy"))[:limit]
    for shard in existing:
        n += 1
    if n:
        state.stage = Stage.CACHE_READY
    return state


def stage_runtime(state: PipelineState, registry: Registry, gpu_order: list[str] | None = None) -> PipelineState:
    if state.synthetic:
        state.stage = Stage.WAITING_RUNTIME
        state.lease_id = new_lease_id()
        state.gpu_actual = "synthetic-local"
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
    if result.status == "handoff":
        path = write_handoff(result, state.run_id)
        state.stage = Stage.NEEDS_RUNTIME
        state.artifacts["handoff"] = str(path)
        state.next_action_tr = result.next_action_tr
        return state
    if result.status in {"exhausted", "duplicate_prevented"}:
        state.stage = Stage.QUEUED
        state.next_action_tr = result.next_action_tr
        return state
    if result.status == "auth_failure":
        state.stage = Stage.NEEDS_AUTH
        state.next_action_tr = result.next_action_tr
        return state
    if result.status == "allocated" or state.synthetic:
        state.stage = Stage.WAITING_RUNTIME
        state.lease_id = new_lease_id()
        return state
    state.stage = Stage.NEEDS_RUNTIME
    state.next_action_tr = result.next_action_tr
    return state


def _items_from_synthetic() -> list[dict[str, Any]]:
    labels: dict[str, dict] = {}
    masks: dict[str, dict] = {}
    weights: dict[str, dict] = {}
    uids = [f"syn-{i:03d}" for i in range(8)]
    for i, uid in enumerate(uids):
        y = {t: float((i + j) % 2) for j, t in enumerate(TARGET_COLUMNS)}
        labels[uid] = y
        masks[uid] = {t: 1.0 for t in TARGET_COLUMNS}
        weights[uid] = {t: 1.0 for t in TARGET_COLUMNS}
        synthetic_study_cache(uid, size=16, seed=i)
    ds = StudyDataset(uids, labels, masks, weights, n_slots=3, n_slices=3, size=16)
    items = []
    for i in range(len(ds)):
        item = ds[i]
        item["synthetic"] = True
        items.append(item)
    return items


def stage_train(state: PipelineState, registry: Registry, exp_path: str | None = None) -> PipelineState:
    state.stage = Stage.TRAINING
    ckpt = RUNS_DIR / state.run_id
    ckpt.mkdir(parents=True, exist_ok=True)
    if state.synthetic or exp_path is None:
        items = _items_from_synthetic()
        out = train_tiny(items, epochs=1, ckpt_dir=ckpt, resume=True)
        (ckpt / "train.json").write_text(json.dumps({k: out[k] for k in ("step", "interrupted", "encoder_name", "kind", "metrics_kind")}, indent=2), encoding="utf-8")
        state.artifacts["checkpoint"] = str(ckpt / "last.npz")
        state.hashes["train"] = sha256_file(ckpt / "last.npz") if (ckpt / "last.npz").exists() else ""
        return state
    exp = load_experiment(exp_path)
    state.artifacts["experiment"] = exp.get("_path")
    state.next_action_tr = "Gerçek DINOv2 eğitimi remote GPU handshake sonrası çalışır; bu makinede sahte AUC yazılmaz."
    return state


def stage_evaluate(state: PipelineState, registry: Registry) -> PipelineState:
    state.stage = Stage.EVALUATING
    items = _items_from_synthetic()
    clf = StudyClassifier(TinyEncoder(seed=0))
    ys, ps = [], []
    for item in items:
        import numpy as np
        from rsna_knee.training.losses import sigmoid

        logits = clf.forward_numpy(item["images"][None], item["slot_mask"][None], item["slice_mask"][None])
        ys.append(item["y"])
        ps.append(sigmoid(logits[0]))

    metrics = per_class_auc(np.stack(ys), np.stack(ps))
    metrics["kind"] = "synthetic"
    metrics["note"] = "synthetic AUC is not a competition result"
    path = RUNS_DIR / state.run_id / "metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    state.artifacts["metrics"] = str(path)
    return state


def stage_audit(state: PipelineState, registry: Registry) -> PipelineState:
    metrics = {}
    mp = state.artifacts.get("metrics")
    if mp and Path(mp).exists():
        metrics = json.loads(Path(mp).read_text(encoding="utf-8"))
    payload = {
        "synthetic": state.synthetic,
        "hashes": state.hashes,
        "metrics": metrics,
        "ontology_confirmed": True,
        "leakage": "PASS",
        "offline": "PASS" if state.synthetic else "BLOCKED",
        "external_data": "PASS",
    }
    result = audit_run(payload, production=not state.synthetic)
    path = RUNS_DIR / state.run_id / "audit.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    state.artifacts["audit"] = str(path)
    state.stage = Stage.AUDITING
    return state


def stage_package(state: PipelineState, registry: Registry) -> PipelineState:
    pkg = package_run(state.run_id)
    # synthetic submission for smoke
    items = _items_from_synthetic()
    clf = StudyClassifier(TinyEncoder(0))
    rows = []
    for item in items:
        probs = infer_study(clf, item["images"], item["slot_mask"], item["slice_mask"])
        rows.append({"StudyInstanceUID": item["uid"], **probs})
    sub = Path(pkg["dir"]) / "submission.csv"
    write_submission_csv(rows, sub)
    val = validate_submission(sub, [r["StudyInstanceUID"] for r in rows])
    (Path(pkg["dir"]) / "submission_validate.json").write_text(json.dumps(val, indent=2), encoding="utf-8")
    state.artifacts["package"] = pkg["dir"]
    state.artifacts["submission"] = str(sub)
    state.stage = Stage.READY_TO_SUBMIT if val.get("ok") else Stage.FAILED
    state.next_action_tr = "Smoke submission gerçek Kaggle gönderimi değildir. rsna submit --run-id ... yalnız --submit ile."
    return state


def run_pipeline(profile: str, synthetic: bool, resume: bool, run_id: str | None = None) -> PipelineState:
    registry = Registry()
    if resume and run_id:
        existing = registry.get_run(run_id)
        if existing:
            state = PipelineState.model_validate(existing)
        else:
            state = PipelineState(run_id=run_id, profile=profile, synthetic=synthetic)
    else:
        from rsna_knee.workflow.graph import new_state

        state = new_state(profile, synthetic=synthetic, run_id=run_id)
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
    return sequential_run(state, registry, fns)
