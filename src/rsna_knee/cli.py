"""rsna CLI. Runtime entrypoint independent of Cursor chat."""

from __future__ import annotations

import json
import time
from pathlib import Path

import typer
from rich.console import Console

from rsna_knee.config import load_config, load_experiment
from rsna_knee.data.folds import create_folds
from rsna_knee.data.metadata import audit_metadata, fetch_metadata_csvs
from rsna_knee.paths import ensure_runtime_dirs
from rsna_knee.pipeline import run_pipeline, stage_cache, stage_evaluate, stage_labels
from rsna_knee.roots import roots_for
from rsna_knee.runtime.gpu_policy import GpuAllocator
from rsna_knee.runtime.handoff import write_handoff
from rsna_knee.runtime.preflight import doctor as run_doctor
from rsna_knee.runtime.providers.colab import ColabProvider
from rsna_knee.runtime.providers.kaggle import KaggleProvider
from rsna_knee.runtime.providers.local import LocalProvider
from rsna_knee.runtime.transport import FileTransport
from rsna_knee.runtime.worker import load_job_bundle, run_job
from rsna_knee.submission.kaggle import submit_run
from rsna_knee.submission.package import package_run
from rsna_knee.workflow.graph import new_state
from rsna_knee.workflow.locks import new_run_id
from rsna_knee.workflow.registry import Registry
from rsna_knee.workflow.state import PipelineState, Stage

app = typer.Typer(no_args_is_help=True, add_completion=False, help="RSNA Knee durable pipeline")
metadata_app = typer.Typer(help="Metadata CSV fetch/audit")
labels_app = typer.Typer(help="Local LM Studio report extraction")
cache_app = typer.Typer(help="DICOM cache")
runtime_app = typer.Typer(help="GPU/runtime providers")
experiment_app = typer.Typer(help="Predefined experiments")
pipeline_app = typer.Typer(help="Durable pipeline")
app.add_typer(metadata_app, name="metadata")
app.add_typer(labels_app, name="labels")
app.add_typer(cache_app, name="cache")
app.add_typer(runtime_app, name="runtime")
app.add_typer(experiment_app, name="experiment")
app.add_typer(pipeline_app, name="pipeline")
console = Console()


def _print(obj: object) -> None:
    console.print_json(json.dumps(obj, default=str))


def _artifact_json(state: PipelineState, key: str) -> object:
    raw = state.artifacts.get(key)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _print_labels(state: PipelineState, live: bool) -> None:
    probe = _artifact_json(state, "json_probe")
    probe_dict = probe if isinstance(probe, dict) else {}
    _print(
        {
            "run_id": state.run_id,
            "stage": str(state.stage),
            "labels": state.artifacts.get("labels"),
            "live": live,
            "reason": state.blocked_reason,
            "next": state.next_action_tr,
            "http_error": probe_dict.get("http_error"),
            "empty_reason": probe_dict.get("empty_reason"),
            "json_probe": probe,
            "label_errors": _artifact_json(state, "label_errors"),
            "response_fields": _artifact_json(state, "response_fields"),
            "label_counts": _artifact_json(state, "label_counts"),
            "label_selection": state.artifacts.get("label_selection"),
        }
    )


@app.command("doctor")
def doctor_cmd() -> None:
    """Measure the real machine and provider capabilities."""
    _print(run_doctor())


@app.command("campaign")
def campaign_cmd(
    run_id: str = typer.Option(..., "--run-id"),
    owner: str = typer.Option(..., "--owner"),
    pilot_run_id: str | None = typer.Option(None, "--pilot-run-id"),
    labels_run_id: str | None = typer.Option(None, "--labels-run-id"),
) -> None:
    """Run local label gates, a private remote cache, and two GPU controls."""
    from rsna_knee.runtime.campaign_controller import run_campaign

    _print(run_campaign(run_id, owner, pilot_run_id=pilot_run_id, labels_run_id=labels_run_id))


@metadata_app.command("fetch")
def metadata_fetch() -> None:
    _print(fetch_metadata_csvs())


@metadata_app.command("audit")
def metadata_audit() -> None:
    _print(audit_metadata())


@app.command("folds")
def folds_cmd(action: str = typer.Argument("create")) -> None:
    if action != "create":
        raise typer.BadParameter("create")
    _print(create_folds())


@labels_app.command("pilot")
def labels_pilot(
    limit: int = typer.Option(100, "--limit"),
    live: bool = typer.Option(False, "--live"),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
    run_id: str | None = typer.Option(None, "--run-id"),
    development_only: bool = typer.Option(False, "--development-only"),
    progress_file: Path | None = typer.Option(None, "--progress-file"),
) -> None:
    ensure_runtime_dirs()
    registry = Registry()
    state = new_state("pilot", synthetic=not live, run_id=run_id)
    def report_progress(counts: dict) -> None:
        console.print(json.dumps({"label_progress": counts}))
        if progress_file:
            progress_file.parent.mkdir(parents=True, exist_ok=True)
            temp = progress_file.with_suffix(".tmp")
            temp.write_text(json.dumps({"run_id": state.run_id, "updated_ts": time.time(), **counts}))
            temp.replace(progress_file)

    state = stage_labels(
        state, registry, limit=limit, live=live, resume=resume,
        development_only=development_only,
        progress=report_progress,
    )
    registry.upsert_run(state.run_id, state.profile, state.stage, state.model_dump(), state.synthetic)
    _print_labels(state, live)


@labels_app.command("run")
def labels_run(
    resume: bool = typer.Option(True, "--resume/--no-resume"),
    live: bool = typer.Option(False, "--live"),
    run_id: str | None = typer.Option(None, "--run-id"),
    development_only: bool = typer.Option(False, "--development-only"),
    progress_file: Path | None = typer.Option(None, "--progress-file"),
) -> None:
    labels_pilot(limit=10**9, live=live, resume=resume, run_id=run_id, development_only=development_only, progress_file=progress_file)


@cache_app.command("pilot")
def cache_pilot(limit: int = typer.Option(20, "--limit")) -> None:
    registry = Registry()
    state = new_state("pilot", synthetic=True)
    state = stage_cache(state, registry, limit=limit)
    _print({"stage": str(state.stage), "index": state.artifacts.get("cache_index")})


@cache_app.command("build")
def cache_build(
    provider: str = typer.Option("kaggle", "--provider"),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
    series_csv: Path | None = typer.Option(None, "--series-csv"),
    dicom_root: Path | None = typer.Option(None, "--dicom-root"),
    dest: Path | None = typer.Option(None, "--dest"),
    limit: int | None = typer.Option(None, "--limit"),
) -> None:
    from rsna_knee.data.cache_build import build_cache
    from rsna_knee.runtime.kaggle_kernel import prepare_cache_kernel

    if series_csv and dicom_root:
        manifest = build_cache(
            series_csv=series_csv,
            dicom_root=dicom_root,
            dest=dest or (Path("data/cache/pilot")),
            limit=limit,
            resume=resume,
        )
        _print(
            {
                "provider": provider,
                "status": "built",
                "n_studies": manifest["n_studies"],
                "n_quarantine": manifest["n_quarantine"],
                "manifest": manifest["path"],
                "downloaded_full_archive": False,
            }
        )
        return
    prepared = prepare_cache_kernel()
    prepared["resume"] = resume
    prepared["executed"] = False
    _print(prepared)


@runtime_app.command("capabilities")
def runtime_capabilities() -> None:
    _print(
        {
            "local": LocalProvider().capabilities().as_dict(),
            "colab": ColabProvider().capabilities().as_dict(),
            "kaggle": KaggleProvider().capabilities().as_dict(),
        }
    )


@runtime_app.command("acquire")
def runtime_acquire(
    gpu_order: str = typer.Option("A100,L4,T4", "--gpu-order"),
    wait_seconds: int = typer.Option(300, "--wait-seconds"),
) -> None:
    cfg = load_config()
    cfg.runtime.allocation.max_wait_per_gpu_seconds = wait_seconds
    provider_cls = {"colab": ColabProvider, "kaggle": KaggleProvider, "local": LocalProvider}[cfg.runtime.provider]
    order = [g.strip() for g in gpu_order.split(",") if g.strip()]
    result = GpuAllocator(provider_cls(), cfg.runtime.allocation).acquire(order)
    path = write_handoff(result, new_run_id("acquire"))
    _print(
        {
            "status": result.status,
            "gpu": result.gpu,
            "handoff": str(path),
            "next": result.next_action_tr,
            "attempts": result.attempts,
            "automatic": ColabProvider().capabilities().automatic_acquisition,
        }
    )


@app.command("worker")
def worker_cmd(job_bundle: Path = typer.Option(..., "--job-bundle")) -> None:
    job = load_job_bundle(job_bundle)
    rec = run_job(job, FileTransport())
    from rsna_knee.runtime.ingest import accept_into_registry

    accepted = accept_into_registry(Registry(), job, rec.model_dump(), job_path=job_bundle)
    payload = rec.model_dump()
    payload["registry"] = accepted
    _print(payload)


@experiment_app.command("run")
def experiment_run(config: Path = typer.Option(..., "--config")) -> None:
    from rsna_knee.runtime.jobs import build_job

    exp = load_experiment(config)
    registry = Registry()
    state = new_state("pilot", synthetic=False)
    state.stage = Stage.NEEDS_RUNTIME
    from rsna_knee.pipeline import resolve_train_config

    train = resolve_train_config(state.profile, exp)
    job_dir = roots_for(False).runs / state.run_id / "job"
    job = build_job(
        run_id=state.run_id,
        stage="TRAINING",
        inputs={"config": str(config)},
        dest=job_dir,
        synthetic=False,
        fencing_token=state.fencing_token,
        resources={"config": str(config)},
        train=train,
        checkpoint_out="checkpoint.pt",
    )
    state.artifacts["job"] = str(job_dir / "job.json")
    state.artifacts["experiment"] = str(config)
    state.blocked_reason = "cuda_required_for_dinov2"
    state.next_action_tr = "Deney config doğrulandı. TinyEncoder çalıştırılmadı."
    registry.upsert_run(state.run_id, state.profile, state.stage, state.model_dump(), False)
    _print(
        {
            "experiment": exp.get("experiment_id"),
            "defined_status": exp.get("status"),
            "run_id": state.run_id,
            "trained": False,
            "tiny_artifact": False,
            "encoder": job["encoder"],
            "job": state.artifacts["job"],
            "note": "Gerçek DINOv2 eğitimi remote GPU job'u olmadan tamamlanmış sayılmaz.",
        }
    )


@app.command("evaluate")
def evaluate_cmd(run_id: str = typer.Option(..., "--run-id")) -> None:
    registry = Registry()
    run = registry.get_run(run_id)
    if run is None:
        registry = Registry(roots_for(True).registry)
        run = registry.get_run(run_id)
    ckpt = (run or {}).get("artifacts", {}).get("checkpoint") if run else None
    if not run or not ckpt or not Path(ckpt).is_file():
        _print({"stage": "BLOCKED", "reason": "evaluate_requires_saved_checkpoint", "run_id": run_id})
        return
    state = PipelineState.model_validate(run)
    state = stage_evaluate(state, registry)
    registry.upsert_run(state.run_id, state.profile, state.stage, state.model_dump(), state.synthetic)
    metrics: dict = {}
    path = state.artifacts.get("metrics")
    if path and Path(path).is_file():
        metrics = json.loads(Path(path).read_text(encoding="utf-8"))
    _print(
        {
            "run_id": run_id,
            "stage": str(state.stage),
            "reason": state.blocked_reason,
            "checkpoint": ckpt,
            "predictions": state.artifacts.get("predictions"),
            "gold_macro_auc": (metrics.get("gold_metrics") or {}).get("macro_auc"),
            "weak_macro_auc": (metrics.get("weak_metrics") or {}).get("macro_auc"),
            "note": metrics.get("note"),
        }
    )


@app.command("audit")
def audit_cmd(run_id: str = typer.Option(..., "--run-id")) -> None:
    from rsna_knee.agents.auditor import revalidate_audit

    registry = Registry()
    run = registry.get_run(run_id) or Registry(roots_for(True).registry).get_run(run_id) or {}
    if not run.get("synthetic") and (run.get("artifacts") or {}).get("offline_proof"):
        from rsna_knee.pipeline import stage_audit

        previous_stage = PipelineState.model_validate(run).stage
        state = stage_audit(PipelineState.model_validate(run), registry)
        result = json.loads(Path(state.artifacts["audit"]).read_text())
        if result.get("overall") == "PASS":
            state.stage = previous_stage if previous_stage in {Stage.SUBMITTED, Stage.SCORED} else Stage.READY_TO_SUBMIT
            state.artifacts["kernel"] = result["kernel"]
            state.artifacts["kernel_version"] = str(result["version"])
            package = Path(state.artifacts["package"])
            package.joinpath("FROZEN").write_text(result["package_sha256"])
        registry.upsert_run(state.run_id, state.profile, state.stage, state.model_dump(), False)
        _print(result)
        return
    _print(revalidate_audit(run))


@app.command("package")
def package_cmd(run_id: str = typer.Option(..., "--run-id")) -> None:
    from rsna_knee.pipeline import stage_package

    registry = Registry()
    run = registry.get_run(run_id)
    if run is None:
        registry = Registry(roots_for(True).registry)
        run = registry.get_run(run_id)
    if run is None:
        _print(package_run(run_id))
        return
    state = PipelineState.model_validate(run)
    state = stage_package(state, registry)
    registry.upsert_run(state.run_id, state.profile, state.stage, state.model_dump(), state.synthetic)
    _print(
        {
            "run_id": run_id,
            "stage": str(state.stage),
            "reason": state.blocked_reason,
            "package": state.artifacts.get("package"),
            "audit": state.artifacts.get("audit"),
            "checkpoint": state.artifacts.get("checkpoint"),
        }
    )


@app.command("submit")
def submit_cmd(
    run_id: str = typer.Option(..., "--run-id"),
    kernel: str = typer.Option(..., "--kernel"),
    version: str = typer.Option(..., "--version"),
    message: str = typer.Option("rsna-knee", "--message"),
    execute: bool = typer.Option(False, "--submit"),
    reconcile: bool = typer.Option(False, "--reconcile"),
) -> None:
    _print(submit_run(run_id, kernel, version, message, execute=execute, reconcile=reconcile))


@pipeline_app.command("run")
def pipeline_run(
    profile: str = typer.Option("pilot", "--profile"),
    resume: bool = typer.Option(False, "--resume"),
    run_id: str | None = typer.Option(None, "--run-id"),
    synthetic: bool = typer.Option(False, "--synthetic"),
) -> None:
    syn = synthetic or profile == "smoke"
    state = run_pipeline(profile, synthetic=syn, resume=resume, run_id=run_id)
    _print(
        {
            "run_id": state.run_id,
            "stage": str(state.stage),
            "next": state.next_action_tr,
            "artifacts": state.artifacts,
            "synthetic": state.synthetic,
        }
    )


@app.command("submission-status")
def submission_status_cmd(
    run_id: str = typer.Option(..., "--run-id"),
    watch: bool = typer.Option(False, "--watch"),
    interval: float = typer.Option(30.0, "--interval", min=5.0, max=60.0),
) -> None:
    from rsna_knee.submission.status import poll_score

    while True:
        result = poll_score(run_id)
        _print({key: result.get(key) for key in ("run_id", "status", "server_receipt", "public_score", "reason")})
        if not watch or result.get("status") != "PENDING":
            return
        time.sleep(interval)


@pipeline_app.command("pause")
def pipeline_pause(run_id: str = typer.Option(..., "--run-id")) -> None:
    reg = Registry()
    found = reg.get_run(run_id)
    synthetic = bool(found and found.get("synthetic"))
    if found is None:
        reg = Registry(roots_for(True).registry)
        found = reg.get_run(run_id)
        synthetic = found is not None
    path = roots_for(synthetic).control
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"run_id": run_id, "command": "pause"}), encoding="utf-8")
    if found:
        found["pause_requested"] = True
        reg.upsert_run(
            run_id,
            found.get("profile") or "pilot",
            found.get("stage") or Stage.CREATED,
            found,
            bool(found.get("synthetic")),
        )
    _print({"control": str(path), "run_id": run_id, "payload_preserved": found is not None})


@app.command("status")
def status_cmd(watch: bool = typer.Option(False, "--watch"), interval: float = typer.Option(2.0, "--interval")) -> None:
    registry = Registry()

    def once() -> dict:
        run = registry.latest_run()
        if not run:
            return {"stage": "CREATED", "next_action_tr": "rsna doctor → rsna smoke --synthetic"}
        return {
            "run_id": run.get("run_id"),
            "stage": run.get("stage"),
            "gpu_desired": run.get("gpu_desired"),
            "gpu_actual": run.get("gpu_actual"),
            "heartbeat": run.get("last_heartbeat"),
            "next_action_tr": run.get("next_action_tr"),
            "synthetic": run.get("synthetic"),
            "artifacts": run.get("artifacts"),
        }

    if not watch:
        _print(once())
        return
    try:
        while True:
            _print(once())
            time.sleep(interval)
    except KeyboardInterrupt:
        _print({"stopped": True})


@app.command("smoke")
def smoke_cmd(synthetic: bool = typer.Option(True, "--synthetic/--live")) -> None:
    state = run_pipeline("smoke", synthetic=True, resume=False)
    _print(
        {
            "run_id": state.run_id,
            "stage": str(state.stage),
            "synthetic": True,
            "metrics_kind": "synthetic",
            "note": "Sentetik AUC gerçek yarışma skoru değildir ve production gate'e geçemez.",
            "artifacts": state.artifacts,
            "next": "rsna doctor → local labels pilot → connect remote worker → rsna pipeline run --profile pilot",
        }
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
