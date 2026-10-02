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
from rsna_knee.pipeline import run_pipeline, stage_cache, stage_labels
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
from rsna_knee.workflow.state import Stage

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


@app.command("doctor")
def doctor_cmd() -> None:
    """Measure the real machine and provider capabilities."""
    _print(run_doctor())


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
) -> None:
    ensure_runtime_dirs()
    registry = Registry()
    state = new_state("pilot", synthetic=not live)
    state = stage_labels(state, registry, limit=limit, live=live, resume=resume)
    registry.upsert_run(state.run_id, state.profile, state.stage, state.model_dump(), state.synthetic)
    _print({"run_id": state.run_id, "stage": str(state.stage), "labels": state.artifacts.get("labels"), "live": live})


@labels_app.command("run")
def labels_run(resume: bool = typer.Option(True, "--resume/--no-resume"), live: bool = typer.Option(False, "--live")) -> None:
    labels_pilot(limit=10**9, live=live, resume=resume)


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
    _print(rec.model_dump())


@experiment_app.command("run")
def experiment_run(config: Path = typer.Option(..., "--config")) -> None:
    from rsna_knee.runtime.jobs import build_job

    exp = load_experiment(config)
    registry = Registry()
    state = new_state("pilot", synthetic=False)
    state.stage = Stage.NEEDS_RUNTIME
    job_dir = roots_for(False).runs / state.run_id / "job"
    job = build_job(
        run_id=state.run_id,
        stage="TRAINING",
        inputs={"config": str(config)},
        dest=job_dir,
        synthetic=False,
        fencing_token=state.fencing_token,
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
    run = Registry().get_run(run_id) or Registry(roots_for(True).registry).get_run(run_id)
    ckpt = (run or {}).get("artifacts", {}).get("checkpoint") if run else None
    if not run or not ckpt or not Path(ckpt).is_file():
        _print({"stage": "BLOCKED", "reason": "evaluate_requires_saved_checkpoint", "run_id": run_id})
        return
    _print({"run_id": run_id, "checkpoint": ckpt, "encoder": (run.get("artifacts") or {}).get("encoder")})


@app.command("audit")
def audit_cmd(run_id: str = typer.Option(..., "--run-id")) -> None:
    run = Registry().get_run(run_id) or Registry(roots_for(True).registry).get_run(run_id) or {}
    path = (run.get("artifacts") or {}).get("audit")
    if not path or not Path(path).is_file():
        _print({"overall": "BLOCKED", "reason": "audit_artifact_missing", "run_id": run_id})
        return
    _print(json.loads(Path(path).read_text(encoding="utf-8")))


@app.command("package")
def package_cmd(run_id: str = typer.Option(..., "--run-id")) -> None:
    _print(package_run(run_id))


@app.command("submit")
def submit_cmd(
    run_id: str = typer.Option(..., "--run-id"),
    kernel: str = typer.Option(..., "--kernel"),
    version: str = typer.Option(..., "--version"),
    message: str = typer.Option("rsna-knee", "--message"),
    execute: bool = typer.Option(False, "--submit"),
) -> None:
    _print(submit_run(run_id, kernel, version, message, execute=execute))


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
