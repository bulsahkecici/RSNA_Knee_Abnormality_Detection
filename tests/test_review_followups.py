from __future__ import annotations

import json
from pathlib import Path

from rsna_knee.cli import experiment_run
from rsna_knee.labels.extract import LabelStore, extract_one
from rsna_knee.labels.lmstudio import LMStudioClient
from rsna_knee.roots import roots_for
from rsna_knee.runtime.colab_session import run_after_handshake
from rsna_knee.runtime.jobs import build_job
from rsna_knee.runtime.providers.kaggle import KaggleProvider


def test_experiment_run_does_not_emit_tiny_checkpoint():
    runs = roots_for(False).runs
    before = {path for path in runs.glob("*") if path.is_dir()} if runs.exists() else set()
    experiment_run(Path("configs/experiments/EXP-002-frozen.yaml"))
    created = [path for path in runs.glob("*") if path.is_dir() and path not in before]
    assert created
    run = created[-1]
    assert (run / "job" / "job.json").is_file()
    assert list(run.rglob("*.npz")) == []
    job = json.loads((run / "job" / "job.json").read_text(encoding="utf-8"))
    assert job["encoder"] == "dinov2_vits14"
    assert job["synthetic"] is False


def test_kaggle_request_does_not_pend_without_kernel():
    handle = KaggleProvider().request_gpu("T4")
    assert handle["status"] == "handoff"
    assert handle["executed"] is False
    assert handle["kernel"] is None
    assert handle["push_argv"][0:3] == ["kaggle", "kernels", "push"]


def test_connected_worker_without_items_is_not_ok(tmp_path):
    job = build_job(run_id="colab-1", stage="TRAINING", inputs={"config": "x"}, dest=tmp_path, synthetic=False)
    out = run_after_handshake(tmp_path / "job.json", inbox=tmp_path / "inbox")
    assert out["exit_reason"] == "no_items"
    assert out["status"] == "BLOCKED"
    assert out["fencing_token"] == job["fencing_token"]


def test_no_resume_does_not_reuse_cache(tmp_path):
    import httpx

    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        payload = {
            "StudyInstanceUID": "u1",
            "targets": {
                t: {"state": "not_mentioned", "evidence_span": "", "criterion_mapping": "none"}
                for t in [
                    "ACL",
                    "MCL",
                    "Medial Meniscus",
                    "Lateral Meniscus",
                    "Medial OA",
                    "Lateral OA",
                    "PF OA",
                    "Effusion",
                    "Synovitis",
                    "Baker's",
                    "Contusion",
                    "Fracture",
                ]
            },
        }
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(payload)}}]})

    client = LMStudioClient(transport=httpx.Client(transport=httpx.MockTransport(handler), base_url="http://127.0.0.1:1234/v1"))
    store = LabelStore(tmp_path / "labels.sqlite")
    kwargs = dict(uid="u1", report="No effusion.", model_id="qwen/qwen3.8-27b", model_revision="r", store=store, json_mode="json_object")
    assert extract_one(client, **kwargs)["cached"] is False
    assert extract_one(client, use_cache=False, **kwargs)["cached"] is False
    assert calls["n"] == 2
    client.close()
