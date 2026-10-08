"""Reconcile a scored result by its real Kaggle submission receipt."""

from __future__ import annotations

import json
import math
from typing import Any

from rsna_knee.roots import roots_for
from rsna_knee.runtime.providers.kaggle import KaggleProvider
from rsna_knee.workflow.registry import Registry
from rsna_knee.workflow.state import PipelineState, Stage


def poll_score(run_id: str, *, registry: Registry | None = None, provider: Any = None) -> dict[str, Any]:
    registry = registry or Registry()
    provider = provider or KaggleProvider()
    run = registry.get_run(run_id)
    if run is None:
        return {"status": "BLOCKED", "reason": "run_not_registered", "run_id": run_id}
    state = PipelineState.model_validate(run)
    kernel = state.artifacts.get("kernel")
    version = state.artifacts.get("kernel_version")
    attempt = registry.submission_for_kernel(str(kernel), str(version))
    if not attempt or not attempt.get("server_receipt"):
        return {"status": "BLOCKED", "reason": "server_receipt_missing", "run_id": run_id}
    receipt = str(attempt["server_receipt"])
    rows = provider.list_submissions()
    matching = [row for row in rows if str(row.get("ref")) == receipt]
    if len(matching) != 1:
        return {"status": "PENDING", "reason": "receipt_not_in_history", "run_id": run_id, "server_receipt": receipt}
    row = matching[0]
    if hasattr(provider, "get_submission"):
        detail = provider.get_submission(receipt)
        if str(detail.get("ref")) != receipt:
            raise RuntimeError("Kaggle submission detail receipt mismatch")
        row = {**row, **detail}
    remote_status = str(row.get("status") or "").upper().split(".")[-1]
    raw_score = row.get("publicScore", row.get("public_score"))
    score = None
    if raw_score not in (None, ""):
        try:
            value = float(raw_score)
            if math.isfinite(value) and 0 <= value <= 1:
                score = value
        except (TypeError, ValueError):
            pass
    failed = remote_status == "ERROR" or bool(row.get("errorDescription") or row.get("error_description"))
    scored = remote_status == "COMPLETE" and score is not None and not failed
    summary = {
        "run_id": run_id, "status": "SCORED" if scored else "ERROR" if failed else "PENDING",
        "server_receipt": receipt, "kernel": kernel, "version": version,
        "public_score": score if scored else None, "kaggle_status": remote_status,
        "kaggle_record": row,
    }
    path = roots_for(False).runs / run_id / "scoring-status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    state.artifacts["scoring_status"] = str(path)
    state.artifacts["server_receipt"] = receipt
    state.stage = Stage.SCORED if scored else Stage.FAILED if failed else Stage.SUBMITTED
    state.blocked_reason = "kaggle_scoring_error" if failed else None
    state.next_action_tr = (
        "Gerçek Kaggle submission başarıyla skorlandı."
        if scored else "Kaggle scoring hatasını inceleyin; aynı sürümü körlemesine tekrar göndermeyin."
        if failed else "Kaggle scorer sonucu bekleniyor; sunucu receipt doğrulandı."
    )
    registry.upsert_run(run_id, state.profile, state.stage, state.model_dump(), False)
    if scored:
        registry.put_submission(
            attempt["request_id"], run_id, "SCORED",
            {**attempt, "scoring": summary, "server_receipt": receipt},
            kernel=kernel, version=str(version),
        )
    return summary
