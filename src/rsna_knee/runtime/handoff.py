"""Manual runtime handoff documents. These are capability limits, not fake completions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rsna_knee.paths import STATE_DIR, ensure_runtime_dirs
from rsna_knee.runtime.gpu_policy import AcquisitionResult


def write_handoff(result: AcquisitionResult, run_id: str, extra: dict[str, Any] | None = None) -> Path:
    ensure_runtime_dirs()
    dest = STATE_DIR / "handoffs"
    dest.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "status": result.status,
        "gpu": result.gpu,
        "automatic": False if result.status == "handoff" else None,
        "next_action_tr": result.next_action_tr,
        "attempts": result.attempts,
        "fencing_token": result.fencing_token,
        "note": (
            "Handoff, tamamlanmış eğitim değildir. Kullanıcı resmi Colab eklentisinde kernel seçer; "
            "worker handshake sonrası controller job'u sürdürür. 300s sonra talimat değişmesi "
            "fiziksel GPU seçiminin otomatik değişmesi değildir."
        ),
        "commands": [
            "Cursor/VS Code: Google Colab eklentisi (Google.colab)",
            "Select Kernel → Colab → New Colab Server → A100 sonra L4 sonra T4",
            "notebooks/02_colab_worker.ipynb çalıştır",
            "rsna status --watch",
        ],
        **(extra or {}),
    }
    path = dest / f"{run_id}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md = dest / f"{run_id}.md"
    md.write_text(
        "\n".join(
            [
                f"# Runtime handoff — {run_id}",
                "",
                f"Durum: `{result.status}`  İstenen GPU: `{result.gpu}`",
                "",
                result.next_action_tr or "",
                "",
                *payload["commands"],
                "",
                payload["note"],
            ]
        ),
        encoding="utf-8",
    )
    return md
