"""Kaggle notebook adapter using documented CLI: kernels push/status/output and competitions submit -k -v."""

from __future__ import annotations

import subprocess
from typing import Any

from rsna_knee.runtime.providers.base import ProviderCapabilities

SLUG = "rsna-knee-abnormality-detection"


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, capture_output=True, text=True, check=False)


class KaggleProvider:
    name = "kaggle"

    def capabilities(self, *, probe: bool = True) -> ProviderCapabilities:
        quota = self.quota() if probe else {"ok": False, "probed": False}
        return ProviderCapabilities(
            name="kaggle",
            supported_gpu_types=["T4", "P100"],  # advertised Kaggle notebook GPUs; verify per-session handshake
            request=True,
            start=True,
            poll=True,
            cancel=False,
            remote_execution=True,
            upload_download="kernels_push_output",
            auth_flow="kaggle_json_or_cli_auth",
            availability_status=("cli" if quota.get("ok") else "not_probed" if not probe else "needs_auth_or_quota"),
            automatic_acquisition=True,
            notes=[
                "Code competition submit uses: kaggle competitions submit -c SLUG -k KERNEL -v VERSION -m MSG",
                "CSV-only competitions submit is not sufficient for this code competition.",
                "GPU type/quota must be checked; do not assume A100 on Kaggle.",
            ],
        )

    def quota(self) -> dict[str, Any]:
        proc = _run(["kaggle", "quota"])
        if proc.returncode != 0:
            return {"ok": False, "error": (proc.stderr or proc.stdout).strip(), "compute_units": "unknown"}
        return {"ok": True, "raw": proc.stdout, "compute_units": "unknown"}

    def request_gpu(self, gpu_type: str, **kwargs: Any) -> dict[str, Any]:
        if gpu_type in {"A100", "L4", "H100"}:
            return {"status": "unsupported", "gpu": gpu_type, "retryable": False}
        from rsna_knee.runtime.kaggle_kernel import prepare_cache_kernel

        prepared = prepare_cache_kernel()
        return {
            "status": "handoff",
            "gpu": gpu_type,
            "retryable": False,
            "executed": False,
            "kernel": None,
            "push_argv": prepared["push_argv"],
            "status_argv": prepared["status_argv"],
            "output_argv": prepared["output_argv"],
            "kernel_dir": prepared["dir"],
            "action_tr": (
                "Kaggle kernel klasörü hazır. kernels push/status/output bu çağrıda çalıştırılmadı. "
                "Kernel kimliği olmadan pending beklenmez."
            ),
        }

    def poll(self, handle: dict[str, Any]) -> dict[str, Any]:
        kernel = handle.get("kernel")
        if not kernel:
            return {"status": "unknown"}
        proc = _run(["kaggle", "kernels", "status", kernel])
        if proc.returncode != 0:
            return {"status": "error", "error": proc.stderr}
        return {"status": "polled", "raw": proc.stdout}

    def cancel(self, handle: dict[str, Any]) -> dict[str, Any]:
        return {"status": "cancel_unsupported"}

    def push_kernel(self, folder: str) -> dict[str, Any]:
        proc = _run(["kaggle", "kernels", "push", "-p", folder])
        return {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}

    def kernel_output(self, kernel: str, dest: str) -> dict[str, Any]:
        proc = _run(["kaggle", "kernels", "output", kernel, "-p", dest])
        return {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}

    def submit_kernel(self, kernel: str, version: str, message: str) -> dict[str, Any]:
        proc = _run(
            [
                "kaggle",
                "competitions",
                "submit",
                "-c",
                SLUG,
                "-k",
                kernel,
                "-v",
                str(version),
                "-m",
                message,
            ]
        )
        return {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}

    def submission_limits(self) -> dict[str, Any]:
        proc = _run(["kaggle", "competitions", "submission-limits", "-c", SLUG])
        return {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}
