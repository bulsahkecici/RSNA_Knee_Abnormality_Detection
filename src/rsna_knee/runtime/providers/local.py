"""Local provider: synthetic/tiny training only. Full MRI training is not enabled on MPS/CPU by default."""

from __future__ import annotations

from typing import Any

from rsna_knee.runtime.providers.base import ProviderCapabilities


class LocalProvider:
    name = "local"

    def __init__(self, mps_full: bool = False, cpu_full: bool = False):
        self.mps_full = mps_full
        self.cpu_full = cpu_full

    def capabilities(self) -> ProviderCapabilities:
        gpus = ["mps"] if self._mps() else []
        return ProviderCapabilities(
            name="local",
            supported_gpu_types=gpus,
            request=True,
            start=True,
            poll=True,
            cancel=True,
            remote_execution=False,
            upload_download="local_fs",
            auth_flow="none",
            availability_status="local",
            automatic_acquisition=True,
            notes=[
                "CUDA is not expected on macOS.",
                "Full training on MPS/CPU is disabled unless config enables it.",
                "Synthetic tiny training is allowed.",
            ],
        )

    def _mps(self) -> bool:
        try:
            import torch

            return bool(torch.backends.mps.is_available())
        except Exception:
            return False

    def request_gpu(self, gpu_type: str, **kwargs: Any) -> dict[str, Any]:
        if gpu_type.upper() in {"A100", "L4", "T4", "H100", "V100"}:
            return {"status": "unsupported", "gpu": gpu_type, "retryable": False}
        if gpu_type.lower() == "mps" and not self._mps():
            return {"status": "unavailable", "gpu": gpu_type, "retryable": False}
        return {"status": "allocated", "gpu": "mps" if self._mps() else "cpu", "handle": "local"}

    def poll(self, handle: dict[str, Any]) -> dict[str, Any]:
        return {"status": handle.get("status", "allocated")}

    def cancel(self, handle: dict[str, Any]) -> dict[str, Any]:
        return {"status": "cancelled"}
