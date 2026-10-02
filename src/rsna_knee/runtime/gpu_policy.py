"""GPU allocation policy on the controller. Sleep cannot change an already-running kernel GPU."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rsna_knee.clock import Clock, WALL
from rsna_knee.config import AllocationConfig


AUTH_STATUSES = {"auth_failure", "needs_auth", "balance_exhausted", "invalid_request"}
UNAVAILABLE = {"unavailable", "unsupported"}
RETRYABLE = {"pending", "queued", "retryable", "retry_after"}


@dataclass
class AcquisitionResult:
    status: str
    gpu: str | None
    handle: dict[str, Any] = field(default_factory=dict)
    waited_seconds: float = 0.0
    attempts: list[dict[str, Any]] = field(default_factory=list)
    fencing_token: str | None = None
    late_ignored: bool = False
    next_action_tr: str = ""


class GpuAllocator:
    def __init__(
        self,
        provider,
        allocation: AllocationConfig | None = None,
        clock: Clock | None = None,
        fencing_token: str | None = None,
        inflight: set[str] | None = None,
    ):
        self.provider = provider
        self.cfg = allocation or AllocationConfig()
        self.clock = clock or WALL
        self.fencing_token = fencing_token
        self.inflight = inflight if inflight is not None else set()
        self.deadline_token = fencing_token

    def acquire(self, gpu_order: list[str] | None = None) -> AcquisitionResult:
        order = gpu_order or list(self.cfg.preferred_gpu_order)
        t0 = self.clock.monotonic()
        total_deadline = t0 + self.cfg.max_total_wait_seconds
        attempts: list[dict[str, Any]] = []
        caps = self.provider.capabilities()
        supported = set(caps.supported_gpu_types)

        if self.cfg.max_concurrent_runtime_requests <= 0:
            return AcquisitionResult(status="blocked", gpu=None, next_action_tr="Eşzamanlı runtime isteği kapalı.")
        if "runtime" in self.inflight:
            return AcquisitionResult(status="duplicate_prevented", gpu=None, next_action_tr="Açık bir runtime isteği var; yenisini açmayın.")
        self.inflight.add("runtime")
        try:
            for gpu in order:
                if self.clock.monotonic() >= total_deadline:
                    break
                if self.cfg.skip_unsupported and gpu not in supported and not caps.automatic_acquisition:
                    # Colab handoff still lists supported types even without headless request.
                    if not caps.request and gpu in caps.supported_gpu_types:
                        pass
                    elif gpu not in supported:
                        attempts.append({"gpu": gpu, "status": "unsupported", "waited": 0})
                        continue
                handle = self.provider.request_gpu(gpu)
                status = handle.get("status")
                attempts.append({"gpu": gpu, "status": status, "waited": 0})
                if status in AUTH_STATUSES:
                    return AcquisitionResult(
                        status="auth_failure",
                        gpu=None,
                        handle=handle,
                        waited_seconds=self.clock.monotonic() - t0,
                        attempts=attempts,
                        fencing_token=self.fencing_token,
                        next_action_tr="Kimlik/kota hatası GPU kıtlığı değildir. Credential'i resmi araçla yenileyin.",
                    )
                if status in UNAVAILABLE or status == "unsupported":
                    continue  # immediate skip, do not wait 300s
                if status in {"allocated", "ready"}:
                    return AcquisitionResult(
                        status="allocated",
                        gpu=handle.get("gpu", gpu),
                        handle=handle,
                        waited_seconds=self.clock.monotonic() - t0,
                        attempts=attempts,
                        fencing_token=self.fencing_token,
                    )
                if status == "handoff":
                    return AcquisitionResult(
                        status="handoff",
                        gpu=gpu,
                        handle=handle,
                        waited_seconds=self.clock.monotonic() - t0,
                        attempts=attempts,
                        fencing_token=self.fencing_token,
                        next_action_tr=handle.get("action_tr") or "Colab eklentisinde kernel seçin.",
                    )
                if status in RETRYABLE or status == "pending_kernel":
                    waited = self._wait_pending(handle, gpu, total_deadline)
                    attempts[-1]["waited"] = waited
                    polled = self.provider.poll(handle)
                    pst = polled.get("status")
                    if pst in {"allocated", "ready"}:
                        # Allocation after the candidate/total deadline is fenced.
                        late = (
                            self.clock.monotonic() > total_deadline
                            or waited >= float(self.cfg.max_wait_per_gpu_seconds) - 1e-9
                        )
                        if late:
                            return AcquisitionResult(
                                status="late_fenced",
                                gpu=None,
                                handle=polled,
                                waited_seconds=self.clock.monotonic() - t0,
                                attempts=attempts,
                                fencing_token=self.fencing_token,
                                late_ignored=True,
                                next_action_tr="Süre dolduktan sonra gelen tahsis bu job'a yazılamaz.",
                            )
                        return AcquisitionResult(
                            status="allocated",
                            gpu=gpu,
                            handle=polled,
                            waited_seconds=self.clock.monotonic() - t0,
                            attempts=attempts,
                            fencing_token=self.fencing_token,
                        )
                    if pst in UNAVAILABLE:
                        continue
            return AcquisitionResult(
                status="exhausted",
                gpu=None,
                waited_seconds=self.clock.monotonic() - t0,
                attempts=attempts,
                fencing_token=self.fencing_token,
                next_action_tr="GPU seçenekleri tükendi. Job QUEUED/NEEDS_RUNTIME. CPU tam eğitime düşülmez.",
            )
        finally:
            self.inflight.discard("runtime")

    def _wait_pending(self, handle: dict[str, Any], gpu: str, total_deadline: float) -> float:
        start = self.clock.monotonic()
        per_deadline = start + self.cfg.max_wait_per_gpu_seconds
        interval = max(1, int(self.cfg.poll_interval_seconds))
        while True:
            now = self.clock.monotonic()
            if now >= per_deadline or now >= total_deadline:
                if hasattr(self.provider, "cancel"):
                    self.provider.cancel(handle)
                break
            retry_after = handle.get("retry_after")
            sleep_s = float(retry_after) if (self.cfg.respect_retry_after and retry_after) else float(interval)
            sleep_s = min(sleep_s, max(0.0, min(per_deadline, total_deadline) - now))
            if sleep_s <= 0:
                break
            self.clock.sleep(sleep_s)
            polled = self.provider.poll(handle)
            handle.update(polled)
            if polled.get("status") not in RETRYABLE and polled.get("status") != "pending_kernel":
                break
        return self.clock.monotonic() - start


class ScriptedProvider:
    """Test double. Production code never uses this unless synthetic=True."""

    def __init__(self, script: dict[str, list[dict[str, Any]]], supported: list[str] | None = None):
        self.script = {k: list(v) for k, v in script.items()}
        self.supported = supported or list(script.keys())
        self.cancels: list[dict[str, Any]] = []

    def capabilities(self):
        from rsna_knee.runtime.providers.base import ProviderCapabilities

        return ProviderCapabilities(
            name="scripted",
            supported_gpu_types=self.supported,
            request=True,
            start=True,
            poll=True,
            cancel=True,
            remote_execution=True,
            upload_download="test",
            auth_flow="none",
            availability_status="scripted",
            automatic_acquisition=True,
            notes=["TEST ONLY"],
        )

    def _next(self, gpu: str) -> dict[str, Any]:
        q = self.script.get(gpu) or self.script.get(gpu.upper()) or []
        if not q:
            return {"status": "unavailable", "gpu": gpu}
        return dict(q.pop(0))

    def request_gpu(self, gpu_type: str, **kwargs: Any) -> dict[str, Any]:
        return self._next(gpu_type)

    def poll(self, handle: dict[str, Any]) -> dict[str, Any]:
        gpu = handle.get("gpu") or handle.get("requested") or ""
        if gpu in self.script and self.script[gpu]:
            return dict(self.script[gpu].pop(0))
        return {"status": handle.get("status", "pending")}

    def cancel(self, handle: dict[str, Any]) -> dict[str, Any]:
        self.cancels.append(handle)
        return {"status": "cancelled"}
