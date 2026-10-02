from __future__ import annotations

from rsna_knee.clock import VirtualClock
from rsna_knee.config import AllocationConfig
from rsna_knee.runtime.gpu_policy import GpuAllocator, ScriptedProvider


def _alloc(script, clock, **kwargs):
    cfg = AllocationConfig(
        preferred_gpu_order=["A100", "L4", "T4"],
        max_wait_per_gpu_seconds=300,
        max_total_wait_seconds=900,
        poll_interval_seconds=15,
        respect_retry_after=True,
        skip_unsupported=True,
    )
    provider = ScriptedProvider(
        script,
        supported=["A100", "L4", "T4"],
        cancel_supported=kwargs.get("cancel_supported", True),
        automatic_acquisition=kwargs.get("automatic_acquisition", True),
    )
    alloc = GpuAllocator(provider, cfg, clock=clock.as_clock(), fencing_token="tok")
    return alloc, provider


def test_retry_after_is_honored():
    clock = VirtualClock()
    script = {"A100": [{"status": "pending", "gpu": "A100", "retry_after": 40}, {"status": "allocated", "gpu": "A100"}]}
    alloc, _provider = _alloc(script, clock)
    res = alloc.acquire()
    assert res.status == "allocated"
    assert clock.now == 40.0


def test_poll_auth_stops():
    clock = VirtualClock()
    script = {
        "A100": [{"status": "pending", "gpu": "A100"}, {"status": "auth_failure", "gpu": "A100"}],
        "L4": [{"status": "allocated", "gpu": "L4"}],
    }
    alloc, provider = _alloc(script, clock)
    res = alloc.acquire()
    assert res.status == "auth_failure"
    assert "L4" not in provider.requested


def test_cancel_unsupported_does_not_start_next_gpu():
    clock = VirtualClock()
    script = {
        "A100": [{"status": "pending", "gpu": "A100"}] * 40,
        "L4": [{"status": "allocated", "gpu": "L4"}],
    }
    alloc, provider = _alloc(script, clock, cancel_supported=False)
    res = alloc.acquire()
    assert res.status == "ambiguous_pending"
    assert provider.cancels == []
    assert provider.requested == ["A100"]


def test_no_allocation_api_does_not_wait():
    clock = VirtualClock()
    script = {"A100": [{"status": "pending", "gpu": "A100"}] * 5}
    alloc, _provider = _alloc(script, clock, automatic_acquisition=False)
    res = alloc.acquire()
    assert clock.now == 0.0
    assert res.waited_seconds == 0.0
