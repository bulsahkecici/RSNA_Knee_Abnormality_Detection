from __future__ import annotations

from rsna_knee.clock import VirtualClock
from rsna_knee.config import AllocationConfig
from rsna_knee.runtime.gpu_policy import GpuAllocator, ScriptedProvider


def _alloc(script, clock=None, **kwargs):
    cfg = AllocationConfig(
        preferred_gpu_order=["A100", "L4", "T4"],
        max_wait_per_gpu_seconds=300,
        max_total_wait_seconds=900,
        poll_interval_seconds=15,
        skip_unsupported=True,
    )
    provider = ScriptedProvider(script, supported=kwargs.get("supported", ["A100", "L4", "T4"]))
    return GpuAllocator(provider, cfg, clock=clock or VirtualClock().as_clock(), fencing_token="tok-1", inflight=kwargs.get("inflight"))


def test_unavailable_skips_immediately():
    clock = VirtualClock()
    alloc = _alloc({"A100": [{"status": "unavailable"}], "L4": [{"status": "allocated", "gpu": "L4"}]}, clock=clock.as_clock())
    res = alloc.acquire()
    assert res.status == "allocated" and res.gpu == "L4"
    assert clock.now == 0.0  # no 300s wait


def test_pending_waits_up_to_300s():
    clock = VirtualClock()
    script = {
        "A100": [{"status": "pending", "gpu": "A100"}] + [{"status": "pending", "gpu": "A100"}] * 30,
        "L4": [{"status": "allocated", "gpu": "L4"}],
    }
    alloc = _alloc(script, clock=clock.as_clock())
    res = alloc.acquire()
    assert clock.now == 300.0 or clock.now >= 300.0
    assert res.gpu == "L4" or res.status in {"allocated", "exhausted", "late_fenced"}


def test_total_900s_cap():
    clock = VirtualClock()
    script = {
        "A100": [{"status": "pending", "gpu": "A100"}] * 50,
        "L4": [{"status": "pending", "gpu": "L4"}] * 50,
        "T4": [{"status": "pending", "gpu": "T4"}] * 50,
    }
    alloc = _alloc(script, clock=clock.as_clock())
    alloc.acquire()
    assert clock.now <= 900.0 + 15.0


def test_auth_no_retry():
    clock = VirtualClock()
    alloc = _alloc({"A100": [{"status": "auth_failure"}], "L4": [{"status": "allocated", "gpu": "L4"}]}, clock=clock.as_clock())
    res = alloc.acquire()
    assert res.status == "auth_failure"
    assert res.gpu is None
    assert clock.now == 0.0


def test_late_allocation_fenced():
    clock = VirtualClock()
    # Stay pending through the 300s candidate window; the post-timeout poll is allocated.
    script = {
        "A100": [{"status": "pending", "gpu": "A100"}] * 21 + [{"status": "allocated", "gpu": "A100"}],
        "L4": [{"status": "unavailable"}],
        "T4": [{"status": "unavailable"}],
    }
    alloc = _alloc(script, clock=clock.as_clock())
    res = alloc.acquire()
    assert res.status == "late_fenced"
    assert res.late_ignored is True
    assert res.fencing_token == "tok-1"


def test_duplicate_runtime_prevented():
    alloc = _alloc({"A100": [{"status": "allocated"}]}, inflight={"runtime"})
    res = alloc.acquire()
    assert res.status == "duplicate_prevented"
