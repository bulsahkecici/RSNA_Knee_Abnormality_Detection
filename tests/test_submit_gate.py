from __future__ import annotations

from rsna_knee.submission.kaggle import submit_run
from rsna_knee.workflow.registry import Registry


class _Provider:
    def __init__(self):
        self.calls = 0

    def submission_limits(self):
        raise AssertionError("limits must not be queried when audit blocks submit")

    def submit_kernel(self, kernel, version, message):
        self.calls += 1
        return {"returncode": 0, "stdout": "Successfully submitted abc", "stderr": ""}


def test_submit_blocked_without_pass_audit(tmp_path):
    registry = Registry(tmp_path / "reg.sqlite")
    provider = _Provider()
    out = submit_run("run", "owner/kernel", "1", "msg", registry=registry, execute=True, provider=provider, audit={"overall": "FAIL"})
    assert out["status"] == "BLOCKED"
    assert out["server_receipt"] is None
    assert out["local_attempt_id"]
    assert provider.calls == 0


def test_unparsed_receipt_is_not_submitted(tmp_path):
    registry = Registry(tmp_path / "reg.sqlite")
    provider = _Provider()
    out = submit_run(
        "run",
        "owner/kernel",
        "1",
        "msg",
        registry=registry,
        execute=True,
        provider=provider,
        audit={"overall": "PASS"},
    )
    assert out["status"] == "BLOCKED"
    assert out["reason"] == "unparsed_receipt"
    assert out["server_receipt"] is None
    assert registry.submission_for_kernel("owner/kernel", "1") is None


def test_server_receipt_differs_from_local_attempt_and_is_unique(tmp_path):
    registry = Registry(tmp_path / "reg.sqlite")

    class Ok(_Provider):
        def submit_kernel(self, kernel, version, message):
            self.calls += 1
            return {
                "returncode": 0,
                "stdout": "Successfully submitted 12345678-1234-1234-1234-123456789abc",
                "stderr": "",
            }

    provider = Ok()
    out = submit_run("run", "owner/kernel", "3", "msg", registry=registry, execute=True, provider=provider, audit={"overall": "PASS"})
    assert out["status"] == "SUBMITTED"
    assert out["server_receipt"] == "12345678-1234-1234-1234-123456789abc"
    assert out["local_attempt_id"] != out["server_receipt"]
    again = submit_run("run", "owner/kernel", "3", "msg", registry=registry, execute=True, provider=provider, audit={"overall": "PASS"})
    assert again["status"] == "BLOCKED"
    assert again["reason"] == "duplicate_kernel_version"
    assert provider.calls == 1
