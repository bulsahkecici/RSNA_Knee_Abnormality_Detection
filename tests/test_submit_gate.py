from __future__ import annotations

from rsna_knee.hashing import sha256_file
from rsna_knee.models.dinov2 import Dinov2ViTS14
from rsna_knee.models.study import StudyModel
from rsna_knee.submission.kaggle import submit_run
from rsna_knee.training.checkpoint import torch_save
from rsna_knee.workflow.registry import Registry
from rsna_knee.workflow.state import Stage


def _bind(tmp_path, registry, version: str) -> dict:
    model = StudyModel(Dinov2ViTS14(img_size=28), freeze_encoder=True)
    ckpt = tmp_path / "checkpoint.pt"
    torch_save(ckpt, {"encoder_name": "dinov2_vits14", "model": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "step": 1, "img_size": 28})
    package = tmp_path / "package.tar"
    package.write_bytes(b"offline-package")
    audit = {
        "overall": "PASS",
        "run_id": "run",
        "kernel": "owner/kernel",
        "version": version,
        "checkpoint_sha256": sha256_file(ckpt),
        "package_sha256": sha256_file(package),
    }
    registry.upsert_run(
        "run",
        "pilot",
        Stage.AUDITING,
        {"run_id": "run", "profile": "pilot", "artifacts": {"checkpoint": str(ckpt), "package_tar": str(package)}},
        False,
    )
    return audit


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


def test_pass_string_without_identity_does_not_submit(tmp_path):
    registry = Registry(tmp_path / "reg.sqlite")
    provider = _Provider()
    out = submit_run("run", "owner/kernel", "9", "msg", registry=registry, execute=True, provider=provider, audit={"overall": "PASS"})
    assert out["status"] == "BLOCKED"
    assert out["reason"] == "audit_not_pass"
    assert provider.calls == 0
    assert "-f" in out["command"]["argv"] or out["command"]["file"] is None


def test_unparsed_receipt_is_not_submitted(tmp_path):
    registry = Registry(tmp_path / "reg.sqlite")
    provider = _Provider()
    audit = _bind(tmp_path, registry, "1")
    out = submit_run(
        "run",
        "owner/kernel",
        "1",
        "msg",
        registry=registry,
        execute=True,
        provider=provider,
        audit=audit,
    )
    assert out["status"] == "BLOCKED"
    assert out["reason"] == "unparsed_receipt"
    assert out["server_receipt"] is None
    assert registry.submission_for_kernel("owner/kernel", "1") is None
    pending = registry.open_submission("owner/kernel", "1")
    assert pending["status"] == "UNKNOWN"
    again = submit_run("run", "owner/kernel", "1", "msg", registry=registry, execute=True, provider=provider, audit=audit)
    assert again["status"] == "BLOCKED"
    assert again["reason"] == "unresolved_attempt"
    assert provider.calls == 1


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
    audit = _bind(tmp_path, registry, "3")
    out = submit_run("run", "owner/kernel", "3", "msg", registry=registry, execute=True, provider=provider, audit=audit)
    assert out["status"] == "SUBMITTED"
    assert out["server_receipt"] == "12345678-1234-1234-1234-123456789abc"
    assert out["local_attempt_id"] != out["server_receipt"]
    again = submit_run("run", "owner/kernel", "3", "msg", registry=registry, execute=True, provider=provider, audit=audit)
    assert again["status"] == "BLOCKED"
    assert again["reason"] == "duplicate_kernel_version"
    assert provider.calls == 1


def test_numeric_receipt_requires_matching_attempt():
    from rsna_knee.submission.kaggle import _matching_receipt

    message = 'pilot [kernel=owner/kernel version=2 attempt=unique-id]'
    rows = [{'ref': 12345, 'description': message}]
    assert _matching_receipt(rows, 'owner/kernel', '2', message) == '12345'
    assert _matching_receipt(rows, 'owner/kernel', '2', 'other attempt=other-id') is None
    assert _matching_receipt([{'ref': 0, 'description': message}], 'owner/kernel', '2', message) is None


def test_cli_text_receipt_is_reconciled_from_actual_history(tmp_path):
    registry = Registry(tmp_path / 'reg.sqlite')
    audit = _bind(tmp_path, registry, '2')

    class Numeric(_Provider):
        def submit_kernel(self, kernel, version, message):
            self.calls += 1
            self.message = message
            return {'returncode': 0, 'stdout': 'Successfully submitted to the competition', 'stderr': ''}

        def list_submissions(self):
            return [{'ref': 98765, 'description': self.message, 'status': 'PENDING'}]

    provider = Numeric()
    out = submit_run('run', 'owner/kernel', '2', 'msg', registry=registry, execute=True, provider=provider, audit=audit)
    assert out['status'] == 'SUBMITTED'
    assert out['server_receipt'] == '98765'
    assert registry.submission_for_kernel('owner/kernel', '2')['server_receipt'] == '98765'
    assert provider.calls == 1
