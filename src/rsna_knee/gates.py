"""Production gates. Synthetic fixtures cannot satisfy a real run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rsna_knee.errors import ContractError, FakeResultError
from rsna_knee.hashing import sha256_file

PRODUCTION_ENCODER = "dinov2_vits14"
TEST_ENCODER = "tiny_test_encoder"


def looks_synthetic_uid(uid: str) -> bool:
    return uid.startswith("syn-") or uid.startswith("synthetic-")


def assert_real_uids(uids: list[str]) -> None:
    bad = [u for u in uids if looks_synthetic_uid(u)]
    if bad:
        raise ContractError(f"synthetic UIDs are not valid production inputs: {bad[:5]}")


def assert_real_encoder(name: str | None) -> None:
    if name in {None, "", TEST_ENCODER, "tiny", "dinov2_small_or_stub", "TorchDinoStub"}:
        raise ContractError(f"encoder {name!r} cannot train or evaluate a real run")
    if name != PRODUCTION_ENCODER:
        raise ContractError(f"production encoder must be {PRODUCTION_ENCODER}, got {name!r}")


def audit_allows_submit(audit: dict[str, Any]) -> bool:
    return audit.get("overall") == "PASS"


def verify_identity(run_id: str, manifest: dict[str, Any], files_root: Path) -> list[str]:
    errors: list[str] = []
    if manifest.get("run_id") != run_id:
        errors.append("run_id_mismatch")
    if manifest.get("synthetic") is True:
        errors.append("synthetic_manifest")
    if manifest.get("encoder") not in {None, PRODUCTION_ENCODER} and manifest.get("encoder") != PRODUCTION_ENCODER:
        errors.append("encoder_mismatch")
    for rel, digest in (manifest.get("files") or {}).items():
        path = files_root / rel
        if not path.is_file():
            errors.append(f"missing_file:{rel}")
            continue
        if sha256_file(path) != digest:
            errors.append(f"hash_mismatch:{rel}")
    ckpt = manifest.get("checkpoint")
    if not ckpt:
        errors.append("checkpoint_missing")
    else:
        ckpt_path = files_root / ckpt if not Path(ckpt).is_absolute() else Path(ckpt)
        if not ckpt_path.is_file():
            errors.append("checkpoint_file_missing")
    return errors


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ContractError(f"expected object: {path}")
    return data


def reject_fake(metrics: dict[str, Any]) -> None:
    if metrics.get("kind") != "real":
        raise FakeResultError("only kind=real metrics can pass a production gate")
