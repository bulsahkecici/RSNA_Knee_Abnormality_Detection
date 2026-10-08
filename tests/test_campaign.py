"""Campaign safety: report isolation, split exclusions, and real bundle inputs."""

import csv
import json

import pytest

from rsna_knee.hashing import sha256_file
from rsna_knee.ontology import TARGET_COLUMNS
from rsna_knee.runtime import campaign, campaign_controller


def _numeric(uid, source="weak"):
    return {"StudyInstanceUID": uid, "source": source,
        "targets": {t: float(i % 2) for i, t in enumerate(TARGET_COLUMNS)},
        "y_mask": {t: 1.0 for t in TARGET_COLUMNS}, "y_weight": {t: 0.25 for t in TARGET_COLUMNS}}


def test_private_kernel_bundle_strips_reports_and_evidence(tmp_path, monkeypatch):
    for folder in ("src/rsna_knee", "configs", "schemas", "state", "artifacts/weights"):
        (tmp_path / folder).mkdir(parents=True)
    (tmp_path / "src/rsna_knee/__init__.py").write_text("")
    (tmp_path / "configs/dinov2_vits14_allowlist.json").write_text("{}")
    (tmp_path / "state/folds.csv").write_text("fixture fold identity")
    (tmp_path / "artifacts/weights/dinov2_vits14_pretrain.pth").write_bytes(b"fixture weights")
    folds = [{"StudyInstanceUID": f"1.2.{i}", "fold": "1", "is_gold": "0", "gold_split": "weak", "group_id": str(i)} for i in range(500)]
    folds.append({"StudyInstanceUID": "gold", "fold": "0", "is_gold": "1", "gold_split": "eval", "group_id": "gold"})
    monkeypatch.setattr(campaign, "ROOT", tmp_path)
    monkeypatch.setattr(campaign, "load_folds", lambda: folds)
    labels = tmp_path / "labels.jsonl"
    rows = [_numeric(f"1.2.{i}") for i in range(500)] + [_numeric("gold", "gold")]
    for row in rows:
        row["Report"] = "DO NOT UPLOAD RAW REPORT"
        row["evidence_span"] = "DO NOT UPLOAD EVIDENCE"
    labels.write_text("".join(json.dumps(row) + "\n" for row in rows))
    prepared = campaign.prepare_campaign_kernel(labels, run_id="test-campaign", owner="owner", dest=tmp_path / "cache-kernel", phase="cache")
    assert prepared["train_studies"] == 500
    assert prepared["gold_eval_studies"] == 1
    script = (tmp_path / "cache-kernel/campaign_entry.py").read_text()
    assert "DO NOT UPLOAD" not in script
    # Decode the embedded numeric JSONL and inspect its keys, rather than merely
    # checking that the literal text was hidden by base64 encoding.
    import ast
    import base64
    import gzip
    tree = ast.parse(script)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "write_bytes":
            encoded = node.args[0].args[0].args[0].value
            exported = [json.loads(line) for line in gzip.decompress(base64.b64decode(encoded)).decode().splitlines()]
            assert len(exported) == 501
            assert all("Report" not in row and "evidence_span" not in row for row in exported)
    metadata = json.loads((tmp_path / "cache-kernel/kernel-metadata.json").read_text())
    assert metadata["is_private"] and not metadata["enable_gpu"]
    with pytest.raises(ValueError, match="verified_cache_required"):
        campaign.prepare_campaign_kernel(labels, run_id="test-campaign", owner="owner", dest=tmp_path / "gpu-kernel", phase="finetune")
    fine = campaign.prepare_campaign_kernel(labels, run_id="test-campaign", owner="owner", dest=tmp_path / "gpu-kernel", phase="finetune", cache_kernel=prepared["kernel"], cache_sha256="fixturehash")
    assert fine["train_studies"] == 500
    gpu = json.loads((tmp_path / "gpu-kernel/kernel-metadata.json").read_text())
    assert gpu["enable_gpu"] and gpu["kernel_sources"] == [prepared["kernel"]]


def test_quality_checks_evidence_and_excludes_gold(tmp_path, monkeypatch):
    meta = tmp_path / "data/metadata"
    meta.mkdir(parents=True)
    with (meta / "train.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=["StudyInstanceUID", "Report"])
        writer.writeheader()
        writer.writerow({"StudyInstanceUID": "weak", "Report": "ACL intact."})
        writer.writerow({"StudyInstanceUID": "gold", "Report": "ACL intact."})
    monkeypatch.setattr(campaign_controller, "ROOT", tmp_path)
    monkeypatch.setattr(campaign_controller, "load_folds", lambda: [
        {"StudyInstanceUID": "weak", "fold": "1", "is_gold": "0", "gold_split": "weak", "group_id": "weak"},
        {"StudyInstanceUID": "gold", "fold": "0", "is_gold": "1", "gold_split": "eval", "group_id": "gold"}])
    payload = {"StudyInstanceUID": "weak", "targets": {t: {"state": "not_mentioned", "evidence_span": "", "criterion_mapping": "unmentioned"} for t in TARGET_COLUMNS}}
    payload["targets"]["ACL"] = {"state": "explicit_negative", "evidence_span": "ACL intact.", "criterion_mapping": "intact"}
    raw = tmp_path / "raw.jsonl"
    raw.write_text(json.dumps({"status": "ok", "payload": payload}) + "\n")
    result = campaign_controller.label_quality(raw, minimum=1)
    assert result["pass"] and not result["clinical_accuracy_verified"]
    payload["StudyInstanceUID"] = "gold"
    raw.write_text(json.dumps({"status": "ok", "payload": payload}) + "\n")
    assert not campaign_controller.label_quality(raw, minimum=1)["pass"]


def test_campaign_rejects_path_traversal_before_writing():
    with pytest.raises(ValueError, match="invalid_campaign_identity"):
        campaign_controller.run_campaign("../../outside", "owner")


def test_adopted_label_job_waits_for_registry_and_checks_identity(tmp_path, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(campaign_controller, "ROOT", tmp_path)
    record = None
    monkeypatch.setattr(campaign_controller, "Registry", lambda: SimpleNamespace(
        get_run=lambda run_id: record, close=lambda: None))
    assert campaign_controller.completed_label_run("existing", 95) is None
    labels = tmp_path / "data/labels"
    labels.mkdir(parents=True)
    raw = labels / "existing-labels.jsonl"
    raw.write_text("fixture")
    numeric = labels / "existing-training.jsonl"
    numeric.write_text("fixture numeric labels")
    record = {"stage": "LABELS_READY", "synthetic": False,
        "hashes": {"labels": sha256_file(numeric)}}
    monkeypatch.setattr(campaign_controller, "label_quality", lambda path, minimum: {"pass": minimum == 95})
    adopted, quality = campaign_controller.completed_label_run("existing", 95)
    assert adopted == numeric and quality["pass"]
    numeric.write_text("changed labels")
    with pytest.raises(ValueError, match="existing_label_hash_mismatch"):
        campaign_controller.completed_label_run("existing", 95)
    record["synthetic"] = True
    with pytest.raises(ValueError, match="existing_label_run_not_ready"):
        campaign_controller.completed_label_run("existing", 95)
