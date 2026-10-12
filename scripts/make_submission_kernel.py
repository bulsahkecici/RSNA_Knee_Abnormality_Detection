"""Build a private, internet-off Kaggle submission kernel for one reviewed Colab arm.

Usage: python scripts/make_submission_kernel.py D3 --slug rsna-d3-submit-20261012

Writes artifacts/kaggle/<slug>-asset/upload/ (private dataset: offline-bundle.tar.gz) and
artifacts/kaggle/<slug>/ (entry.py + kernel-metadata.json). Nothing is pushed or submitted here.

The kernel preprocesses test DICOMs with the exact source archive embedded in the
cache kernel that built the arm's training cache (same build_cache call and policy),
asserts the training preprocess hash, and predicts with the exact training source of
the arm's notebook (load_trained_study + _predict over every cached window, as in the
Colab evaluation). Outside the competition rerun it first re-preprocesses the 58 gold
training studies and requires their probabilities to match the arm's reviewed Colab
predictions.
"""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPETITION = "rsna-knee-abnormality-detection"
CACHE_KERNELS = {"171ef4fff32fdb62d850c0787dab55d7f4473d9cc0667a5699ec2fe9436afdf4": "rsna-v4-dense12-cache-20261009"}
WHEELS = ROOT / "artifacts/kaggle/v3-offline-asset/wheels"
CODEC_PINS = {"pydicom": "3.0.1", "pylibjpeg": "2.1.0", "pylibjpeg-libjpeg": "2.4.0", "pylibjpeg-openjpeg": "2.6.0"}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _assigned(text: str, names: set[str]) -> dict:
    return {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(text).body
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in names}


def preprocess_source(cache_manifest_sha: str) -> tuple[bytes, dict, str]:
    """The cache kernel's embedded source, its build_cache policy and the training preprocess hash."""
    slug = CACHE_KERNELS[cache_manifest_sha]
    folder = ROOT / "artifacts/kaggle" / slug
    values = _assigned((folder / "entry.py").read_text(), {"SOURCE_B64", "SOURCE_SHA256", "POLICY"})
    data = base64.b64decode(values["SOURCE_B64"])
    if _sha(data) != values["SOURCE_SHA256"]:
        raise SystemExit("cache kernel source archive does not match its recorded sha256")
    result = json.loads((folder / "output/campaign-result.json").read_text())
    if result["status"] != "ok" or result["manifest_sha256"] != cache_manifest_sha:
        raise SystemExit("cache kernel result does not describe the arm's cache")
    policy = dict(values["POLICY"], max_cache_bytes=18 * 1024**3, strict_series=True)
    versions = {name: result["packages"][name] for name in ("numpy", "Pillow", "pydicom")}
    return data, policy, result["preprocess_hash"], versions


def training_source(exp: str, build: dict) -> bytes:
    notebook = json.loads((ROOT / "notebooks/generated" / f"{exp}.ipynb").read_text())
    cell = next("".join(c["source"]) for c in notebook["cells"] if "SOURCE_B64 = " in "".join(c["source"]))
    data = base64.b64decode(ast.literal_eval(cell.split("SOURCE_B64 = ", 1)[1].split("\n", 1)[0]))
    if _sha(data) != build["source_sha256"]:
        raise SystemExit("notebook source archive does not match build.json")
    return data


def build(exp: str, slug: str, parity_file: Path | None = None) -> tuple[Path, Path]:
    out = ROOT / "artifacts/colab" / f"p1-{exp}" / "output"
    review = json.loads((out / "review.json").read_text())
    ran = json.loads((out / "experiment.json").read_text())
    build_info = json.loads((ROOT / "notebooks/generated" / f"{exp}.build.json").read_text())
    metrics = json.loads((out / "job/metrics.json").read_text())
    if review["experiment"] != exp or ran["id"] != exp or ran["source_sha256"] != build_info["source_sha256"]:
        raise SystemExit("reviewed result, run record and notebook build disagree")
    checkpoint = out / "job/checkpoint.pt"
    checkpoint_sha = _sha(checkpoint.read_bytes())
    if checkpoint_sha != metrics["oof_provenance"]["checkpoint_sha256"]:
        raise SystemExit("checkpoint does not match the evaluated checkpoint")
    pre, policy, preprocess_hash, pixel_versions = preprocess_source(ran["cache_manifest_sha256"])
    model = training_source(exp, build_info)
    if parity_file is None:
        rows = [json.loads(x) for x in (out / "job/predictions.jsonl").read_text().splitlines()]
        parity = "".join(json.dumps(r) + "\n" for r in rows if r["split"] == "gold").encode()
    else:
        parity = parity_file.read_bytes()
    if not parity.strip():
        raise SystemExit("no parity predictions")

    asset = ROOT / "artifacts/kaggle" / f"{slug}-asset"
    staged = asset / "bundle"
    if staged.exists():
        shutil.rmtree(staged)
    (staged / "wheels").mkdir(parents=True)
    # .blob, not .tar.gz: Kaggle may unpack archives inside an uploaded dataset.
    (staged / "preprocess-source.blob").write_bytes(pre)
    (staged / "model-source.blob").write_bytes(model)
    shutil.copyfile(checkpoint, staged / "checkpoint.pt")
    (staged / "parity.jsonl").write_bytes(parity)
    for wheel in sorted(WHEELS.glob("*.whl")):
        shutil.copyfile(wheel, staged / "wheels" / wheel.name)
    files = {str(p.relative_to(staged)): _sha(p.read_bytes()) for p in sorted(staged.rglob("*")) if p.is_file()}
    identity = {"experiment": exp, "checkpoint_sha256": checkpoint_sha, "training_source_sha256": _sha(model),
                "preprocess_source_sha256": _sha(pre), "preprocess_hash": preprocess_hash, "policy": policy,
                "training_pixel_versions": pixel_versions,
                "cache_manifest_sha256": ran["cache_manifest_sha256"], "img_size": int(policy["size"]),
                "codec_pins": CODEC_PINS, "git_commit": ran["git_commit"],
                "gold_all_macro_auc": review["splits"]["gold_all"]["macro_auc"]}
    (staged / "MANIFEST.json").write_text(json.dumps({"internet_required": False, "identity": identity, "files": files},
                                                     indent=2))
    upload = asset / "upload"
    upload.mkdir(parents=True, exist_ok=True)
    with tarfile.open(upload / "offline-bundle.tar.gz", "w:gz") as tar:
        for name in [*files, "MANIFEST.json"]:
            tar.add(staged / name, arcname=name)
    dataset = f"bulsahkecici/{slug}-asset"
    (upload / "dataset-metadata.json").write_text(json.dumps(
        {"title": f"{slug}-asset", "id": dataset, "licenses": [{"name": "CC0-1.0"}]}, indent=2))

    kernel_dir = ROOT / "artifacts/kaggle" / slug
    kernel_dir.mkdir(parents=True, exist_ok=True)
    entry = ENTRY.replace("__IDENTITY__", repr(json.dumps(identity, sort_keys=True)))
    compile(entry, "entry.py", "exec")
    (kernel_dir / "entry.py").write_text(entry)
    (kernel_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": f"bulsahkecici/{slug}", "title": slug, "code_file": "entry.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": False,
        "competition_sources": [COMPETITION], "dataset_sources": [dataset], "kernel_sources": []}, indent=2))
    return upload, kernel_dir


ENTRY = r'''#!/usr/bin/env python3
"""RSNA knee offline submission. Generated by scripts/make_submission_kernel.py; edit the generator, not this file."""
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

IDENTITY = json.loads(__IDENTITY__)
INPUTS = Path(os.environ.get("RSNA_INPUT_DIR", "/kaggle/input"))
WORK = Path(os.environ.get("RSNA_WORK_DIR", "/kaggle/working"))
TMP = Path(os.environ.get("RSNA_TMP_DIR", "/tmp/rsna-submit"))
RERUN = bool(os.environ.get("KAGGLE_IS_COMPETITION_RERUN"))
PARITY_MAX_ABS = 0.02     # largest single-probability difference vs the reviewed Colab run
PARITY_MEAN_ABS = 0.002   # mean difference; GPU type and TF32 convolutions differ from Colab
MAX_FALLBACK_FRACTION = 0.10

PREPROCESS = """
import json, sys
from pathlib import Path
from rsna_knee.data import cache_build
from rsna_knee.data.preprocess import preprocess_hash
if __name__ == "__main__":  # cache workers may re-import this module
    spec = json.load(open(sys.argv[1]))
    p = spec["policy"]
    m = cache_build.build_cache(series_csv=Path(spec["series_csv"]), dicom_root=Path(spec["dicom_root"]),
        dest=Path(spec["dest"]), study_ids=spec["study_ids"], size=p["size"], n_centers=p["n_centers"],
        center_lower=p["center_lower"], center_upper=p["center_upper"], slots=getattr(cache_build, p["slots"]),
        strict_series=spec["strict"], resume=True, max_cache_bytes=p["max_cache_bytes"], workers=spec["workers"],
        progress_every=100, index_flush_every=100)
    config = m["preprocess_config"]
    print(json.dumps({"n_studies": m["n_studies"], "n_quarantine": m["n_quarantine"],
                      "preprocess_hash": m["preprocess_hash"], "quarantine": m["quarantine"],
                      "pixel_runtime_versions": config["pixel_runtime_versions"],
                      "hash_with_training_versions": preprocess_hash(
                          dict(config, pixel_runtime_versions=spec["training_versions"]))}), flush=True)
"""

PREDICT = """
import json, sys
from pathlib import Path
spec = json.load(open(sys.argv[1]))
from rsna_knee.evaluation.heldout import _predict, load_trained_study, select_device
from rsna_knee.ontology import TARGET_COLUMNS
from rsna_knee.runtime.train_payload import LazyStudyBatch
device = select_device(spec.get("device"))
model, _ = load_trained_study(Path(spec["checkpoint"]), spec["img_size"], device)
done = {}
for manifest_path in spec["manifests"]:
    manifest = json.load(open(manifest_path))
    uids = [u for u in spec["study_ids"] if u in manifest["studies"] and u not in done]
    batches = [LazyStudyBatch(u, manifest["studies"][u], {}, Path(manifest_path).parent) for u in uids]
    got, _, probs, _ = _predict(model, batches, device)
    for uid, prob in zip(got, probs, strict=True):
        done[uid] = dict(zip(TARGET_COLUMNS, map(float, prob), strict=True))
with open(spec["out"], "w") as handle:
    for uid, prob in done.items():
        handle.write(json.dumps({"StudyInstanceUID": uid, **prob}) + "\\n")
print(json.dumps({"predicted": len(done), "device": str(device), "targets": TARGET_COLUMNS}), flush=True)
"""


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def find_asset():
    found = [p.parent for p in INPUTS.rglob("MANIFEST.json") if (p.parent / "checkpoint.pt").is_file()]
    if not found:
        archives = list(INPUTS.rglob("offline-bundle.tar.gz"))
        assert len(archives) == 1, f"offline bundle missing or ambiguous: {archives}"
        with tarfile.open(archives[0]) as tar:
            tar.extractall(TMP / "asset", filter="data")
        found = [TMP / "asset"]
    assert len(found) == 1, f"expected one offline asset, found {found}"
    asset = found[0]
    manifest = json.loads((asset / "MANIFEST.json").read_text())
    assert json.loads(json.dumps(manifest["identity"], sort_keys=True)) == IDENTITY, "asset identity differs from kernel"
    for name, expected in manifest["files"].items():
        assert sha256(asset / name) == expected, f"asset integrity mismatch: {name}"
    return asset


def find_competition():
    roots = [p.parent for p in INPUTS.rglob("test_series.csv") if (p.parent / "test_series").is_dir()]
    assert len(roots) == 1, f"expected one competition root, found {roots}"
    return roots[0]


def installed(name):
    import importlib.metadata
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def setup_packages(asset):
    """Same codec pins as the cache kernel, installed offline from bundled wheels."""
    target = TMP / "deps"
    needed = [f"{n}=={v}" for n, v in IDENTITY["codec_pins"].items() if installed(n) != v]
    if needed and os.environ.get("RSNA_SKIP_WHEELS") != "1":
        target.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check", "--no-index",
                        "--no-deps", "--upgrade", "--target", str(target), "--find-links", str(asset / "wheels"),
                        *needed], check=True)
    return {"installed": {n: installed(n) for n in IDENTITY["codec_pins"]}, "target": str(target), "needed": needed}


def unpack(blob, dest):
    if not (dest / "src").is_dir():
        with tarfile.open(blob, "r:gz") as tar:
            tar.extractall(dest, filter="data")
    return dest / "src"


def run(script, spec, src, deps, name):
    TMP.mkdir(parents=True, exist_ok=True)
    (TMP / f"{name}.py").write_text(script)
    (TMP / f"{name}.json").write_text(json.dumps(spec))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([deps, str(src)]))  # cache workers inherit both
    done = subprocess.run([sys.executable, str(TMP / f"{name}.py"), str(TMP / f"{name}.json")], env=env,
                          capture_output=True, text=True, cwd=str(TMP))
    sys.stdout.write(done.stdout[-4000:])
    if done.returncode:
        raise RuntimeError(f"{name} failed:\n{done.stderr[-4000:]}")
    return json.loads(done.stdout.strip().splitlines()[-1])


def preprocess(comp_csv, dicom_root, uids, dest, strict, pre_src, deps, workers, name):
    return run(PREPROCESS, {"series_csv": str(comp_csv), "dicom_root": str(dicom_root), "dest": str(dest),
                            "study_ids": uids, "policy": IDENTITY["policy"], "strict": strict, "workers": workers,
                            "training_versions": IDENTITY["training_pixel_versions"]},
               pre_src, deps, name)


def predict(manifests, uids, out, model_src, deps, asset):
    return run(PREDICT, {"manifests": [str(m) for m in manifests], "study_ids": uids, "out": str(out),
                         "checkpoint": str(asset / "checkpoint.pt"), "img_size": IDENTITY["img_size"]},
               model_src, deps, "predict-" + out.stem)


def check_policy(built, name):
    """Code, policy and slots must hash to training; differing numpy/Pillow/pydicom must pass pixel parity."""
    assert built["hash_with_training_versions"] == IDENTITY["preprocess_hash"], f"{name} preprocessing differs from training"
    return built["pixel_runtime_versions"] == IDENTITY["training_pixel_versions"]


def read_jsonl(path):
    return {r["StudyInstanceUID"]: r for r in map(json.loads, path.read_text().splitlines())}


def main():
    started = time.time()
    WORK.mkdir(parents=True, exist_ok=True)
    result = {"identity": IDENTITY, "rerun": RERUN, "status": "running"}
    asset = find_asset()
    comp = find_competition()
    result["packages"] = setup_packages(asset)
    deps = result["packages"]["target"]
    pre_src = unpack(asset / "preprocess-source.blob", TMP / "preprocess-source")
    model_src = unpack(asset / "model-source.blob", TMP / "model-source")
    workers = max(1, min(4, os.cpu_count() or 1))

    can_check = (comp / "train_series.csv").is_file() and (comp / "train_series").is_dir()
    if not RERUN and can_check:
        parity = read_jsonl(asset / "parity.jsonl")
        t0 = time.time()
        built = preprocess(comp / "train_series.csv", comp / "train_series", sorted(parity), TMP / "parity-cache",
                           True, pre_src, deps, workers, "preprocess-parity")
        result["pixel_versions_match_training"] = check_policy(built, "parity")
        result["pixel_runtime_versions"] = built["pixel_runtime_versions"]
        assert built["n_quarantine"] == 0, f"parity studies quarantined: {built['quarantine']}"
        predict([TMP / "parity-cache/manifest.json"], sorted(parity), TMP / "parity-predictions.jsonl", model_src, deps, asset)
        mine = read_jsonl(TMP / "parity-predictions.jsonl")
        assert set(mine) == set(parity), "parity prediction set differs"
        targets = [k for k in parity[next(iter(parity))] if k not in {"StudyInstanceUID", "split"}]
        diffs = [abs(mine[u][t] - parity[u][t]) for u in parity for t in targets]
        result["parity"] = {"n_studies": len(parity), "max_abs": max(diffs), "mean_abs": sum(diffs) / len(diffs),
                            "seconds": round(time.time() - t0, 1)}
        atomic_json(WORK / "offline-result.json", result)
        print("parity", json.dumps(result["parity"]), flush=True)
        assert result["parity"]["max_abs"] <= PARITY_MAX_ABS and result["parity"]["mean_abs"] <= PARITY_MEAN_ABS, \
            "test-time pipeline does not reproduce the reviewed Colab predictions"

    with (comp / "test.csv").open(newline="") as handle:
        studies = [row["StudyInstanceUID"] for row in csv.DictReader(handle)]
    assert studies and len(set(studies)) == len(studies), "test.csv study list is empty or duplicated"
    with (comp / "sample_submission.csv").open(newline="") as handle:
        columns = next(csv.reader(handle))
    t0 = time.time()
    strict = preprocess(comp / "test_series.csv", comp / "test_series", studies, TMP / "test-cache", True,
                        pre_src, deps, workers, "preprocess-test")
    same_versions = check_policy(strict, "test")
    # The competition rerun uses the image of the version that passed parity; elsewhere parity is required.
    assert same_versions or RERUN or "parity" in result, "pixel runtime differs from training and parity did not run"
    manifests = [TMP / "test-cache/manifest.json"]
    built = set(json.loads(manifests[0].read_text())["studies"])
    retry = [u for u in studies if u not in built]
    relaxed = None
    if retry:  # strict decoding quarantined these; never zero-fill, try once with tolerant series decoding
        relaxed = preprocess(comp / "test_series.csv", comp / "test_series", retry, TMP / "test-cache-relaxed", False,
                             pre_src, deps, workers, "preprocess-test-relaxed")
        manifests.append(TMP / "test-cache-relaxed/manifest.json")
    preprocess_seconds = time.time() - t0
    predicted = predict(manifests, studies, TMP / "test-predictions.jsonl", model_src, deps, asset)
    probs = read_jsonl(TMP / "test-predictions.jsonl")
    targets = columns[1:]
    assert targets == predicted["targets"], "submission columns differ from the model's targets"
    missing = [u for u in studies if u not in probs]
    assert len(missing) <= MAX_FALLBACK_FRACTION * len(studies), f"{len(missing)} test studies could not be preprocessed"
    fallback = {t: sum(p[t] for p in probs.values()) / len(probs) if probs else 0.5 for t in targets}
    with (WORK / "submission.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for uid in studies:
            row = probs.get(uid, fallback)
            writer.writerow([uid, *[repr(float(row[t])) for t in targets]])
    with (WORK / "submission.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [r["StudyInstanceUID"] for r in rows] == studies
    assert all(math.isfinite(float(r[t])) and 0 <= float(r[t]) <= 1 for r in rows for t in targets)
    result.update(status="ok", n_test=len(studies), n_strict=len(built & set(studies)),
                  n_relaxed=0 if relaxed is None else relaxed["n_studies"], n_fallback=len(missing),
                  fallback_uids=missing, strict_quarantine=strict["quarantine"],
                  preprocess_seconds=round(preprocess_seconds, 1),
                  seconds_per_test_study=round((time.time() - t0) / len(studies), 3),
                  wall_seconds=round(time.time() - started, 1), device=predicted["device"])
    atomic_json(WORK / "offline-result.json", result)
    print(json.dumps({k: result[k] for k in ("status", "n_test", "n_strict", "n_relaxed", "n_fallback",
                                             "seconds_per_test_study", "wall_seconds")}), flush=True)
    shutil.rmtree(TMP, ignore_errors=True)


if __name__ == "__main__":
    main()
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment")
    parser.add_argument("--slug", required=True)
    parser.add_argument("--parity-file", type=Path, help="rehearsal only: replaces the reviewed gold predictions")
    args = parser.parse_args()
    upload, kernel = build(args.experiment, args.slug, args.parity_file)
    print(upload)
    print(kernel)


if __name__ == "__main__":
    main()
