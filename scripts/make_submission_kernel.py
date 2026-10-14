"""Build a private, internet-off Kaggle submission kernel for one reviewed Colab arm.

Usage: python scripts/make_submission_kernel.py D3 --slug rsna-d3-submit-20261012
       python scripts/make_submission_kernel.py D3LR2 LR2F1 LR2F2 LR2F3 LR2F4 --slug <slug>   # equal-weight mean

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
READER = "goodpjw2008/rsna-knee-2-5d-convnext-reader"  # eligible public reader (2026-10-14 check)
READER_MODES = ("rank50", "reader_only", "route")  # route: listed findings from ours, the rest from the reader
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


def _member(exp: str, parity_dir: Path | None) -> dict:
    out = ROOT / "artifacts/colab" / f"p1-{exp}" / "output"
    review = json.loads((out / "review.json").read_text())
    ran = json.loads((out / "experiment.json").read_text())
    build_info = json.loads((ROOT / "notebooks/generated" / f"{exp}.build.json").read_text())
    metrics = json.loads((out / "job/metrics.json").read_text())
    if review["experiment"] != exp or ran["id"] != exp or ran["source_sha256"] != build_info["source_sha256"]:
        raise SystemExit(f"{exp}: reviewed result, run record and notebook build disagree")
    checkpoint = out / "job/checkpoint.pt"
    checkpoint_sha = _sha(checkpoint.read_bytes())
    if checkpoint_sha != metrics["oof_provenance"]["checkpoint_sha256"]:
        raise SystemExit(f"{exp}: checkpoint does not match the evaluated checkpoint")
    if parity_dir is None:
        rows = [json.loads(x) for x in (out / "job/predictions.jsonl").read_text().splitlines()]
        parity = "".join(json.dumps(r) + "\n" for r in rows if r["split"] == "gold").encode()
    else:
        parity = (parity_dir / f"{exp}.jsonl").read_bytes()
    if not parity.strip():
        raise SystemExit(f"{exp}: no parity predictions")
    model = training_source(exp, build_info)
    return {"exp": exp, "checkpoint": checkpoint, "checkpoint_sha256": checkpoint_sha, "model": model,
            "parity": parity, "cache": ran["cache_manifest_sha256"], "git_commit": ran["git_commit"],
            "gold": review["splits"]["gold_all"]["macro_auc"]}


def build(exps: list[str], slug: str, parity_dir: Path | None = None, reader_eval: Path | None = None,
          reader_mode: str | None = None, reader_weight: float = 0.5,
          route_ours: tuple[str, ...] = ()) -> tuple[Path, Path]:
    """One reviewed arm, or several averaged with equal weight; all must share one training cache.

    With reader_eval (the output folder of scripts/make_reader_eval_kernel.py) the public reader is
    pinned by its recorded file hashes, parity-checked against its recorded gold predictions, and
    combined with ours per finding: rank50 = (1 - w) rank(ours) + w rank(reader), w = reader_weight
    (0.5 unless set); reader_only = reader.
    """
    if (reader_eval is None) != (reader_mode is None) or (reader_mode and reader_mode not in READER_MODES):
        raise SystemExit("give --reader-eval and --reader-mode together")
    if not 0 < reader_weight < 1:
        raise SystemExit("reader weight must be strictly between 0 and 1")
    from rsna_knee.ontology import TARGET_COLUMNS
    if (reader_mode == "route") != bool(route_ours) or set(route_ours) - set(TARGET_COLUMNS):
        raise SystemExit("route mode needs --route-ours with valid finding names (and only route mode takes them)")
    members = [_member(exp, parity_dir) for exp in exps]
    if len({m["cache"] for m in members}) != 1 or len(set(exps)) != len(exps):
        raise SystemExit("ensemble members must be distinct and share one training cache")
    pre, policy, preprocess_hash, pixel_versions = preprocess_source(members[0]["cache"])

    asset = ROOT / "artifacts/kaggle" / f"{slug}-asset"
    staged = asset / "bundle"
    if staged.exists():
        shutil.rmtree(staged)
    (staged / "wheels").mkdir(parents=True)
    # .blob, not .tar.gz: Kaggle may unpack archives inside an uploaded dataset.
    (staged / "preprocess-source.blob").write_bytes(pre)
    for m in members:
        (staged / f"model-source-{_sha(m['model'])[:16]}.blob").write_bytes(m["model"])
        (staged / "members" / m["exp"]).mkdir(parents=True)
        shutil.copyfile(m["checkpoint"], staged / "members" / m["exp"] / "checkpoint.pt")
        (staged / "members" / m["exp"] / "parity.jsonl").write_bytes(m["parity"])
    reader = None
    if reader_eval is not None:
        pinned = json.loads((reader_eval / "reader-files.json").read_text())
        if pinned["dataset"] != READER or "infer.py" not in pinned["files"]:
            raise SystemExit("reader-files.json does not describe the vetted reader")
        (staged / "reader").mkdir()
        shutil.copyfile(reader_eval / "reader-gold.csv", staged / "reader" / "parity.csv")
        reader = {"dataset": READER, "files": pinned["files"], "mode": reader_mode, "weight": reader_weight,
                  "route_ours": sorted(route_ours),
                  "checkpoints": [f"cnxt_v0_fold{k}.pt" for k in range(3)]}
    for wheel in sorted(WHEELS.glob("*.whl")):
        shutil.copyfile(wheel, staged / "wheels" / wheel.name)
    files = {str(p.relative_to(staged)): _sha(p.read_bytes()) for p in sorted(staged.rglob("*")) if p.is_file()}
    identity = {"experiment": "+".join(exps), "ensemble": "equal-weight mean of probabilities" if len(exps) > 1 else None,
                "members": [{"experiment": m["exp"], "checkpoint_sha256": m["checkpoint_sha256"],
                             "training_source_sha256": _sha(m["model"]), "git_commit": m["git_commit"],
                             "gold_all_macro_auc": m["gold"]} for m in members],
                "preprocess_source_sha256": _sha(pre), "preprocess_hash": preprocess_hash, "policy": policy,
                "training_pixel_versions": pixel_versions, "cache_manifest_sha256": members[0]["cache"],
                "img_size": int(policy["size"]), "codec_pins": CODEC_PINS, "reader": reader}
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
        "competition_sources": [COMPETITION], "dataset_sources": [dataset] + ([READER] if reader else []),
        "kernel_sources": []}, indent=2))
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


READER_RUN = """
import json, os, sys
spec = json.load(open(sys.argv[1]))
sys.path.insert(0, spec["reader_dir"])
os.chdir(spec["workdir"])
import infer  # the reader's own inference code
df = infer.run(spec["data_dir"], spec["split"], spec["ckpts"], studies=spec["studies"])
assert len(df) == len(spec["studies"]) and df.notna().all().all(), "reader did not score every study"
df.to_csv(spec["out"])
print(json.dumps({"n": len(df), "targets": list(df.columns)}), flush=True)
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
    found = [p.parent for p in INPUTS.rglob("MANIFEST.json") if (p.parent / "preprocess-source.blob").is_file()]
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


def predict(manifests, uids, out, member, deps, asset):
    """One member's predictions with that member's own training source."""
    src = unpack(asset / f"model-source-{member['training_source_sha256'][:16]}.blob",
                 TMP / ("model-source-" + member["training_source_sha256"][:16]))
    return run(PREDICT, {"manifests": [str(m) for m in manifests], "study_ids": uids, "out": str(out),
                         "checkpoint": str(asset / "members" / member["experiment"] / "checkpoint.pt"),
                         "img_size": IDENTITY["img_size"]}, src, deps, "predict-" + out.stem)


def predict_all(manifests, uids, stem, deps, asset):
    """Equal-weight mean over members; every member must predict the same studies and targets."""
    per, targets = [], None
    for member in IDENTITY["members"]:
        out = TMP / f"{stem}-{member['experiment']}.jsonl"
        info = predict(manifests, uids, out, member, deps, asset)
        assert targets in (None, info["targets"]), "members disagree on target order"
        targets, device = info["targets"], info["device"]
        per.append(read_jsonl(out))
    assert all(set(p) == set(per[0]) for p in per), "members predicted different study sets"
    mean = {u: {t: sum(p[u][t] for p in per) / len(per) for t in targets} for u in per[0]}
    return mean, per, targets, device


def check_policy(built, name):
    """Code, policy and slots must hash to training; differing numpy/Pillow/pydicom must pass pixel parity."""
    assert built["hash_with_training_versions"] == IDENTITY["preprocess_hash"], f"{name} preprocessing differs from training"
    return built["pixel_runtime_versions"] == IDENTITY["training_pixel_versions"]


def find_reader():
    spec = IDENTITY["reader"]
    found = [p.parent for p in INPUTS.rglob("infer.py") if (p.parent / spec["checkpoints"][0]).is_file()]
    assert len(found) == 1, f"expected one reader dataset, found {found}"
    for name, expected in spec["files"].items():
        assert sha256(found[0] / name) == expected, f"reader file changed since it was vetted: {name}"
    return found[0]


def reader_predict(reader, comp, split, uids, name, deps):
    out = TMP / f"{name}.csv"
    (TMP / name).mkdir(parents=True, exist_ok=True)
    info = run(READER_RUN, {"reader_dir": str(reader), "workdir": str(TMP / name), "data_dir": str(comp), "split": split,
                            "ckpts": [str(reader / c) for c in IDENTITY["reader"]["checkpoints"]], "studies": uids,
                            "out": str(out)}, reader, deps, name)
    with out.open(newline="") as handle:
        rows = {r["StudyInstanceUID"]: {t: float(r[t]) for t in info["targets"]} for r in csv.DictReader(handle)}
    return rows, info["targets"]


def normalized_ranks(values):
    """Average ranks scaled to (0, 1]; per-finding AUC only depends on order."""
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks, i = [0.0] * len(values), 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = ((i + j) / 2 + 1) / len(values)
        i = j + 1
    return ranks


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
    workers = max(1, min(4, os.cpu_count() or 1))

    can_check = (comp / "train_series.csv").is_file() and (comp / "train_series").is_dir()
    if not RERUN and can_check:
        members = IDENTITY["members"]
        parity = {m["experiment"]: read_jsonl(asset / "members" / m["experiment"] / "parity.jsonl") for m in members}
        uids = sorted(parity[members[0]["experiment"]])
        assert all(sorted(p) == uids for p in parity.values()), "members have different parity studies"
        t0 = time.time()
        built = preprocess(comp / "train_series.csv", comp / "train_series", uids, TMP / "parity-cache",
                           True, pre_src, deps, workers, "preprocess-parity")
        result["pixel_versions_match_training"] = check_policy(built, "parity")
        result["pixel_runtime_versions"] = built["pixel_runtime_versions"]
        assert built["n_quarantine"] == 0, f"parity studies quarantined: {built['quarantine']}"
        _, mine, targets, _ = predict_all([TMP / "parity-cache/manifest.json"], uids, "parity", deps, asset)
        result["parity"] = {"n_studies": len(uids), "seconds": None, "members": {}}
        for member, got in zip(members, mine, strict=True):
            ref = parity[member["experiment"]]
            assert set(got) == set(ref), f"{member['experiment']}: parity prediction set differs"
            diffs = [abs(got[u][t] - ref[u][t]) for u in ref for t in targets]
            result["parity"]["members"][member["experiment"]] = {"max_abs": max(diffs), "mean_abs": sum(diffs) / len(diffs)}
        result["parity"]["max_abs"] = max(v["max_abs"] for v in result["parity"]["members"].values())
        result["parity"]["mean_abs"] = max(v["mean_abs"] for v in result["parity"]["members"].values())
        result["parity"]["seconds"] = round(time.time() - t0, 1)
        atomic_json(WORK / "offline-result.json", result)
        print("parity", json.dumps(result["parity"]), flush=True)
        assert result["parity"]["max_abs"] <= PARITY_MAX_ABS and result["parity"]["mean_abs"] <= PARITY_MEAN_ABS, \
            "test-time pipeline does not reproduce the reviewed Colab predictions"
        if IDENTITY["reader"]:
            with (asset / "reader" / "parity.csv").open(newline="") as handle:
                ref = {r["StudyInstanceUID"]: r for r in csv.DictReader(handle)}
            got, rt = reader_predict(find_reader(), comp, "train", sorted(ref), "reader-parity", deps)
            diffs = [abs(got[u][t] - float(ref[u][t])) for u in ref for t in rt]
            result["reader_parity"] = {"n_studies": len(ref), "max_abs": max(diffs), "mean_abs": sum(diffs) / len(diffs)}
            atomic_json(WORK / "offline-result.json", result)
            print("reader parity", json.dumps(result["reader_parity"]), flush=True)
            assert result["reader_parity"]["max_abs"] <= PARITY_MAX_ABS and \
                result["reader_parity"]["mean_abs"] <= PARITY_MEAN_ABS, "reader does not reproduce its vetted gold predictions"

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
    probs, _, model_targets, device = predict_all(manifests, studies, "test", deps, asset)
    targets = columns[1:]
    assert targets == model_targets, "submission columns differ from the model's targets"
    missing = [u for u in studies if u not in probs]
    assert len(missing) <= MAX_FALLBACK_FRACTION * len(studies), f"{len(missing)} test studies could not be preprocessed"
    fallback = {t: sum(p[t] for p in probs.values()) / len(probs) if probs else 0.5 for t in targets}
    final = {uid: probs.get(uid, fallback) for uid in studies}
    if IDENTITY["reader"]:
        theirs, rt = reader_predict(find_reader(), comp, "test", studies, "reader-test", deps)
        assert sorted(rt) == sorted(targets), "reader targets differ from the submission columns"
        if IDENTITY["reader"]["mode"] == "reader_only":
            final = {uid: theirs[uid] for uid in studies}
        elif IDENTITY["reader"]["mode"] == "route":  # per-finding AUC: each column comes whole from one model
            ours = set(IDENTITY["reader"]["route_ours"])
            final = {uid: {t: (final[uid][t] if t in ours else theirs[uid][t]) for t in targets} for uid in studies}
        else:  # per-finding rank average, equal weight
            w = IDENTITY["reader"]["weight"]
            mixed = {uid: {} for uid in studies}
            for t in targets:
                ro = normalized_ranks([final[u][t] for u in studies])
                rr = normalized_ranks([theirs[u][t] for u in studies])
                for u, a, b in zip(studies, ro, rr, strict=True):
                    mixed[u][t] = (1 - w) * a + w * b
            final = mixed
        result["reader_mode"] = IDENTITY["reader"]["mode"]
    with (WORK / "submission.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for uid in studies:
            writer.writerow([uid, *[repr(float(final[uid][t])) for t in targets]])
    with (WORK / "submission.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [r["StudyInstanceUID"] for r in rows] == studies
    assert all(math.isfinite(float(r[t])) and 0 <= float(r[t]) <= 1 for r in rows for t in targets)
    result.update(status="ok", n_test=len(studies), n_strict=len(built & set(studies)),
                  n_relaxed=0 if relaxed is None else relaxed["n_studies"], n_fallback=len(missing),
                  fallback_uids=missing, strict_quarantine=strict["quarantine"],
                  preprocess_seconds=round(preprocess_seconds, 1),
                  seconds_per_test_study=round((time.time() - t0) / len(studies), 3),
                  wall_seconds=round(time.time() - started, 1), device=device, n_members=len(IDENTITY["members"]))
    atomic_json(WORK / "offline-result.json", result)
    print(json.dumps({k: result[k] for k in ("status", "n_test", "n_strict", "n_relaxed", "n_fallback",
                                             "seconds_per_test_study", "wall_seconds")}), flush=True)
    if os.environ.get("RSNA_KEEP_TMP") != "1":  # local rehearsal keeps per-member predictions
        shutil.rmtree(TMP, ignore_errors=True)


if __name__ == "__main__":
    main()
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiments", nargs="+", help="one reviewed arm, or several to average")
    parser.add_argument("--slug", required=True)
    parser.add_argument("--parity-dir", type=Path, help="rehearsal only: <experiment>.jsonl files replace the reviewed gold predictions")
    parser.add_argument("--reader-eval", type=Path, help="output folder of the reader gold-eval kernel")
    parser.add_argument("--reader-mode", choices=READER_MODES)
    parser.add_argument("--reader-weight", type=float, default=0.5, help="reader share in the rank50 blend")
    parser.add_argument("--route-ours", nargs="+", default=[], help="route mode: findings taken from our model")
    args = parser.parse_args()
    upload, kernel = build(args.experiments, args.slug, args.parity_dir, args.reader_eval, args.reader_mode,
                           args.reader_weight, tuple(args.route_ours))
    print(upload)
    print(kernel)


if __name__ == "__main__":
    main()
