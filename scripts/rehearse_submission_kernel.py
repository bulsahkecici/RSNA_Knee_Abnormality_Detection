"""Rehearse a KneeNet submission kernel locally (CPU) before pushing it.

Usage: python scripts/rehearse_submission_kernel.py K5ALL [K5ALLS2 ...] --scratch <dir>

Builds the kernel for the reviewed KneeNet arm(s) and runs its entry.py against the synthetic DICOM studies of
rehearse_colab_notebook.py laid out as competition input (test_series/ for the test studies, train_series/ for
parity). Pass 1 runs as the competition rerun (no parity) and keeps per-member predictions; each member's must equal
the notebook's own evaluation path (its V5Study class, executed from the generated notebook, every cached window,
fp32, same checkpoint) on the kernel's test cache, so the kernel feeds the model exactly what the notebook evaluated.
Pass 2 takes pass-1 predictions of three studies as the parity reference and runs the interactive path, which must
pass parity and write a valid submission.csv. Nothing is pushed; the rehearsal kernel folders are removed.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

import make_submission_kernel as msk
import rehearse_colab_notebook as rh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "state/external-code/goodpjw2008-reader"))
import knee  # noqa: E402  (sha256 checked against the notebook's embedded copy below)

from rsna_knee.ontology import TARGET_COLUMNS  # noqa: E402
from rsna_knee.runtime.train_payload import load_volume  # noqa: E402


def competition_input(scratch: Path) -> tuple[Path, list[str]]:
    root, uids = rh.synthetic_inputs(scratch)
    comp = scratch / "kernel-input/competition"
    if not (comp / "test.csv").is_file():
        shutil.copytree(root / "train_series", comp / "train_series")
        shutil.copytree(root / "train_series", comp / "test_series")
        shutil.copy(root / "train_series.csv", comp / "train_series.csv")
        shutil.copy(root / "train_series.csv", comp / "test_series.csv")
        (comp / "test.csv").write_text("StudyInstanceUID\n" + "".join(u + "\n" for u in uids))
        (comp / "sample_submission.csv").write_text(",".join(["StudyInstanceUID", *TARGET_COLUMNS]) + "\n")
    return comp, uids


def run_kernel(kernel: Path, upload: Path, comp: Path, scratch: Path, name: str, rerun: bool) -> dict:
    inputs = scratch / f"kernel-input-{name}"
    shutil.rmtree(inputs, ignore_errors=True)
    (inputs / "asset").mkdir(parents=True)
    shutil.copytree(comp, inputs / "competition")  # rglob does not descend into symlinked folders
    shutil.copy(upload / "offline-bundle.tar.gz", inputs / "asset/offline-bundle.tar.gz")
    work, tmp = scratch / f"kernel-work-{name}", scratch / f"kernel-tmp-{name}"
    for d in (work, tmp):
        shutil.rmtree(d, ignore_errors=True)
    env = dict(os.environ, RSNA_INPUT_DIR=str(inputs), RSNA_WORK_DIR=str(work), RSNA_TMP_DIR=str(tmp),
               RSNA_SKIP_WHEELS="1", RSNA_KEEP_TMP="1")
    env.pop("KAGGLE_IS_COMPETITION_RERUN", None)
    if rerun:
        env["KAGGLE_IS_COMPETITION_RERUN"] = "1"
    done = subprocess.run([sys.executable, str(kernel / "entry.py")], env=env, capture_output=True, text=True, cwd=scratch)
    if done.returncode:
        raise SystemExit(f"kernel {name} failed\n{done.stdout[-3000:]}\n{done.stderr[-4000:]}")
    result = json.loads((work / "offline-result.json").read_text())
    assert result["status"] == "ok", result
    return {"result": result, "work": work, "tmp": tmp}


def notebook_eval(exp: str, manifest_path: Path, uids: list[str]) -> dict[str, np.ndarray]:
    """The generated notebook's V5Study evaluation, with that run's checkpoint, in strict fp32."""
    nb = json.loads((ROOT / "notebooks/generated" / f"{exp}.ipynb").read_text())
    cell = next("".join(c["source"]) for c in nb["cells"] if "class V5Study" in "".join(c["source"]))
    if msk._sha(Path(knee.__file__).read_bytes()) != msk._sha(msk.notebook_blob(exp, "KNEE_B64")):
        raise SystemExit("local knee.py differs from the notebook's embedded reader code")
    manifest = json.loads(manifest_path.read_text())
    policy = manifest["preprocess_config"]
    n_c = int(policy["n_centers"])
    ns = {"np": np, "torch": torch, "knee": knee, "load_volume": load_volume, "TARGET_COLUMNS": TARGET_COLUMNS,
          "STUDIES": manifest["studies"], "CACHE": manifest_path.parent, "N_C": n_c, "SOFT_T": {},
          "SLOT_MAP": [knee.slot_of(plane, kind == "fluid") for plane, kind in policy["slots"]],
          "POS": np.linspace(float(policy["center_lower"]), float(policy["center_upper"]), n_c).astype(np.float32)}
    exec(cell[cell.index("class V5Study"):cell.index("\nseed = int(")], ns)
    ck = torch.load(ROOT / "artifacts/colab" / f"p1-{exp}" / "output/job/checkpoint.pt", map_location="cpu", weights_only=False)
    cfg = ck["kneenet"]
    model = knee.KneeNet(cfg["backbone"], pretrained=False, dim=int(cfg["dim"]), depth=int(cfg["depth"]))
    model.load_state_dict(ck["model"])
    model = model.to(memory_format=torch.channels_last).eval()
    out = {}
    with torch.no_grad():
        for x, mask, pos, _, _, ids in torch.utils.data.DataLoader(ns["V5Study"](uids, n_c, False), batch_size=2):
            p = model(x, mask, pos, res=int(cfg["res"])).float().sigmoid().numpy()
            out.update(zip(ids, p))
    return out


def read_jsonl(path: Path) -> dict[str, np.ndarray]:
    return {r["StudyInstanceUID"]: np.array([r[t] for t in TARGET_COLUMNS]) for r in map(json.loads, path.read_text().splitlines())}


def rehearse(exps: list[str], scratch: Path) -> None:
    comp, uids = competition_input(scratch)
    slug = "rehearsal-" + "-".join(e.lower() for e in exps)
    upload, kernel = msk.build(exps, slug)
    try:
        first = run_kernel(kernel, upload, comp, scratch, "rerun", rerun=True)
        manifest = first["tmp"] / "test-cache/manifest.json"
        worst = 0.0
        for exp in exps:
            got = read_jsonl(first["tmp"] / f"test-{exp}.jsonl")
            ref = notebook_eval(exp, manifest, sorted(got))
            assert set(got) == set(uids) == set(ref), "kernel did not predict every synthetic test study"
            worst = max(worst, max(float(np.abs(got[u] - ref[u]).max()) for u in uids))
        assert worst < 1e-4, f"kernel predictions differ from the notebook evaluation path: max abs {worst}"
        parity_dir = scratch / "kernel-parity"
        parity_dir.mkdir(exist_ok=True)
        for exp in exps:
            rows = [json.loads(x) for x in (first["tmp"] / f"test-{exp}.jsonl").read_text().splitlines()]
            (parity_dir / f"{exp}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows if r["StudyInstanceUID"] in uids[:3]))
        upload, kernel = msk.build(exps, slug, parity_dir)
        second = run_kernel(kernel, upload, comp, scratch, "interactive", rerun=False)
        result = second["result"]
        assert result["parity"]["n_studies"] == 3 and result["parity"]["max_abs"] <= 1e-6, result["parity"]
        with (second["work"] / "submission.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert [r["StudyInstanceUID"] for r in rows] == uids and result["n_fallback"] == 0
        print(json.dumps({"experiments": exps, "rehearsal": "ok", "kernel_vs_notebook_max_abs": worst,
                          "parity": result["parity"], "n_test": result["n_test"],
                          "members": [(m["experiment"], m["kind"], m["parity_reference"]) for m in result["identity"]["members"]]}))
    finally:
        for path in (kernel, upload.parent):
            shutil.rmtree(path, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiments", nargs="+")
    parser.add_argument("--scratch", type=Path, required=True)
    args = parser.parse_args()
    rehearse(args.experiments, args.scratch)


if __name__ == "__main__":
    main()
