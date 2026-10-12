"""Execute a generated Colab notebook locally before handing it to Colab.

Runs the config, payload, train and collect cells verbatim. Only the Colab-specific
setup cell (GPU/Drive/pip) and the Kaggle download are replaced: the cache parts
are tiny synthetic DICOM caches built with the experiment's own slot/size/crop/
intensity policy and sharded like the real cache. After the payload cell verifies
the embedded source, labels and folds, synthetic folds/labels/train UIDs are swapped
in and the run uses the CPU pilot profile for one epoch. A pass means the notebook
code paths work end to end; it says nothing about model quality.

Each notebook runs twice in a fresh interpreter with an empty working directory (as in
Colab): first with the weight allowlist made unavailable, which must fail at the weight
check, then again in the same output folder, which must finish — the "Run All again
after a failure" path.

Usage: python scripts/rehearse_colab_notebook.py D1 [D0 L1 D2 ...] --scratch /path/to/tmp
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tests.test_cache_manifest import _write_slice  # noqa: E402

from rsna_knee.data import cache_build  # noqa: E402
from rsna_knee.ontology import TARGET_COLUMNS  # noqa: E402

SERIES = {"sag-fs": ("Sagittal", "1"), "sag-t1": ("Sagittal", "0"), "cor-fs": ("Coronal", "1"),
          "cor-t1": ("Coronal", "0"), "ax-fs": ("Axial", "1")}
N_STUDIES = 15


def synthetic_inputs(scratch: Path) -> tuple[Path, list[str]]:
    root = scratch / "input"
    uids = [f"1.2.{k}" for k in range(N_STUDIES)]
    if (root / "train_series.csv").is_file():
        return root, uids
    rows, rng = [], np.random.default_rng(0)
    for uid in uids:
        for name, (plane, fluid) in SERIES.items():
            folder = root / "train_series" / uid / name
            folder.mkdir(parents=True)
            for i in range(1, 9):
                _write_slice(folder / f"{i}.dcm", uid, name, i, 3.0 * i)
            rows.append({"StudyInstanceUID": uid, "SeriesInstanceUID": name, "Fluid_Sensitive": fluid,
                         "Fat_Suppression": fluid, "Anatomical_Plane": plane})
    with (root / "train_series.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (scratch / "folds.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["StudyInstanceUID", "fold", "is_gold", "gold_split", "group_id", "gold_set"])
        writer.writeheader()
        for k, uid in enumerate(uids):
            gold = k < 3
            writer.writerow({"StudyInstanceUID": uid, "fold": str(k % 5), "is_gold": int(gold),
                             "gold_split": "eval" if gold else "weak", "group_id": uid,
                             "gold_set": ("A_historic_eval" if k < 2 else "B_former_train") if gold else ""})
    with (scratch / "labels.jsonl").open("w") as hard, (scratch / "soft.jsonl").open("w") as soft:
        for k, uid in enumerate(uids):
            y = {t: float(rng.integers(0, 2)) for t in TARGET_COLUMNS}
            hard.write(json.dumps({"StudyInstanceUID": uid, "source": "gold" if k < 3 else "weak", "targets": y,
                                   "y_mask": dict.fromkeys(TARGET_COLUMNS, 1.0),
                                   "y_weight": dict.fromkeys(TARGET_COLUMNS, 1.0), "ontology_version": "x"}) + "\n")
            if k >= 3:
                soft.write(json.dumps({"StudyInstanceUID": uid,
                                       "soft_targets": {t: float(rng.uniform()) for t in TARGET_COLUMNS}}) + "\n")
    return root, uids


def synthetic_parts(scratch: Path, exp: dict, root: Path, uids: list[str]) -> list[Path]:
    config = yaml.safe_load((ROOT / exp["config_file"]).read_text())
    cache = config["cache"]
    slots = [tuple(s.split(":")) for s in cache["slots"]]
    n_parts = len(exp["cache_parts"])
    parts = []
    for i in range(n_parts):
        dest = scratch / f"{exp['id']}-part{i}"
        shutil.rmtree(dest, ignore_errors=True)
        cache_build.build_cache(series_csv=root / "train_series.csv", dicom_root=root / "train_series", dest=dest,
                                study_ids=uids[i::n_parts], size=int(cache["img_size"]), n_centers=int(cache["centers"]),
                                slots=slots, crop_mm=float(cache.get("crop_mm", 160.0)),
                                intensity=cache.get("intensity", "slice"), progress_every=100)
        parts.append(dest / "manifest.json")
    return parts


# Runs in a fresh interpreter whose working directory is empty, like Colab's /content:
# rsna_knee must come from the archive embedded in the notebook, never from this repo.
CHILD = r"""
import json, os, shutil, subprocess, sys
from pathlib import Path
args = json.loads(Path(sys.argv[1]).read_text())
assert 'rsna_knee' not in sys.modules
import torch
torch.cuda.get_device_name = lambda *a: 'local-rehearsal'
torch.cuda.max_memory_allocated = lambda *a: 0
src = args['cells']
ns = {'__name__': '__main__'}
exec(src[0], ns)
work = Path(args['work']); work.mkdir(parents=True, exist_ok=True)
ns.update(WORK=work, sys=sys, os=os, json=json, Path=Path, shutil=shutil, subprocess=subprocess, torch=torch,
          INPUT_ROOT=Path(args['input_root']))
exec(src[3], ns)
assert Path(ns['SOURCE']).resolve() == Path.cwd().resolve(), 'payload cell must chdir into the embedded source'
import rsna_knee
assert Path(rsna_knee.__file__).resolve().is_relative_to(Path(ns['SOURCE']).resolve()), rsna_knee.__file__
shutil.copy(args['folds'], ns['FOLDS']); shutil.copy(args['labels'], ns['LABELS'])
if ns.get('SOFT') is not None: shutil.copy(args['soft'], ns['SOFT'])
exp = ns['EXPERIMENT']
exp['train_uids'] = args['train_uids']
exp['train'].update(profile='pilot', epochs=1, effective_batch=2)
ns['part_manifests'] = [Path(p) for p in args['parts']]
shutil.copy(args['weights'], work / exp.get('weights_file', 'dinov2_vits14_pretrain.pth'))
if args['break_allowlist']:
    os.environ['RSNA_DINOV2_ALLOWLIST'] = str(work / 'missing-allowlist.json')
    os.environ['RSNA_' + exp['train']['encoder'].upper() + '_ALLOWLIST'] = str(work / 'missing-allowlist.json')
try:
    exec(src[4], ns)
except AssertionError as error:
    print(json.dumps({'train_cell': 'assertion', 'detail': str(error)})); sys.exit(3)
exec(src[5], ns)
print(json.dumps({'train_cell': 'ok', 'archive_bytes': ns['archive_path'].stat().st_size}))
"""


def _child(exp_id: str, src: list[str], run: Path, scratch: Path, parts: list[Path], uids: list[str],
           *, weights: Path, break_allowlist: bool, outer_fold: str) -> subprocess.CompletedProcess:
    cwd = run / "empty-cwd"
    cwd.mkdir(parents=True, exist_ok=True)
    spec = {"cells": src, "work": str(run / "work"), "input_root": str(run / "input-root"),
            "folds": str(scratch / "folds.csv"), "labels": str(scratch / "labels.jsonl"),
            "soft": str(scratch / "soft.jsonl"), "parts": [str(p) for p in parts],
            "train_uids": [u for k, u in enumerate(uids) if k >= 3 and str(k % 5) != outer_fold],
            "weights": str(weights), "break_allowlist": break_allowlist}
    (run / "spec.json").write_text(json.dumps(spec))
    (run / "child.py").write_text(CHILD)
    return subprocess.run([sys.executable, "-I", str(run / "child.py"), str(run / "spec.json")], cwd=cwd,
                          capture_output=True, text=True)


def rehearse(exp_id: str, scratch: Path) -> None:
    nb = json.loads((ROOT / "notebooks/generated" / f"{exp_id}.ipynb").read_text())
    src = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    exp = json.loads((ROOT / "notebooks/generated" / f"{exp_id}.build.json").read_text())
    root, uids = synthetic_inputs(scratch)
    run = scratch / f"run-{exp_id}"
    shutil.rmtree(run, ignore_errors=True)
    parts = synthetic_parts(scratch, {"id": exp_id, **exp}, root, uids)
    weights = ROOT / "artifacts/weights" / exp.get("weights_file", "dinov2_vits14_pretrain.pth")  # real pretrained file
    # 1) a run that fails at the weight check, 2) Run All again in the same Drive folder.
    outer_fold = str(exp.get("outer_fold", "0"))
    failed = _child(exp_id, src, run, scratch, parts, uids, weights=weights, break_allowlist=True, outer_fold=outer_fold)
    if failed.returncode != 3 or "unverified_weights" not in failed.stdout:
        raise SystemExit(f"{exp_id}: expected an unverified_weights failure\n{failed.stdout[-1500:]}\n{failed.stderr[-3000:]}")
    done = _child(exp_id, src, run, scratch, parts, uids, weights=weights, break_allowlist=False, outer_fold=outer_fold)
    if done.returncode != 0:
        raise SystemExit(f"{exp_id}: rehearsal failed\n{done.stdout[-1500:]}\n{done.stderr[-4000:]}")
    print(json.dumps({"experiment": exp_id, "rehearsal": "ok", "rerun_after_failure": "ok",
                      "child": json.loads(done.stdout.strip().splitlines()[-1])}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiments", nargs="+")
    parser.add_argument("--scratch", type=Path, required=True)
    args = parser.parse_args()
    args.scratch.mkdir(parents=True, exist_ok=True)
    for exp_id in args.experiments:
        rehearse(exp_id, args.scratch)


if __name__ == "__main__":
    main()
