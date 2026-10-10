"""Execute a generated Colab notebook locally before handing it to Colab.

Runs the config, payload, train and collect cells verbatim. Only the Colab-specific
setup cell (GPU/Drive/pip) and the Kaggle download are replaced: the cache parts
are tiny synthetic DICOM caches built with the experiment's own slot/size/crop/
intensity policy and sharded like the real cache. After the payload cell verifies
the embedded source, labels and folds, synthetic folds/labels/train UIDs are swapped
in and the run uses the CPU pilot profile for one epoch. A pass means the notebook
code paths work end to end; it says nothing about model quality.

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


def rehearse(exp_id: str, scratch: Path) -> None:
    nb = json.loads((ROOT / "notebooks/generated" / f"{exp_id}.ipynb").read_text())
    src = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    root, uids = synthetic_inputs(scratch)
    import torch
    torch.cuda.get_device_name = lambda *a: "local-rehearsal"
    torch.cuda.max_memory_allocated = lambda *a: 0
    run = scratch / f"run-{exp_id}"
    shutil.rmtree(run, ignore_errors=True)
    ns: dict = {"__name__": "__main__"}
    exec(src[0], ns)  # config cell
    work = run / "work"
    work.mkdir(parents=True)
    ns.update(WORK=work, sys=sys, os=__import__("os"), json=json, Path=Path, shutil=shutil, subprocess=subprocess,
              torch=torch, INPUT_ROOT=run / "input-root")
    exec(src[3], ns)  # payload cell: unpack and hash-verify embedded source/labels/folds
    exp = ns["EXPERIMENT"]
    shutil.copy(scratch / "folds.csv", ns["FOLDS"])
    shutil.copy(scratch / "labels.jsonl", ns["LABELS"])
    if ns.get("SOFT") is not None:
        shutil.copy(scratch / "soft.jsonl", ns["SOFT"])
    exp["train_uids"] = [u for k, u in enumerate(uids) if k >= 3 and k % 5 != 0]
    exp["train"].update(profile="pilot", epochs=1, effective_batch=2)
    ns["part_manifests"] = synthetic_parts(scratch, exp, root, uids)
    shutil.copy(ROOT / "artifacts/weights/dinov2_vits14_pretrain.pth", work / "dinov2_vits14_pretrain.pth")
    exec(src[4], ns)  # train cell
    exec(src[5], ns)  # collect cell
    print(json.dumps({"experiment": exp_id, "rehearsal": "ok", "archive_bytes": ns["archive_path"].stat().st_size}))


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
