"""Rehearse a generated KneeNet notebook locally (CPU) before handing it over.

Usage: python scripts/rehearse_kneenet_notebook.py K5F0 --scratch <dir>

Builds tiny synthetic DICOM studies, caches them with the experiment's own v5 policy (as many parts as the
notebook downloads), then runs the payload and training cells verbatim except for CUDA-only calls, which are
swapped for CPU stand-ins (device, capability, memory, name). Synthetic UIDs and soft labels replace the real
ones; two epochs, batch 2, 64 px, 2 windows per slot. One run is stopped after epoch 0 (a simulated disconnect) and resumed in the same
work folder; it must end bit-identical (EMA weights and predictions) to an uninterrupted run in another folder,
which checks that the saved state holds everything the second epoch depends on. Finally the collect cell runs
and the result files are checked the way review_experiment.py reads them. Real pretrained ConvNeXt weights from
artifacts/weights/ are loaded.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import rehearse_colab_notebook as rh
from make_submission_kernel import notebook_cells
import torch

ROOT = Path(__file__).resolve().parents[1]
STATE_SAVED = "    os.replace(str(state_path) + '.tmp', state_path)\n"
CPU = [("torch.device('cuda')", "torch.device('cpu')"),
       ("torch.cuda.get_device_capability(0)", "(8, 0)"),
       ("torch.cuda.get_device_properties(0).total_memory", "16e9"),
       ("torch.cuda.get_device_name(0)", "'local-cpu'"),
       ("torch.cuda.max_memory_allocated()", "0"),
       ("torch.cuda.synchronize()", "None"),
       ("torch.cuda.get_rng_state_all()", "[]"),
       ("torch.cuda.set_rng_state_all(st['cuda_rng'])", "None"),
       ("torch.autocast('cuda', dtype=amp_dtype)", "torch.autocast('cpu', dtype=amp_dtype)"),
       ("torch.autocast('cuda', dtype=amp_dtype, enabled=not strict_eval)",
        "torch.autocast('cpu', dtype=amp_dtype, enabled=not strict_eval)"),
       ("max(2, min(8, os.cpu_count() or 2))", "0")]

CHILD = r"""
import json, os, shutil, sys
from pathlib import Path
args = json.loads(Path(sys.argv[1]).read_text())
src = args['cells']
ns = {'__name__': '__main__'}
exec(src[0], ns)
work = Path(args['work']); work.mkdir(parents=True, exist_ok=True)
import torch
ns.update(WORK=work, sys=sys, os=os, json=json, Path=Path, shutil=shutil, torch=torch, INPUT_ROOT=Path(args['input_root']))
exec(src[3], ns)
import rsna_knee, knee
assert Path(rsna_knee.__file__).resolve().is_relative_to(Path(ns['SOURCE']).resolve()), rsna_knee.__file__
assert Path(knee.__file__).resolve().is_relative_to(Path(ns['SOURCE']).resolve()), knee.__file__
exp = ns['EXPERIMENT']
exp['train_uids'], exp['gold_uids'], exp['weak_uids'] = args['train_uids'], args['gold_uids'], args['weak_uids']
exp['train'].update(epochs=2, batch=2, res=64, k=2)  # small enough for a CPU; resume logic does not depend on size
if exp.get('runtime_guard'):
    exp['runtime_guard'].update(warmup_micro_batches=2, measure_micro_batches=2,
                                max_projected_training_seconds=1e9)
ns['DIE_AFTER_EPOCH'] = args['die_after']
ns['SOFT'].write_text(Path(args['soft']).read_text())
ns['part_manifests'] = [Path(p) for p in args['parts']]
if exp.get('cache_access') == 'mounted_symlinks':
    import hashlib
    exp['cache_parts'] = [{'kernel': f'synthetic/rehearsal-{i}',
                           'manifest_sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                          for i, p in enumerate(ns['part_manifests'])]
    exp['cache_manifest_sha256'] = hashlib.sha256(','.join(p['manifest_sha256'] for p in exp['cache_parts']).encode()).hexdigest()
shutil.copy(args['weights'], work / exp['weights_file'])
exec(src[4], ns)
# Synthetic rehearsal output must never masquerade as production evidence, even in its collected ZIP.
metrics_path = work / 'job/metrics.json'
metrics = json.loads(metrics_path.read_text()); metrics['kind'] = 'synthetic'
metrics_path.write_text(json.dumps(metrics, indent=2))
if (work / 'resource-gate.json').is_file():
    gate = json.loads((work / 'resource-gate.json').read_text())
    gate['kind'] = 'synthetic_engineering_resource_gate'
    (work / 'resource-gate.json').write_text(json.dumps(gate))
(work / 'campaign-result.json').write_text(json.dumps({'exit_reason': 'synthetic_rehearsal', 'model': 'kneenet'}))
exec(src[5], ns)
print(json.dumps({'train_cell': 'ok', 'archive_bytes': ns['archive_path'].stat().st_size}))
"""


def rehearse(exp_id: str, scratch: Path, notebook_dir: Path | None = None) -> None:
    notebook_dir = notebook_dir or ROOT / 'notebooks/generated'
    nb = json.loads((notebook_dir / f"{exp_id}.ipynb").read_text())
    src = [s for c, s in zip(nb['cells'], notebook_cells(exp_id, notebook_dir), strict=True)
           if c['cell_type'] == 'code']
    for old, new in CPU + [(STATE_SAVED, STATE_SAVED + "    if ep == DIE_AFTER_EPOCH: raise SystemExit(3)\n")]:
        if old == 'torch.cuda.synchronize()' and old not in src[4]:
            continue  # archived notebooks predate the optional resource guard
        assert old in src[4], f"training cell no longer contains {old!r}"
        src[4] = src[4].replace(old, new)
    exp = json.loads((notebook_dir / f"{exp_id}.build.json").read_text())
    fixture = scratch / ('canonical-inputs' if exp.get('cache_layout') == 'canonical_volumes' else 'v5-inputs')
    root, uids = rh.synthetic_inputs(fixture)
    run = scratch / f"run-{exp_id}"
    shutil.rmtree(run, ignore_errors=True)
    if exp.get('cache_layout') == 'canonical_volumes':
        import pydicom

        from rsna_knee.data.volume_cache import build_volume_cache
        # Correct plane-specific DICOM geometry; generic v5 fixture is axial in every slot.
        iops = {'sag': [0, 1, 0, 0, 0, -1], 'cor': [1, 0, 0, 0, 0, -1], 'ax': [1, 0, 0, 0, 1, 0]}
        for path in (root / 'train_series').rglob('*.dcm'):
            ds = pydicom.dcmread(path)
            plane = path.parent.name.split('-')[0]
            ds.ImageOrientationPatient = iops[plane]
            ipp = [0., 0., 0.]
            ipp[['sag', 'cor', 'ax'].index(plane)] = float(ds.InstanceNumber) * 3
            ds.ImagePositionPatient = ipp
            ds.PixelSpacing = [19.2, 19.2]
            ds.save_as(path, enforce_file_format=True)
        parts = []
        for i in range(len(exp['cache_parts'])):
            dest = scratch / f'{exp_id}-part{i}'
            shutil.rmtree(dest, ignore_errors=True)
            build_volume_cache(series_csv=root / 'train_series.csv', dicom_root=root / 'train_series',
                               dest=dest, study_ids=uids[i::len(exp['cache_parts'])], size=64)
            parts.append(dest / 'manifest.json')
    else:
        parts = rh.synthetic_parts(scratch, {"id": exp_id, **exp}, root, uids)
    gold, weak = uids[:3], ([] if exp["outer_fold"] == "none" else uids[3:5])
    train = [u for u in uids if u not in gold and u not in weak]
    soft = fixture / "soft.jsonl"
    rows = [json.loads(x) for x in soft.read_text().splitlines()]
    covered = {r["StudyInstanceUID"] for r in rows}
    assert set(train) <= covered | set(uids[:3]), "synthetic soft table misses training studies"
    weights = ROOT / "artifacts/weights" / exp["weights_file"]
    spec = {"cells": src, "input_root": str(run / "input-root"), "parts": [str(p) for p in parts],
            "train_uids": train, "gold_uids": gold, "weak_uids": weak, "soft": str(soft), "weights": str(weights)}
    run.mkdir(parents=True, exist_ok=True)
    (run / "child.py").write_text(CHILD)
    cwd = run / "empty-cwd"
    cwd.mkdir(parents=True, exist_ok=True)
    outputs = []
    for attempt, folder, die_after in (("straight", "straight-work", None), ("interrupted", "work", 0), ("resume", "work", None)):
        (run / f"spec-{attempt}.json").write_text(json.dumps(spec | {"work": str(run / folder), "die_after": die_after}))
        done = subprocess.run([sys.executable, "-I", str(run / "child.py"), str(run / f"spec-{attempt}.json")], cwd=cwd,
                              capture_output=True, text=True)
        if attempt == "interrupted":
            if done.returncode != 3 or not (run / "work/kneenet-state.pt").is_file() or (run / "work/job").exists():
                raise SystemExit(f"{exp_id}: the simulated disconnect after epoch 0 did not happen as planned\n"
                                 f"{done.stdout[-2000:]}\n{done.stderr[-4000:]}")
            continue
        if done.returncode:
            raise SystemExit(f"{exp_id} {attempt}: failed\n{done.stdout[-2000:]}\n{done.stderr[-4000:]}")
        if attempt == "resume" and ("Resumed after epoch 0" not in done.stdout or "epoch 1 loss" not in done.stdout):
            raise SystemExit(f"{exp_id}: second run did not resume and train epoch 1\n{done.stdout[-1500:]}")
        outputs.append(json.loads(done.stdout.strip().splitlines()[-1]))
    work = run / "work"
    straight, resumed = (torch.load(run / f / "job/checkpoint.pt", weights_only=False) for f in ("straight-work", "work"))
    assert straight["step"] == resumed["step"] > 0, (straight["step"], resumed["step"])
    differ = [k for k in straight["model"] if not torch.equal(straight["model"][k], resumed["model"][k])]
    assert not differ, f"resumed EMA weights differ from the uninterrupted run: {differ[:5]}"
    assert (run / "straight-work/job/predictions.jsonl").read_bytes() == (work / "job/predictions.jsonl").read_bytes(), \
        "resumed predictions differ from the uninterrupted run"
    metrics = json.loads((work / "job/metrics.json").read_text())
    preds = [json.loads(x) for x in (work / "job/predictions.jsonl").read_text().splitlines()]
    assert metrics["kind"] == "synthetic" and sorted(metrics["train_uids"]) == sorted(train)
    assert metrics["attempted_updates"] == metrics["applied_updates"] + metrics["skipped_updates"] == resumed["step"] + metrics["skipped_updates"]
    assert {p["StudyInstanceUID"] for p in preds if p["split"] == "gold"} == set(gold)
    assert set(metrics["gold_fp32"]) == set(gold) and all(len(v) == 12 for v in metrics["gold_fp32"].values())
    assert {p["StudyInstanceUID"] for p in preds if p["split"] == "weak"} == set(weak)
    assert json.loads((work / "campaign-result.json").read_text())["exit_reason"] == "synthetic_rehearsal"
    print(json.dumps({"experiment": exp_id, "rehearsal": "ok", "resume": "bit-identical to uninterrupted", "child": outputs[-1],
                      "n_predictions": len(preds)}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiments", nargs="+")
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument('--notebook-dir', type=Path)
    args = parser.parse_args()
    for exp in args.experiments:
        rehearse(exp, args.scratch, args.notebook_dir)


if __name__ == "__main__":
    main()
