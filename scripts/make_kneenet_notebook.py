"""Generate a Colab notebook that trains the vetted public reader's KneeNet on our v5 cache.

Usage: python scripts/make_kneenet_notebook.py configs/experiments/P4-K5F0-kneenet-v5-fold0.yaml \
          --cache-kernel <part0> --cache-manifest-sha256 <sha0> --cache-kernel <part1> --cache-manifest-sha256 <sha1>

KneeNet (model code: goodpjw2008/rsna-knee-2-5d-convnext-reader knee.py, Apache-2.0, pinned by sha256) is
trained with the reader's optimisation (train.py: AdamW on the whole backbone, warm-up + cosine, EMA, grad
clip, BCE on soft targets) on our own v5 shards, folds and train UIDs, with the mean-of-four-CC0-tables
labels. The notebook writes the same result files as our worker notebooks, so scripts/review_experiment.py
reviews it unchanged. Cache download/Drive reuse and result collection are the cells of
make_colab_notebook.py. It never submits.
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import make_colab_notebook as mcn
import yaml

ROOT = Path(__file__).resolve().parents[1]
KNEE = ROOT / "state/external-code/goodpjw2008-reader/knee.py"
WEIGHTS_URL = "https://huggingface.co/timm/convnext_tiny.in12k_ft_in1k/resolve/main/model.safetensors"
WEIGHTS_FILE = "convnext_tiny.in12k_ft_in1k.safetensors"

PAYLOAD = r'''
import base64, io, tarfile, hashlib
def unpack(b64, sha):
    data = base64.b64decode(b64)
    assert hashlib.sha256(data).hexdigest() == sha, 'embedded payload integrity failure'
    return data
SOURCE = Path('/tmp/rsna-src-' + EXPERIMENT['source_sha256'][:16])
with tarfile.open(fileobj=io.BytesIO(unpack(SOURCE_B64, EXPERIMENT['source_sha256'])), mode='r:gz') as archive:
    archive.extractall(SOURCE, filter='data')
(SOURCE / 'third_party').mkdir(exist_ok=True)
(SOURCE / 'third_party/knee.py').write_bytes(unpack(KNEE_B64, EXPERIMENT['reader_code_sha256']))  # Apache-2.0, goodpjw2008
sys.path.insert(0, str(SOURCE / 'src')); sys.path.insert(0, str(SOURCE / 'third_party'))
os.chdir(SOURCE)
SOFT = WORK / 'soft-labels.jsonl'; SOFT.write_bytes(unpack(SOFT_B64, EXPERIMENT['soft_labels_sha256']))
(WORK / 'experiment.json').write_text(json.dumps(EXPERIMENT, indent=1))
print('Source, reader model code and labels verified.')
'''

TRAIN = r'''
import copy, math, random, time, urllib.request
import numpy as np, torch, torch.nn.functional as F
from safetensors.torch import load_file
from rsna_knee.data.cache_build import manifest_ready
from rsna_knee.hashing import sha256_file
from rsna_knee.ontology import TARGET_COLUMNS
from rsna_knee.runtime.train_payload import load_volume
import knee
assert list(knee.LABELS) == list(TARGET_COLUMNS), 'reader label order differs'
if len(part_manifests) == 1:
    manifest_path = part_manifests[0]
else:  # move shard parts into one directory; manifest_ready below re-verifies every shard
    from rsna_knee.data.cache_build import merge_cache_parts
    merged_dir = INPUT_ROOT / ('merged-' + EXPERIMENT['cache_manifest_sha256'][:12]) / 'cache'
    if not (merged_dir / 'manifest.json').is_file():
        merge_cache_parts([m.parent for m in part_manifests], merged_dir)
    manifest_path = merged_dir / 'manifest.json'
EXPERIMENT['merged_manifest_sha256'] = sha256_file(manifest_path)
(WORK / 'experiment.json').write_text(json.dumps(EXPERIMENT, indent=1))
t0 = time.time()
ready, reason = manifest_ready(manifest_path)  # re-hashes every shard, checks masks and shapes
assert ready, reason
print(f'Cache verified in {time.time() - t0:.0f}s:', reason, flush=True)
weights = WORK / EXPERIMENT['weights_file']
if not weights.is_file() or sha256_file(weights) != EXPERIMENT['weights_sha256']:
    urllib.request.urlretrieve(EXPERIMENT['weights_url'], weights)
assert sha256_file(weights) == EXPERIMENT['weights_sha256'], 'pinned weights hash mismatch'

T = EXPERIMENT['train']
manifest = json.loads(manifest_path.read_text()); STUDIES, CACHE = manifest['studies'], manifest_path.parent
policy = manifest['preprocess_config']
SLOT_MAP = [knee.slot_of(plane, kind == 'fluid') for plane, kind in policy['slots']]  # fluid-sensitive ~ fat-suppressed
assert len(set(SLOT_MAP)) == len(SLOT_MAP)
N_C = int(policy['n_centers'])
POS = np.linspace(float(policy['center_lower']), float(policy['center_upper']), N_C).astype(np.float32)
SOFT_T = {r['StudyInstanceUID']: r['soft_targets'] for r in map(json.loads, SOFT.read_text().splitlines())}


class V5Study(torch.utils.data.Dataset):
    """Our v5 shard [slots, 12, 3, H, W] -> the reader's layout [6, k, 3, H, W], slot mask, positions, soft target."""
    def __init__(self, uids, k, train):
        self.uids, self.k, self.train = list(uids), k, train
    def __len__(self):
        return len(self.uids)
    def __getitem__(self, i):
        uid = self.uids[i]; meta = STUDIES[uid]
        vol = load_volume(uid, meta, CACHE)
        present = meta.get('slot_mask') or [1] * vol.shape[0]
        x = np.zeros((knee.N_SLOTS, self.k) + vol.shape[2:], np.uint8)
        mask, pos = np.zeros(knee.N_SLOTS, bool), np.zeros((knee.N_SLOTS, self.k), np.float32)
        for s in range(vol.shape[0]):
            if not present[s]:
                continue
            if self.train:  # one window per stratum, as the reader's window_centres(train=True)
                edges = np.linspace(0, N_C, self.k + 1)
                idx = np.minimum((edges[:-1] + np.random.rand(self.k) * (edges[1:] - edges[:-1])).astype(int), N_C - 1)
            else:
                idx = np.round(np.linspace(0, N_C - 1, self.k)).astype(int)
            t = SLOT_MAP[s]; x[t], mask[t], pos[t] = vol[s, idx], True, POS[idx]
        row = SOFT_T.get(uid)
        y = np.array([0.0 if row is None or row[c] is None else row[c] for c in TARGET_COLUMNS], np.float32)
        m = np.array([0.0 if row is None or row[c] is None else 1.0 for c in TARGET_COLUMNS], np.float32)
        return torch.from_numpy(x), torch.from_numpy(mask), torch.from_numpy(pos), torch.from_numpy(y), torch.from_numpy(m), uid


seed = int(T['seed']); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
dev = torch.device('cuda')
cap = torch.cuda.get_device_capability(0)
amp_dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16  # T4 has no bf16; fp16 needs a GradScaler
scaler = torch.amp.GradScaler('cuda', enabled=amp_dtype == torch.float16)
mem = torch.cuda.get_device_properties(0).total_memory
micro = 8 if mem >= 38e9 else (4 if mem >= 20e9 else 2)
micro = min(micro, int(T['batch'])); accum = int(T['batch']) // micro
workers = max(2, min(8, os.cpu_count() or 2))
train_uids = [u for u in EXPERIMENT['train_uids'] if u in STUDIES]
assert len(train_uids) == len(EXPERIMENT['train_uids']), 'train studies missing from the cache'
# workers are re-seeded from the main torch RNG every epoch (no persistent workers), so a resumed run draws the
# same augmentation streams as an uninterrupted one
train_dl = torch.utils.data.DataLoader(V5Study(train_uids, int(T['k']), True), batch_size=micro, shuffle=True,
                                       drop_last=True, num_workers=workers, pin_memory=True)
model = knee.KneeNet(T['backbone'], pretrained=False, dim=int(T['dim']), depth=int(T['depth']))
missing, unexpected = model.enc.load_state_dict(load_file(str(weights)), strict=False)
assert not missing and set(unexpected) <= {'head.fc.weight', 'head.fc.bias'}, (missing, unexpected)
model = model.to(dev).to(memory_format=torch.channels_last)
ema = copy.deepcopy(model).eval()
for p in ema.parameters():
    p.requires_grad_(False)
enc = list(model.enc.parameters()); head = [p for n, p in model.named_parameters() if not n.startswith('enc.')]
opt = torch.optim.AdamW([{'params': enc, 'lr': float(T['lr'])}, {'params': head, 'lr': float(T['lr_head'])}],
                        weight_decay=float(T['wd']))
per_epoch = len(train_dl) // accum
steps, warm = int(T['epochs']) * per_epoch, per_epoch
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
identity = {'id': EXPERIMENT['id'], 'config_sha256': EXPERIMENT['config_sha256'], 'train_uids_sha256': EXPERIMENT['train_uids_sha256'],
            'micro': micro, 'accum': accum}
state_path = WORK / 'kneenet-state.pt'
step, skipped, start_ep, epoch_seconds = 0, 0, 0, []  # step counts applied updates; fp16 overflows are skipped
if state_path.is_file():  # resume after a disconnect: Drive keeps the last finished epoch
    st = torch.load(state_path, map_location='cpu', weights_only=False)
    assert st['identity'] == identity, 'state file belongs to another run; move it away to start over'
    model.load_state_dict(st['model']); ema.load_state_dict(st['ema']); opt.load_state_dict(st['opt'])
    sched.load_state_dict(st['sched']); scaler.load_state_dict(st['scaler'])
    step, skipped, start_ep, epoch_seconds = st['step'], st['skipped'], st['epoch'] + 1, st['epoch_seconds']
    torch.set_rng_state(st['rng']); torch.cuda.set_rng_state_all(st['cuda_rng'])
    np.random.set_state(st['np_rng']); random.setstate(st['py_rng'])
    print(f'Resumed after epoch {st["epoch"]}', flush=True)
print(f'GPU {torch.cuda.get_device_name(0)} | {amp_dtype} | micro {micro} x accum {accum} | {len(train_uids)} studies | '
      f'{per_epoch} steps/epoch | workers {workers}', flush=True)
started = time.time()
for ep in range(start_ep, int(T['epochs'])):
    model.train(); t_ep, tot, n = time.time(), 0.0, 0
    opt.zero_grad(set_to_none=True)
    for i, (x, mask, pos, y, m, _) in enumerate(train_dl):
        x, mask, pos, y, m = x.to(dev, non_blocking=True), mask.to(dev), pos.to(dev), y.to(dev), m.to(dev)
        with torch.autocast('cuda', dtype=amp_dtype):
            logit = model(x, mask, pos, res=int(T['res']), train=True)
        loss = (F.binary_cross_entropy_with_logits(logit.float(), y, reduction='none') * m).sum() / m.sum().clamp_min(1)
        scaler.scale(loss / accum).backward()
        tot += loss.item(); n += 1
        if (i + 1) % accum:
            continue
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(T['clip']))
        before = scaler.get_scale()
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        if scaler.is_enabled() and scaler.get_scale() < before:
            skipped += 1  # fp16 overflow: the optimizer step was skipped, so schedule and EMA stay put
            continue
        sched.step(); step += 1
        d = min(float(T['ema']), (1 + step) / (10 + step))
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - d)
            for be, bm in zip(ema.buffers(), model.buffers()):
                be.copy_(bm)
        if step % 100 == 0:
            done = (time.time() - t_ep) / max(1, (i + 1)) * len(train_dl)
            print(f'  epoch {ep} step {(i + 1) // accum}/{per_epoch} loss {tot / n:.4f} | epoch ETA {done / 60:.0f} min', flush=True)
    epoch_seconds.append(round(time.time() - t_ep, 1))
    print(f'epoch {ep} loss {tot / max(n, 1):.4f} {epoch_seconds[-1] / 60:.1f} min', flush=True)
    torch.save({'identity': identity, 'model': model.state_dict(), 'ema': ema.state_dict(), 'opt': opt.state_dict(),
                'sched': sched.state_dict(), 'scaler': scaler.state_dict(), 'step': step, 'skipped': skipped, 'epoch': ep,
                'epoch_seconds': epoch_seconds, 'rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                'np_rng': np.random.get_state(), 'py_rng': random.getstate()},
               str(state_path) + '.tmp')
    os.replace(str(state_path) + '.tmp', state_path)

job = WORK / 'job'; job.mkdir(exist_ok=True)
ckpt = job / 'checkpoint.pt'
torch.save({'model': ema.state_dict(), 'kneenet': {k: T[k] for k in ('backbone', 'dim', 'depth', 'res')},
            'step': step, 'reader_code_sha256': EXPERIMENT['reader_code_sha256']}, ckpt)
ema.eval(); rows = []
for split, uids in (('gold', EXPERIMENT['gold_uids']), ('weak', EXPERIMENT['weak_uids'])):
    uids = [u for u in uids if u in STUDIES]
    dl = torch.utils.data.DataLoader(V5Study(uids, N_C, False), batch_size=micro, num_workers=workers)
    with torch.no_grad():
        for x, mask, pos, _, _, ids in dl:
            with torch.autocast('cuda', dtype=amp_dtype):
                p = ema(x.to(dev), mask.to(dev), pos.to(dev), res=int(T['res'])).float().sigmoid().cpu().numpy()
            for uid, prob in zip(ids, p):
                rows.append({'StudyInstanceUID': uid, 'split': split, **{c: float(v) for c, v in zip(TARGET_COLUMNS, prob)}})
(job / 'predictions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
# Gold again in strict fp32 (no autocast, no TF32): the offline kernel predicts in fp32, so this is its parity reference
torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
gold_fp32 = {}
with torch.no_grad():
    for x, mask, pos, _, _, ids in torch.utils.data.DataLoader(V5Study([u for u in EXPERIMENT['gold_uids'] if u in STUDIES], N_C, False),
                                                                batch_size=micro, num_workers=workers):
        p = ema(x.to(dev), mask.to(dev), pos.to(dev), res=int(T['res'])).float().sigmoid().cpu().numpy()
        gold_fp32.update({uid: [float(v) for v in prob] for uid, prob in zip(ids, p)})
(job / 'metrics.json').write_text(json.dumps({
    'kind': 'real', 'model': 'kneenet', 'train_uids': sorted(train_uids), 'epoch_wall_seconds': epoch_seconds,
    'skipped_updates': skipped, 'applied_updates': step, 'attempted_updates': step + skipped, 'amp_dtype': str(amp_dtype), 'micro_batch': micro,
    'oof_provenance': {'checkpoint_sha256': sha256_file(ckpt), 'encoder': 'kneenet_' + T['backbone'], 'step': step},
    'n_gold': sum(r['split'] == 'gold' for r in rows), 'n_weak': sum(r['split'] == 'weak' for r in rows),
    'gold_fp32': gold_fp32}, indent=2))
(WORK / 'campaign-result.json').write_text(json.dumps({'exit_reason': 'ok', 'model': 'kneenet'}))
(WORK / 'timing.json').write_text(json.dumps({'run_job_wall_seconds': round(time.time() - started, 1),
    'gpu': torch.cuda.get_device_name(0), 'peak_cuda_bytes': torch.cuda.max_memory_allocated()}, indent=2))
print('Training status: ok', flush=True)
'''


def generate(config_path: Path, *, cache_kernels: list[str], cache_manifest_sha256s: list[str], ref: str) -> Path:
    if not cache_kernels or len(cache_kernels) != len(cache_manifest_sha256s):
        raise SystemExit("give one manifest sha256 per cache kernel")
    parts = [{"kernel": k, "manifest_sha256": s} for k, s in zip(cache_kernels, cache_manifest_sha256s, strict=True)]
    cache_id = parts[0]["manifest_sha256"] if len(parts) == 1 else mcn._sha(",".join(cache_manifest_sha256s).encode())
    spec = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    exp_id = spec["experiment_id"]
    if spec.get("train", {}).get("model") != "kneenet":
        raise SystemExit("not a KneeNet config")
    knee = KNEE.read_bytes()
    if mcn._sha(knee) != spec["reader_code"]["sha256"]:
        raise SystemExit("knee.py differs from the vetted reader file")
    soft = (ROOT / spec["labels"]["file"]).read_bytes()
    if mcn._sha(soft) != spec["labels"]["sha256"]:
        raise SystemExit("soft label table differs from the config")
    allow = yaml.safe_load((ROOT / "configs/models/convnext_tiny.yaml").read_text())["weights_sha256_allowlist"]
    if spec["weights"]["sha256"] not in allow:
        raise SystemExit("backbone weights are not allowlisted")
    outer_fold = str(spec.get("outer_fold", "0"))
    folds_path = ROOT / spec["folds"]
    labels, folds = mcn.LABELS.read_bytes(), folds_path.read_bytes()
    uids = mcn.train_uids(folds_path, labels, outer_fold)
    rows = mcn.load_folds(folds_path)
    gold = sorted(r["StudyInstanceUID"] for r in rows if r.get("gold_set"))
    weak = [] if outer_fold == "none" else sorted(
        r["StudyInstanceUID"] for r in rows if str(r["fold"]) == outer_fold and not r.get("gold_set"))
    archive, names = mcn.source_archive(ref, extra=("configs", "pyproject.toml"))
    experiment = {
        "id": exp_id, "hypothesis": spec.get("hypothesis"), "config_file": str(config_path.relative_to(ROOT)),
        "config_sha256": mcn._sha(config_path.read_bytes()), "train": spec["train"], "outer_fold": outer_fold,
        "train_uids": uids, "train_uids_sha256": mcn._sha("\n".join(uids).encode()), "gold_uids": gold, "weak_uids": weak,
        "cache_parts": parts, "cache_manifest_sha256": cache_id,
        "labels_sha256": mcn._sha(labels), "folds_file": spec["folds"], "folds_sha256": mcn._sha(folds),
        "source_sha256": mcn._sha(archive), "n_source_files": len(names), **mcn.git_identity(ref),
        "weights_url": WEIGHTS_URL, "weights_file": WEIGHTS_FILE, "weights_sha256": spec["weights"]["sha256"],
        "soft_labels_sha256": spec["labels"]["sha256"], "reader_code_sha256": spec["reader_code"]["sha256"],
        "submission": False,
    }
    header = (f"# RSNA {exp_id} (KneeNet on our v5 cache)\n\n**Hypothesis:** {spec.get('hypothesis', '')}\n\n"
              f"Control: {spec['control']}. Outer holdout fold {outer_fold}; all 58 gold studies excluded from training. "
              "Model code: goodpjw2008/rsna-knee-2-5d-convnext-reader knee.py (Apache-2.0), pinned by sha256. "
              "Local validation only; this notebook does not submit.\n\nSelect a Colab GPU runtime, then Run All. "
              "Checkpoints live on Drive after every epoch, so re-running after a disconnect resumes.")
    cells = [
        mcn._md(header),
        mcn._code("import json\nEXPERIMENT = json.loads(" + repr(json.dumps(experiment, indent=1)) + ")"),
        mcn._code(mcn.SETUP),
        mcn._code(mcn.AUTH_AND_CACHE),
        mcn._code(f"SOURCE_B64 = {base64.b64encode(archive).decode()!r}\n"
                  f"KNEE_B64 = {base64.b64encode(knee).decode()!r}\n"
                  f"SOFT_B64 = {base64.b64encode(soft).decode()!r}\n" + PAYLOAD),
        mcn._code(TRAIN),
        mcn._code(mcn.COLLECT),
    ]
    namespace: dict = {}
    exec("".join(cells[1]["source"]), namespace)
    if namespace["EXPERIMENT"] != experiment:
        raise SystemExit("EXPERIMENT cell does not round-trip")
    notebook = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
                "metadata": {"accelerator": "GPU", "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}}
    out = mcn.OUT_DIR / f"{exp_id}.ipynb"
    mcn.OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    (mcn.OUT_DIR / f"{exp_id}.build.json").write_text(json.dumps(
        {k: v for k, v in experiment.items() if k not in ("train_uids", "gold_uids", "weak_uids")}
        | {"n_train_uids": len(uids), "n_weak_uids": len(weak)}, indent=2), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--cache-kernel", required=True, action="append")
    parser.add_argument("--cache-manifest-sha256", required=True, action="append")
    parser.add_argument("--ref", default="HEAD")
    args = parser.parse_args()
    print(generate(args.config.resolve(), cache_kernels=args.cache_kernel,
                   cache_manifest_sha256s=args.cache_manifest_sha256, ref=args.ref))


if __name__ == "__main__":
    main()
