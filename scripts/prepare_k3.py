"""One new encoder family; bounded resource pilot and an immutable full-job package."""
import copy
import hashlib
import json
from pathlib import Path
import prepare_k1_kaggle_training as prep
import make_colab_notebook as mcn
import make_kneenet_notebook as kn
from rsna_knee.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[1]
PILOT_SLUG = 'rsna-k3ev2-resource-pilot-20261016'
FULL_SLUG = 'rsna-k3ev2-kneenet-fold0-20261017'
DERIVE = '''
from rsna_knee.data.calibration_view import calibration_view
part_manifests = [calibration_view([p.parent for p in part_manifests], INPUT_ROOT / 'pilot-cache',
    expected_hashes=[p['manifest_sha256'] for p in EXPERIMENT['cache_parts']], uids=EXPERIMENT['train_uids'])]
'''

def main():
    existing = ROOT / 'artifacts/research/k3-campaign.json'
    if existing.exists():
        old = json.loads(existing.read_text())
        if old['status'] != 'PREPARED_NOT_RUN' or any((ROOT / old[k]['folder'] / 'push-attempt.json').exists() for k in ('pilot', 'full')):
            raise RuntimeError('Preserve receipt-bound campaign')
    audit = prep.prepare(config_path=ROOT / 'configs/experiments/P5-K3EV2-kneenet-fold0.yaml', slug=FULL_SLUG)
    folder = ROOT / 'artifacts/kaggle' / FULL_SLUG
    notebook = json.loads((folder / 'K3EV2.ipynb').read_text())
    ns = {};exec(''.join(notebook['cells'][1]['source']), ns)
    exp = copy.deepcopy(ns['EXPERIMENT'])
    full_count = len(exp['train_uids'])
    sizes = {}
    for part in exp['cache_parts']:
        m = json.loads((ROOT / 'artifacts/kaggle' / part['kernel'].split('/')[-1] / 'output/cache/manifest.json').read_text())
        sizes.update({u: sum(meta['slot_mask']) for u, meta in m['studies'].items()})
    # No target values or validation images enter the selection: most present slots first, stable hash ties.
    selected = sorted(exp['train_uids'], key=lambda u: (-sizes[u], hashlib.sha256(u.encode()).hexdigest()))[:192]
    exp.update(id='K3EV2PILOT', train_uids=selected, gold_uids=[], weak_uids=[],
               train_uids_sha256=hashlib.sha256('\n'.join(selected).encode()).hexdigest(),
               calibration_scope={'selection': 'train_only_most_slots_then_uid_sha256', 'n': len(selected),
                                  'full_training_studies': full_count, 'full_experiment': 'K3EV2',
                                  'full_notebook_sha256': sha256_file(folder / 'K3EV2.ipynb'),
                                  'full_build_sha256': sha256_file(ROOT / 'notebooks/generated/K3EV2.build.json')})
    exp['runtime_guard'].update(calibration_only=True, projection_training_studies=full_count)
    # Own identity: this is not a completed K3 training or validation experiment.
    exp['config_sha256'] = hashlib.sha256(json.dumps(exp, sort_keys=True).encode()).hexdigest()
    train = DERIVE + '\ntry:\n' + '\n'.join('    ' + line for line in kn.TRAIN.splitlines()) + '\nexcept CalibrationComplete:\n    print("Bounded real GPU calibration completed; no AUC claimed.", flush=True)\n'
    exp['training_cell_sha256'] = hashlib.sha256(train.encode()).hexdigest()
    notebook['cells'][0] = mcn._md('# K3EV2 resource calibration only\n192 train-only studies; no gold/weak validation, no AUC, no completed-model claim.')
    notebook['cells'][1] = prep.packed_code('import json\nEXPERIMENT = json.loads(' + repr(json.dumps(exp)) + ')', 'K3-pilot-config')
    notebook['cells'][5] = prep.packed_code(train, 'K3-pilot-training')
    notebook['cells'][6] = mcn._code("print((WORK / 'calibration-result.json').read_text())")
    pilot = ROOT / 'artifacts/kaggle' / PILOT_SLUG;pilot.mkdir(parents=True, exist_ok=True)
    (pilot / 'K3EV2PILOT.ipynb').write_text(json.dumps(notebook, indent=1))
    metadata = json.loads((folder / 'kernel-metadata.json').read_text())
    metadata.update(id='bulsahkecici/' + PILOT_SLUG, title=PILOT_SLUG, code_file='K3EV2PILOT.ipynb')
    (pilot / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2))
    (pilot / 'build.json').write_text(json.dumps(exp, indent=2))
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':compile(''.join(cell['source']), 'K3-pilot', 'exec')
    assert len(json.dumps({**notebook, 'cells': [{**c, 'source': ''.join(c['source'])} for c in notebook['cells']]}).encode()) < 1_000_000
    plan = {'status': 'PREPARED_NOT_RUN', 'max_full_runs': 1, 'minimum_remaining_gpu_hours': 10.5,
            'no_automatic_submission': True, 'no_automatic_second_fold': True,
            'pilot': {'id': 'K3EV2PILOT', 'kernel': metadata['id'], 'folder': str(pilot.relative_to(ROOT)),
                      'notebook_sha256': sha256_file(pilot / 'K3EV2PILOT.ipynb'),
                      'metadata_sha256': sha256_file(pilot / 'kernel-metadata.json'),
                      'config_sha256': exp['config_sha256'], 'training_cell_sha256': exp['training_cell_sha256']},
            'full': {'id': 'K3EV2', 'kernel': 'bulsahkecici/' + FULL_SLUG, 'folder': str(folder.relative_to(ROOT)),
                     'notebook_sha256': sha256_file(folder / 'K3EV2.ipynb'),
                     'metadata_sha256': sha256_file(folder / 'kernel-metadata.json'),
                     'build_sha256': sha256_file(ROOT / 'notebooks/generated/K3EV2.build.json')},
            'audit': audit}
    dest = ROOT / 'artifacts/research/k3-campaign.json'
    dest.write_text(json.dumps(plan, indent=2))
    print(json.dumps({k: plan[k] for k in ('status','pilot','full')}, indent=2))

if __name__ == '__main__':main()
