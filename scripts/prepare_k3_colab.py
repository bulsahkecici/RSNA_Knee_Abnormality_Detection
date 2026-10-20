"""Make a Colab handoff from immutable K3 bytes; never overwrite Kaggle identities."""
import hashlib
import json
from pathlib import Path
import make_colab_notebook as mcn
from make_submission_kernel import notebook_cells
from rsna_knee.hashing import sha256_file, sha256_json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'notebooks/generated/colab'
SETUP=r'''
import sys, subprocess, shutil, json, os
from pathlib import Path
import torch
assert torch.cuda.is_available(), 'Colab: Runtime > Change runtime type > GPU, then Run all.'
assert sys.version_info >= (3, 11)
# MRI volumes stay on the remote runtime, and are never copied to Drive or Mac.
assert shutil.disk_usage('/content').free >= 75 * 1024**3, 'At least 75 GiB free Colab disk required.'
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'kaggle>=2.2', 'timm', 'safetensors',
    'pydicom==3.0.1', 'pydantic>=2.7', 'pyyaml', 'filelock', 'scikit-learn', 'jsonschema'], check=True)
from google.colab import drive
drive.mount('/content/drive')
WORK=Path('/content/drive/MyDrive/RSNA/colab/K3EV2-colab')
WORK.mkdir(parents=True, exist_ok=True)
import importlib.metadata
handshake={'gpu':torch.cuda.get_device_name(0), 'vram_bytes':torch.cuda.get_device_properties(0).total_memory,
    'capability':list(torch.cuda.get_device_capability(0)), 'torch':torch.__version__,
    'platform':'colab', 'cache_access':'remote_ephemeral_downloads_symlink_view',
    'packages':{n:importlib.metadata.version(n) for n in ('numpy','timm','safetensors','pydicom')}}
(WORK/'handshake.json').write_text(json.dumps(handshake,indent=2))
print('Verified GPU:',handshake,flush=True)
print('Durable checkpoints/results:',WORK,flush=True)
'''


def auth_code():
    prefix=mcn.AUTH_AND_CACHE.split('import tarfile\n',1)[0]
    prefix=prefix.replace("'--page-size', '200']", "'--page-size', '200', '--file-pattern', r'^cache/.*$']")
    auth, rest = prefix.split('def redact', 1)
    # Authenticate only after all local/Drive reuse paths have been exhausted.
    auth = auth.replace('import getpass, re\n', '')
    deferred = 'import getpass, re\ndef ensure_kaggle_auth():\n' + '\n'.join('    ' + line for line in auth.splitlines()) + '\n'
    helper=(ROOT/'scripts/colab_cache_reuse.py').read_text()
    return deferred + 'def redact' + rest + '\n' + helper + r'''
INPUT_ROOT.mkdir(parents=True,exist_ok=True)
DRIVE_CACHE=Path('/content/drive/MyDrive/RSNA/cache')
part_manifests=[]; cache_sources=[]
for part in EXPERIMENT['cache_parts']:
    part_dir=INPUT_ROOT / part['kernel'].split('/')[-1]
    print('Checking existing runtime/Drive cache:',part['kernel'],flush=True)
    source=restore_pinned_drive_cache(DRIVE_CACHE,part,part_dir)
    if source is None:
        print('No matching complete K1 volume cache on Drive; downloading this part only.',flush=True)
        ensure_kaggle_auth()
        part_dir.mkdir(parents=True,exist_ok=True)
        download_part(part['kernel'],part_dir,INPUT_ROOT/(part_dir.name+'.log'))
        assert pinned_cache_ready(part_dir,part['manifest_sha256']), 'Pinned cache missing or incomplete'
        source={'kind':'kaggle_download','manifest_sha256':part['manifest_sha256']}
    manifest=part_dir/'cache/manifest.json'
    part_manifests.append(manifest)
    cache_sources.append({'kernel':part['kernel'],**source})
    print('Cache source:',source['kind'],flush=True)
(WORK/'cache-sources.json').write_text(json.dumps(cache_sources,indent=2))
print('Pinned manifests ready. Every payload is rehashed and checked before training.',flush=True)
'''


def prepare():
    plan=json.loads((ROOT/'artifacts/research/k3-campaign.json').read_text())
    source=ROOT/plan['full']['folder']/'K3EV2.ipynb'
    assert sha256_file(source)==plan['full']['notebook_sha256']
    original=json.loads(source.read_text());cells=notebook_cells('K3EV2',source.parent)
    ns={};exec(cells[1],ns);exp=ns['EXPERIMENT']
    auth=auth_code()
    collect=mcn.COLLECT.replace("'campaign-result.json', 'handshake.json',", "'campaign-result.json', 'handshake.json', 'resource-gate.json', 'throughput.json', 'cache-sources.json',")
    exp.update(platform='colab',platform_recipe_sha256=sha256_json({'setup':SETUP,'auth_and_cache':auth,'collect':collect}),
        platform_generator_sha256=sha256_file(Path(__file__)),
        colab_handoff={'kaggle_full_notebook_sha256':sha256_file(source),
                       'requires_exclusive_full_execution':True,'mri_storage':'existing_drive_cache_read_only_then_remote_ephemeral',
                       'gpu_precision_note':'A100/L4 use bf16; T4 fp16; evaluation remains fp32. Log runtime differences.'})
    header='# K3EV2 — COLAB DRIVE CACHE — 17 Ekim 2026\n\nOpen a GPU runtime and Run all. 75 GiB free runtime disk required. Kaggle credentials are entered through a hidden prompt. Existing MyDrive/RSNA/cache archives are reused first after exact manifest checks. Only missing or incompatible parts download from Kaggle to Colab. Existing Drive cache is never modified; new checkpoints and numeric results are saved to Drive.\n\nOne full run only: the Kaggle auto-launch must be on hold before starting this notebook. Same frozen K3 encoder, labels, cohorts, seed and training cell. Hardware/AMP changes are recorded and must be reported in review. No automatic submission.'
    original['cells'][0]=mcn._md(header)
    original['cells'][1]=mcn._code('import json\nEXPERIMENT=json.loads('+repr(json.dumps(exp))+')\nprint("K3EV2 COLAB DRIVE CACHE — 2026-10-17")')
    original['cells'][2]=mcn._code(SETUP)
    original['cells'][3]=mcn._code(auth)
    original['cells'][6]=mcn._code(collect)
    original['metadata'].update(accelerator='GPU',colab={'name':'K3EV2_COLAB_DRIVE.ipynb','provenance':[]})
    for cell in original['cells']:
        if cell['cell_type']=='code':compile(''.join(cell['source']),'K3EV2-colab','exec')
    OUT.mkdir(parents=True,exist_ok=True)
    out=OUT/'K3EV2.ipynb';out.write_text(json.dumps(original,indent=1))
    delivery=OUT/'K3EV2_COLAB_DRIVE.ipynb';delivery.write_bytes(out.read_bytes())
    # A separate build record prevents overwriting the receipt-bound Kaggle build.
    build=json.loads((ROOT/'notebooks/generated/K3EV2.build.json').read_text())
    build.update({k:v for k,v in exp.items() if k not in ('train_uids','gold_uids','weak_uids')})
    (OUT/'K3EV2.build.json').write_text(json.dumps(build,indent=2))
    assert notebook_cells('K3EV2',OUT)[5]==cells[5]
    for key in ('source_sha256','train_uids_sha256','config_sha256','weights_sha256','training_cell_sha256'):
        assert build[key]==json.loads((ROOT/'notebooks/generated/K3EV2.build.json').read_text())[key]
    result={'status':'PREPARED_NOT_STARTED','notebook':str(delivery),'notebook_sha256':sha256_file(delivery),'canonical_notebook':str(out),
        'kaggle_source_unchanged':sha256_file(source)==plan['full']['notebook_sha256'],
        'training_cell_identical':True,'minimum_ephemeral_disk_gib':75,'cache_gib':plan['audit']['total_payload_gib'],
        'runtime_acquisition':'manual_colab_gpu_and_oauth','mri_downloaded_to_mac':False,'drive_cache_reuse_first':True,'kaggle_auth_only_for_missing_parts':True}
    (ROOT/'artifacts/research/k3-colab-handoff.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':prepare()
