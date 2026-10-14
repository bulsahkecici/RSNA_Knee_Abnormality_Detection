"""Generate a private, numeric-only v4 cache job from the audited v3 campaign."""
from __future__ import annotations
import argparse, ast, base64, gzip, hashlib, io, json, tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def prepare(structural=False):
    previous=ROOT/'artifacts/kaggle/v3-sixcenters-cache/entry.py'
    text=previous.read_text();tree=ast.parse(text)
    values={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body
            if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)
            and n.targets[0].id in {'SOURCE_B64','CONFIG_B64','LABELS_B64'}}
    files={}
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(values['SOURCE_B64']))) as archive:
        for m in archive.getmembers():
            assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
            files[m.name]=archive.extractfile(m).read()
    # Current production worker/evaluation avoids stale embedded modules.
    for path in (ROOT/'src/rsna_knee').rglob('*.py'):
        if '__pycache__' not in path.parts:files[str(path.relative_to(ROOT))]=path.read_bytes()
    for path in (ROOT/'scripts/cache_v4_source/rsna_knee/data').glob('*.py'):
        files['src/rsna_knee/data/'+path.name]=path.read_bytes()
    assert b'npz' in files['src/rsna_knee/data/cache_build.py']
    buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
        for name,data in sorted(files.items()):
            info=tarfile.TarInfo(name);info.size=len(data);info.mtime=0;archive.addfile(info,io.BytesIO(data))
    source=buffer.getvalue();configuration=json.loads(gzip.decompress(base64.b64decode(values['CONFIG_B64'])))
    tag='v4-five-slots12' if structural else 'v4-dense12'
    configuration['run_id']='campaign-20261009-'+tag
    config_bytes=json.dumps(configuration,sort_keys=True,separators=(',',':')).encode()
    replacements={'SOURCE_B64':base64.b64encode(source).decode(),
                  'SOURCE_SHA256':hashlib.sha256(source).hexdigest(),
                  'CONFIG_B64':base64.b64encode(gzip.compress(config_bytes,mtime=0)).decode(),
                  'CONFIG_SHA256':hashlib.sha256(config_bytes).hexdigest()}
    lines=text.splitlines()
    for i,line in enumerate(lines):
        for key,value in replacements.items():
            if line.startswith(key+' = '):lines[i]=key+' = '+repr(value)
    text='\n'.join(lines)+'\n'
    text=text.replace('N_CENTERS = 6','N_CENTERS = 12').replace('CENTER_LOWER = 0.15','CENTER_LOWER = 0.05').replace('CENTER_UPPER = 0.85','CENTER_UPPER = 0.95')
    slots=[['Sagittal','fluid'],['Coronal','fluid'],['Axial','fluid']]
    if structural:slots += [['Sagittal','structural'],['Coronal','structural']]
    text=text.replace('# ------------------------------',f'SLOTS = {slots!r}\nMAX_CACHE_BYTES = 18 * 1024**3\n# ------------------------------',1)
    text=text.replace('"v3-sixcenters"',repr(tag)).replace('cache-v3-source.tar.gz','cache-v4-source.tar.gz')
    text=text.replace('(".npy", ".json")','(".npz", ".json")')
    text=text.replace('volume = np.load(cache / (uid + ".npy"), mmap_mode="r", allow_pickle=False)', 'with np.load(cache / (uid + ".npz"), allow_pickle=False) as shard:\n            volume = shard["images"]')
    text=text.replace('3 * 154 + 26','len(SLOTS) * 154 + 26').replace('enumerate(("Sagittal", "Coronal", "Axial"))','enumerate([plane+"/"+kind for plane,kind in SLOTS])')
    start=text.index('        per_study_bytes = ');end=text.index('        atomic_json(work / "preprocess-config.json"',start)
    replacement='''        import numpy as np, time
        free = shutil.disk_usage(work).free
        if free < MAX_CACHE_BYTES + 256 * 1024**2:
            raise RuntimeError('Insufficient free disk for bounded cache output')
        options = dict(series_csv=root/'train_series.csv', dicom_root=root/'train_series', dest=work/'cache',
                       size=IMAGE_SIZE,n_centers=N_CENTERS,resume=True,strict_series=STRICT_DECODE,
                       center_strategy=CENTER_STRATEGY,center_lower=CENTER_LOWER,center_upper=CENTER_UPPER,
                       min_present_slots=MIN_PRESENT_PLANES,slots=SLOTS,max_cache_bytes=MAX_CACHE_BYTES,
                       progress_every=PROGRESS_EVERY,index_flush_every=INDEX_FLUSH_EVERY)
        pilot_uids=np.random.default_rng(2026).permutation(config['requested'])[:64].tolist()
        started=time.perf_counter()
        pilot=build_cache(study_ids=pilot_uids,**options)
        ready,reason=manifest_ready(work/'cache/manifest.json')
        if not ready:raise RuntimeError('Pilot cache QC failed: '+reason)
        elapsed=time.perf_counter()-started
        pilot_bytes=sum((work/'cache'/(uid+'.npz')).stat().st_size for uid in pilot_uids)
        projected_bytes=pilot_bytes/len(pilot_uids)*len(config['requested'])
        projected_seconds=elapsed/len(pilot_uids)*len(config['requested'])
        measurement={'n_studies':len(pilot_uids),'compressed_bytes':pilot_bytes,'elapsed_seconds':elapsed,
                     'projected_bytes':projected_bytes,'projected_seconds':projected_seconds,
                     'budget_bytes':MAX_CACHE_BYTES,'note':'Sample projection; hard byte guard also applies during full build.'}
        atomic_json(work/'pilot-measurement.json',measurement)
        print(json.dumps({'event':'real_cache_pilot',**measurement}),flush=True)
        if projected_bytes * 1.15 > MAX_CACHE_BYTES:
            raise RuntimeError('Pilot projected cache exceeds byte budget with 15 percent margin; revise layout')
        if projected_seconds * 1.2 > 8 * 3600:
            raise RuntimeError('Pilot projected build exceeds 8-hour planning budget with 20 percent margin')
        manifest=build_cache(study_ids=config['requested'],**options)
'''
    text=text[:start]+replacement+text[end:]
    text=text.replace('parser.add_argument("--self-test", action="store_true", help="Run small synthetic DICOM regression checks")','parser.add_argument("--self-test", action="store_true", help="Reserved: run the v4 test suite locally instead")')
    a=text.index('    if args.self_test:\n');b=text.index('    from rsna_knee.data.cache_build',a)
    text=text[:a]+"    if args.self_test:\n        raise RuntimeError('Use pytest tests/test_cache_v4.py; legacy self-tests do not validate compressed layout')\n"+text[b:]
    ast.parse(text)
    folder=ROOT/'artifacts/kaggle'/tag;folder.mkdir(parents=True,exist_ok=True)
    (folder/'entry.py').write_text(text)
    metadata={'id':'bulsahkecici/rsna-'+tag+'-cache-20261009','title':'RSNA '+tag+' cache 20261009',
              'code_file':'entry.py','language':'python','kernel_type':'script','is_private':True,
              'enable_gpu':False,'enable_internet':True,'competition_sources':['rsna-knee-abnormality-detection'],
              'dataset_sources':[],'kernel_sources':[]}
    (folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (folder/'preparation.json').write_text(json.dumps({'source_archive_sha256':replacements['SOURCE_SHA256'],
        'labels_sha256':configuration['labels_sha256'],'n_requested':len(configuration['requested']),
        'n_train':len(configuration['selected']),'slots':slots,'n_centers':12,'scope':'preprocessing only, no submission'},indent=2)+'\n')
    print(folder)
    return folder


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--structural',action='store_true')
    prepare(parser.parse_args().structural)
