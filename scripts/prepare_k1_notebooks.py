"""Write an executable Kaggle K1 pilot notebook and an explicit pending-cache training notebook.

No remote execution, upload or submission. Existing K0/K0BASE builds are untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import make_colab_notebook as mcn
import make_kaggle_kernel as mkg
import make_kneenet_notebook as kn

ROOT = Path(__file__).resolve().parents[1]
PILOT_SLUG = 'rsna-k1-volume-pilot-20261015'

QC = r'''
from collections import Counter
import json, os, zipfile
from pathlib import Path

work = Path(os.environ.get('RSNA_WORK_DIR', '/kaggle/working'))
work.mkdir(parents=True, exist_ok=True)
result = (json.loads((work / 'campaign-result.json').read_text())
          if (work / 'campaign-result.json').is_file() else {'status': 'failed', 'error': PILOT_ERROR})
manifest = (json.loads((work / 'cache/manifest.json').read_text())
            if (work / 'cache/manifest.json').is_file() else {'studies': {}, 'quarantine': {}})
counts = Counter()
failures = Counter()
for study in manifest['studies'].values():
    counts['studies_with_alternative_series'] += int(any(len(c) > 1 for c in study['candidates']))
    for slot, candidates in enumerate(study['candidates']):
        counts[f'slot_{slot}_studies'] += bool(candidates)
        for candidate in candidates:
            counts['usable_series'] += 1
            counts['order_' + candidate['order']] += 1
            counts['series_with_padding'] += int(any(any(v for v in r['padding_pixels']) for r in candidate['resampling']))
            for transform in candidate['transforms']:
                for key, value in transform.items():
                    counts['slices_' + key] += int(value)
    for failure in study['series_failures']:
        failures[failure['error']] += 1
qc = {'kind': 'dicom_preprocessing_pilot_not_training', 'status': result['status'],
      'error': PILOT_ERROR or result.get('error'),
      'n_studies': len(manifest['studies']), 'n_quarantine': len(manifest['quarantine']),
      'counts': dict(counts), 'series_failure_reasons': dict(failures),
      'seconds_per_study': result.get('seconds_per_study'), 'payload_bytes': result.get('payload_bytes'),
      'manifest_sha256': result.get('manifest_sha256')}
(work / 'pilot-qc.json').write_text(json.dumps(qc, indent=2))
archive = work / 'K1-PILOT-results.zip'
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
    for relative in ('pilot-qc.json', 'campaign-result.json', 'cache/manifest.json', 'cache/build-summary.json'):
        if (work / relative).is_file():
            z.write(work / relative, relative)
print(json.dumps(qc, indent=2))
print('QC için indirilecek dosya:', archive)
print('Bu ZIP yalnız JSON metadata içerir; MRI hacimleri Kaggle üzerinde kalır.')
if qc['status'] != 'ok':
    raise RuntimeError('K1 pilotu başarısız; K1-PILOT-results.zip içindeki hata/QC raporunu inceleyin.')
'''


def prepare(ref='WORKTREE'):
    folder = mkg.generate('volume_cache', PILOT_SLUG, internet=True, gpu=False, ref=ref)
    script = (folder / 'entry.py').read_text()
    run_cell = ('PILOT_ERROR = None\ntry:\n'
                f'    exec(compile({script!r}, "K1-PILOT-entry.py", "exec"), globals())\n'
                'except Exception as exc:\n'
                '    PILOT_ERROR = f"{type(exc).__name__}: {exc}"\n'
                '    print("Pilot başarısız:", PILOT_ERROR)\n'
                '    print("Hata/QC raporunu toplamak için son hücre çalışacak.")\n')
    notebook = {
        'nbformat': 4, 'nbformat_minor': 5,
        'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                     'kaggle': {'isInternetEnabled': True, 'isGpuEnabled': False},
                     'rsna': {'required_competition': 'rsna-knee-abnormality-detection'}},
        'cells': [mcn._md('# K1 — gerçek DICOM geometri/decode pilotu\n\n'
                         '**Şimdi çalıştırılacak notebook budur.** Kaggle üzerinde açın; '
                         '`rsna-knee-abnormality-detection` yarışma verisini bağlayın, Internet açık ve '
                         'CPU oturumu seçili olsun. **Run All** çalıştırın.\n\n'
                         '64 gerçek çalışma üzerinde ortak yön, fiziksel FOV/padding, altı slot ve '
                         'tüm kullanılabilir seri adayları hazırlanır. Eğitim/submission yapılmaz. '
                         'Eksik/bozuk seriler kaydedilir; kullanılamayan çalışma başarı sayılmaz.\n\n'
                         'İşlem bitince **K1-PILOT-results.zip** dosyasını indirin. Bu rapor, '
                         'tam hacim cache’inin boyutunu ve QC sorunlarını değerlendirmek içindir. '
                         'Mac’e yarışmanın MRI arşivini indirmez.'),
                  mcn._code(run_cell), mcn._code(QC)],
    }
    out = mcn.OUT_DIR / 'K1-PILOT.ipynb'
    out.parent.mkdir(parents=True, exist_ok=True)
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), str(out), 'exec')
    out.write_text(json.dumps(notebook, indent=1))
    build = json.loads((folder / 'build.json').read_text())
    build.update({'scope': 'dicom_pilot_no_training', 'kernel_type': 'notebook',
                  'notebook_sha256': hashlib.sha256(out.read_bytes()).hexdigest()})
    (mcn.OUT_DIR / 'K1-PILOT.build.json').write_text(json.dumps(build, indent=2))
    # The CLI push must execute the same notebook (including QC export) that was rehearsed locally.
    (folder / 'K1-PILOT.ipynb').write_bytes(out.read_bytes())
    metadata_path = folder / 'kernel-metadata.json'
    metadata = json.loads(metadata_path.read_text())
    metadata.update({'kernel_type': 'notebook', 'code_file': 'K1-PILOT.ipynb'})
    metadata_path.write_text(json.dumps(metadata, indent=2))
    (folder / 'build.json').write_text(json.dumps(build, indent=2))
    config = ROOT / 'configs/experiments/P5-K1-kneenet-fold0.yaml'
    # Never replace a K1 that has already been bound to genuine cache manifests.
    training_build = mcn.OUT_DIR / 'K1.build.json'
    if training_build.exists() and json.loads(training_build.read_text()).get('cache_parts'):
        training = mcn.OUT_DIR / 'K1.ipynb'
    else:
        training = kn.generate(config, cache_kernels=[], cache_manifest_sha256s=[], ref=ref,
                               prepare_without_cache=True)
    return {'run_now': str(out), 'training_notebook': str(training),
            'training_status': json.loads(training_build.read_text()).get('cache_status', 'cache_pins_bound'),
            'remote_job_started': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', default='WORKTREE')
    args = parser.parse_args()
    print(json.dumps(prepare(args.ref), indent=2))
