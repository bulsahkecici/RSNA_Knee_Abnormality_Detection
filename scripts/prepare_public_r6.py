"""Prepare an attributed, unchanged public R6/CoAtNet recipe for remote inference."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'artifacts/research/open-source-20261018/code/goodpjw2008__rsna-knee-coatnet-r6-2-5d-mil-lb-0-958'
OUT = ROOT / 'artifacts/kaggle/rsna-r6-coatnet-reference-20261018'
EXPECTED = 'a906df9328b6a53d7e0fe4e18962c655867c8617c54e6745a8af647033b2399c'


def main():
    original = next(SOURCE.glob('*.ipynb')).read_bytes()
    assert hashlib.sha256(original).hexdigest() == EXPECTED
    notebook = json.loads(original)
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            ast.parse(source.split('\n', 1)[1] if source.startswith('%%writefile') else source)
            cell['outputs'] = []
            cell['execution_count'] = None
    audit = '''import json as _audit_json, hashlib as _audit_hash, time as _audit_time
from pathlib import Path as _audit_Path
import numpy as _audit_np
import pandas as _audit_pd
_audit_root = _audit_Path('/kaggle/input')
_audit_candidates = [_audit_root/'rsna-knee-abnormality-detection', _audit_root/'competitions/rsna-knee-abnormality-detection']
_audit_data = next(p for p in _audit_candidates if (p/'sample_submission.csv').exists())
_audit_expected = _audit_pd.read_csv(_audit_data/'sample_submission.csv', dtype={'StudyInstanceUID':str})
_audit_file = _audit_Path('/kaggle/working/submission.csv')
_audit_frame = _audit_pd.read_csv(_audit_file, dtype={'StudyInstanceUID':str})
assert list(_audit_frame.columns) == list(_audit_expected.columns)
assert not _audit_frame.StudyInstanceUID.duplicated().any()
assert len(_audit_frame) == len(_audit_expected) > 0
assert set(_audit_frame.StudyInstanceUID) == set(_audit_expected.StudyInstanceUID)
_audit_values = _audit_frame.iloc[:,1:].to_numpy(dtype=float)
assert _audit_np.isfinite(_audit_values).all()
assert ((_audit_values >= 0) & (_audit_values <= 1)).all()
_audit_result = {'status':'PASS', 'kind':'real_remote_public_reference_inference',
    'source_notebook_sha256':'SOURCE_DIGEST', 'n_test':len(_audit_frame),
    'submission_sha256':_audit_hash.sha256(_audit_file.read_bytes()).hexdigest(),
    'elapsed_seconds':_audit_time.time()-_reference_started,
    'independent_holdout':False, 'training_performed':False,
    'upstream_gold_exclusion':'manifest and pinned trainer; not independently reconstructed',
    'r6_manifest_sha256':_audit_hash.sha256((M_SRC/'manifest.json').read_bytes()).hexdigest()}
_audit_Path('/kaggle/working/reference-audit.json').write_text(_audit_json.dumps(_audit_result, indent=2))
print(_audit_json.dumps(_audit_result, indent=2))
'''.replace('SOURCE_DIGEST', EXPECTED)
    def code(source):
        return {'cell_type':'code','metadata':{},'source':source.splitlines(keepends=True),'outputs':[],'execution_count':None}
    notebook['cells'].insert(0, {'cell_type':'markdown','metadata':{},'source':[
        '# Public reference reproduction\n',
        'Original: goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958, retrieved 2026-10-18.\n',
        'Credit: goodpjw2008, heliosli, nartaa, dreaddevelopment and upstream component authors.\n',
        'Original recipe and attribution retained. Added only timing and output validation.\n',
        'The upstream 0.958 score is not a measured score for this reproduction.\n']})
    notebook['cells'].insert(1, code('import time\n_reference_started = time.time()\n'))
    notebook['cells'].append(code(audit))
    metadata = json.loads((SOURCE/'kernel-metadata.json').read_text())
    metadata.pop('id_no', None)
    metadata.update(id='bulsahkecici/'+OUT.name, title='RSNA R6 CoAtNet public reference 20261018',
                    code_file='reference.ipynb', is_private=True, enable_internet=False)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'reference.ipynb').write_text(json.dumps(notebook, indent=1)+'\n')
    (OUT/'kernel-metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    (OUT/'provenance.json').write_text(json.dumps({
        'source':json.loads((SOURCE/'kernel-metadata.json').read_text())['id'],
        'source_sha256':EXPECTED,'status':'PREPARED',
        'notebook_sha256':hashlib.sha256((OUT/'reference.ipynb').read_bytes()).hexdigest(),
        'authorization':'User: ne yapmamız gerekiyorsa yap; reference run and submission',
        'changes':['clear historical outputs','attribution','timing','final output validation'],
        'promotion':False,'independent_holdout':False},indent=2)+'\n')
    print(OUT)


if __name__ == '__main__':
    main()
