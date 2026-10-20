"""Engineering checks only; synthetic fixtures do not authorize a production submission."""
import csv
import importlib.util
import json
from pathlib import Path

from rsna_knee.ontology import TARGET_COLUMNS


def generator():
    path = Path(__file__).resolve().parents[1] / 'scripts/make_submission_kernel.py'
    spec = importlib.util.spec_from_file_location('submission_generator', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_canonical_test_chunks_keep_predictions_and_release_cache(tmp_path):
    module = generator()
    identity = {'policy': {'layout': 'canonical_volumes'}, 'members': [{}], 'reader': None}
    ns = {'__name__': 'streaming_test'}
    exec(compile(module.ENTRY.replace('__IDENTITY__', repr(json.dumps(identity))), 'entry.py', 'exec'), ns)
    comp, work, scratch = (tmp_path / n for n in ('competition', 'working', 'tmp'))
    comp.mkdir()
    scratch.mkdir()
    uids = [f'syn-{i}' for i in range(65)]
    (comp / 'test.csv').write_text('StudyInstanceUID\n' + '\n'.join(uids) + '\n')
    (comp / 'sample_submission.csv').write_text(','.join(['StudyInstanceUID', *TARGET_COLUMNS]) + '\n')
    chunks, previous_cache = [], []

    def preprocess(_csv, _root, ids, dest, *_args):
        assert not dest.exists(), 'the preceding chunk was not deleted'
        dest.mkdir()
        (dest / 'payload.npz').write_bytes(b'synthetic payload')
        chunks.append(list(ids))
        previous_cache.append(dest)
        return {'quarantine': {}}

    def predict_all(manifests, ids, *_args):
        assert (manifests[0].parent / 'payload.npz').is_file()
        return ({u: {t: int(u.split('-')[1]) / 100 for t in TARGET_COLUMNS} for u in ids},
                [], list(TARGET_COLUMNS), 'cpu')

    ns.update(WORK=work, TMP=scratch, RERUN=True, find_asset=lambda: tmp_path,
              find_competition=lambda: comp, setup_packages=lambda _: {'target': ''},
              unpack=lambda *_: tmp_path, preprocess=preprocess, predict_all=predict_all,
              check_policy=lambda *_: True)
    ns['main']()
    assert list(map(len, chunks)) == [32, 32, 1]
    assert not any(p.exists() for p in previous_cache)
    with (work / 'submission.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    assert [r['StudyInstanceUID'] for r in rows] == uids
    assert all(float(r[TARGET_COLUMNS[0]]) == i / 100 for i, r in enumerate(rows))
    result = json.loads((work / 'offline-result.json').read_text())
    assert result['n_strict'] == 65 and result['n_fallback'] == 0
