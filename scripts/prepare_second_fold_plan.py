"""Verify the matched K0F1/K1F1 pair and write a launch plan; never launches jobs."""
import json
from datetime import datetime, timezone
from pathlib import Path

from make_submission_kernel import notebook_cells
from rsna_knee.data.folds import assert_group_integrity, assert_train_uids, load_folds
from rsna_knee.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'artifacts/research/second-fold-campaign.json'


def prepare():
    assert not PLAN.exists(), 'Preserve the existing campaign; reconcile its receipts instead of regenerating.'
    first = json.loads((ROOT / 'artifacts/colab/p1-K1/output/review.json').read_text())
    assert first['decision']['status'] == 'passes_this_fold_only', 'fold0 prerequisite did not pass'
    experiments, jobs = [], []
    for exp_id in ('K0F1', 'K1F1'):
        cells = notebook_cells(exp_id)
        ns = {};exec(cells[1], ns)
        exp = ns['EXPERIMENT'];experiments.append(exp)
        assert exp['id'] == exp_id and exp['outer_fold'] == '1' and not exp['submission']
        folds = load_folds(ROOT / exp['folds_file']);assert_group_integrity(folds)
        assert_train_uids(set(exp['train_uids']), folds)
        roles = {r['StudyInstanceUID']: r for r in folds}
        assert all(roles[u]['fold'] != '1' for u in exp['train_uids'])
        assert set(exp['weak_uids']) == {r['StudyInstanceUID'] for r in folds if r['fold'] == '1' and not r.get('gold_set')}
        assert set(exp['train_uids']).isdisjoint(exp['weak_uids'] + exp['gold_uids'])
        assert len(exp['gold_uids']) == 58
        folder = ROOT / f'artifacts/kaggle/rsna-{exp_id.lower()}-kneenet-fold1-20261015'
        metadata = json.loads((folder / 'kernel-metadata.json').read_text())
        assert metadata['is_private'] and metadata['enable_gpu'] and not metadata['competition_sources']
        assert metadata['kernel_sources'] == [p['kernel'] for p in exp['cache_parts']]
        jobs.append({'experiment': exp_id, 'kernel': metadata['id'], 'folder': str(folder.relative_to(ROOT)),
                     'notebook_sha256': sha256_file(folder / metadata['code_file']),
                     'build_sha256': sha256_file(ROOT / f'notebooks/generated/{exp_id}.build.json'),
                     'metadata_sha256': sha256_file(folder / 'kernel-metadata.json'),
                     'source_sha256': exp['source_sha256'], 'status': 'PREPARED_NOT_STARTED',
                     'requested_accelerator': 'NvidiaTeslaT4', 'timeout_seconds': 21600})
    matched = ['train', 'train_uids', 'gold_uids', 'weak_uids', 'folds_sha256', 'labels_sha256',
               'soft_labels_sha256', 'source_sha256', 'training_cell_sha256', 'platform_recipe_sha256',
               'weights_sha256', 'reader_code_sha256', 'eval_precision', 'outer_fold']
    assert all(experiments[0][k] == experiments[1][k] for k in matched), 'pair is not matched'
    assert experiments[0]['eval_precision'] == 'fp32'
    old = {};exec(notebook_cells('K1')[1], old)
    assert set(experiments[1]['weak_uids']).isdisjoint(old['EXPERIMENT']['weak_uids'])
    plan = {'kind': 'authorized_second_outer_fold_pair_not_training_result',
            'status': 'PREPARED_NOT_STARTED', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'authorization': 'User: ne gerekiyorsa yap, following the proposal to train and compare K0F1/K1F1',
            'n_train': len(experiments[0]['train_uids']), 'n_weak': len(experiments[0]['weak_uids']),
            'n_gold_reused_development': 58, 'matched_fields': matched, 'gold_exclusion': True,
            'outer_fold_exclusion': True, 'weak_fold0_fold1_disjoint': True,
            'fold0_decision': first['decision']['status'],
            'confirmation_rule': 'fold0 fixed threshold+CI already passed; require same positive weak delta on fold1, report its CI and all class regressions',
            'promotion': False, 'reader_only_candidate_changed': False, 'jobs': jobs}
    PLAN.write_text(json.dumps(plan, indent=2) + '\n')
    return plan


if __name__ == '__main__':
    print(json.dumps(prepare(), indent=2))
