"""Ranking diagnostics must handle ties and avoid dependence on input row order."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

spec=importlib.util.spec_from_file_location('diagnostic',Path(__file__).parents[1]/'scripts/diagnose_oa_synovitis.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_pair_cost_matches_auc_with_ties_and_reordering():
    y=np.array([1,0,1,0,0]);s=np.array([.5,.5,.2,.7,.1])
    for order in (np.arange(5),np.array([4,2,0,3,1])):
        p,n,e=m.pair_errors(y[order],s[order])
        assert e.shape==(2,3)
        assert set(np.unique(e))=={0,.5,1}
        assert 1-e.mean()==pytest.approx(roc_auc_score(y,s))


def test_pair_errors_reject_masked_targets_and_nonfinite_scores():
    with pytest.raises(AssertionError):m.pair_errors([1,-1,0],[.7,.5,.2])
    with pytest.raises(AssertionError):m.pair_errors([1,0],[np.nan,.2])


def test_csv_alignment_rejects_duplicate_and_missing_uids(tmp_path):
    path=tmp_path/'predictions.csv'
    path.write_text('StudyInstanceUID,score\nb,0.8\na,0.1\n')
    assert m.aligned_csv(path,['a','b']).score.tolist()==[.1,.8]
    with pytest.raises(AssertionError):m.aligned_csv(path,['a','c'])
    path.write_text('StudyInstanceUID,score\na,0.8\na,0.1\n')
    with pytest.raises(AssertionError):m.aligned_csv(path,['a','b'])
