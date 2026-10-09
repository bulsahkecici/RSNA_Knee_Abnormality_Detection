import importlib.util
from pathlib import Path
import pytest
from rsna_knee.ontology import TARGET_COLUMNS

spec=importlib.util.spec_from_file_location('campaign_review',Path(__file__).resolve().parents[1]/'scripts/campaign_review.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def row(split='gold'):
 return {'StudyInstanceUID':'fixture-study-'+split,'split':split,**{t:0.5 for t in TARGET_COLUMNS}}

@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-0.01,1.01])
def test_weak_probabilities_are_checked_even_for_unlabelled_targets(bad):
 r=row('weak');r['MCL']=bad
 with pytest.raises(ValueError,match='invalid_prediction_probability'):
  module.validate_prediction_table([row(),r])

def test_duplicate_uid_rejected():
 with pytest.raises(ValueError,match='duplicate_prediction_uid'):
  module.validate_prediction_table([row(),row()])

def test_unknown_split_rejected():
 with pytest.raises(ValueError,match='invalid_prediction_split'):
  module.validate_prediction_table([row('train')])

def test_same_uid_in_different_splits_rejected():
 weak=row('weak');weak['StudyInstanceUID']=row()['StudyInstanceUID']
 with pytest.raises(ValueError,match='prediction_split_overlap'):
  module.validate_prediction_table([row(),weak])
