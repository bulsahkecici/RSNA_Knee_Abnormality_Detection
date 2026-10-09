import hashlib
import json
from pathlib import Path
import pytest
from rsna_knee.submission.offline import inference_cache_options


def config(tmp_path,centers=6):
 data=Path(__file__).resolve().parents[1]/'src/rsna_knee/data'
 policy={'size':224,'n_centers':centers,'dtype':'uint8','center_strategy':'interior','center_lower':.15,'center_upper':.85,'crop_mm':160.,'clip_percentiles':[.5,99.5],'slots':[['Sagittal','fluid'],['Coronal','fluid'],['Axial','fluid']],'code_sha256':{n:hashlib.sha256((data/n).read_bytes()).hexdigest() for n in ['cache_build.py','cache.py','preprocess.py','dicom.py']}}
 p=tmp_path/'preprocess.json';p.write_text(json.dumps(policy));return p,policy


def test_six_centers_come_from_training_policy(tmp_path):
 p,_=config(tmp_path)
 options=inference_cache_options(224,None,p)
 assert options['n_centers']==6 and options['strict_series']
 assert options['center_lower']==.15


def test_conflicting_center_request_rejected(tmp_path):
 p,_=config(tmp_path)
 with pytest.raises(ValueError,match='conflicts'):
  inference_cache_options(224,3,p)


@pytest.mark.parametrize('key,value,error',[('crop_mm',120.,'pixel_policy'),('size',256,'layout'),('n_centers',0,'layout'),('dtype','float32','preprocess_policy')])
def test_unsupported_training_policy_is_not_silently_ignored(tmp_path,key,value,error):
 p,d=config(tmp_path);d[key]=value;p.write_text(json.dumps(d))
 with pytest.raises(ValueError,match=error):inference_cache_options(224,None,p)


def test_wrong_source_hash_rejected(tmp_path):
 p,d=config(tmp_path);d['code_sha256']['dicom.py']='wrong';p.write_text(json.dumps(d))
 with pytest.raises(ValueError,match='source_hash'):inference_cache_options(224,None,p)


def test_legacy_default_preserved():
 assert inference_cache_options(224,None,None)=={'size':224,'n_centers':3,'strict_series':False}
