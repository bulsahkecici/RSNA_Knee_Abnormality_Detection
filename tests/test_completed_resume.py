import torch
from tests.test_resume_equivalence import _fresh, _batches
from rsna_knee.training.loop import train_study_model


def test_completed_resume_retains_supervision_without_new_updates(tmp_path):
    path=tmp_path/'done.pt'
    args=dict(epochs=1,effective_batch=1,seed=0,input_id='verified-input',config_id='verified-config')
    first=train_study_model(_fresh(),_batches(),ckpt_path=path,**args)
    second=train_study_model(_fresh(),_batches(),ckpt_path=path,resume=True,**args)
    assert second['step']==first['step']
    assert second['supervised_weight']==first['supervised_weight'] > 0
    assert second['validated_completed_resume'] is True


def test_legacy_completed_resume_requires_identity_and_real_optimizer_state(tmp_path):
    path=tmp_path/'legacy.pt'
    args=dict(epochs=1,effective_batch=1,seed=0,input_id='verified-input',config_id='verified-config')
    train_study_model(_fresh(),_batches(),ckpt_path=path,**args)
    blob=torch.load(path,weights_only=False);blob.pop('supervised_weight');torch.save(blob,path)
    out=train_study_model(_fresh(),_batches(),ckpt_path=path,resume=True,**args)
    assert out['validated_completed_resume'] is True and out['supervised_weight']==0
    blob=torch.load(path,weights_only=False);blob['input_id']=None;torch.save(blob,path)
    out=train_study_model(_fresh(),_batches(),ckpt_path=path,resume=True,**args)
    assert out['validated_completed_resume'] is False


def test_completed_legacy_amp_skips_do_not_require_planned_successful_steps(tmp_path):
    path=tmp_path/'amp-skipped.pt'
    args=dict(epochs=1,effective_batch=1,seed=0,input_id='verified-input',config_id='verified-config')
    train_study_model(_fresh(),_batches(),ckpt_path=path,**args)
    blob=torch.load(path,weights_only=False)
    blob.pop('supervised_weight')
    blob['step']=1
    blob['attempted_updates']=2
    blob['skipped_updates']=1
    torch.save(blob,path)
    out=train_study_model(_fresh(),_batches(),ckpt_path=path,resume=True,**args)
    assert out['step'] < out['updates_planned']
    assert out['validated_completed_resume'] is True
    assert all(out['resume_validation'].values())
    blob=torch.load(path,weights_only=False);blob['attempted_updates']=1;torch.save(blob,path)
    out=train_study_model(_fresh(),_batches(),ckpt_path=path,resume=True,**args)
    assert out['validated_completed_resume'] is False
    assert out['resume_validation']['update_counters_consistent'] is False
