"""External mutation safety: ambiguous receipts are never blindly repeated."""
import importlib.util
import json
from pathlib import Path
import pytest


def load_module(name):
    path=Path(__file__).resolve().parents[1]/'scripts'/f'{name}.py'
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bundle_excludes_mac_metadata_and_hidden_paths(tmp_path):
    module=load_module('candidate_publish')
    (tmp_path/'src').mkdir()
    (tmp_path/'src/model.py').write_text('real source')
    (tmp_path/'src/.DS_Store').write_bytes(b'mac metadata')
    (tmp_path/'.hidden').mkdir()
    (tmp_path/'.hidden/file').write_text('hidden')
    (tmp_path/'MANIFEST.json').write_text('{}')
    assert set(module.bundle_files(tmp_path))=={'src/model.py'}


@pytest.mark.parametrize('phase',['SUBMIT_UNKNOWN','ASSET_PUSH_UNKNOWN','KERNEL_PUSH_UNKNOWN'])
def test_publication_requires_reconciliation_before_external_mutation(tmp_path,phase):
    module=load_module('candidate_publish')
    (tmp_path/'publication.json').write_text(json.dumps({'phase':phase}))
    calls=[]
    with pytest.raises(RuntimeError,match='Reconcile'):
        module.publish({'kernel':'owner/test'},tmp_path,lambda args:calls.append(args),lambda *args:None,lambda *args:None)
    assert calls==[]


def test_submitted_resume_records_exact_score_without_resubmission(tmp_path):
    module=load_module('candidate_publish')
    (tmp_path/'publication.json').write_text(json.dumps({'phase':'SUBMITTED','submission_description':'unique candidate'}))
    calls=[]
    def call(args):
        calls.append(args)
        return 'ref,description,status,publicScore\n1,other,SubmissionStatus.COMPLETE,0.999\n2,unique candidate,SubmissionStatus.COMPLETE,0.741\n'
    def write(path,value):
        path.write_text(json.dumps(value))
    state=module.publish({},tmp_path,call,write,lambda message:None)
    assert state['public_score']==0.741
    assert state['submission_ref']=='2'
    assert state['phase']=='SCORED'
    assert all(args[:2]==['competitions','submissions'] for args in calls)


def test_duplicate_description_refuses_score_attribution(tmp_path):
    module=load_module('candidate_publish')
    with pytest.raises(RuntimeError,match='Ambiguous'):
        module.track_submission({'submission_description':'same'},tmp_path/'publication.json',
            lambda args:'ref,description,status,publicScore\n1,same,SubmissionStatus.COMPLETE,0.7\n2,same,SubmissionStatus.COMPLETE,0.8\n',
            lambda *args:None,lambda *args:None)


def test_daily_limit_waits_without_external_mutation(monkeypatch):
    module=load_module('candidate_publish')
    limits=iter(['Remaining today: 0','Remaining today: 0','Remaining today: 5'])
    calls=[];pending=[];sleeps=[]
    monkeypatch.setattr(module.time,'sleep',sleeps.append)
    def call(args):
        calls.append(args)
        return next(limits)
    assert module.wait_submission_slot(call,pending.append)=='Remaining today: 5'
    assert len(pending)==2 and sleeps==[60,60]
    assert all(args[:2]==['competitions','submission-limits'] for args in calls)


def test_queue_refuses_ambiguous_kernel_push(tmp_path,monkeypatch):
    module=load_module('autonomous_campaign')
    monkeypatch.setattr(module,'FOLDER',tmp_path)
    monkeypatch.setattr(module,'notify',lambda message:None)
    (tmp_path/'plan.json').write_text(json.dumps({'jobs':[{'id':'test'}],'max_jobs':1,'max_elapsed_hours':24}))
    (tmp_path/'status.json').write_text(json.dumps({'started_ts':module.time.time(),'jobs':{'test':{'phase':'PUSH_UNKNOWN'}},'gpu_hours_observed':0}))
    calls=[]
    monkeypatch.setattr(module,'call',lambda args:calls.append(args))
    with pytest.raises(RuntimeError,match='Ambiguous push'):
        module.run()
    assert calls==[]
    assert json.loads((tmp_path/'status.json').read_text())['phase']=='NEEDS_ATTENTION'
