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


@pytest.mark.parametrize('phase',['SUBMIT_UNKNOWN','SUBMITTED','ASSET_PUSH_UNKNOWN','KERNEL_PUSH_UNKNOWN'])
def test_publication_requires_reconciliation_before_external_mutation(tmp_path,phase):
    module=load_module('candidate_publish')
    (tmp_path/'publication.json').write_text(json.dumps({'phase':phase}))
    calls=[]
    with pytest.raises(RuntimeError,match='Reconcile'):
        module.publish({'kernel':'owner/test'},tmp_path,lambda args:calls.append(args),lambda *args:None,lambda *args:None)
    assert calls==[]


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
