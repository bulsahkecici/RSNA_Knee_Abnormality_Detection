"""No full-job launch before a passed pilot and sufficient observed quota."""
import importlib
import json
from pathlib import Path
import pytest

@pytest.fixture
def controller(tmp_path, monkeypatch):
    import sys
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    module = importlib.import_module('monitor_k3')
    monkeypatch.setattr(module, 'PLAN', tmp_path / 'plan.json')
    monkeypatch.setattr(module, 'bindings', lambda job: tmp_path)
    return module


def test_waiting_quota_cannot_launch_until_enough_gpu_time(controller, monkeypatch):
    m = controller
    m.PLAN.write_text(json.dumps({'status':'WAITING_QUOTA','minimum_remaining_gpu_hours':10.5,'full':{}}))
    launched=[]
    monkeypatch.setattr(m, 'quota', lambda: (2.57, {'remaining':'2.57h'}))
    monkeypatch.setattr(m, 'push', lambda plan,key,timeout: launched.append((key,timeout)))
    assert m.tick() is False
    assert launched == []
    monkeypatch.setattr(m, 'quota', lambda: (30., {'remaining':'30.00h'}))
    assert m.tick() is False
    assert launched == [('full',36000)]


@pytest.mark.parametrize('status', ['PUSH_UNKNOWN', 'COLAB_HANDOFF'])
def test_uncertain_upload_is_terminal_and_never_retried(controller, monkeypatch, status):
    m = controller
    m.PLAN.write_text(json.dumps({'status':status}))
    monkeypatch.setattr(m, 'quota', lambda: pytest.fail('Must reconcile uncertain upload before further action'))
    assert m.tick() is True


def test_existing_push_attempt_prevents_repeat_even_without_receipt(controller, monkeypatch, tmp_path):
    m = controller
    (tmp_path / 'push-attempt.json').write_text('{}')
    monkeypatch.setattr(m.subprocess,'run',lambda *a,**k: pytest.fail('Duplicate GPU launch'))
    with pytest.raises(AssertionError,match='repeat'):
        m.push({'pilot':{'kernel':'synthetic/pilot'}},'pilot',3600)
