"""Server status protocol tests; no competition requests are sent."""
from types import SimpleNamespace

from rsna_knee.submission.status import poll_score
from rsna_knee.workflow.registry import Registry
from rsna_knee.workflow.state import PipelineState, Stage


def _registry(tmp_path, monkeypatch):
    monkeypatch.setattr('rsna_knee.submission.status.roots_for', lambda _: SimpleNamespace(runs=tmp_path/'runs'))
    registry = Registry(tmp_path/'registry.sqlite')
    state = PipelineState(run_id='run', profile='pilot', stage=Stage.SUBMITTED,
                          artifacts={'kernel': 'owner/kernel', 'kernel_version': '2'})
    registry.upsert_run('run', 'pilot', state.stage, state.model_dump(), False)
    registry.put_submission('local-attempt', 'run', 'SUBMITTED',
                            {'server_receipt': '12345'}, kernel='owner/kernel', version='2')
    return registry


def test_complete_without_public_score_is_not_scored(tmp_path, monkeypatch):
    registry = _registry(tmp_path, monkeypatch)
    provider = SimpleNamespace(list_submissions=lambda: [{'ref': 12345, 'status': 'SubmissionStatus.COMPLETE', 'publicScore': ''}])
    result = poll_score('run', registry=registry, provider=provider)
    assert result['status'] == 'PENDING'
    assert registry.get_run('run')['stage'] == 'SUBMITTED'


def test_another_receipt_cannot_complete_the_run(tmp_path, monkeypatch):
    registry = _registry(tmp_path, monkeypatch)
    provider = SimpleNamespace(list_submissions=lambda: [{'ref': 67890, 'status': 'SubmissionStatus.COMPLETE', 'publicScore': '0.61'}])
    result = poll_score('run', registry=registry, provider=provider)
    assert result['status'] == 'PENDING'
    assert result['reason'] == 'receipt_not_in_history'


def test_scored_receipt_stays_duplicate_protected(tmp_path, monkeypatch):
    registry = _registry(tmp_path, monkeypatch)
    provider = SimpleNamespace(list_submissions=lambda: [{'ref': 12345, 'status': 'SubmissionStatus.COMPLETE', 'publicScore': '0.61'}])
    result = poll_score('run', registry=registry, provider=provider)
    assert result['status'] == 'SCORED'
    assert result['public_score'] == 0.61
    assert registry.get_run('run')['stage'] == 'SCORED'
    assert registry.submission_for_kernel('owner/kernel', '2')['status'] == 'SCORED'


def test_detail_error_overrides_stale_pending_history(tmp_path, monkeypatch):
    registry = _registry(tmp_path, monkeypatch)
    provider = SimpleNamespace(
        list_submissions=lambda: [{'ref': 12345, 'status': 'SubmissionStatus.PENDING', 'publicScore': ''}],
        get_submission=lambda receipt: {'ref': int(receipt), 'status': 'SubmissionStatus.COMPLETE',
                                        'publicScore': '', 'errorDescription': 'Notebook Threw Exception'},
    )
    result = poll_score('run', registry=registry, provider=provider)
    assert result['status'] == 'ERROR'
    assert registry.get_run('run')['stage'] == 'FAILED'
