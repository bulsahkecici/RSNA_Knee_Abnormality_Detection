import pytest
from rsna_knee.runtime.training_budget import assess_training_budget


def measure(**changes):
    args = dict(measured_studies=128, measured_seconds=50, total_training_studies=48706,
                peak_cuda_bytes=12_000_000_000, device_vram_bytes=15_636_037_632,
                max_projected_training_seconds=21600, max_peak_vram_fraction=.9)
    return assess_training_budget(**(args | changes))


def test_budget_rejects_slow_training_and_low_vram_headroom():
    assert measure()['pass']
    assert not measure(measured_seconds=90)['pass']
    assert 'projected_training_time_exceeded' in measure(measured_seconds=90)['reasons']
    assert not measure(peak_cuda_bytes=15_000_000_000)['pass']
    for kwargs in ({'measured_seconds': 0}, {'measured_studies': 0}, {'measured_seconds': float('nan')},
                   {'peak_cuda_bytes': float('inf')}):
        with pytest.raises(ValueError):
            measure(**kwargs)
