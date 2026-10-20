"""A bounded, measured-throughput gate; never a performance/AUC result."""
from __future__ import annotations
import math


def assess_training_budget(*, measured_studies, measured_seconds, total_training_studies,
                           peak_cuda_bytes, device_vram_bytes, max_projected_training_seconds,
                           max_peak_vram_fraction):
    values = (measured_studies, measured_seconds, total_training_studies,
              device_vram_bytes, max_projected_training_seconds, max_peak_vram_fraction)
    if any(not math.isfinite(float(v)) or v <= 0 for v in values) or peak_cuda_bytes < 0:
        raise ValueError('invalid throughput/budget measurement')
    if max_peak_vram_fraction > 1 or not math.isfinite(float(peak_cuda_bytes)):
        raise ValueError('invalid VRAM budget')
    rate = measured_studies / measured_seconds
    projection = total_training_studies / rate
    fraction = peak_cuda_bytes / device_vram_bytes
    reasons = []
    if projection > max_projected_training_seconds:
        reasons.append('projected_training_time_exceeded')
    if fraction > max_peak_vram_fraction:
        reasons.append('peak_vram_fraction_exceeded')
    return {'kind': 'real_measured_training_resource_gate_not_AUC', 'pass': not reasons,
            'studies_per_second': rate, 'projected_training_seconds': projection,
            'peak_cuda_bytes': peak_cuda_bytes, 'device_vram_bytes': device_vram_bytes,
            'peak_vram_fraction': fraction, 'reasons': reasons}
