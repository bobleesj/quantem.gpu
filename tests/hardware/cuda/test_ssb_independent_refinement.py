"""Refinement batches preserve the scalar CUDA objective exactly."""

import numpy as np
import pytest

pytestmark = pytest.mark.slow
cp = pytest.importorskip('cupy')


def test_independent_candidates_match_singleton_objectives():
    from quantem.gpu import SSB

    rng = np.random.default_rng(7)
    counts = rng.poisson(40.0, size=(128, 128, 64, 64)).astype(np.uint16)
    row, col = np.meshgrid(np.arange(64) - 31.5, np.arange(64) - 31.5, indexing='ij')
    counts[:, :, np.hypot(row, col) < 22] += 200
    parameters = np.column_stack([
        rng.uniform(-300, 300, 17), rng.uniform(0, 200, 17),
        rng.uniform(-1.5, 1.5, 17), rng.uniform(-20, 20, 17),
        rng.uniform(-20, 20, 17), rng.choice([0.0, 150.0, 400.0], 17),
    ])
    with SSB(counts, backend='cuda', voltage_kV=300, semiangle_mrad=30,
             scan_sampling_A=(0.3, 0.3), detector_sampling_mrad=(30 / 22, 30 / 22)) as ssb:
        ssb.preview({'C10_nm': 0.0, 'C12_nm': 0.0, 'phi12_deg': 0.0}, compute_loss=False)
        model = ssb._backend()._get_accelerator().thick
        expected = np.array([model.fit_batch(point)[0] for point in parameters])
        for size in (2, 3, 8, 17):
            actual = model.fit_batch(parameters[:size], independent=True)
            np.testing.assert_array_equal(actual, expected[:size])
