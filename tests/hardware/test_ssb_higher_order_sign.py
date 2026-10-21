"""Reconstruction with odd aberrations (coma, trefoil) on the session backend, against a weak-phase simulation.

For a thin object G(q, k) ~ P(k - q) conj P(k) - P(k) conj P(k + q); evaluating the first term at q - k reverses the
odd orders on that sideband. With the probe's own coefficients the reconstruction must then be closer to the object
than without the higher orders or with the odd ones reversed, for both estimators. With the probe phase at q - k the
true coefficients scored no better than leaving them out (0.941 against 0.942 on a similar simulation).
"""

import pytest

from tests.hardware.thin_object import (
    TRUTH,
    backend_or_skip,
    correlation,
    session,
    simulate,
)

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def simulation():
    backend = backend_or_skip()
    # CUDA: 256 x 256, where mean_phase runs the fused full-aberration FFT kernels and phase_of_mean the wave sum
    return (backend, *simulate(TRUTH, tile=4 if backend == "cuda" else 1))


@pytest.mark.parametrize("estimator", ["phase_of_mean", "mean_phase"])
def test_reconstruction_corrects_odd_orders_with_their_sign(simulation, estimator):
    backend, data, phase, detector_mrad = simulation
    reversed_odd = {**TRUTH, "phi21_deg": TRUTH["phi21_deg"] + 180.0, "phi23_deg": TRUTH["phi23_deg"] + 60.0}
    low = {name: TRUTH[name] for name in ("C10_nm", "C12_nm", "phi12_deg")}
    with session(data, detector_mrad, backend) as ssb:
        scores = {}
        for name, coefficients in (("truth", TRUTH), ("low", low), ("reversed", reversed_odd)):
            image, _ = ssb.preview(coefficients, compute_loss=False, phase_estimator=estimator)
            scores[name] = correlation(image.get() if hasattr(image, "get") else image, phase)
    assert scores["truth"] > 0.92, scores
    assert scores["truth"] > scores["low"] + 0.02, scores
    assert scores["truth"] > scores["reversed"] + 0.05, scores
