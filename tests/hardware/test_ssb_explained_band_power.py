"""``sample_agreement`` is the fraction of G's band power the model explains, the same for every fit.

The band power P = sum_q w_q sum_k |G(q, k)|^2 bounds the agreement S (Cauchy-Schwarz at each q), so S / P is in
[0, 1]. On the noise-free weak-phase simulation of ``thin_object`` the SSB model is exact: the fifth-order fit, which
holds the probe's coma, trefoil and C30, explains nearly all of it, and the low-order fits less. The fifth-order
objective at the low-order fit is the low-order fit's own fraction, and its ``tilt_fit_gain`` follows from the
primary's. Runs on this machine's backend (MPS on Apple silicon, else CUDA).
"""

import numpy as np
import pytest

from tests.hardware.thin_object import TRUTH, backend_or_skip, session, simulate

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def ptycho():
    backend = backend_or_skip()
    # CUDA reconstructs 128 x 128 scans without padding: two periods of the same specimen
    data, _, detector_mrad = simulate(TRUTH, tile=2 if backend == "cuda" else 1)
    with session(data, detector_mrad, backend) as ssb:
        yield ssb


def _float64_band_power(ssb, band=(0.2, 0.9)) -> float:
    """sum_q sum_k |G|^2 over every q of the band in float64, the stored half plane's mirror columns counted twice."""
    backend = ssb._backend()
    if ssb.backend == "mps":
        backend.cache_rotation(0.0)
        prepared = backend._prepared
        g = np.asarray(prepared.g_qk).astype(np.complex128)
        q_row = np.asarray(prepared.q_row, np.float64).ravel()
        q_col = np.asarray(prepared.q_col, np.float64).ravel()
    else:
        engine = backend._get_accelerator()
        g = engine.G_qk.get().astype(np.complex128)
        q_row, q_col = engine._cache["qx_1d"].get().astype(np.float64), engine._cache["qy_1d"].get().astype(np.float64)
    nx, cols = len(q_col), g.shape[-1]
    weight = np.ones(cols)
    if cols != nx:
        weight[1:nx // 2 if nx % 2 == 0 else cols] = 2.0
    q = np.hypot(q_row[:, None], q_col[None, :cols])
    inside = (q > band[0]) & (q < band[1])
    return float(((np.abs(g) ** 2).sum(axis=0) * weight[None] * inside).sum())


def test_band_power_is_the_float64_sum_and_bounds_the_agreement(ptycho):
    backend = ptycho._backend()
    power = backend.band_power((0.2, 0.9))
    assert power == pytest.approx(_float64_band_power(ptycho), rel=1e-6)
    assert backend.band_power((0.2, 0.9)) == power
    rng = np.random.default_rng(0)
    rows = np.column_stack([rng.uniform(-200, 200, 16), rng.uniform(0, 100, 16), rng.uniform(-1.5, 1.5, 16),
                            rng.uniform(-10, 10, (16, 2)), rng.uniform(0, 200, 16)])
    explained = backend.sample_agreement(rows, (0.2, 0.9)) / power
    assert np.all((explained > 0) & (explained <= 1))


def test_every_fit_reports_the_explained_fraction_and_the_fifth_order_gain(ptycho):
    primaries = []
    fit = ptycho.fit_aberrations(search="standard", trials=50, show=False, verbose=False,
                                  on_initial_fit=primaries.append)
    primary = primaries[0]
    standard = ptycho.fit_aberrations(search="primary", tilt=False, trials=50, verbose=False)
    assert 0 < standard.sample_agreement < primary.sample_agreement < fit.sample_agreement <= 1
    assert fit.sample_agreement > 0.99
    # the fifth-order objective at the low-order fit is that fit's own fraction of the band power
    assert fit.optimizer["normalization"] / fit.optimizer["band_power"] == pytest.approx(
        primary.sample_agreement, rel=1e-5)
    assert primary.tilt_fit_gain > 1
    assert fit.tilt_fit_gain == pytest.approx(primary.tilt_fit_gain * fit.optimizer["normalized_agreement"])

    unmeasured = ptycho.fit_aberrations(search="standard", trials=50, show=False, verbose=False,
                                         compare_standard=False)
    assert unmeasured.tilt_fit_gain is None
    assert unmeasured.sample_agreement == pytest.approx(fit.sample_agreement, rel=1e-6)
