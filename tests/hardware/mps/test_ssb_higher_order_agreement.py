"""The MPS per-term agreement kernel (``mps.higher_order_agreement.agreement_batch``) against independent models.

``tests/hardware/mps/test_ssb_fifth_order.py`` checks the fifth-order objective against this kernel, so the kernel is
checked here on its own: with zero higher orders it is the joint fit's two-term kernel, and with coma, trefoil and
C30 it equals a float64 sum over every q of the band. The data are the weak-phase simulation of ``thin_object`` (one
slice, a probe with known coma, trefoil and spherical aberration).
"""

import math

import numpy as np
import pytest

from tests.hardware.thin_object import ORDERS, TRUTH, chi, session, simulate

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def simulation():
    pytest.importorskip("mlx.core")
    return simulate(TRUTH, tile=1)


def _rows(aberrations, depth_A=0.0, tilt_mrad=(0.0, 0.0)):
    """Agreement rows: (depth, tilt row, tilt col) and the Cartesian Krivanek coefficients C10 ... C30 (A)."""
    row = [depth_A, *tilt_mrad]
    for name, (_, m) in ORDERS.items():
        magnitude = 10.0 * aberrations.get(f"{name}_nm", 0.0)
        angle = math.radians(aberrations.get(f"phi{name[1:]}_deg", 0.0))
        row += [magnitude * math.cos(m * angle), magnitude * math.sin(m * angle)]
    return np.array([row])


def _evidence(prepared):
    """The prepared G(q, k) on the full q plane, k, q and the probe geometry, in float64 on the host."""
    g = np.asarray(prepared.g_qk).astype(np.complex128)
    kr, kc = np.asarray(prepared.kx, np.float64), np.asarray(prepared.ky, np.float64)
    q_row, q_col = np.asarray(prepared.q_row, np.float64).ravel(), np.asarray(prepared.q_col, np.float64).ravel()
    native = len(q_col)
    if g.shape[-1] != native:
        mirror = (-np.arange(len(q_row))) % len(q_row)
        cols = (native - np.arange(native // 2 + 1, native)) % native
        g = np.concatenate((g, g[:, mirror][:, :, cols].conj()), axis=-1)
    geometry = prepared.wavelength, prepared.semiangle_rad, prepared.ang_y_rad, prepared.ang_x_rad
    return g, kr[:, None], kc[:, None], q_row, q_col, geometry


def test_agreement_reduces_to_the_two_term_kernel_and_matches_the_physical_model(simulation):
    """Zero higher orders: the joint fit's kernel. With coma, trefoil and C30: a float64 sum over the full q plane."""
    from quantem.gpu.direct_ptycho.mps.higher_order_agreement import agreement_batch

    data, _, detector_mrad = simulation
    with session(data, detector_mrad, "mps") as ssb:
        backend = ssb._backend()
        backend.cache_rotation(0.0)
        prepared = backend._prepared
        low = {"C10_nm": -4.0, "C12_nm": 1.5, "phi12_deg": 25.0}
        two_term = backend.sample_agreement(np.array([[-40.0, 15.0, math.radians(25.0), 2.0, -3.0, 120.0]]), (0.2, 0.9))
        np.testing.assert_allclose(agreement_batch(prepared, _rows(low, 120.0, (2.0, -3.0))), two_term, rtol=3e-6)

        # independent float64 model on every q of the band (not the half plane the kernel sums with weight 2)
        g, kr, kc, q_row, q_col, (wavelength, semiangle, ang_y, ang_x) = _evidence(prepared)
        band = (np.hypot(q_row[:, None], q_col[None]) > 0.2) & (np.hypot(q_row[:, None], q_col[None]) < 0.9)
        qr, qc = np.broadcast_to(q_row[:, None], band.shape)[band], np.broadcast_to(q_col[None], band.shape)[band]

        def probe(row, col):
            r = np.hypot(row, col)
            with np.errstate(divide="ignore", invalid="ignore"):
                edge = np.hypot(row * ang_y, col * ang_x) / np.where(r > 0, r, 1.0)
                aperture = np.clip(np.where(r > 0, (semiangle - r * wavelength) / edge + 0.5, 1.0), 0.0, 1.0)
            return aperture * np.exp(-1j * chi(row, col, wavelength, TRUTH))

        gamma = probe(kr - qr, kc - qc) * probe(kr, kc).conj() - probe(kr, kc) * probe(kr + qr, kc + qc).conj()
        source = g.reshape(len(g), -1)[:, band.ravel()]
        numerator = np.abs((source * gamma.conj()).sum(0)) ** 2
        denominator = (np.abs(gamma) ** 2).sum(0)
        expected = float((numerator[denominator > 0] / denominator[denominator > 0]).sum())
        np.testing.assert_allclose(agreement_batch(prepared, _rows(TRUTH)), expected, rtol=1e-5)
