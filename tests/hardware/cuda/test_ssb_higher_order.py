"""Higher-order SSB preview on CUDA: the 14-term Krivanek path an interactive viewer calls with ``higher_order_magnitudes``.

Top: the preview matches an independent float64-chi Torch reconstruction with every aberration order active. Bottom: with
only C10/C12 non-zero, the 14-term kernel reproduces the two-term preview, because chi is the same polynomial.
"""

import math

import numpy as np
import pytest

pytestmark = pytest.mark.slow

cp = pytest.importorskip("cupy")
torch = pytest.importorskip("torch")

from tests.parity.torch_ssb import TorchSSB

# Krivanek (n, m) per packed slot: C10, C12, C21, C23, C30, C32, C34, C41, C43, C45, C50, C52, C54, C56.
_ORDERS = ((1, 0), (1, 2), (2, 1), (2, 3), (3, 0), (3, 2), (3, 4), (4, 1), (4, 3), (4, 5), (5, 0), (5, 2), (5, 4), (5, 6))


class KrivanekSSB(TorchSSB):
    """The Torch SSB reference with chi(v) = (2 pi / lambda) sum_nm C_nm alpha^(n+1) / (n+1) cos(m (phi_v - phi_nm)).

    chi is evaluated in float64 from atan2, independent of the Chebyshev recurrence in the CUDA kernel. The C10/C12/phi12
    arguments of ``reconstruct`` are ignored; ``magnitudes_A`` and ``angles_rad`` carry all 14 coefficients.
    """

    magnitudes_A: np.ndarray
    angles_rad: np.ndarray

    def _chi(self, vx, vy):
        vx, vy = vx.double(), vy.double()
        alpha = torch.sqrt(vx * vx + vy * vy) * self.wavelength
        phi = torch.atan2(vy, vx)
        total = torch.zeros_like(alpha)
        for (n, m), magnitude, angle in zip(_ORDERS, self.magnitudes_A, self.angles_rad):
            total = total + float(magnitude) * alpha ** (n + 1) / (n + 1) * torch.cos(m * (phi - float(angle)))
        return 2.0 * math.pi / self.wavelength * total

    def _gamma(self, qx, qy, kx, ky, *_):
        *_, ap_k = self._geometry(kx, ky)
        *_, ap_m = self._geometry(qx - kx, qy - ky)
        *_, ap_p = self._geometry(qx + kx, qy + ky)
        chi_k, chi_m, chi_p = self._chi(kx, ky), self._chi(qx - kx, qy - ky), self._chi(qx + kx, qy + ky)
        t1 = (ap_m * ap_k) * torch.exp(-1j * (chi_m - chi_k))
        t2 = (ap_p * ap_k) * torch.exp(1j * (chi_p - chi_k))
        return t1 - t2


def _session_256():
    """Poisson counts with a bright-field disk on a 256 x 256 scan: the 128 kernels carry C10/C12 only."""
    from quantem.gpu import SSB

    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("No CUDA device")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("No CUDA runtime")
    rng = np.random.default_rng(11)
    counts = rng.poisson(40.0, size=(256, 256, 32, 32)).astype(np.uint16)
    rr, cc = np.meshgrid(np.arange(32) - 15.5, np.arange(32) - 15.5, indexing="ij")
    counts[:, :, np.hypot(rr, cc) < 10] += 200
    ssb = SSB(counts, backend="cuda", voltage_kV=300.0, semiangle_mrad=30.0, scan_sampling_A=0.3, det_sampling=3.0,
              rotation_angle_deg=0.0)
    ssb.reconstruct(aberrations={"C10": 0.0, "C12": 0.0, "phi12": 0.0})
    return ssb


def _packed(magnitudes_nm: dict[int, float], angles_rad: dict[int, float]) -> tuple[np.ndarray, np.ndarray]:
    magnitudes = np.zeros(14, np.float32)
    angles = np.zeros(14, np.float32)
    for slot, value in magnitudes_nm.items():
        magnitudes[slot] = value
    for slot, value in angles_rad.items():
        angles[slot] = value
    return magnitudes, angles


def test_higher_order_preview_matches_float64_chi_reference():
    """Every aberration order changes chi by about a radian at the 30 mrad aperture edge."""
    ssb = _session_256()
    try:
        magnitudes, angles = _packed(
            {0: -8.0, 1: 4.0, 2: 40.0, 3: 30.0, 4: 1500.0, 5: 900.0, 6: 700.0, 7: 3.0e4, 8: 2.0e4, 9: 2.5e4,
             10: 9.0e5, 11: 6.0e5, 12: 5.0e5, 13: 4.0e5},
            {1: -0.7, 2: 0.4, 3: 1.1, 5: -0.3, 6: 0.9, 7: -1.2, 8: 0.2, 9: 0.6, 11: -0.5, 12: 1.4, 13: -0.1},
        )
        aberrations = {"C10": -8.0, "C12": 4.0, "phi12": -0.7}
        phase, loss = ssb.preview(aberrations, higher_order_magnitudes=magnitudes, higher_order_angles=angles)
        dragged, no_loss = ssb.preview(aberrations, compute_loss=False, higher_order_magnitudes=magnitudes,
                                       higher_order_angles=angles)
        reference = KrivanekSSB.from_ssb(ssb)
        reference.magnitudes_A = magnitudes.astype(np.float64) * 10.0    # the engine takes Angstrom
        reference.angles_rad = angles.astype(np.float64)
        expected, expected_loss = reference.reconstruct(0.0, 0.0, 0.0)
        # measured 2026-10-07: 2.0e-6 rad max (float32 Chebyshev chi against float64 atan2 chi), loss 2.4e-7 relative
        np.testing.assert_allclose(phase, expected.cpu().numpy(), atol=5e-6)
        assert loss == pytest.approx(expected_loss, rel=1e-6)
        np.testing.assert_array_equal(dragged, phase)
        assert no_loss is None
        two_term, _ = ssb.preview(aberrations, phase_estimator="mean_phase")
        # the higher orders change the image on the scale of the image itself, far above the parity tolerance
        assert (phase - two_term).std() > 0.5 * two_term.std()
    finally:
        ssb.close()


def test_two_term_coefficients_through_the_full_kernel_equal_the_standard_preview():
    """With C21..C56 zero, the 14-term chi is the C10/C12 polynomial; an angle left on a zero magnitude has no effect."""
    ssb = _session_256()
    try:
        aberrations = {"C10": -8.0, "C12": 4.0, "phi12": -0.7}
        magnitudes, angles = _packed({0: -8.0, 1: 4.0}, {1: -0.7, 2: 1.2})
        standard, standard_loss = ssb.preview(aberrations, phase_estimator="mean_phase")
        full, full_loss = ssb.preview(aberrations, phase_estimator="mean_phase", higher_order_magnitudes=magnitudes,
                                      higher_order_angles=angles)
        # measured 2026-10-07: 1.7e-8 rad max, identical loss
        np.testing.assert_allclose(full, standard, atol=1e-7)
        assert full_loss == pytest.approx(standard_loss, rel=1e-7)
    finally:
        ssb.close()
