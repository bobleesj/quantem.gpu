"""The SSB transfer function with odd aberrations: the probe of the first sideband is at k - q, not q - k.

For a thin object G(q, k) ~ P(k - q) conj P(k) - P(k) conj P(k + q) (Pennycook et al., Ultramicroscopy 151, 2015).
P(q - k) equals P(k - q) for C10, C12 and every even order but not for coma, trefoil and the other odd orders, so
the kernels must evaluate the first term at k - q. This file checks the physics on a CPU simulation, independent of
any kernel: the simulated G matches the k - q form and not the q - k form, and only the k - q form keeps
G(-q) = conj G(q), which the data obey and the half-plane fit objective relies on.
tests/hardware/test_ssb_higher_order_sign.py checks the reconstruction kernels against the same physics.
"""

import math

import numpy as np

WAVELENGTH_A = 0.019687      # 300 kV
SEMIANGLE = 0.030
# Krivanek (n, m) of the terms used; magnitudes in A, azimuths in rad: coma and trefoil are odd, C30 even
TERMS = {"C10": (1, 0, -50.0, 0.0), "C12": (1, 2, 20.0, 0.5), "C21": (2, 1, 500.0, 1.0),
         "C23": (2, 3, 400.0, -0.35), "C30": (3, 0, 3.0e4, 0.0)}


def _probe(kr, kc):
    alpha = WAVELENGTH_A * np.hypot(kr, kc)
    phi = np.arctan2(kc, kr)
    chi = sum(magnitude * alpha ** (n + 1) / (n + 1) * np.cos(m * (phi - angle))
              for n, m, magnitude, angle in TERMS.values())
    return (alpha < SEMIANGLE) * np.exp(-1j * 2 * np.pi / WAVELENGTH_A * chi)


def _gamma(qr, qc, kr, kc, sign):
    first = _probe(kr - qr, kc - qc) if sign == "k-q" else _probe(qr - kr, qc - kc)
    return first * np.conj(_probe(kr, kc)) - _probe(kr, kc) * np.conj(_probe(kr + qr, kc + qc))


def _simulated_g(phase_scale):
    """G(q, k) of an exact thin-object 4D-STEM simulation: a periodic random phase, the probe p(r - r0)."""
    cell, n, scan = 12.0, 64, 16
    k = np.fft.fftfreq(n, cell / n)
    kr, kc = np.meshgrid(k, k, indexing="ij")
    rng = np.random.default_rng(3)
    spectrum = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) * (np.hypot(kr, kc) < 0.6)
    phase = np.fft.ifft2(spectrum).real
    phase *= phase_scale / phase.std()
    probe = np.fft.ifft2(_probe(kr, kc))
    stride = n // scan
    intensity = np.empty((scan, scan, n, n))
    for i in range(scan):
        for j in range(scan):
            wave = np.roll(probe, (i * stride, j * stride), axis=(0, 1)) * np.exp(1j * phase)
            intensity[i, j] = np.abs(np.fft.fft2(wave)) ** 2
    bright = np.hypot(kr, kc) * WAVELENGTH_A < SEMIANGLE - 1e-3
    g = np.fft.fft2(intensity[:, :, bright], axes=(0, 1))           # FFT over the scan, e^{-2 pi i q r0}
    q = np.fft.fftfreq(scan, cell / scan)
    qr, qc = np.meshgrid(q, q, indexing="ij")
    return g, qr[..., None], qc[..., None], kr[bright][None, None], kc[bright][None, None]


def _residual(g, model):
    """Relative misfit of g by the best complex multiple of model at each q (one object coefficient per q)."""
    scale = (np.conj(model) * g).sum(-1, keepdims=True) / np.maximum((np.abs(model) ** 2).sum(-1, keepdims=True), 1e-30)
    keep = np.ones(g.shape[:2], bool)
    keep[0, 0] = False
    return float(np.linalg.norm((g - scale * model)[keep]) / np.linalg.norm(g[keep]))


def test_weak_object_g_follows_the_probe_at_k_minus_q():
    # the linear (weak-phase) part: G(phi) - G(-phi) cancels every even power of the phase
    plus, qr, qc, kr, kc = _simulated_g(0.005)
    minus = _simulated_g(-0.005)[0]
    linear = 0.5 * (plus - minus)
    assert _residual(linear, _gamma(qr, qc, kr, kc, "k-q")) < 1e-4
    assert _residual(linear, _gamma(qr, qc, kr, kc, "q-k")) > 0.2


def test_only_the_k_minus_q_model_keeps_the_data_symmetry():
    """G(-q, k) = conj G(q, k) for real intensities; Gamma(-q, k) = -conj Gamma(q, k) only with k - q."""
    g, qr, qc, kr, kc = _simulated_g(0.05)
    scan = g.shape[0]
    mirror = (-np.arange(scan)) % scan
    assert np.abs(g[mirror][:, mirror] - np.conj(g)).max() < 1e-9 * np.abs(g).max()
    for sign, expected_small in (("k-q", True), ("q-k", False)):
        gamma = _gamma(qr, qc, kr, kc, sign)
        keep = np.ones((scan, scan), bool)
        keep[0, 0] = keep[scan // 2] = keep[:, scan // 2] = False      # Nyquist is its own mirror index, not -q
        error = np.linalg.norm((gamma[mirror][:, mirror] + np.conj(gamma))[keep]) / np.linalg.norm(gamma[keep])
        assert (error < 1e-12) == expected_small, (sign, error)
        assert math.isfinite(error)
