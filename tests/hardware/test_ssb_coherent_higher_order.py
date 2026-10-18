"""Shared higher-order reconstruction against a small float64 polar oracle.

The oracle uses the retained Fourier evidence but independently evaluates the
documented transfer function, depth average, aliases, and complex summation.
Only a 128-square synthetic scan is copied to the CPU; real acquisitions stay
on their physical accelerator.
"""

import math

import numpy as np
import pytest

from tests.hardware.test_ssb_scan_mean import _session

NAMES = ("C10", "C12", "C21", "C23", "C30", "C32", "C34", "C41",
         "C43", "C45", "C50", "C52", "C54", "C56")
ORDERS = ((1, 0), (1, 2), (2, 1), (2, 3), (3, 0), (3, 2), (3, 4),
          (4, 1), (4, 3), (4, 5), (5, 0), (5, 2), (5, 4), (5, 6))
MAGNITUDES = (-8., 4., 40., 30., 1500., 900., 700., 3e4, 2e4, 2.5e4, 9e5, 6e5, 5e5, 4e5)
ANGLES = (0., -.7, .4, 1.1, 0., -.3, .9, -1.2, .2, .6, 0., -.5, 1.4, -.1)
COEFS = {f"{name}_nm": value for name, value in zip(NAMES, MAGNITUDES)} | {
    f"phi{name[1:]}_deg": math.degrees(angle)
    for name, (_, m), angle in zip(NAMES, ORDERS, ANGLES) if m
}


def reference(ssb, coefs, factor, tilt, depth, estimator="phase_of_mean"):
    """Evaluate polar chi and each detector wave independently in float64."""
    if ssb.backend == "mps":
        p = ssb._backend()._prepared
        g = np.asarray(p.g_qk)
        kr, kc = np.asarray(p.kx), np.asarray(p.ky)
        qr0, qc0 = np.asarray(p.q_row), np.asarray(p.q_col)
        wl, semi, ay, ax = p.wavelength, p.semiangle_rad, p.ang_y_rad, p.ang_x_rad
        dc, num_bf = p.dc_value, p.num_bright_field_pixels
    else:
        engine = ssb._backend()._get_accelerator()
        cache = engine._cache
        g = engine.G_qk.get()
        kr, kc = cache["kx_bf"].get(), cache["ky_bf"].get()
        qr0, qc0 = cache["qx_1d"].get(), cache["qy_1d"].get()
        wl, semi, ay, ax = map(float, engine.thick._kernel_scalars(0., 0., 0., (0., 0.), 0.)[:4])
        dc, num_bf = engine._dc_value_host, ssb.num_bright_field_pixels
    native = ssb.scan_shape[0]
    if g.shape[-1] != native:
        cols = (native - np.arange(native // 2 + 1, native)) % native
        g = np.concatenate((g, g[:, (-np.arange(native)) % native][:, :, cols].conj()), axis=-1)
    n = native * factor
    qr = (np.fft.fftfreq(n) * n * float(qr0[1]))[:, None]
    qc = (np.fft.fftfreq(n) * n * float(qc0[1]))[None, :]
    total = np.zeros((n, n), dtype=np.complex128)
    moments = np.zeros((2, n, n), dtype=np.float64)
    shift = 2 * np.pi * (qr * tilt[0] + qc * tilt[1]) * 1e-3

    def geometry(row, col):
        alpha = np.hypot(row, col) * wl
        phi = np.arctan2(col, row)
        denom = np.hypot(np.cos(phi) * ay, np.sin(phi) * ax)
        aperture = np.clip((semi - alpha) / denom + .5, 0., 1.)
        chi = np.zeros(np.broadcast_shapes(np.shape(row), np.shape(col)))
        for name, (order, m) in zip(NAMES, ORDERS):
            magnitude = float(np.float32(np.float32(coefs.get(f"{name}_nm", 0.)) * np.float32(10.)))
            angle = float(np.float32(np.deg2rad(coefs.get(f"phi{name[1:]}_deg", 0.))))
            chi += magnitude * alpha ** (order + 1) / (order + 1) * np.cos(m * (phi - angle))
        return alpha**2, aperture * np.exp(-2j * np.pi / wl * chi)

    for k, source in enumerate(g):
        ak, pk = geometry(float(kr[k]), float(kc[k]))
        am, pm = geometry(qr - float(kr[k]), qc - float(kc[k]))
        ap, pp = geometry(qr + float(kr[k]), qc + float(kc[k]))
        w1 = np.sinc(.5 * depth * 10 * (-np.pi / wl * (am-ak) - shift) / np.pi)
        w2 = np.sinc(.5 * depth * 10 * (np.pi / wl * (ap-ak) - shift) / np.pi)
        gamma = w1 * pm * pk.conjugate() - w2 * pp.conjugate() * pk
        corrected = np.tile(source.astype(np.complex128), (factor, factor)) * (
            gamma / np.maximum(np.abs(gamma), 1e-8)).conj()
        corrected[::native, ::native] = 0
        corrected[0, 0] = dc
        total += corrected
        if estimator == "mean_phase":
            phase = np.angle(np.fft.ifft2(corrected))
            moments[0] += phase
            moments[1] += phase * phase
    if estimator == "phase_of_mean":
        total[0, 0] += (num_bf - len(g)) * dc
        return np.angle(np.fft.ifft2(total)), None
    mean = moments[0] / num_bf
    return mean, float(np.mean(moments[1] / num_bf - mean**2))


@pytest.mark.parametrize("factor", [1, 2, 3, 4])
@pytest.mark.parametrize("depth", [0., 18.])
def test_all_orders_coherent_tilt_and_upsampling_match_polar_oracle(factor, depth):
    with _session(constant=False) as ssb:
        phase, loss = ssb.preview(COEFS, upsampling_factor=factor,
                                  sample_tilt_mrad=(-9., 5.), depth_spread_nm=depth)
        expected, _ = reference(ssb, COEFS, factor, (-9., 5.), depth)
        np.testing.assert_allclose(phase, expected, rtol=0, atol=5e-6)
        _, native_loss = reference(ssb, COEFS, 1, (-9., 5.), depth, "mean_phase")
        assert loss == pytest.approx(native_loss, rel=1e-5, abs=1e-10)
        explicit, _ = ssb.preview(COEFS, upsampling_factor=factor,
                                 sample_tilt_mrad=(-9., 5.), depth_spread_nm=depth,
                                 phase_estimator="phase_of_mean", compute_loss=False)
        np.testing.assert_array_equal(phase, explicit)


@pytest.mark.parametrize("factor", [1, 3, 4, 8])
def test_higher_order_constant_scan_has_no_alias_pattern(factor):
    with _session() as ssb:
        phase, _ = ssb.preview(COEFS, upsampling_factor=factor, compute_loss=False,
                              sample_tilt_mrad=(-9., 5.), depth_spread_nm=18.)
        np.testing.assert_allclose(phase, 0., rtol=0, atol=1e-7)


def test_each_active_higher_order_changes_coherent_output():
    with _session(constant=False) as ssb:
        full, _ = ssb.preview(COEFS, compute_loss=False)
        for name in NAMES[2:]:
            removed, _ = ssb.preview(COEFS | {f"{name}_nm": 0.}, compute_loss=False)
            assert np.std(full - removed) > 1e-7, name


def test_auto_sampling_preserves_higher_orders_when_saving_and_reloading(tmp_path):
    from quantem.gpu import SSB

    source = tmp_path / "counts.npy"
    with _session(constant=False, source_path=source) as seed:
        backend = seed.backend
    with SSB.open(source, backend=backend, voltage_kV=300, semiangle_mrad=30,
                  scan_sampling_A=(.5, .5), detector_sampling_mrad=(10., 10.)) as ssb:
        ssb.sample_tilt_mrad = (-9., 5.)
        ssb.depth_spread_nm = 18.
        saved = ssb.reconstruct(aberrations=COEFS, upsampling_factor="auto", save_to=tmp_path / "result")
        assert saved.phase_estimator == "phase_of_mean"
        assert saved.upsampling_factor == 4
        assert saved.aberrations == COEFS
        loaded = ssb.reconstruct(aberrations=COEFS, upsampling_factor="auto", save_to=tmp_path / "result")
        assert loaded.reused
        before = saved.phase.get() if backend == "cuda" else saved.phase
        after = loaded.phase.get() if backend == "cuda" else loaded.phase
        np.testing.assert_array_equal(before, after)
