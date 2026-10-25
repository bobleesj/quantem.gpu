"""Regularized direct reconstructions (``method="phase"`` and ``"complex"``) on MPS, on a synthetic scene.

The scene is the native tests' synthetic acquisition: 128 x 128 scan, 12 x 12 detector, 39 bright-field pixels,
300 kV, 30 mrad, 0.4 Angstrom step, 7.5 mrad detector sampling, rotation 17.25 degrees. ``_reference`` is an
independent float64 implementation of the model (polar aberration function, complex128 transforms and solves), so
these tests need no CUDA reference and no measured data.
"""

import math

import numpy as np
import pytest

from .test_ssb_thick_sample_mps import _require_mps

SCAN, DETECTOR, CENTER, RADIUS = 128, 12, (5.4, 5.7), 3.6
STEP_A, SAMPLING_MRAD, SEMIANGLE_MRAD, ROTATION_DEG = 0.4, 7.5, 30.0, 17.25
ORDERS = ((1, 0), (1, 2), (2, 1), (2, 3), (3, 0), (3, 2), (3, 4), (4, 1), (4, 3), (4, 5), (5, 0), (5, 2), (5, 4), (5, 6))
NAMES = ("C10", "C12", "C21", "C23", "C30", "C32", "C34", "C41", "C43", "C45", "C50", "C52", "C54", "C56")
POINTS = {
    "primary": ({"C10_nm": 6.1, "C12_nm": 1.4, "phi12_deg": math.degrees(0.7)}, (0.0, 0.0), 0.0),
    "fifth": ({"C10_nm": -3.8, "C12_nm": 0.9, "phi12_deg": math.degrees(-0.4), "C21_nm": 50.0,
               "phi21_deg": math.degrees(0.3), "C23_nm": 30.0, "phi23_deg": math.degrees(-1.1), "C30_nm": 2.0e3,
               "C34_nm": 4.0e2, "phi34_deg": math.degrees(0.5), "C56_nm": 1.0e6, "phi56_deg": math.degrees(0.2)},
              (2.5, -4.0), 14.0),
}
PRESETS = {
    "default": dict(phase_regularization=0.03, amplitude_regularization=0.03, smoothness_inv_A=1.0,
                    detector_weighting="uniform"),
    "flat": dict(phase_regularization=0.1, amplitude_regularization=0.01, smoothness_inv_A=0.0,
                 detector_weighting="uniform"),
    "inverse_mean": dict(phase_regularization=0.03, amplitude_regularization=0.03, smoothness_inv_A=1.0,
                         detector_weighting="inverse_mean"),
}


def _scene(scale=1, shift=(0, 0)):
    """uint16 counts 400 + 40 cos(x + 0.3 dr) cos(y - 0.2 dc) + 25 sin(x + y + 0.1 dr dc) + a 13-periodic pattern."""
    r, c = np.indices((SCAN, SCAN))
    x, y = 2 * np.pi * r / 6.4, 2 * np.pi * c / 5.12
    data = np.zeros((SCAN, SCAN, DETECTOR, DETECTOR), np.uint16)
    for pixel in range(DETECTOR * DETECTOR):
        dr, dc = pixel // DETECTOR - CENTER[0], pixel % DETECTOR - CENTER[1]
        value = (400 + 40 * np.cos(x + 0.3 * dr) * np.cos(y - 0.2 * dc) + 25 * np.sin(x + y + 0.1 * dr * dc)
                 + ((r * 31 + c * 17 + pixel * 7) % 13))
        data[:, :, pixel // DETECTOR, pixel % DETECTOR] = np.floor(value + 0.5).astype(np.uint16) * scale
    return np.roll(data, shift, axis=(0, 1))


def _session(data, point, radius=RADIUS):
    from quantem.gpu import DirectPtycho

    ptycho = DirectPtycho(data, backend="mps", voltage_kV=300, semiangle_mrad=SEMIANGLE_MRAD,
                          scan_sampling_A=STEP_A, detector_sampling_mrad=SAMPLING_MRAD,
                          bright_field_center_px=CENTER, bright_field_radius_px=radius,
                          scan_detector_rotation_deg=ROTATION_DEG)
    aberrations, tilt, depth = POINTS[point]
    ptycho.aberrations = aberrations
    ptycho.sample_tilt_mrad, ptycho.depth_spread_nm = tilt, depth
    return ptycho


def _run(ptycho, method, factor, preset="default", **changes):
    options = {**PRESETS[preset], **changes}
    if method == "phase":
        options.pop("amplitude_regularization")
    return ptycho.run(None, method=method, upsampling_factor=factor, **options)


@pytest.fixture(scope="module")
def sessions():
    _require_mps()
    data = _scene()
    made = {point: _session(data, point) for point in POINTS}
    yield made
    for ptycho in made.values():
        ptycho.close()


# --- independent float64 reference -----------------------------------------------------------------------------


def _chi(pr, pc, wavelength, aberrations):
    """(2 pi / wavelength) sum C_nm / (n + 1) |wavelength p|^(n + 1) cos(m (phi_p - phi_nm)), in float64."""
    x, y = wavelength * pr, wavelength * pc
    rho, azimuth = np.hypot(x, y), np.arctan2(y, x)
    chi = np.zeros(np.broadcast(pr, pc).shape)
    for (n, m), name in zip(ORDERS, NAMES):
        magnitude = aberrations.get(f"{name}_nm", 0.0) * 10
        angle = math.radians(aberrations.get(f"phi{name[1:]}_deg", 0.0)) if m else 0.0
        chi += magnitude / (n + 1) * rho ** (n + 1) * np.cos(m * (azimuth - angle))
    return 2 * np.pi / wavelength * chi


def _reference_prepare(data, point, factor, weighting, radius=RADIUS, aberrations=None):
    """Gram, right side and layout of every solved frequency (MATH: dose, weights, damping scale, aliases).

    ``aberrations`` replaces the point's coefficients, its tilt and depth kept.
    """
    from quantem.gpu.optics.physics import wavelength_A_from_kV

    point_aberrations, tilt, depth = POINTS[point]
    aberrations = point_aberrations if aberrations is None else aberrations
    wavelength = float(wavelength_A_from_kV(300.0))
    alpha, dtheta = SEMIANGLE_MRAD * 1e-3, SAMPLING_MRAD * 1e-3
    rows, cols = np.indices((DETECTOR, DETECTOR))
    selected = (rows - CENTER[0]) ** 2 + (cols - CENTER[1]) ** 2 <= radius**2
    counts = np.moveaxis(data[:, :, selected].astype(np.float64), -1, 0)
    spectra = np.fft.fft2(counts)
    angle = -math.radians(ROTATION_DEG)
    k0r, k0c = (rows[selected] - CENTER[0]) * dtheta / wavelength, (cols[selected] - CENTER[1]) * dtheta / wavelength
    kr = k0r * math.cos(angle) + k0c * math.sin(angle)
    kc = -k0r * math.sin(angle) + k0c * math.cos(angle)

    def aperture(pr, pc):
        return np.clip((alpha - wavelength * np.hypot(pr, pc)) / dtheta + 0.5, 0, 1)

    aperture_k = aperture(kr, kc)
    dose = spectra[:, 0, 0].real.sum() / (SCAN * SCAN * np.sum(aperture_k**2))
    weights = np.ones(kr.size)
    if weighting == "inverse_mean":
        mean = np.maximum(spectra[:, 0, 0].real / SCAN**2, 0)
        weights = 1 / np.maximum(mean, 0.05 * np.median(mean))
        weights /= weights.mean()
    scale = np.sum(weights * aperture_k**4)
    chi_k = _chi(kr, kc, wavelength, aberrations)
    u, v = np.indices((SCAN, SCAN // 2 + 1))
    keep = ((v > 0) | ((u > 0) & (u <= (SCAN - 1) // 2))) & (u != SCAN // 2) & (v != SCAN // 2)
    u, v = u[keep], v[keep]
    size = factor * SCAN
    a, b = (index.ravel() for index in np.indices((factor, factor)))
    out_r, out_c = u[:, None] + a * SCAN, v[:, None] + b * SCAN
    step = 1 / (SCAN * STEP_A)
    qr = np.where(out_r <= (size - 1) // 2, out_r, out_r - size) * step
    qc = np.where(out_c <= (size - 1) // 2, out_c, out_c - size) * step
    active = qr**2 + qc**2 < ((2 * alpha + dtheta) / wavelength) ** 2
    t, theta = depth * 10, np.array(tilt) * 1e-3

    def side(sign, qr, qc):
        pr, pc = kr + sign * qr[..., None], kc + sign * qc[..., None]
        shift = 2 * np.pi * (qr * theta[0] + qc * theta[1])[..., None]
        x = t / 2 * (sign * np.pi / wavelength * wavelength**2 * (pr**2 + pc**2 - kr**2 - kc**2) - shift)
        weight = np.where(np.abs(x) < 1e-6, 1.0, np.sin(x) / np.where(x == 0, 1, x))
        return aperture(pr, pc) * aperture_k * weight * np.exp(sign * 1j * (_chi(pr, pc, wavelength, aberrations) - chi_k))

    groups = []
    counts_per = active.sum(axis=1)
    for count in range(1, factor * factor + 1):
        members = np.flatnonzero(counts_per == count)
        if not members.size:
            continue
        aliases = np.nonzero(active[members])[1].reshape(members.size, count)
        q_r = np.take_along_axis(qr[members], aliases, axis=1)
        q_c = np.take_along_axis(qc[members], aliases, axis=1)
        minus, plus = side(-1, q_r, q_c), side(1, q_r, q_c)
        transfer = np.concatenate((minus + plus, 1j * (minus - plus)), axis=1)  # (members, [eta | phi], BF)
        measured = spectra[:, u[members], v[members]].T / dose
        gram = np.einsum("fak,k,fbk->fab", transfer.conj(), weights, transfer)
        rhs = np.einsum("fak,k,fk->fa", transfer.conj(), weights, measured)
        groups.append((count, q_r**2 + q_c**2, gram, rhs,
                       np.take_along_axis(out_r[members], aliases, axis=1),
                       np.take_along_axis(out_c[members], aliases, axis=1)))
    return {"factor": factor, "dose": dose, "scale": scale, "groups": groups, "shape": (size, size)}


def _reference_solve(prepared, method, preset):
    options = PRESETS[preset]
    factor, size = prepared["factor"], prepared["shape"][0]
    spectra = {"eta": np.zeros((size, size), complex), "phi": np.zeros((size, size), complex)}
    for count, q2, gram, rhs, out_r, out_c in prepared["groups"]:
        q0 = options["smoothness_inv_A"]
        prior = (1 + q2 / q0**2) ** 2 if q0 else np.ones_like(q2)
        if method == "complex":
            damping = np.concatenate((np.full(count, options["amplitude_regularization"]),
                                      np.full(count, options["phase_regularization"])))
            penalty = np.concatenate((prior, prior), axis=1) * damping * prepared["scale"]
        else:
            gram, rhs = gram[:, count:, count:], rhs[:, count:]
            penalty = prior * options["phase_regularization"] * prepared["scale"]
        system = gram + penalty[:, :, None] * np.eye(gram.shape[1])
        x = np.linalg.solve(system, rhs[..., None])[..., 0] * factor**2
        channels = [("phi", x[:, -count:])] + ([("eta", x[:, :count])] if method == "complex" else [])
        for name, values in channels:
            spectra[name][out_r, out_c] = values
            spectra[name][(-out_r) % size, (-out_c) % size] = values.conj()
    wave = np.exp(np.fft.ifft2(spectra["eta"]).real + 1j * np.fft.ifft2(spectra["phi"]).real)
    return np.angle(wave), np.abs(wave)


@pytest.fixture(scope="module")
def references():
    _require_mps()
    data, cache = _scene(), {}

    def reference(point, factor, method, preset):
        key = point, factor, PRESETS[preset]["detector_weighting"]
        if key not in cache:
            cache[key] = _reference_prepare(data, point, factor, key[2])
        return _reference_solve(cache[key], method, preset)

    return reference


# --- tests ------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("point", sorted(POINTS))
@pytest.mark.parametrize("factor", [1, 2, 3])
def test_direct_matches_double_precision_definition(sessions, references, point, factor):
    """Phase and complex images agree with the float64 model for every preset (2x includes the output Nyquist)."""
    for preset in PRESETS:
        for method in ("phase", "complex"):
            result = _run(sessions[point], method, factor, preset)
            phase, amplitude = references(point, factor, method, preset)
            assert result.phase.shape == phase.shape == (factor * SCAN,) * 2
            error = np.asarray(result.phase, np.float64) - phase
            label = f"{point} {factor}x {method} {preset}"
            assert np.sqrt(np.mean(error**2)) <= 1e-5 * np.sqrt(np.mean(phase**2)), label
            assert np.abs(error).max() <= 1e-4 * np.abs(phase).max(), label
            if method == "complex":
                assert result.amplitude_estimated
                # relative to the log-amplitude's size, plus one float32 step at 1 (the stored amplitude's rounding)
                deviation = np.asarray(result.amplitude, np.float64) - amplitude
                bound = 1e-5 * np.sqrt(np.mean((amplitude - 1) ** 2)) + np.finfo(np.float32).eps
                assert np.sqrt(np.mean(deviation**2)) <= bound, label
            else:
                assert not result.amplitude_estimated
                # |cos(phi) + i sin(phi)| in float32: 1 to a few rounding steps
                np.testing.assert_allclose(result.amplitude, 1, rtol=0, atol=4 * np.finfo(np.float32).eps)


def test_strength_change_reuses_preparation_bitwise(sessions):
    """A damping change solves the prepared system again; it equals a fresh preparation bit for bit."""
    ptycho = sessions["fifth"]
    first = _run(ptycho, "complex", 3)
    prepared = ptycho._prepared_direct[1]["complex"]
    changed = _run(ptycho, "complex", 3, amplitude_regularization=0.1)
    assert ptycho._prepared_direct[1]["complex"] is prepared
    assert not np.array_equal(changed.object_wave, first.object_wave)
    again = _run(ptycho, "complex", 3)
    np.testing.assert_array_equal(again.object_wave, first.object_wave)
    rebuilt = ptycho.run(None, method="complex", upsampling_factor=3, amplitude_regularization=0.1, force=True)
    assert ptycho._prepared_direct[1]["complex"] is not prepared
    np.testing.assert_array_equal(rebuilt.object_wave, changed.object_wave)
    assert first.reconstruction_options["amplitude_regularization"] == 0.03
    assert first.reconstruction_diagnostics["training_relative_sse"] >= 0


def test_preview_returns_the_run_phase(sessions):
    """The interactive preview of a direct method is the committed result's phase, as a NumPy array."""
    ptycho = sessions["primary"]
    aberrations, tilt, depth = POINTS["primary"]
    result = _run(ptycho, "phase", 2, phase_regularization=0.1)
    preview, loss = ptycho.preview(aberrations, method="phase", phase_regularization=0.1, upsampling_factor=2,
                                   sample_tilt_mrad=tilt, depth_spread_nm=depth)
    assert loss is None and isinstance(preview, np.ndarray) and preview.dtype == np.float32
    np.testing.assert_array_equal(preview, result.phase)


def test_complex_with_suppressed_amplitude_is_phase(sessions):
    """With the log-amplitude block damped away, complex reduces to the phase method ([eta | phi] layout)."""
    ptycho = sessions["fifth"]
    for factor in (1, 3):
        phase = _run(ptycho, "phase", factor).phase
        suppressed = _run(ptycho, "complex", factor, amplitude_regularization=1e8).phase
        assert np.sqrt(np.mean((suppressed - phase) ** 2)) <= 1e-5 * np.sqrt(np.mean(phase**2))


def test_prior_and_damping(sessions):
    """A flat prior equals a very wide one; more damping shrinks the image monotonically towards zero."""
    ptycho = sessions["fifth"]
    flat = _run(ptycho, "phase", 3, smoothness_inv_A=0.0).phase
    wide = _run(ptycho, "phase", 3, smoothness_inv_A=1e9).phase
    assert np.sqrt(np.mean((flat - wide) ** 2)) <= 1e-6 * np.sqrt(np.mean(flat**2))
    rms = [float(np.sqrt(np.mean(_run(ptycho, "phase", 3, phase_regularization=strength).phase ** 2)))
           for strength in (0.003, 0.03, 0.3, 3.0, 1e6)]
    assert all(later < earlier for earlier, later in zip(rms, rms[1:]))
    assert rms[-1] < 1e-5 * rms[1]


def test_spectral_support(sessions):
    """Nothing outside the solved set: DC replicas, the native Nyquist families and beyond the cutoff stay empty."""
    from quantem.gpu.optics.physics import wavelength_A_from_kV

    ptycho = sessions["fifth"]
    cutoff = (2 * SEMIANGLE_MRAD + SAMPLING_MRAD) * 1e-3 / float(wavelength_A_from_kV(300.0))
    for factor in (1, 3):
        result = _run(ptycho, "complex", factor)
        size = factor * SCAN
        index = np.fft.fftfreq(size, 1 / size)
        q = np.hypot(*np.meshgrid(index, index, indexing="ij")) / (SCAN * STEP_A)
        rows, cols = np.indices((size, size))
        replica = (rows % SCAN == 0) & (cols % SCAN == 0)
        nyquist = (rows % SCAN == SCAN // 2) | (cols % SCAN == SCAN // 2)
        empty = replica | nyquist | (q > cutoff * (1 + 1e-6))
        for image in (np.angle(result.object_wave), np.log(np.abs(result.object_wave))):
            spectrum = np.abs(np.fft.fft2(image.astype(np.float64)))
            assert spectrum[empty].max() <= 1e-6 * spectrum.max()
            assert abs(image.mean()) <= 1e-6 * np.abs(image).max()


@pytest.mark.slow
def test_counts_scale_and_scan_shift(sessions):
    """Doubling every count leaves the images; a cyclic scan shift rolls them by the shift times the factor."""
    for factor in (1, 3):
        base = _run(sessions["fifth"], "complex", factor)
        with _session(_scene(scale=2), "fifth") as doubled:
            twice = _run(doubled, "complex", factor)
        np.testing.assert_allclose(twice.phase, base.phase, rtol=0, atol=1e-6 * np.abs(base.phase).max())
        with _session(_scene(shift=(5, -3)), "fifth") as shifted:
            moved = _run(shifted, "complex", factor)
        expected = np.roll(base.phase, (5 * factor, -3 * factor), axis=(0, 1))
        assert np.sqrt(np.mean((moved.phase - expected) ** 2)) <= 1e-5 * np.sqrt(np.mean(expected**2))


@pytest.mark.parametrize("weighting", ["uniform", "inverse_mean"])
def test_pixels_outside_the_aperture_count_in_dose_and_weights(references, weighting):
    """BF pixels beyond the soft aperture (zero transfer, not stored in G) still enter the dose and the weights."""
    preset = "default" if weighting == "uniform" else "inverse_mean"
    # 86 BF pixels within 5.3 px (none near that circle); the soft aperture closes beyond 4.5 px
    with _session(_scene(), "primary", radius=5.3) as ptycho:
        backend = ptycho._backend()
        backend.cache_rotation(math.radians(ROTATION_DEG))
        assert ptycho.num_bright_field_pixels == 86
        assert backend._prepared.bf_storage_indices_np.size < 86
        prepared = _reference_prepare(_scene(), "primary", 1, weighting, radius=5.3)
        for method in ("phase", "complex"):
            result = _run(ptycho, method, 1, preset)
            layout = backend._prepared.direct_layout[1]
            # without the 24 closed-aperture pixels the dose would be about 28 % low
            assert layout.dose == pytest.approx(prepared["dose"], rel=1e-6)
            assert layout.scale == pytest.approx(prepared["scale"], rel=1e-5)
            phase, _ = _reference_solve(prepared, method, preset)
            # this weak 1x phase (rms 2e-4 rad) sits near the float32 transform's noise floor of G
            assert np.sqrt(np.mean((result.phase - phase) ** 2)) <= 3e-5 * np.sqrt(np.mean(phase**2))
