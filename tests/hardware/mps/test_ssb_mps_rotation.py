"""MPS SSB after a rotation change: the same answer as a session prepared at that rotation."""

import math
import platform

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() != "arm64", reason="MPS SSB requires Apple silicon"
)


def test_rotation_change_keeps_every_aperture_pixel_with_anisotropic_sampling():
    """With rows and columns sampled differently, set_rotation gives what a session prepared at the new rotation gives.

    A rotation keeps each bright-field pixel's scattering angle but turns the aperture's soft edge, one detector pixel
    wide along k, and that width changes with direction when the samplings differ. Preparation stored only the pixels
    inside the aperture at the first rotation, so after set_rotation the pixels that had entered it were missing while
    the bright-field count still included them.
    """
    pytest.importorskip("mlx.core")
    from quantem.gpu import SSB

    rng = np.random.default_rng(5)
    rows, cols = np.indices((64, 64)) - 31.5
    counts = rng.poisson(30.0, (128, 128, 64, 64)).astype(np.uint16)
    counts[:, :, np.hypot(rows, cols) < 30] += 300
    # the selection (30 px) reaches past the aperture (25 mrad: 25 px along rows, 17 px along columns)
    settings = dict(backend="mps", voltage_kV=300.0, semiangle_mrad=25.0, scan_sampling_A=(0.3, 0.3), detector_sampling_mrad=(1.0, 1.5),
                    bright_field_radius_px=30.0, bright_field_center_px=(31.5, 31.5))
    aberrations = {"C10_nm": 4.0, "C12_nm": 1.0, "phi12_deg": math.degrees(0.4)}

    with SSB(counts, scan_detector_rotation_deg=0.0, **settings) as turned:
        turned.reconstruct(aberrations=aberrations)
        turned.set_rotation(45.0)
        phase, loss = turned.preview(aberrations)
    with SSB(counts, scan_detector_rotation_deg=45.0, **settings) as prepared:
        expected, expected_loss = prepared.preview(aberrations)

    np.testing.assert_array_equal(phase, expected)
    assert loss == expected_loss


def _astigmatic_crystal(n=64, scan_A=0.25, C10_A=-80.0, C12_A=25.0, phi12_deg=20.0):
    """Noise-free 4D-STEM of a periodic phase object (bright columns) under a 300 kV, 30 mrad probe with astigmatism.

    The scan steps are the object pixels, so each frame is the far field of the object times the probe shifted by whole
    pixels. Returns the frames (a dose of 1e5 electrons) and the detector sampling in mrad.
    """
    energy = 300e3
    wavelength_A = 12.2643 / math.sqrt(energy * (1 + energy / (2 * 511e3)))
    k = np.fft.fftfreq(n, scan_A)
    kr, kc = np.meshgrid(k, k, indexing="ij")
    alpha, azimuth = np.hypot(kr, kc) * wavelength_A, np.arctan2(kc, kr)
    chi = math.pi / wavelength_A * alpha**2 * (C10_A + C12_A * np.cos(2 * (azimuth - math.radians(phi12_deg))))
    probe = np.fft.ifft2((alpha < 30e-3) * np.exp(-1j * chi))
    rows, cols = np.indices((n, n)) * scan_A
    field, a = n * scan_A, 4.0
    potential = np.zeros((n, n))
    for i in range(int(field / a)):
        for j in range(int(field / a)):
            for offset, weight in ((0.0, 1.0), (0.5, 0.6)):
                dr = (rows - (i + offset) * a + field / 2) % field - field / 2
                dc = (cols - (j + offset) * a + field / 2) % field - field / 2
                potential += weight * np.exp(-(dr**2 + dc**2) / (2 * 0.35**2))
    transmission = np.exp(0.4j * potential)
    frames = np.empty((n, n, n, n), np.float32)
    for i in range(n):
        for j in range(n):
            frames[i, j] = np.fft.fftshift(np.abs(np.fft.fft2(np.roll(probe, (i, j), axis=(0, 1)) * transmission)) ** 2)
    frames *= 1e5 / frames[0, 0].sum()
    return frames, wavelength_A / (n * scan_A) * 1e3


def test_rotation_check_keeps_astigmatism_on_the_opposite_branch():
    """Starting 180 degrees off, the polarity check lands where a direct fit at the right rotation does.

    The other branch has the opposite C12 at the same angle, which is the same magnitude at phi12 + 90 degrees. Before
    2026-10-19 the refit started from a negative C12, the fit clamped it to 0, and the check returned C12 = 0.0 nm with
    a loss 9x the direct fit's (simulated truth C10 -8 nm, C12 2.5 nm at 20 degrees).
    """
    pytest.importorskip("mlx.core")
    from quantem.gpu import SSB

    frames, detector_mrad = _astigmatic_crystal()
    settings = dict(backend="mps", voltage_kV=300.0, semiangle_mrad=30.0, scan_sampling_A=(0.25, 0.25),
                    detector_sampling_mrad=(detector_mrad, detector_mrad))
    with SSB(frames, scan_detector_rotation_deg=0.0, **settings) as session:
        direct = session.find_aberrations(check_rotation=False, verbose=False)
    with SSB(frames, scan_detector_rotation_deg=180.0, **settings) as session:
        checked = session.find_aberrations(check_rotation=True, verbose=False)

    assert checked.initial_scan_detector_rotation_deg == 180.0
    assert checked.scan_detector_rotation_deg == 0.0
    assert direct.aberrations["C12_nm"] == pytest.approx(2.5, abs=0.1)
    assert checked.aberrations["C12_nm"] == pytest.approx(direct.aberrations["C12_nm"], abs=0.05)
    assert checked.aberrations["phi12_deg"] == pytest.approx(direct.aberrations["phi12_deg"], abs=1.0)
    assert checked.aberrations["C10_nm"] == pytest.approx(direct.aberrations["C10_nm"], abs=0.05)
    assert checked.loss <= direct.loss * (1.0 + 1e-3)


def test_reconstruct_keeps_the_rotation_check_of_the_fit_it_applies():
    """reconstruct() after a fit that moved the branch reports the check, as the fit does.

    Before 2026-10-20 reconstruct() built a new result with no column sign, no suggested rotation and no checked
    phases: report() had no "Suggested rotation" column and show("rotation") raised.
    """
    pytest.importorskip("mlx.core")
    import matplotlib

    matplotlib.use("Agg")
    from quantem.gpu import SSB

    frames, detector_mrad = _astigmatic_crystal()
    settings = dict(backend="mps", voltage_kV=300.0, semiangle_mrad=30.0, scan_sampling_A=(0.25, 0.25),
                    detector_sampling_mrad=(detector_mrad, detector_mrad), scan_detector_rotation_deg=180.0)
    with SSB(frames, **settings) as session:
        fitted = session.find_aberrations(num_trials=40, verbose=False)
        assert fitted.initial_scan_detector_rotation_deg == 180.0 and fitted.scan_detector_rotation_deg == 0.0
        for result in (session.reconstruct(), session.reconstruct(fitted, upsampling_factor=2)):
            assert result.column_sign == fitted.column_sign
            assert result.initial_scan_detector_rotation_deg == 180.0
            assert result.initial_column_sign == fitted.initial_column_sign
            np.testing.assert_array_equal(result.rotation_check_phases, fitted.rotation_check_phases)
            assert result.report().loc["standard SSB", "Suggested rotation (deg)"] == 180.0
            result.show("rotation")
        changed = session.reconstruct(aberrations={"C10_nm": fitted.aberrations["C10_nm"] + 1.0})
        assert changed.column_sign is None and changed.initial_scan_detector_rotation_deg is None
    with SSB(frames, **settings) as fresh:
        auto = fresh.reconstruct()
        assert auto.initial_scan_detector_rotation_deg == 180.0 and auto.column_sign is not None


def test_joint_fit_takes_a_c10_range_and_holds_c12_at_zero():
    """The tutorial's constrained joint fit runs: a C10 range, C12 held at 0, the others fitted.

    Before 2026-10-20 search_ranges was refused with tilt=True and fixed_aberrations needed all three coefficients
    with check_rotation=False.
    """
    pytest.importorskip("mlx.core")
    from quantem.gpu import SSB

    frames, detector_mrad = _astigmatic_crystal()
    settings = dict(backend="mps", voltage_kV=300.0, semiangle_mrad=30.0, scan_sampling_A=(0.25, 0.25),
                    detector_sampling_mrad=(detector_mrad, detector_mrad), scan_detector_rotation_deg=0.0)
    with SSB(frames, **settings) as session:
        held = session.find_aberrations(tilt=True, num_trials=40, verbose=False,
                                        search_ranges={"C10_nm": (-50, 50)}, fixed_aberrations={"C12_nm": 0.0})
        assert held.aberrations["C12_nm"] == 0.0 and held.aberrations["phi12_deg"] == 0.0
        assert -50.0 <= held.aberrations["C10_nm"] <= 50.0 and held.column_sign is not None
        bounded = session.find_aberrations(tilt=True, num_trials=40, verbose=False,
                                           search_ranges={"C10_nm": (-12, -4), "C12_nm": (0, 5)})
        assert -12.0 <= bounded.aberrations["C10_nm"] <= -4.0 and 0.0 <= bounded.aberrations["C12_nm"] <= 5.0
        assert bounded.aberrations["C10_nm"] == pytest.approx(-8.0, abs=0.1)
        fixed_defocus = session.find_aberrations(tilt=True, num_trials=40, verbose=False,
                                                 fixed_aberrations={"C10_nm": -8.0})
        assert fixed_defocus.aberrations["C10_nm"] == pytest.approx(-8.0, abs=1e-6)


@pytest.mark.parametrize("constraints,match", [
    (dict(search_ranges={"phi12_deg": (0, 30)}), "phi12_deg over its whole period"),
    (dict(fixed_aberrations={"phi12_deg": 10.0}), "phi12_deg over its whole period"),
    (dict(fixed_aberrations={"C12_nm": 2.0}), "C12_nm from 0"),
    (dict(search_ranges={"C12_nm": (1, 5)}), "C12_nm from 0"),
    (dict(search_ranges={"C10_nm": (5, -5)}), "low <= high"),
    (dict(search_ranges={"C10_nm": (0, 5)}, fixed_aberrations={"C10_nm": 1.0}), "both"),
    (dict(fixed_aberrations={"C10_nm": 1.0, "C12_nm": 2.0, "phi12_deg": 3.0}), "check_rotation=False"),
    (dict(refine_lock=["C12_nm"]), "fixed_aberrations"),
])
def test_unsupported_joint_constraints_say_what_to_pass(constraints, match):
    from quantem.gpu import SSB

    session = SSB.__new__(SSB)  # validation runs before any data is touched
    session._device = None
    with pytest.raises(ValueError, match=match):
        SSB.find_aberrations(session, tilt=True, verbose=False, **constraints)
