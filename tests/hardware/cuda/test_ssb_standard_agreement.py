"""The CUDA standard fit (tilt=False): the joint fit's agreement objective and search with depth spread 0, and the
phase-variance objective kept as ``objective="phase_variance"``.

The synthetic crystal (128 x 128 scan, 300 kV, 30 mrad, a periodic phase object, the CUDA twin of
``test_ssb_mps_rotation._astigmatic_crystal``) has known aberrations, so the fits are checked against the truth, not
against a previous fit. The agreement kernel is checked against ``tests/parity/torch_ssb.py``.
"""

import math

import numpy as np
import pytest

cp = pytest.importorskip("cupy")


def _astigmatic_crystal(scan_size=128, scan_sampling_A=0.25, C10_A=-80.0, C12_A=25.0, phi12_deg=20.0):
    """Noise-free 4D-STEM of a periodic phase object (bright columns) under a 300 kV, 30 mrad probe with astigmatism.

    The scan steps are the object pixels, so each frame is the far field of the object times the probe shifted by whole
    pixels. Returns the frames on the GPU (a dose of 1e5 electrons) and the detector sampling in mrad.
    """
    energy_eV = 300e3
    wavelength_A = 12.2643 / math.sqrt(energy_eV * (1 + energy_eV / (2 * 511e3)))
    k_inv_A = cp.fft.fftfreq(scan_size, scan_sampling_A)
    k_row, k_col = cp.meshgrid(k_inv_A, k_inv_A, indexing="ij")
    alpha, azimuth = cp.hypot(k_row, k_col) * wavelength_A, cp.arctan2(k_col, k_row)
    chi = math.pi / wavelength_A * alpha**2 * (C10_A + C12_A * cp.cos(2 * (azimuth - math.radians(phi12_deg))))
    probe = cp.fft.ifft2((alpha < 30e-3) * cp.exp(-1j * chi))
    rows_A, cols_A = cp.indices((scan_size, scan_size)) * scan_sampling_A
    field_A, lattice_A = scan_size * scan_sampling_A, 4.0
    potential = cp.zeros((scan_size, scan_size))
    for cell_row in range(int(field_A / lattice_A)):
        for cell_col in range(int(field_A / lattice_A)):
            for offset, weight in ((0.0, 1.0), (0.5, 0.6)):
                row_distance = (rows_A - (cell_row + offset) * lattice_A + field_A / 2) % field_A - field_A / 2
                col_distance = (cols_A - (cell_col + offset) * lattice_A + field_A / 2) % field_A - field_A / 2
                potential += weight * cp.exp(-(row_distance**2 + col_distance**2) / (2 * 0.35**2))
    transmission = cp.exp(0.4j * potential)
    # every column shift of the probe at once: shifted[shift, r, c] = probe[(r - scan_row) % n, (c - shift) % n]
    column_shift = (cp.arange(scan_size)[None, :] - cp.arange(scan_size)[:, None]) % scan_size
    frames = cp.empty((scan_size,) * 4, cp.float32)
    for scan_row in range(scan_size):
        shifted = cp.roll(probe, scan_row, axis=0)[:, column_shift].transpose(1, 0, 2)
        frames[scan_row] = cp.fft.fftshift(cp.abs(cp.fft.fft2(shifted * transmission)) ** 2, axes=(-2, -1))
    frames *= 1e5 / frames[0, 0].sum()
    return frames, wavelength_A / (scan_size * scan_sampling_A) * 1e3


def _settings(detector_sampling_mrad):
    """SSB settings of the synthetic crystal: its scan step and detector sampling, rotation 0."""
    return dict(backend="cuda", voltage_kV=300.0, semiangle_mrad=30.0, scan_sampling_A=(0.25, 0.25),
                detector_sampling_mrad=(detector_sampling_mrad, detector_sampling_mrad), scan_detector_rotation_deg=0.0)


def _astigmatism_error_nm(fit, C12_nm, phi12_deg):
    """|C12 exp(2 i phi12) - truth|: the astigmatism as a vector, defined through C12 = 0."""
    fitted = fit["C12_nm"] * np.exp(2j * math.radians(fit["phi12_deg"]))
    return float(abs(fitted - C12_nm * np.exp(2j * math.radians(phi12_deg))))


@pytest.mark.parametrize("C10_nm,C12_nm", [(-8.0, 2.5), (5.0, 0.0), (-20.0, 5.0)])
@pytest.mark.parametrize("refinement", ["auto", "nelder-mead"])
def test_standard_fit_recovers_known_aberrations(C10_nm, C12_nm, refinement):
    """The agreement fit lands on the true aberrations with Newton ("auto") and with Nelder-Mead."""
    from quantem.gpu import SSB

    frames, detector_sampling_mrad = _astigmatic_crystal(C10_A=10 * C10_nm, C12_A=10 * C12_nm, phi12_deg=20.0)
    with SSB(frames, **_settings(detector_sampling_mrad)) as session:
        fit = session.find_aberrations(search="standard", tilt=False, refinement=refinement, check_rotation=False, verbose=False)
        _, loss = session.preview(fit.aberrations)
    assert fit.refinement == ("newton" if refinement == "auto" else refinement)
    assert fit.aberrations["C10_nm"] == pytest.approx(C10_nm, abs=0.1)
    assert _astigmatism_error_nm(fit.aberrations, C12_nm, 20.0) < 0.05
    assert fit.sample_tilt_mrad is None and fit.depth_spread_nm is None
    # the agreement is the objective; the phase variance at the fit is the reported loss
    assert fit.sample_agreement > 0 and fit.loss == pytest.approx(loss, rel=1e-6)
    assert {record["objective"] for record in fit.trial_records} == {"negative_thick_agreement"}
    assert len(fit.trial_records) == 200


def test_phase_variance_objective_is_the_older_standard_fit():
    """The phase-variance standard fit minimizes the phase variance with TPE then Nelder-Mead, and needs tilt=False."""
    from quantem.gpu import SSB

    frames, detector_sampling_mrad = _astigmatic_crystal()  # truth C10 -8 nm, C12 2.5 nm, phi12 20 degrees
    with SSB(frames, **_settings(detector_sampling_mrad)) as session:
        fit = session.find_aberrations(search="standard", tilt=False, objective="phase_variance", check_rotation=False, verbose=False)
        _, loss = session.preview(fit.aberrations)
        assert fit.refinement == "nelder-mead" and fit.sample_agreement is None
        assert fit.aberrations["C10_nm"] == pytest.approx(-8.0, abs=0.15)
        assert _astigmatism_error_nm(fit.aberrations, 2.5, 20.0) < 0.1
        assert fit.loss == pytest.approx(loss, rel=1e-6)
        assert len(fit.trials) == 200 and set(fit.trials.objective) == {"phase_variance"}
        with pytest.raises(ValueError, match="tilt=False"):
            session.find_aberrations(search="standard", objective="phase_variance", verbose=False)
        with pytest.raises(ValueError, match="newton"):
            session.find_aberrations(search="standard", tilt=False, objective="phase_variance", refinement="newton",
                                     check_rotation=False, verbose=False)


def test_fixed_aberrations_hold_a_coefficient_in_the_standard_fit():
    """fixed_aberrations holds C12 at 0 in every trial and through the refinement; C10 is still fitted."""
    from quantem.gpu import SSB

    frames, detector_sampling_mrad = _astigmatic_crystal(C12_A=0.0)  # truth C10 -8 nm, no astigmatism
    with SSB(frames, **_settings(detector_sampling_mrad)) as session:
        fit = session.find_aberrations(search="standard", tilt=False, fixed_aberrations={"C12_nm": 0.0}, num_trials=60,
                                       check_rotation=False, verbose=False)
        assert fit.refinement == "nelder-mead"  # a locked coefficient: Newton cannot hold it
        assert fit.aberrations["C10_nm"] == pytest.approx(-8.0, abs=0.1)
        assert fit.aberrations["C12_nm"] == 0.0 and fit.aberrations["phi12_deg"] == 0.0
        held = session.find_aberrations(search="standard", tilt=False, fixed_aberrations={"C10_nm": -7.0}, num_trials=40,
                                        check_rotation=False, verbose=False)
        assert held.aberrations["C10_nm"] == -7.0
        variance = session.find_aberrations(search="standard", tilt=False, objective="phase_variance", fixed_aberrations={"C12_nm": 0.0},
                                            num_trials=40, check_rotation=False, verbose=False)
        assert variance.aberrations["C12_nm"] == 0.0
        assert variance.aberrations["C10_nm"] == pytest.approx(-8.0, abs=0.15)
        with pytest.raises(ValueError, match="nothing to fit"):
            session.find_aberrations(search="standard", tilt=False, fixed_aberrations={"C10_nm": -7.0, "C12_nm": 0.0, "phi12_deg": 0.0},
                                     check_rotation=False, verbose=False)


def test_phi12_range_bounds_the_standard_fit():
    """A phi12 range bounds the search and the refinement (Nelder-Mead, as Newton cannot bound phi12)."""
    from quantem.gpu import SSB

    frames, detector_sampling_mrad = _astigmatic_crystal()  # phi12 20 degrees
    with SSB(frames, **_settings(detector_sampling_mrad)) as session:
        inside = session.find_aberrations(search="standard", tilt=False, search_ranges={"phi12_deg": (0.0, 45.0)}, num_trials=60,
                                          check_rotation=False, verbose=False)
        outside = session.find_aberrations(search="standard", tilt=False, search_ranges={"phi12_deg": (40.0, 80.0)}, num_trials=60,
                                           check_rotation=False, verbose=False)
    assert inside.refinement == "nelder-mead" and inside.aberrations["phi12_deg"] == pytest.approx(20.0, abs=0.5)
    assert 40.0 <= outside.aberrations["phi12_deg"] <= 80.0


def test_depth_zero_objective_matches_torch_reference():
    """The fused agreement kernel at depth spread 0 equals tests/parity/torch_ssb.py fit_power, and ignores the tilt
    exactly: at depth 0 both depth weights are 1, so the model is ordinary SSB."""
    from quantem.gpu import SSB

    frames, detector_sampling_mrad = _astigmatic_crystal()
    with SSB(frames, **_settings(detector_sampling_mrad)) as session:
        session.preview({"C10_nm": 0.0, "C12_nm": 0.0, "phi12_deg": 0.0}, compute_loss=False)
        pytest.importorskip("torch")
        from tests.parity.torch_ssb import TorchSSB

        reference = TorchSSB.from_ssb(session)
        point = np.array([-80.0, 25.0, math.radians(20.0), 0.0, 0.0, 0.0])
        rows = np.array([point, point + [20, 0, 0, 0, 0, 0], point + [0, -10, 0.3, 0, 0, 0],
                         point + [-150, 40, -0.6, 0, 0, 0], point + [0, 0, 0, 3.0, -4.0, 0]])
        fused = session._backend().sample_agreement(rows, (0.2, 0.9))
        expected = np.array([reference.fit_power(row[0], row[1], row[2], (row[3], row[4]), row[5]) for row in rows])
        bound = 2.0 * 2.0 ** -24 * math.sqrt(session.num_bright_field_pixels)
    assert np.max(np.abs(fused - expected) / np.abs(expected)) <= bound
    assert fused[4] == fused[0]
