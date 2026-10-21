"""The MPS standard fit (tilt=False): the joint fit's agreement objective and search with depth spread 0.

The synthetic crystal (``test_ssb_mps_rotation._astigmatic_crystal``: 64 x 64 scan, 300 kV, 30 mrad, a periodic phase
object) has known aberrations, so the fit is checked against the truth, not against a previous fit. On 15 such crystals
(C10 -20 to +15 nm, C12 0 to 5 nm, noise-free and Poisson at 1e4 and 1e5 electrons per pattern) this objective landed
0.024 nm from the true C10 on average (0.046 nm at worst) and the phase-variance search it replaced 0.078 nm (0.127 nm);
docs/maintainer/2026-10-20-ssb-standard-fit-objective.md.
"""

import math
import platform

import numpy as np
import pytest

from tests.hardware.mps.test_ssb_mps_rotation import _astigmatic_crystal

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() != "arm64", reason="MPS SSB requires Apple silicon"
)


def _settings(detector_mrad):
    return dict(backend="mps", voltage_kV=300.0, semiangle_mrad=30.0, scan_sampling_A=(0.25, 0.25),
                detector_sampling_mrad=(detector_mrad, detector_mrad), scan_detector_rotation_deg=0.0)


def _astigmatism_error_nm(fit, C12_nm, phi12_deg):
    """|C12 exp(2 i phi12) - truth|: the astigmatism as a vector, defined through C12 = 0."""
    fitted = fit["C12_nm"] * np.exp(2j * math.radians(fit["phi12_deg"]))
    return float(abs(fitted - C12_nm * np.exp(2j * math.radians(phi12_deg))))


@pytest.mark.parametrize("C10_nm,C12_nm", [(-8.0, 2.5), (5.0, 0.0), (-20.0, 5.0)])
@pytest.mark.parametrize("refinement", ["auto", "nelder-mead"])
def test_standard_fit_recovers_known_aberrations(C10_nm, C12_nm, refinement):
    pytest.importorskip("mlx.core")
    from quantem.gpu import SSB

    frames, detector_mrad = _astigmatic_crystal(C10_A=10 * C10_nm, C12_A=10 * C12_nm, phi12_deg=20.0)
    with SSB(frames, **_settings(detector_mrad)) as session:
        fit = session.find_aberrations(tilt=False, refinement=refinement, check_rotation=False, verbose=False)
        _, loss = session.preview(fit.aberrations)
    assert fit.refinement == ("newton" if refinement == "auto" else refinement)
    assert fit.aberrations["C10_nm"] == pytest.approx(C10_nm, abs=0.1)
    assert _astigmatism_error_nm(fit.aberrations, C12_nm, 20.0) < 0.05
    assert fit.sample_tilt_mrad is None and fit.depth_spread_nm is None
    # the agreement is the objective; the phase variance at the fit is the reported loss
    assert fit.sample_agreement > 0 and fit.loss == pytest.approx(loss, rel=1e-6)
    assert {record["objective"] for record in fit.trial_records} == {"negative_thick_agreement"}
    assert len(fit.trial_records) == 200


def test_fixed_aberrations_hold_a_coefficient_in_the_standard_fit():
    """fixed_aberrations holds C12 at 0 in every trial and through the refinement; C10 is still fitted."""
    pytest.importorskip("mlx.core")
    from quantem.gpu import SSB

    frames, detector_mrad = _astigmatic_crystal(C12_A=0.0)  # truth C10 -8 nm, no astigmatism
    with SSB(frames, **_settings(detector_mrad)) as session:
        fit = session.find_aberrations(tilt=False, fixed_aberrations={"C12_nm": 0.0}, num_trials=60,
                                       check_rotation=False, verbose=False)
        assert fit.refinement == "nelder-mead"  # a locked coefficient: Newton cannot hold it
        assert fit.aberrations["C10_nm"] == pytest.approx(-8.0, abs=0.1)
        assert fit.aberrations["C12_nm"] == 0.0 and fit.aberrations["phi12_deg"] == 0.0
        held = session.find_aberrations(tilt=False, fixed_aberrations={"C10_nm": -7.0}, num_trials=40,
                                        check_rotation=False, verbose=False)
        assert held.aberrations["C10_nm"] == -7.0
        with pytest.raises(ValueError, match="nothing to fit"):
            session.find_aberrations(tilt=False, fixed_aberrations={"C10_nm": -7.0, "C12_nm": 0.0, "phi12_deg": 0.0},
                                     check_rotation=False, verbose=False)
        with pytest.raises(ValueError, match="nelder-mead"):
            session.find_aberrations(tilt=False, refinement="newton", refine_lock=["C10_nm"],
                                     check_rotation=False, verbose=False)


def test_phi12_range_bounds_the_standard_fit():
    pytest.importorskip("mlx.core")
    from quantem.gpu import SSB

    frames, detector_mrad = _astigmatic_crystal()  # phi12 20 degrees
    with SSB(frames, **_settings(detector_mrad)) as session:
        inside = session.find_aberrations(tilt=False, search_ranges={"phi12_deg": (0.0, 45.0)}, num_trials=60,
                                          check_rotation=False, verbose=False)
        outside = session.find_aberrations(tilt=False, search_ranges={"phi12_deg": (40.0, 80.0)}, num_trials=60,
                                           check_rotation=False, verbose=False)
    assert inside.refinement == "nelder-mead" and inside.aberrations["phi12_deg"] == pytest.approx(20.0, abs=0.5)
    assert 40.0 <= outside.aberrations["phi12_deg"] <= 80.0


def test_depth_zero_objective_matches_torch_reference():
    """thick_fit_batch at depth spread 0 equals tests/parity/torch_ssb.py fit_power, and ignores the tilt exactly."""
    pytest.importorskip("mlx.core")
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("needs the Apple GPU (MPS)")
    from quantem.gpu import SSB
    from quantem.gpu.ssb.mps.thick_sample import thick_fit_batch
    from tests.parity.torch_ssb import TorchSSB

    frames, detector_mrad = _astigmatic_crystal()
    with SSB(frames, **_settings(detector_mrad)) as session:
        session.preview({"C10_nm": 0.0, "C12_nm": 0.0, "phi12_deg": 0.0}, compute_loss=False)
        prepared = session._backend()._prepared
        reference = TorchSSB.from_mps(prepared, device="mps")
        point = np.array([-80.0, 25.0, math.radians(20.0), 0.0, 0.0, 0.0])
        rows = np.array([point, point + [20, 0, 0, 0, 0, 0], point + [0, -10, 0.3, 0, 0, 0],
                         point + [-150, 40, -0.6, 0, 0, 0], point + [0, 0, 0, 3.0, -4.0, 0]])
        fused = thick_fit_batch(prepared, rows)
        expected = np.array([reference.fit_power(r[0], r[1], r[2], (r[3], r[4]), r[5]) for r in rows])
        bound = 2.0 * 2.0 ** -24 * math.sqrt(prepared.num_bright_field_pixels)
    assert np.max(np.abs(fused - expected) / np.abs(expected)) <= bound
    assert fused[4] == fused[0]
