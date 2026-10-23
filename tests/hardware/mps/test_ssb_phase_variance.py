"""The older standard fit on MPS: ``find_aberrations(tilt=False, objective="phase_variance")``.

Optuna TPE in pairs on the exact phase-variance loss (``mps/loss.py``), then Nelder-Mead
(``optimizer.optimize_phase_variance``). The synthetic crystal (``test_ssb_mps_rotation._astigmatic_crystal``) has
known aberrations; on 15 such crystals this search landed 0.078 nm from the true C10 on average (0.127 nm at worst),
docs/maintainer/2026-10-20-ssb-standard-fit-objective.md. The gold test pins the search to the fit of the search that
was removed and restored on 2026-10-20 (``find_aberrations(tilt=False)`` at commit 4d1ea007).
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


def test_phase_variance_fit_recovers_known_aberrations():
    pytest.importorskip("mlx.core")
    from quantem.gpu import SSB

    frames, detector_mrad = _astigmatic_crystal(C10_A=-80.0, C12_A=25.0, phi12_deg=20.0)
    with SSB(frames, **_settings(detector_mrad)) as session:
        fit = session.find_aberrations(search="primary", tilt=False, objective="phase_variance", check_rotation=False, verbose=False)
        _, loss = session.preview(fit.aberrations)
    assert fit.objective == "phase_variance" and fit.refinement == "nelder-mead"
    assert fit.aberrations["C10_nm"] == pytest.approx(-8.0, abs=0.15)
    astigmatism = fit.aberrations["C12_nm"] * np.exp(2j * math.radians(fit.aberrations["phi12_deg"]))
    assert abs(astigmatism - 2.5 * np.exp(2j * math.radians(20.0))) < 0.05
    # the phase variance is the objective: the fit's loss is its minimum, the same value a preview reports
    assert fit.sample_agreement is None and fit.loss == pytest.approx(loss, rel=1e-6)
    assert len([record for record in fit.trial_records if record["stage"] == "search"]) == 200


def test_phase_variance_fit_holds_a_fixed_coefficient():
    pytest.importorskip("mlx.core")
    from quantem.gpu import SSB

    frames, detector_mrad = _astigmatic_crystal(C12_A=0.0)  # truth C10 -8 nm, no astigmatism
    with SSB(frames, **_settings(detector_mrad)) as session:
        fit = session.find_aberrations(search="primary", tilt=False, objective="phase_variance", fixed_aberrations={"C12_nm": 0.0},
                                       num_trials=60, check_rotation=False, verbose=False)
    assert fit.aberrations["C10_nm"] == pytest.approx(-8.0, abs=0.15)
    assert fit.aberrations["C12_nm"] == 0.0 and fit.aberrations["phi12_deg"] == 0.0


@pytest.mark.slow
def test_gold_phase_variance_fit_matches_the_removed_search():
    """Gold at the benchmark harness calibration (0.46 mrad) and a given rotation, one branch. The search before its
    removal gave C10 -23.225593050625694 nm, C12 3.40092432545935 nm, phi12 -40.17076537705569 deg, loss
    0.04498843848705292 after 53 Nelder-Mead evaluations on an M5 Max; the restored one gives the same bits there."""
    pytest.importorskip("mlx.core")
    from quantem.gpu import SSB, io
    from tests.hardware.mps.test_ssb_thick_torch_reference import _gold_harness

    harness = _gold_harness()
    master = harness.gold_master()
    if master is None:
        pytest.skip("the gold scan is not in the Hugging Face cache (tests/benchmarks/mps_ssb.py run --download fetches it)")
    with io.load(str(master), hot_pixel_correction=harness.HOT_PIXELS, verbose=False) as data:
        harness.calibrate(data)
        with SSB(data, scan_detector_rotation_deg=358.3108) as ssb:
            fit = ssb.find_aberrations(search="primary", tilt=False, objective="phase_variance", verbose=False)
    assert fit.aberrations["C10_nm"] == pytest.approx(-23.225593050625694, abs=1e-4)
    assert fit.aberrations["C12_nm"] == pytest.approx(3.40092432545935, abs=1e-4)
    assert fit.aberrations["phi12_deg"] == pytest.approx(-40.17076537705569, abs=1e-3)
    assert fit.loss == pytest.approx(0.04498843848705292, rel=1e-6)
