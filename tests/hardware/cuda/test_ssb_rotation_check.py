"""find_aberrations settles the 180-degree ambiguity of an estimated scan-detector rotation.

The centre-of-mass rotation search cannot tell omega from omega + 180 degrees, and SSB fits both equally well: the second
is the negated phase with the opposite defocus. For an estimated rotation the fit runs twice, independently, at both
angles, and keeps the branch whose dark-field atomic columns have positive phase (the larger phase tail when no columns
are resolved). A given rotation is fitted once and kept. Parity: starting from the wrong branch must land on the same
reconstruction as a direct fit at the right one.
"""

import math
import os
from pathlib import Path

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from tests.hardware.cuda.test_ssb_thick_sample import _simulated_crystal  # noqa: E402

pytestmark = pytest.mark.slow

# an Arina master of a real 512 x 512 x 192 x 192 acquisition whose right rotation is 158.9 deg (atoms bright)
SOURCE = Path(os.environ.get("QUANTEM_SSB_ARINA_MASTER", ""))
REAL = dict(backend="cuda", voltage_kV=300.0, semiangle_mrad=30.0, scan_sampling_A=(0.264, 0.264), detector_sampling_mrad=(0.5554, 0.5554))
REAL_ROTATION_DEG = 158.9


def _require_cuda():
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("No CUDA device")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("No CUDA runtime")


def _host(array) -> np.ndarray:
    return np.asarray(cp.asnumpy(array) if isinstance(array, cp.ndarray) else array, np.float64)


def _correlation(first, second) -> float:
    first, second = _host(first), _host(second)
    first, second = first - first.mean(), second - second.mean()
    return float((first * second).sum() / np.sqrt((first * first).sum() * (second * second).sum()))


def _astigmatism(aberrations) -> complex:
    return aberrations["C12_nm"] * np.exp(2j * math.radians(aberrations["phi12_deg"]))


def _simulated(scan_detector_rotation_deg: float):
    from quantem.gpu import SSB

    data, detector_sampling_mrad = _simulated_crystal((3.0, -4.0))
    return SSB(data, backend="cuda", voltage_kV=300.0, semiangle_mrad=30.0, scan_sampling_A=(0.25, 0.25),
                          detector_sampling_mrad=detector_sampling_mrad, scan_detector_rotation_deg=scan_detector_rotation_deg)


# ---


@pytest.fixture
def estimate(monkeypatch):
    """Set the centre-of-mass estimate a session starts from (the curl knows it only modulo 180 degrees)."""
    from quantem.gpu.ssb import workflow

    def set_estimate(rotation):
        monkeypatch.setattr(workflow, "_estimate_rotation", lambda *args: (rotation, False))

    return set_estimate


def _estimated():
    from quantem.gpu import SSB

    data, detector_sampling_mrad = _simulated_crystal((3.0, -4.0))
    return SSB(data, backend="cuda", voltage_kV=300.0, semiangle_mrad=30.0, scan_sampling_A=(0.25, 0.25),
               detector_sampling_mrad=detector_sampling_mrad)


def test_simulated_crystal_estimated_on_the_wrong_branch_keeps_the_right_one(tmp_path, estimate):
    """Known answer (atoms positive, right rotation 0 deg): an estimate of 180 deg ends at 0 deg after two fits."""
    _require_cuda()
    direct = _simulated(0.0).find_aberrations(search="standard", check_rotation=False, verbose=False)
    estimate(180.0)
    session = _estimated()
    checked = session.find_aberrations(search="standard", verbose=False)
    branches = checked.rotation_branches
    assert [branch.scan_detector_rotation_deg for branch in branches.branches] == [180.0, 0.0]
    assert branches.kept == branches.chosen == 1
    assert checked.scan_detector_rotation_deg == pytest.approx(0.0, abs=1e-9)
    assert session.scan_detector_rotation_deg == pytest.approx(0.0, abs=1e-9)   # session keeps it
    # the branch at 0 deg is an independent fit with the direct fit's settings and start
    assert checked.aberrations == direct.aberrations
    np.testing.assert_array_equal(_host(checked.phase), _host(direct.phase))
    assert _correlation(branches.branches[0].phase, -_host(direct.phase)) > 0.99
    figure = checked.show("rotation")
    # the phase-tail panel (log axis) sits below the two branch phases
    assert any(axis.get_yscale() == "log" for axis in figure.axes)

    from quantem.gpu.ssb.persistence import load_result, result_paths, save_result

    paths = result_paths(tmp_path / "rotation", "find_aberrations")
    save_result(checked, paths=paths, signature={"test": "rotation"}, input_metadata={})
    restored = load_result(paths=paths, signature={"test": "rotation"}, backend="cuda")
    assert restored.rotation_branches.summary == branches.summary
    np.testing.assert_array_equal(restored.rotation_branches.branches[0].phase, branches.branches[0].phase)


def test_simulated_crystal_estimated_on_the_right_branch_is_left_exactly_as_fitted(estimate):
    """Estimated at the right rotation, the kept branch is the first fit: same result as check_rotation=False."""
    _require_cuda()
    estimate(0.0)
    kept = _estimated().find_aberrations(search="standard", verbose=False)
    direct = _simulated(0.0).find_aberrations(search="standard", check_rotation=False, verbose=False)
    assert kept.rotation_branches.kept == 0 and kept.scan_detector_rotation_deg == 0.0
    np.testing.assert_array_equal(_host(kept.phase), _host(direct.phase))
    assert kept.aberrations == direct.aberrations
    assert kept.loss == direct.loss


def test_a_given_rotation_is_kept():
    """A given rotation is fitted once and kept, dark atoms included, with no branch comparison."""
    _require_cuda()
    from quantem.gpu.ssb.results import column_sign

    result = _simulated(180.0).find_aberrations(search="standard", verbose=False)
    assert result.scan_detector_rotation_deg == pytest.approx(180.0, abs=1e-9)
    assert result.rotation_branches is None
    assert column_sign(result.phase) < -0.8


def test_real_acquisition_compares_the_same_two_branches_from_either_angle():
    """Real 512 x 512 data: comparing from either branch selects the same rotation and the same fits."""
    if not SOURCE.is_file():
        pytest.skip("set QUANTEM_SSB_ARINA_MASTER to a real Arina master file")
    _require_cuda()
    from quantem.gpu import SSB

    with SSB.open(str(SOURCE), scan_detector_rotation_deg=REAL_ROTATION_DEG, **REAL) as session:
        first = session.compare_rotation_branches(verbose=False)
        assert session.scan_detector_rotation_deg == REAL_ROTATION_DEG   # given: reported, not applied
    cp.get_default_memory_pool().free_all_blocks()
    with SSB.open(str(SOURCE), scan_detector_rotation_deg=REAL_ROTATION_DEG + 180.0, **REAL) as session:
        second = session.compare_rotation_branches(apply=True, verbose=False)
        assert session.scan_detector_rotation_deg == pytest.approx(second.chosen_rotation_deg)
    assert first.chosen_rotation_deg == pytest.approx(second.chosen_rotation_deg, abs=1e-9)
    assert first.decided_by == second.decided_by
    for one, other in zip(first.branches, second.branches[::-1], strict=True):
        assert one.loss == pytest.approx(other.loss, rel=1e-5)
        assert one.aberrations["C10_nm"] == pytest.approx(other.aberrations["C10_nm"], abs=0.05)
        assert abs(_astigmatism(one.aberrations) - _astigmatism(other.aberrations)) < 0.05


def test_rotation_plus_180_is_the_conjugate_object_with_every_aberration_flipped():
    """The rotation plus 180 degrees reconstructs the negated phase with C10 and C12 flipped, whether the session is
    turned there with set_rotation or built at that rotation."""
    _require_cuda()
    aberrations = {"C10_nm": 1.5, "C12_nm": 0.8, "phi12_deg": math.degrees(0.4)}
    flipped = {"C10_nm": -1.5, "C12_nm": -0.8, "phi12_deg": math.degrees(0.4)}
    measured = _simulated(0.0).reconstruct(aberrations=aberrations)
    reversed_ = _simulated(0.0)
    reversed_.set_rotation(180.0)
    conjugate = reversed_.reconstruct(aberrations=flipped)
    assert reversed_.scan_detector_rotation_deg == 180.0
    assert conjugate.scan_detector_rotation_deg == 180.0
    assert _correlation(conjugate.phase, -_host(measured.phase)) > 0.9999
    same = _simulated(180.0).reconstruct(aberrations=flipped)     # a session built at the same physical rotation
    np.testing.assert_array_equal(_host(same.phase), _host(conjugate.phase))


@pytest.mark.parametrize("rotation", [0.0, 180.0])
def test_joint_fit_runs_two_independent_searches(rotation, estimate, monkeypatch, tmp_path):
    """Joint calibration fits both branches with fresh searches and keeps the one with bright columns."""
    from quantem.gpu import SSB
    from quantem.gpu.ssb.persistence import load_result, result_paths

    _require_cuda()
    estimate(rotation)
    data, detector_sampling_mrad = _simulated_crystal((3.0, -4.0))
    source = tmp_path / "tilted_crystal.npy"
    np.save(source, data)
    session = SSB(
        data, source_path=source, backend="cuda", voltage_kV=300.0,
        semiangle_mrad=30.0, scan_sampling_A=(0.25, 0.25), detector_sampling_mrad=detector_sampling_mrad,
    )
    fit_calls = []
    fit = session._fit_tilt

    def measured_fit(*args, **kwargs):
        fit_calls.append((session.scan_detector_rotation_deg, session._aberrations_explicit))
        return fit(*args, **kwargs)

    monkeypatch.setattr(session, "_fit_tilt", measured_fit)
    try:
        result = session.find_aberrations(search="standard", tilt=True, save_dir=tmp_path / "joint", verbose=False)
        # two searches, neither started from the other's fit
        assert fit_calls == [(rotation, False), ((rotation + 180.0) % 360.0, False)]
        assert result.scan_detector_rotation_deg == 0.0
        assert result.sample_tilt_mrad == pytest.approx((3.0, -4.0), abs=1.0)
        assert len(result.trial_records) == 2 * result.num_trials == 400
        assert {record["scan_detector_rotation_deg"] for record in result.trial_records} == {0.0, 180.0}
        kept = result.rotation_branches.branches[result.rotation_branches.kept]
        np.testing.assert_array_equal(_host(result.phase), kept.phase)

        restored = load_result(
            paths=result_paths(tmp_path / "joint", "find_aberrations"),
            signature=result.metadata["signature"], backend="cuda",
        )
        assert restored.scan_detector_rotation_deg == result.scan_detector_rotation_deg
        assert restored.sample_tilt_mrad == result.sample_tilt_mrad
        assert restored.rotation_branches.summary == result.rotation_branches.summary
        np.testing.assert_array_equal(_host(restored.phase), _host(result.phase))
        finer = session.reconstruct(restored, upsampling_factor=2)
        assert finer.phase.shape == tuple(2 * size for size in result.phase.shape)
        assert finer.rotation_branches is restored.rotation_branches
        assert len(fit_calls) == 2
    finally:
        session.close()


def test_no_signal_scan_reports_unidentifiable_fit_without_changing_rotation():
    """No fit-band signal cannot support joint fitting or a rotation decision."""
    from quantem.gpu import SSB

    _require_cuda()
    counts = np.full((128, 128, 16, 16), 20, dtype=np.uint16)
    with SSB(counts, backend="cuda", voltage_kV=300, semiangle_mrad=20,
             scan_sampling_A=(0.5, 0.5), detector_sampling_mrad=(5, 5), bright_field_center_px=(8, 8),
             bright_field_radius_px=4, scan_detector_rotation_deg=180) as session:
        with pytest.raises(ValueError, match="no positive finite agreement"):
            session.compare_rotation_branches(apply=True, tilt=True, num_trials=8,
                                              refinement=None, verbose=False)
        assert session.scan_detector_rotation_deg == 180.0


def test_preview_context_follows_a_rotation_set_after_it_was_made():
    """A reduced-BF drag context made before set_rotation previews, and leaves the session, at the new rotation.

    Leaving the context used to restore the geometry of the rotation it was made at while the engine recorded the new
    rotation as prepared, so this and every later reconstruction used the old rotation.
    """
    _require_cuda()
    from tests.hardware.cuda.test_ssb_thick_sample import _synthetic_session

    aberrations = {"C10_nm": 3.0, "C12_nm": 0.5, "phi12_deg": math.degrees(0.3)}
    ssb, _ = _synthetic_session()
    context = ssb.preview_context(ssb.num_bright_field_pixels // 4)
    ssb.set_rotation(40.0)
    subset_phase, _ = ssb.preview(aberrations, context=context)
    phase, loss = ssb.preview(aberrations)
    reference, _ = _synthetic_session()
    reference.set_rotation(40.0)
    expected_subset, _ = reference.preview(aberrations, context=reference.preview_context(reference.num_bright_field_pixels // 4))
    expected, expected_loss = reference.preview(aberrations)

    np.testing.assert_array_equal(subset_phase, expected_subset)
    np.testing.assert_array_equal(phase, expected)
    assert loss == expected_loss

