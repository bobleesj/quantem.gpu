"""fit() settles the 180-degree ambiguity of the scan-detector rotation.

The centre-of-mass rotation search cannot tell omega from omega + 180 degrees, and SSB fits both equally well: the second
is the negated phase with the opposite defocus. Atom columns carry positive phase, so ``fit()`` keeps the rotation whose
phase has a positive column sign and otherwise moves the session's rotation by 180 degrees (``scan_detector_rotation_deg``;
``initial_scan_detector_rotation_deg`` keeps the angle before the check) and refits with every aberration flipped in sign (C10 -> -C10, C12 -> -C12). Parity: starting from the wrong rotation must land on the same reconstruction as a direct fit at the right one.
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


def test_simulated_crystal_wrong_rotation_is_turned_and_refit(tmp_path):
    """Known answer (atoms positive, right rotation 0 deg): starting at 180 deg, fit() moves the rotation back to 0 deg."""
    _require_cuda()
    direct = _simulated(0.0).find_aberrations(check_rotation=False, verbose=False)
    session = _simulated(180.0)
    checked = session.find_aberrations(verbose=False)
    assert checked.initial_scan_detector_rotation_deg != checked.scan_detector_rotation_deg
    assert checked.scan_detector_rotation_deg == pytest.approx(0.0, abs=1e-9)
    assert session.scan_detector_rotation_deg == pytest.approx(0.0, abs=1e-9)   # session keeps it
    # validated 2026-09-26: column sign +0.88; phase correlation 0.993 with the direct fit, whose loss is 0.8% higher
    # (the refit from the flipped aberrations lands in a slightly deeper minimum of the same branch)
    assert checked.column_sign > 0.8
    assert _correlation(checked.phase, direct.phase) > 0.99
    assert checked.loss <= direct.loss * (1.0 + 1e-6)

    from quantem.gpu.ssb.persistence import load_result, result_paths, save_result
    from quantem.gpu.ssb.results import column_sign

    assert checked.initial_scan_detector_rotation_deg == 180.0
    assert checked.initial_column_sign == pytest.approx(column_sign(checked.rotation_check_phases[0]))
    assert checked.column_sign == pytest.approx(column_sign(checked.rotation_check_phases[1]))
    comparison = checked.show("rotation")
    assert len([axis for axis in comparison.axes if axis.images]) == 2
    colorbars = [axis for axis in comparison.axes if not axis.images]
    assert colorbars[0].get_ylim() == colorbars[1].get_ylim()
    histogram = checked.show("rotation", histogram=True)
    histogram_axes = [axis for axis in histogram.axes if axis.get_xlabel() == "Phase − median (rad)"]
    assert len(histogram_axes) == 2
    assert histogram_axes[0].get_xlim() == histogram_axes[1].get_xlim()
    assert histogram_axes[0].get_ylim() == histogram_axes[1].get_ylim()
    assert "Suggested rotation: 180.00°" in histogram.texts[0].get_text()
    assert "SSB selected 0.00°" in histogram.texts[0].get_text()
    paths = result_paths(tmp_path / "rotation", "find_aberrations")
    save_result(checked, paths=paths, signature={"test": "rotation"}, input_metadata={})
    restored = load_result(paths=paths, signature={"test": "rotation"}, backend="cuda")
    np.testing.assert_array_equal(restored.rotation_check_phases, checked.rotation_check_phases)
    assert restored.initial_scan_detector_rotation_deg == checked.initial_scan_detector_rotation_deg


def test_simulated_crystal_right_rotation_is_left_exactly_as_fitted(monkeypatch):
    """Starting at the right rotation, the check only measures: same result as check_rotation=False, bit for bit."""
    _require_cuda()
    import quantem.gpu.ssb.workflow as workflow

    drawn = []
    monkeypatch.setattr(workflow, "_in_notebook", lambda: True)
    monkeypatch.setattr(workflow, "draw_column_histogram", lambda *args, **kwargs: drawn.append(args))
    kept = _simulated(0.0).find_aberrations(verbose=True)
    direct = _simulated(0.0).find_aberrations(check_rotation=False, verbose=False)
    assert kept.initial_scan_detector_rotation_deg == kept.scan_detector_rotation_deg
    assert kept.column_sign > 0.8
    np.testing.assert_array_equal(_host(kept.phase), _host(direct.phase))
    assert kept.aberrations == direct.aberrations
    assert kept.loss == direct.loss
    assert drawn == []
    assert kept.show("rotation", histogram=True) is None


def test_check_rotation_false_keeps_the_given_rotation():
    """Opting out keeps the rotation as given, dark atoms included, and records no column sign."""
    _require_cuda()
    from quantem.gpu.ssb.results import column_sign

    result = _simulated(180.0).find_aberrations(check_rotation=False, verbose=False)
    assert result.scan_detector_rotation_deg == pytest.approx(180.0, abs=1e-9)
    assert result.initial_scan_detector_rotation_deg is None and result.column_sign is None
    assert column_sign(result.phase) < -0.8


def test_real_acquisition_from_the_wrong_rotation_matches_a_direct_fit():
    """Real 512 x 512 data started 180 deg off: same rotation, aberrations and phase as a direct fit at the right one."""
    if not SOURCE.is_file():
        pytest.skip("set QUANTEM_SSB_ARINA_MASTER to a real Arina master file")
    _require_cuda()
    from quantem.gpu import SSB

    direct = SSB.open(str(SOURCE), scan_detector_rotation_deg=REAL_ROTATION_DEG, **REAL).find_aberrations(check_rotation=False, verbose=False)
    direct_phase, direct_loss, direct_aberrations = _host(direct.phase), direct.loss, dict(direct.aberrations)
    del direct
    cp.get_default_memory_pool().free_all_blocks()
    checked = SSB.open(str(SOURCE), scan_detector_rotation_deg=REAL_ROTATION_DEG + 180.0, **REAL).find_aberrations(verbose=False)
    assert checked.initial_scan_detector_rotation_deg != checked.scan_detector_rotation_deg
    assert checked.scan_detector_rotation_deg == pytest.approx(REAL_ROTATION_DEG, abs=1e-9)
    # validated 2026-09-26: column sign +0.665, phase correlation 0.99995, loss 1.4e-6 relative, C10 0.018 nm apart
    # (two Nelder-Mead polishes from different starts on the same minimum)
    assert checked.column_sign > 0.5
    assert _correlation(checked.phase, direct_phase) > 0.9999
    assert checked.loss == pytest.approx(direct_loss, rel=1e-5)
    assert checked.aberrations["C10_nm"] == pytest.approx(direct_aberrations["C10_nm"], abs=0.05)
    # astigmatism as one physical quantity, C12 * exp(2 i phi12): (-C12, phi12) and (C12, phi12 + 90 deg) are the same
    # aberration, and the refit starts from every aberration flipped, so compare the vector, not the signed C12
    assert abs(_astigmatism(checked.aberrations) - _astigmatism(direct_aberrations)) < 0.05


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
def test_joint_rotation_check_reuses_one_search(rotation, monkeypatch, tmp_path):
    """Joint calibration selects bright columns with one search from either branch."""
    from quantem.gpu import SSB
    from quantem.gpu.ssb.persistence import load_result, result_paths

    _require_cuda()
    data, detector_sampling_mrad = _simulated_crystal((3.0, -4.0))
    source = tmp_path / "tilted_crystal.npy"
    np.save(source, data)
    session = SSB(
        data, source_path=source, backend="cuda", voltage_kV=300.0,
        semiangle_mrad=30.0, scan_sampling_A=(0.25, 0.25),
        detector_sampling_mrad=detector_sampling_mrad, scan_detector_rotation_deg=rotation,
    )
    fit_calls, reconstruction_calls = [], []
    fit = session._fit_tilt
    reconstruct = session.reconstruct

    def measured_fit(*args, **kwargs):
        result = fit(*args, **kwargs)
        fit_calls.append((dict(result.aberrations), result.sample_tilt_mrad))
        return result

    def measured_reconstruct(*args, **kwargs):
        reconstruction_calls.append(kwargs)
        return reconstruct(*args, **kwargs)

    monkeypatch.setattr(session, "_fit_tilt", measured_fit)
    monkeypatch.setattr(session, "reconstruct", measured_reconstruct)
    try:
        result = session.find_aberrations(
            tilt=True, save_to=tmp_path / "joint", verbose=False,
        )
        assert len(fit_calls) == 1
        assert len(reconstruction_calls) == (2 if rotation == 180 else 1)
        assert (result.initial_scan_detector_rotation_deg != result.scan_detector_rotation_deg) == (rotation == 180)
        assert result.scan_detector_rotation_deg == 0.0
        assert result.column_sign > 0.2
        assert result.sample_tilt_mrad == pytest.approx((3.0, -4.0), abs=1.0)
        assert len(result.trial_records) == result.num_trials == 200
        assert all(record["scan_detector_rotation_deg"] == rotation
                   for record in result.trial_records)
        if rotation == 180:
            initial_aberrations, initial_tilt = fit_calls[0]
            assert result.aberrations["C10_nm"] == -initial_aberrations["C10_nm"]
            assert result.aberrations["C12_nm"] == -initial_aberrations["C12_nm"]
            assert result.sample_tilt_mrad == tuple(-value for value in initial_tilt)

        restored = load_result(
            paths=result_paths(tmp_path / "joint", "find_aberrations"),
            signature=result.metadata["signature"], backend="cuda",
        )
        assert restored.scan_detector_rotation_deg == result.scan_detector_rotation_deg
        assert restored.sample_tilt_mrad == result.sample_tilt_mrad
        np.testing.assert_array_equal(_host(restored.phase), _host(result.phase))
        # The diagnostic belongs to the selected geometry, and sampling reuses it.
        native = reconstruct(restored, phase_estimator="mean_phase")
        np.testing.assert_allclose(
            _host(native.phase), result.rotation_check_phases[1], atol=1e-6,
        )
        assert native.loss == pytest.approx(result.loss, rel=1e-6)
        finer = reconstruct(restored, upsampling_factor=2)
        assert finer.phase.shape == tuple(2 * size for size in native.phase.shape)
        assert finer.loss == pytest.approx(native.loss, rel=1e-6)
        assert finer.sample_tilt_mrad == restored.sample_tilt_mrad
        assert finer.scan_detector_rotation_deg == restored.scan_detector_rotation_deg
        assert len(fit_calls) == 1
    finally:
        session.close()


def test_joint_rotation_check_keeps_an_unresolved_scan(monkeypatch):
    """A scan with no phase contrast provides no evidence for a branch change."""
    from quantem.gpu import SSB

    _require_cuda()
    counts = np.full((128, 128, 16, 16), 20, dtype=np.uint16)
    with SSB(counts, backend="cuda", voltage_kV=300, semiangle_mrad=20,
             scan_sampling_A=(0.5, 0.5), detector_sampling_mrad=(5, 5), bright_field_center_px=(8, 8),
             bright_field_radius_px=4, scan_detector_rotation_deg=180) as session:
        original = session.reconstruct
        calls = []

        def measured(*args, **kwargs):
            calls.append(kwargs)
            return original(*args, **kwargs)

        monkeypatch.setattr(session, "reconstruct", measured)
        result = session.find_aberrations(
            tilt=True, trials=8, refinement=None, verbose=False,
        )
        assert result.column_sign == 0
        assert result.initial_scan_detector_rotation_deg == result.scan_detector_rotation_deg == 180.0
        assert len(calls) == 1  # Only the normal final display reconstruction.


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

