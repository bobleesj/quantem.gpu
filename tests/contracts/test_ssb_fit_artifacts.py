"""A fit can be reviewed and exported without running scientific computation."""

import json
from copy import deepcopy
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest

from quantem.gpu.direct_ptycho.calibration import capture_calibration
from quantem.gpu.direct_ptycho.persistence import load_result, save_result, source_identity
from quantem.gpu.direct_ptycho.results import SSBResult


def _fit():
    return SSBResult(
        object_wave=np.exp(1j * np.arange(16).reshape(4, 4) / 100).astype(np.complex64),
        backend="mps", aberrations={"C10_nm": -8., "C12_nm": 0.2, "phi12_deg": 35., "C50_nm": 10_000.},
        sample_tilt_mrad=(1., -2.), depth_spread_nm=12., phase_estimator="phase_of_mean",
        num_bright_field_pixels=17, loss=0.2, voltage_kV=300., semiangle_mrad=30.,
        object_sampling_A=(.5, .5), elapsed_s=3., refinement="newton",
        optimizer={"coarse_scores": [.1, .3, .2], "refinement_trace": [[1, .3, 0.], [100, .4, 1.]],
                   "stop": "evaluation cap"},
    )


def test_convergence_uses_saved_values_and_states_budget_stop(tmp_path):
    fit = _fit()
    wave, optimizer = fit.object_wave.copy(), deepcopy(fit.optimizer)
    figure = fit.show("convergence", size="small", save=tmp_path / "convergence.png")
    ax = figure.axes[0]
    np.testing.assert_array_equal(ax.lines[0].get_ydata(), [.1, .3, .3])
    np.testing.assert_array_equal(ax.lines[1].get_xdata(), [4, 103])
    assert "evaluation cap" in ax.texts[0].get_text()
    assert not plt.fignum_exists(figure.number)
    assert (tmp_path / "convergence.png").is_file()
    assert fit.optimizer == optimizer
    assert fit.elapsed_s == 3.
    np.testing.assert_array_equal(fit.object_wave, wave)


def test_standard_branches_and_objectives_are_not_combined():
    fit = _fit()
    fit.optimizer = {}
    fit.trial_records = [
        {"search": 0, "objective": "negative_thick_agreement", "loss": -2.},
        {"search": 0, "objective": "negative_thick_agreement", "loss": -1.},
        {"search": 1, "objective": "negative_thick_agreement", "loss": -5.},
        {"search": 1, "objective": "phase_variance", "loss": .3},
    ]
    figure = fit.show("convergence", columns=1)
    assert len(figure.axes) == 3
    np.testing.assert_array_equal(figure.axes[0].lines[0].get_ydata(), [-2., -2.])
    assert "not recorded" in figure.axes[0].texts[0].get_text()
    fit.trial_records = None
    with pytest.raises(ValueError, match="no recorded search history"):
        fit.show("convergence")


def test_calibration_survives_result_roundtrip_without_session(tmp_path, monkeypatch):
    source = tmp_path / "scan.bin"
    source.write_bytes(b"synthetic detector counts")
    fit = _fit()
    fit.source_path = str(source)
    session = SimpleNamespace(
        source_kind="detector", source_path=str(source), input_scan_shape=(4, 4), scan_region=None,
        source_dtype="uint16", source_bytes=128, source_detector_bin=1,
        voltage_kV=300., semiangle_mrad=30., scan_sampling_A=(.5, .5), detector_sampling_mrad=(.5, .5),
        scan_detector_rotation_flipped=False, bright_field_radius_px=2., bright_field_center_px=(1.5, 1.5),
        bright_field_intensity_threshold=.3, source_provenance={}, source_storage_path=None,
        calibration_path=None, source_manifest_path=None,
    )
    capture_calibration(session, fit)
    session.scan_sampling_A = (9., 9.)  # changing a session must not alter its earlier fit
    wave = fit.object_wave.copy()
    path = fit.save_calibration(tmp_path / "review" / "calibration.json")
    record = json.loads(path.read_text())
    assert record["schema"] == "quantem.ssb-calibration/6"
    assert record["aberrations"] == fit.aberrations
    assert record["sample_tilt_mrad"] == [1., -2.]
    assert record["scan_sampling_A"] == [.5, .5]
    assert record["upsampling_mode"] == "auto" and record["upsampling_factor"] > 1
    assert str(tmp_path) not in path.read_text()
    fingerprint = source_identity(**fit._calibration_source)
    signature = {"source": fingerprint}
    paths = (tmp_path / "result.npz", tmp_path / "result.json")
    save_result(fit, paths=paths, signature=signature, input_metadata={})
    metadata = json.loads(paths[1].read_text())
    assert "calibration_context" in metadata
    assert "calibration_context" not in metadata["result"]  # older result readers ignore optional metadata
    restored = load_result(paths=paths, signature=signature, backend="mps")
    # The persisted fingerprint is sufficient even after closing the source/session.
    monkeypatch.setattr("quantem.gpu.direct_ptycho.persistence.source_identity",
                        lambda **kwargs: pytest.fail("export rehashed a saved acquisition"))
    second = json.loads(restored.save_calibration(tmp_path / "restored.json").read_text())
    for key in ("aberrations", "session_identity", "sample_tilt_mrad", "depth_spread_nm", "scan_region"):
        assert second[key] == record[key]
    assert "_calibration_source" not in paths[1].read_text()
    assert str(tmp_path) not in paths[1].read_text()
    np.testing.assert_array_equal(fit.object_wave, wave)
    np.testing.assert_array_equal(restored.object_wave, wave)
    restored.show("convergence")


def test_calibration_rejects_unidentified_array_and_partial_model(tmp_path):
    fit = _fit()
    with pytest.raises(ValueError, match="persistent acquisition"):
        fit.save_calibration(tmp_path / "array.json")
    assert not (tmp_path / "array.json").exists()
    fit.calibration_context = {"session_identity": {"source": {}, "geometry": {
        "voltage_kV": 300., "semiangle_mrad": 30., "scan_sampling_A": [.5, .5],
        "detector_sampling_mrad": [.5, .5], "num_bright_field_pixels": 17,
    }}}
    fit.depth_spread_nm = None
    with pytest.raises(ValueError, match="complete model"):
        fit.save_calibration(tmp_path / "partial.json")


def test_timed_budget_overlay_uses_one_clock_and_common_objective(tmp_path):
    fits = {}
    for count, normalization in ((200, 2.), (400, 4.), (600, 8.)):
        fit = _fit()
        fit.num_trials = count
        fit.optimizer.update(
            coarse_scores=[.1 / normalization, .3 / normalization, .2 / normalization],
            refinement_trace=[[1, .3 / normalization, 0.1], [100, .4 / normalization, 1.]],
            coarse_elapsed_s=[2., 3., 4.], refinement_start_s=4.5,
            convergence_clock="fit_start", normalization=normalization,
        )
        fits[count] = fit
    fits[600].refinement = "nelder-mead"
    before = [deepcopy(fit.optimizer) for fit in fits.values()]
    figure = fits[200].show("convergence", compare=fits, size="large", save=tmp_path / "compare.png")
    ax = figure.axes[0]
    assert len(ax.lines) == 6  # the reference is already in the mapping
    for line in ax.lines[::2]:
        np.testing.assert_array_equal(line.get_xdata(), [2., 3., 4.])
        np.testing.assert_allclose(line.get_ydata(), [.1, .3, .3])
    for line in ax.lines[1::2]:
        np.testing.assert_array_equal(line.get_xdata(), [4.6, 5.5])
        np.testing.assert_allclose(line.get_ydata(), [.3, .4])
    assert "Nelder–Mead" in ax.lines[-1].get_label()
    assert ax.get_xlabel() == "Elapsed fit time (s)"
    assert not plt.fignum_exists(figure.number)
    assert before == [fit.optimizer for fit in fits.values()]
    fits[400].scan_detector_rotation_deg += 180
    with pytest.raises(ValueError, match="same data, calibration"):
        fits[200].show("convergence", compare=fits)


def test_timed_history_roundtrip_and_old_history_comparison(tmp_path):
    fit = _fit()
    fit.optimizer.update(coarse_elapsed_s=[.2, .4, .6], refinement_start_s=.7,
                         convergence_clock="fit_start", normalization=2.)
    paths = (tmp_path / "fit.npz", tmp_path / "fit.json")
    save_result(fit, paths=paths, signature={"test": "time"}, input_metadata={})
    restored = load_result(paths=paths, signature={"test": "time"}, backend="mps")
    assert restored.optimizer == fit.optimizer
    figure = restored.show("convergence")
    np.testing.assert_array_equal(figure.axes[0].lines[1].get_xdata(), [.7, 1.7])
    old = _fit()
    old.optimizer["normalization"] = 2.
    figure = restored.show("convergence", compare={"current": restored, "old": old})
    assert "evaluations" in figure.axes[0].get_xlabel()
    with pytest.raises(ValueError, match="lack elapsed times"):
        restored.show("convergence", x="time", compare=old)
    old.objective = "phase_variance"
    with pytest.raises(ValueError, match="same data, calibration"):
        restored.show("convergence", compare=old)


def test_joint_refinement_without_extra_tpe_has_timed_convergence(tmp_path):
    fit = _fit()
    fit.optimizer.update(
        search="standard", coarse_scores=[], coarse_elapsed_s=[],
        refinement_start_s=2., convergence_clock="fit_start",
        primary_trials=200, primary_seed=42, normalization=1.,
    )
    before = deepcopy(fit.optimizer)
    figure = fit.show("convergence", x="time")
    ax = figure.axes[0]
    assert len(ax.lines) == 1
    np.testing.assert_array_equal(ax.lines[0].get_xdata(), [2., 3.])
    assert "joint refinement" in ax.get_title()
    assert "Initial fit + setup" in ax.patches[0].get_label()
    assert fit.optimizer == before
    paths = (tmp_path / "fit.npz", tmp_path / "fit.json")
    save_result(fit, paths=paths, signature={"search": "standard"}, input_metadata={})
    restored = load_result(paths=paths, signature={"search": "standard"}, backend="mps")
    assert restored.optimizer == before
    restored.show("convergence", x="time")
