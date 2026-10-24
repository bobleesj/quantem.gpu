"""Saved direct results retain the method and its controls for widget review."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from quantem.gpu.ssb.calibration import capture_calibration
from quantem.gpu.ssb.persistence import load_result, save_result, source_identity
from quantem.gpu.ssb.results import SSBResult


@pytest.mark.parametrize("method", ["phase", "complex"])
def test_direct_calibration_survives_saved_result_roundtrip(tmp_path, method):
    source = tmp_path / "scan.bin"
    source.write_bytes(b"disposable acquisition identity")
    options = {
        "method": method,
        "phase_regularization": 0.05,
        "smoothness_inv_A": 0.7,
        "detector_weighting": "inverse_mean",
    }
    if method == "complex":
        options["amplitude_regularization"] = 0.1
    result = SSBResult(
        object_wave=np.ones((8, 8), np.complex64), backend="mps",
        source_path=str(source), aberrations={"C10_nm": -8., "C12_nm": 0.2, "phi12_deg": 35.},
        sample_tilt_mrad=(1., -2.), depth_spread_nm=12., num_bright_field_pixels=17,
        voltage_kV=300., semiangle_mrad=30., object_sampling_A=(0.25, 0.25),
        upsampling_factor=2, phase_estimator=method, amplitude_estimated=method == "complex",
        reconstruction_method=method, reconstruction_options=options,
        reconstruction_diagnostics={"training_relative_sse": 0.12, "normalization": "relative amplitude"},
    )
    session = SimpleNamespace(
        source_kind="detector", source_path=str(source), input_scan_shape=(4, 4), scan_region=None,
        source_dtype="uint16", source_bytes=128, source_detector_bin=1,
        voltage_kV=300., semiangle_mrad=30., scan_sampling_A=(0.5, 0.5), detector_sampling_mrad=(0.5, 0.5),
        scan_detector_rotation_flipped=False, bright_field_radius_px=2., bright_field_center_px=(1.5, 1.5),
        bright_field_intensity_threshold=0.3, source_provenance={}, source_storage_path=None,
        calibration_path=None, source_manifest_path=None,
    )
    capture_calibration(session, result)
    original = json.loads(result.save_calibration(tmp_path / "calibration.json").read_text())
    assert original["phase_estimator"] == "phase_of_mean"  # Conventional comparison view.
    assert original["reconstruction_method"] == method
    assert original["reconstruction_options"] == options
    assert original["reconstruction_diagnostics"] == result.reconstruction_diagnostics
    assert original["upsampling_factor"] == 2
    assert original["upsampling_mode"] == "explicit"
    assert original["loss"] is None  # Prediction error is not the SSB fit objective.

    paths = (tmp_path / "result.npz", tmp_path / "result.json")
    signature = {"source": source_identity(**result._calibration_source)}
    save_result(result, paths=paths, signature=signature, input_metadata={})
    # Use host loading for this serialization contract, not an MPS solver claim.
    restored = load_result(paths=paths, signature=signature, backend="mps")
    reopened = json.loads(restored.save_calibration(tmp_path / "reopened.json").read_text())
    for key in ("reconstruction_method", "reconstruction_options", "reconstruction_diagnostics",
                "phase_estimator", "upsampling_factor", "upsampling_mode", "aberrations", "session_identity"):
        assert reopened[key] == original[key]
    np.testing.assert_array_equal(restored.object_wave, result.object_wave)
    assert result.phase_estimator == method
