"""Physical Apple GPU coverage for detector-to-SSB buffer ownership."""

import math
import platform

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(platform.system() != "Darwin", reason="Requires physical Apple GPU")


def _counts():
    rng = np.random.default_rng(42)
    rows, cols = np.mgrid[:12, :12]
    disk = (rows - 5.5)**2 + (cols - 5.5)**2 < 16
    return rng.integers(1, 20, (128, 128, 12, 12), dtype=np.uint16) * disk


@pytest.mark.parametrize("dtype", [np.uint16, np.int32, np.float32])
def test_mps_tensor_matches_existing_array_input_without_download(dtype, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("mlx.core")
    if not torch.backends.mps.is_available():
        pytest.skip("MPS device unavailable")
    from quantem.gpu import SSB

    counts = _counts().astype(dtype)
    if dtype is np.float32:
        counts /= 1024  # Preserve sub-unit simulated intensities.
    tensor = torch.from_numpy(counts).to("mps")
    settings = dict(backend="mps", voltage_kV=300, semiangle_mrad=30,
                    scan_sampling_A=(.99, .99), detector_sampling_mrad=(6., 6.), bright_field_radius_px=4., bright_field_center_px=(5.5, 5.5))
    coefficients = {"C10_nm": -11.52, "C12_nm": 4.88, "phi12_deg": -32.8}
    original_cpu = torch.Tensor.cpu

    def no_volume_download(value, *args, **kwargs):
        if value.ndim > 2:
            raise AssertionError("SSB downloaded detector volume")
        return original_cpu(value, *args, **kwargs)

    with SSB(counts, **settings) as baseline, SSB(tensor, **settings) as candidate:
        expected, expected_loss = baseline.preview(coefficients)
        monkeypatch.setattr(torch.Tensor, "cpu", no_volume_download)
        actual, loss = candidate.preview(coefficients)
        np.testing.assert_array_equal(actual, expected)
        assert loss == expected_loss
        # Closing the session never releases storage owned by the caller.
    assert bool(torch.all(tensor.to(torch.float32) >= 0))


def test_open_encoded_source_keeps_mps_crop_encoded(tmp_path, monkeypatch):
    """SSB.open keeps an integer MPS acquisition encoded and reads its bright-field columns without a volume download."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("mlx.core")
    if not torch.backends.mps.is_available():
        pytest.skip("MPS device unavailable")
    import h5py
    import hdf5plugin
    from quantem.gpu import SSB
    from quantem.gpu.direct_ptycho.mps.frames import EncodedBrightFieldCrop

    counts = _counts()
    source = tmp_path / "counts.h5"
    with h5py.File(source, "w") as handle:
        handle.create_dataset("entry/data/data", data=counts.reshape(-1, 12, 12),
                              chunks=(1, 12, 12), **hdf5plugin.Bitshuffle(cname="lz4"))
    original_cpu = torch.Tensor.cpu

    def no_volume_download(value, *args, **kwargs):
        if value.ndim > 2:
            raise AssertionError("SSB.open downloaded detector volume")
        return original_cpu(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", no_volume_download)
    with SSB.open(str(source), backend="mps", voltage_kV=300, semiangle_mrad=30,
                  scan_sampling_A=(.99, .99), detector_sampling_mrad=(6., 6.), scan_shape=(128, 128)) as session:
        assert isinstance(session._data, EncodedBrightFieldCrop)
        assert session._data.source is session._owned_dataset
        phase, loss = session.preview({"C10_nm": -11.52, "C12_nm": 4.88, "phi12_deg": math.degrees(-.5)})
        assert phase.shape == (128, 128)
        assert np.isfinite(phase).all() and np.isfinite(loss)


def test_open_uses_recorded_calibration_and_keeps_explicit_overrides(tmp_path):
    """Opening calibrated counts needs no duplicated microscope settings."""
    pytest.importorskip("mlx.core")
    from quantem.gpu import SSB, io

    counts = _counts()
    calibration = dict(voltage_kV=300, semiangle_mrad=30,
                       scan_sampling_A=(.99, .99), detector_sampling_mrad=(6., 6.))
    source = tmp_path / "calibrated.qem"
    io.save(source, counts, format="quantem", compression="ans", backend="cpu", metadata=calibration)
    recorded_step = tuple(io.inspect(source).metadata["scan_sampling_A"])
    coefficients = {"C10_nm": -11.52, "C12_nm": 4.88, "phi12_deg": -32.8}
    with SSB.open(str(source)) as automatic, SSB.open(str(source), **calibration) as explicit:
        actual, actual_loss = automatic.preview(coefficients)
        expected, expected_loss = explicit.preview(coefficients)
        np.testing.assert_array_equal(actual, expected)
        assert actual_loss == expected_loss
        assert automatic.scan_sampling_A == recorded_step
    with SSB.open(str(source), detector_sampling_mrad=(7., 7.)) as overridden:
        assert overridden.detector_sampling_mrad == (7., 7.)
        assert overridden.scan_sampling_A == recorded_step
    uncalibrated = tmp_path / "uncalibrated.qem"
    io.save(uncalibrated, counts, format="quantem", compression="ans", backend="cpu",
            metadata={"voltage_kV": 300, "semiangle_mrad": 30})
    with pytest.raises(ValueError, match="missing scan_sampling_A"):
        SSB.open(str(uncalibrated))


def test_transposed_detector_matches_explicit_swap_on_mps(tmp_path):
    """Compare the automatic orientation against an explicitly exchanged detector."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("mlx.core")
    from quantem.gpu import SSB

    counts = _counts()[:, :, :11, :12]
    tensor = torch.from_numpy(counts.copy()).to("mps")
    settings = dict(backend="mps", voltage_kV=300, semiangle_mrad=30,
                    scan_sampling_A=(.99, 1.1), scan_detector_rotation_deg=31.,
                    bright_field_radius_px=4.)
    coefficients = {"C10_nm": -11.52, "C12_nm": 4.88, "phi12_deg": -32.8}
    with SSB(tensor, scan_detector_rotation_flipped=True, detector_sampling_mrad=(6., 7.),
             bright_field_center_px=(5., 5.5), **settings) as candidate:
        actual = candidate.reconstruct(aberrations=coefficients)
        state = candidate.browser_state()
        assert actual.scan_detector_rotation_flipped is True
        assert actual.bright_field_center_px == (5., 5.5)
    with SSB(tensor.swapaxes(-1, -2).contiguous(), detector_sampling_mrad=(7., 6.),
             bright_field_center_px=(5.5, 5.), **settings) as reference:
        expected = reference.reconstruct(aberrations=coefficients)
        reference_state = reference.browser_state()
    np.testing.assert_array_equal(actual.phase, expected.phase)
    assert actual.loss == expected.loss
    assert state.scan_shape == reference_state.scan_shape


def test_saved_geometry_reopens_the_same_encoded_brightfield(tmp_path):
    """Reopening a fit retains its detector centre, fractional mask and calibration."""
    import h5py
    import hdf5plugin
    from quantem.gpu import SSB

    counts = _counts()
    counts[:, :, :, 6:] *= 2
    source = tmp_path / "counts.h5"
    with h5py.File(source, "w") as handle:
        handle.create_dataset("entry/data/data", data=counts.reshape(-1, 12, 12),
                              chunks=(1, 12, 12), **hdf5plugin.Bitshuffle(cname="lz4"))
    settings = dict(backend="mps", voltage_kV=300, semiangle_mrad=30,
                    scan_sampling_A=(.99, 1.1), scan_shape=(128, 128),
                    scan_detector_rotation_deg=31., scan_detector_rotation_flipped=True,
                    detector_sampling_mrad=(6., 7.))
    coefficients = {"C10_nm": -11.52, "C12_nm": 4.88, "phi12_deg": -32.8}
    with SSB.open(str(source), **settings) as initial:
        expected = initial.reconstruct(aberrations=coefficients)
    with SSB.open(str(source), **settings,
                  bright_field_center_px=expected.bright_field_center_px,
                  bright_field_radius_px=expected.bright_field_radius_px) as reopened:
        actual = reopened.reconstruct(aberrations=coefficients)
    np.testing.assert_array_equal(actual.phase, expected.phase)
    assert actual.loss == expected.loss
