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
                    scan_sampling_A=.99, det_sampling=6., bf_radius=4., bf_center=(5.5, 5.5))
    coefficients = {"C10": -11.52, "C12": 4.88, "phi12": math.radians(-32.8)}
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


def test_open_encoded_source_retains_mps_crop(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("mlx.core")
    if not torch.backends.mps.is_available():
        pytest.skip("MPS device unavailable")
    import h5py
    import hdf5plugin
    from quantem.gpu import SSB

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
                  scan_sampling_A=.99, det_sampling=6., scan_shape=(128, 128)) as session:
        assert torch.is_tensor(session._data)
        assert session._data.device.type == "mps"
        phase, loss = session.preview({"C10": -11.52, "C12": 4.88, "phi12": -.5})
        assert phase.shape == (128, 128)
        assert np.isfinite(phase).all() and np.isfinite(loss)
