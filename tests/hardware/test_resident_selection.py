"""Scientific selections and previews use the loaded object's public reader."""

import os

import numpy as np
import pytest

from quantem.gpu.io import load, _read


def test_selected_and_binned_reads_preserve_counts(tmp_path, monkeypatch):
    """Inspect selections larger than the decode budget without a dense input."""
    backend = os.environ.get("QEM_TEST_BACKEND")
    if backend not in {"cuda", "mps"}:
        pytest.skip("Set QEM_TEST_BACKEND to the physical accelerator.")
    values = np.arange(5 * 7 * 4 * 6, dtype=np.uint16).reshape(5, 7, 4, 6)
    path = tmp_path / "scan.npy"
    np.save(path, values)
    monkeypatch.setattr(_read, "_READ_BYTES", 3 * 4 * 6 * 2)
    with load(path, backend=backend, verbose=False) as data:
        binned = data.read(detector_bin=2)
        expected = values.reshape(5, 7, 2, 2, 3, 2).sum((3, 5), dtype=np.uint64)
        np.testing.assert_array_equal(binned.cpu().numpy(), expected)
        region = data.read(scan_region=(1, 5, 2, 7), detector_region=(1, 4, 2, 6))
        np.testing.assert_array_equal(region.cpu().numpy(), values[1:5, 2:7, 1:4, 2:6])
        np.testing.assert_array_equal(data.read().cpu().numpy(), values)
        assert region.device.type == binned.device.type == backend


def test_resample_scan_preserves_detector_measurements(tmp_path):
    """Resample a linear scan field with known fractional and border values."""
    import torch
    from quantem.gpu.geometry import resample_scan

    backend = os.environ.get("QEM_TEST_BACKEND")
    if backend not in {"cuda", "mps"}:
        pytest.skip("Set QEM_TEST_BACKEND to the physical accelerator.")
    rows, columns = np.indices((5, 7))
    detector = np.arange(24).reshape(4, 6)
    values = (rows[..., None, None] * 40 + columns[..., None, None] * 5
              + detector).astype(np.uint16)
    coordinates = np.array([[[1.25, 2.5], [-1, 3.25]],
                            [[10, 10], [2.5, 1.5]]], np.float32)
    expected = (np.clip(coordinates[..., 0], 0, 4) * 40
                + np.clip(coordinates[..., 1], 0, 6) * 5)[..., None, None] + detector
    path = tmp_path / "scan.npy"
    np.save(path, values)
    with load(path, backend=backend, verbose=False) as data:
        positions = torch.as_tensor(coordinates, device=backend)
        corrected = resample_scan(data, positions)
        np.testing.assert_allclose(corrected.cpu().numpy(), expected, rtol=1e-6, atol=1e-5)
        destination = np.empty(expected.shape, np.float32)
        actual = resample_scan(data, positions, output=destination)
        assert actual is destination
        np.testing.assert_array_equal(actual, corrected.cpu().numpy())
