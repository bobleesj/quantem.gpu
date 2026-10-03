"""Exact screening of uint32 detector files whose counts fit uint16."""

import h5py
import hdf5plugin
import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from quantem.gpu import screening


def write_counts(path, counts):
    """Write a small detector acquisition with per-pattern compression."""
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "entry/data/data", data=counts.reshape(-1, 32, 32),
            chunks=(1, 32, 32), **hdf5plugin.Bitshuffle(cname="lz4"),
        )
        handle["entry/instrument/detector/detectorSpecific/ntrigger"] = 16


def test_uint32_screening_preserves_measured_counts(tmp_path):
    """Wider on-disk storage must give the same exact detector products."""
    rows, columns = np.indices((32, 32))
    disk = ((rows - 16) ** 2 + (columns - 16) ** 2) < 8 ** 2
    counts = (10 + 500 * disk)[None, None] + np.arange(16).reshape(4, 4, 1, 1)
    outputs = []
    for dtype in (np.uint16, np.uint32):
        path = tmp_path / f"counts-{np.dtype(dtype).name}.h5"
        write_counts(path, counts.astype(dtype))
        outputs.append(screening.prepare(
            path, backend="cuda", cache=False, memory_budget_gb=1,
            rotation_steps=4,
        ))
    for field in (
        "mean_dp", "total_intensity", "bright_field", "annular_bright_field",
        "annular_dark_field", "dark_field", "com_row", "com_col",
    ):
        np.testing.assert_array_equal(getattr(outputs[0], field), getattr(outputs[1], field))
    np.testing.assert_array_equal(outputs[1].total_intensity, counts.sum(axis=(-2, -1)))
    assert outputs[1].metadata["parameters"]["narrowing_verified"]
    assert outputs[1].metadata["memory"]["dtype"] == "uint32"


def test_uint32_screening_does_not_truncate_large_counts(tmp_path):
    """An unrepresentable count stops screening before products are saved."""
    counts = np.ones((4, 4, 32, 32), dtype=np.uint32)
    counts[-1, -1, 16, 16] = 65536
    path = tmp_path / "high-counts.h5"
    write_counts(path, counts)
    with pytest.raises(ValueError, match="65536"):
        screening.prepare(path, backend="cuda", cache=False, memory_budget_gb=1)
