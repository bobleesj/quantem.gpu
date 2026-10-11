"""A finished ``.qem`` save reports its wall time and sizes, shows them as a table, and reopens unchanged.

The summary comes from the file size and the array geometry alone, so it costs
no detector read. The host reference writer runs everywhere; the encoded
residents run where ``QEM_TEST_BACKEND`` names a physical accelerator.
"""

import os

import numpy as np
import pytest

from quantem.gpu import io


def _measurements(dtype):
    """Small scan with distinct values; the float one is fractional, negative and holds a signed zero."""
    values = np.arange(4 * 5 * 16 * 16).reshape(4, 5, 16, 16).astype(dtype)
    if dtype == np.float32:
        values = values / 8 - 100
        values[0, 0, 0, 0] = -0.0
    return values


@pytest.mark.parametrize("dtype", [np.uint16, np.float32])
def test_host_qem_save_reports_time_and_sizes(tmp_path, dtype):
    values = _measurements(dtype)
    path = tmp_path / "measurements.qem"
    result = io.save(path, values, backend="cpu")
    assert (result.complete, result.backend, result.path) == (True, "cpu", str(path))
    assert result.logical_bytes == values.nbytes
    assert result.file_bytes == path.stat().st_size
    assert result.elapsed_seconds > 0
    text, html = repr(result), result._repr_html_()
    megabytes = f"{path.stat().st_size / 2**20:.2f} MiB"
    for label in ("Saved", "Save time", "Uncompressed array", "Lossless QEM", megabytes):
        assert label in text and label in html
    assert "<table" in html
    with io.load(path, backend="cpu") as reopened:
        assert reopened.data.tobytes() == values.tobytes()


def test_summary_says_when_the_file_is_larger_than_the_array(tmp_path):
    """Four detector pixels compress to less than the file's own metadata."""
    result = io.save(tmp_path / "small.qem", np.zeros((1, 1, 2, 2), dtype=np.uint8), backend="cpu")
    assert result.file_bytes > result.logical_bytes == 4
    assert "larger than uncompressed" in repr(result)


def test_hdf5_save_summary_has_no_sizes(tmp_path):
    result = io.save(tmp_path / "scan_master.h5", _measurements(np.uint16), backend="cpu")
    assert (result.complete, result.file_bytes, result.elapsed_seconds) == (True, None, None)
    assert "Saved" in repr(result) and "Lossless QEM" not in repr(result)


@pytest.mark.parametrize("dtype", [np.uint16, np.float32])
def test_encoded_resident_save_reports_time_and_sizes(tmp_path, dtype):
    backend = os.environ.get("QEM_TEST_BACKEND")
    if backend not in {"cuda", "mps"}:
        pytest.skip("Set QEM_TEST_BACKEND=cuda or mps on physical hardware.")
    values = _measurements(dtype)
    np.save(tmp_path / "measurements.npy", values)
    path = tmp_path / "measurements.qem"
    with io.load(tmp_path / "measurements.npy", backend=backend) as loaded:
        result = io.save(path, loaded)
    assert (result.complete, result.backend) == (True, backend)
    assert result.logical_bytes == values.nbytes
    assert result.file_bytes == path.stat().st_size
    assert "Lossless QEM" in repr(result)
    with io.load(path, backend=backend) as reopened:
        restored = reopened.read().cpu().numpy()
    assert restored.dtype == values.dtype and restored.tobytes() == values.tobytes()
