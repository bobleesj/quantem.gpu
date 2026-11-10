"""A finished ``.qem`` save reports its wall time and sizes, shows them as a table, and reopens unchanged.

The summary comes from the file size and the array geometry alone, so it costs
no detector read. The table itself is checked everywhere; the encoded
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


def test_summary_is_a_table_of_time_and_sizes() -> None:
    """The summary is built from the recorded numbers alone, in text and as an HTML table."""
    from quantem.gpu.io.save import SaveResult

    result = SaveResult("measurements.qem", "cuda", complete=True, elapsed_s=0.25,
                        logical_bytes=4 * 2**20, file_bytes=3 * 2**20)
    text, html = repr(result), result._repr_html_()
    for label in ("Saved", "Save time", "Uncompressed array", "Lossless QEM", "3.00 MiB", "25.0% smaller"):
        assert label in text and label in html
    assert "<table" in html


def test_summary_says_when_the_file_is_larger_than_the_array() -> None:
    """Four detector pixels compress to less than the file's own metadata."""
    from quantem.gpu.io.save import SaveResult

    result = SaveResult("small.qem", "cuda", complete=True, elapsed_s=0.01, logical_bytes=4, file_bytes=900)
    assert "larger than uncompressed" in repr(result)


def test_hdf5_save_summary_has_no_sizes() -> None:
    from quantem.gpu.io.save import SaveResult

    result = SaveResult("scan_master.h5", "cuda", complete=True)
    assert "Saved" in repr(result) and "Lossless QEM" not in repr(result)


@pytest.mark.parametrize("dtype", [np.uint16, np.float32])
def test_encoded_resident_save_reports_time_and_sizes(tmp_path, dtype):
    backend = os.environ.get("QEM_TEST_BACKEND")
    if backend not in {"cuda", "mps"}:
        pytest.skip("Set QEM_TEST_BACKEND=cuda or mps on physical hardware.")
    values = _measurements(dtype)
    np.save(tmp_path / "measurements.npy", values)
    path = tmp_path / "measurements.qem"
    with io.load(tmp_path / "measurements.npy", device=backend) as loaded:
        result = io.save(path, loaded)
    assert (result.complete, result.backend) == (True, backend)
    assert result.logical_bytes == values.nbytes
    assert result.file_bytes == path.stat().st_size
    assert "Lossless QEM" in repr(result)
    with io.load(path, device=backend) as reopened:
        restored = reopened.read().cpu().numpy()
    assert restored.dtype == values.dtype and restored.tobytes() == values.tobytes()
