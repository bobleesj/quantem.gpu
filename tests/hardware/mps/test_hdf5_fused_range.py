"""Full original-count and dtype checks for the encoder's fused range flag."""

import numpy as np
import pytest
import torch

from quantem.gpu import io
from tests.hardware.mps.test_mps_buffer_release import _write_bslz4_master
from tests.hardware.mps.test_streamed_h5_ans import _decoded


@pytest.fixture(autouse=True)
def physical_mps():
    pytest.importorskip("Metal")
    if not torch.backends.mps.is_available():
        pytest.skip("Physical MPS device required.")


@pytest.mark.parametrize("maximum", [255, 256, 65535])
@pytest.mark.parametrize("batch", [0, 1])
def test_range_flag_preserves_late_and_constant_counts(tmp_path, maximum, batch):
    values = np.random.default_rng(41).integers(0, 33, (32768, 4, 8), dtype=np.uint16)
    first = batch * 16384
    # A constant stream returns early from model selection. The last outlier
    # is a flagged detector pixel whose original count must still be preserved.
    values[first:first + 512, 0, 0] = maximum
    values[first + 16383, 2, 5] = maximum
    mask = np.zeros((4, 8), np.uint8)
    mask[2, 5] = 1
    master = _write_bslz4_master(tmp_path, "fused-range", values, pixel_mask=mask)
    saved = tmp_path / "saved.qem"
    expected = values.reshape(128, 256, 4, 8)
    with io.load(master, backend="mps", scan_shape=(128, 256),
                 hot_pixel_correction="none", verbose=False) as loaded:
        assert loaded.dtype == np.dtype("uint8" if maximum <= 255 else "uint16")
        assert loaded.metadata["source_dtype"] == "uint16"
        assert loaded.metadata["file_counts_exact"]
        assert len(loaded.data.chunks) >= 2
        assert loaded.data._encoding_exceeds_uint8 is (maximum > 255)
        np.testing.assert_array_equal(_decoded(loaded.data), expected)
        io.save(saved, loaded)
    assert loaded.data.is_released and not loaded.data.chunks
    with io.load(saved, backend="mps", verbose=False) as reopened:
        np.testing.assert_array_equal(_decoded(reopened.data), expected)
    # A new owner starts with a clear flag after a preceding high-count owner.
    values.fill(255)
    low = _write_bslz4_master(tmp_path, "fresh-low", values)
    with io.load(low, backend="mps", scan_shape=(128, 256),
                 hot_pixel_correction="none", verbose=False) as fresh:
        assert fresh.dtype == np.dtype("uint8")
        assert fresh.data._encoding_exceeds_uint8 is False
        np.testing.assert_array_equal(_decoded(fresh.data), values.reshape(128, 256, 4, 8))
