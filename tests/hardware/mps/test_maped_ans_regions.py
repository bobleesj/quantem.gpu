"""Metal ANS and bounded MAPED parity against a NumPy count fixture."""

import numpy as np
import pytest

pytest.importorskip("Metal")
torch = pytest.importorskip("torch")

from quantem.gpu import io
from quantem.gpu._maped.mps import _automatic_region_frames
from quantem.gpu.io.backends.mps._streamed import MPSStreamedCounts
from quantem.gpu.io.backends.mps.precision import upload
from quantem.gpu.maped import merge_to_scaled_h5


def test_mps_region_planner_uses_bounded_scan_rows():
    assert _automatic_region_frames((512, 512, 192, 192)) == 4096
    large = _automatic_region_frames((512, 512, 512, 512))
    assert 512 <= large <= 4096
    assert large % 512 == 0


def test_mps_ans_bounded_merge_preserves_counts_mask_and_late_regions(
    tmp_path, monkeypatch
):
    """Exact ANS input and scaled output agree past the first merge region."""
    shape = (4, 400, 2, 4)
    values = (
        np.arange(np.prod(shape), dtype=np.uint32).reshape(shape) * 17 % 997
    ).astype(np.uint16)
    valid = np.ones(shape[2:], bool)
    valid[0, 1] = False
    values[:, :, 0, 1] = np.uint16(65535)
    source = MPSStreamedCounts(shape, np.uint16, valid)
    raw = upload(values.reshape(-1, *shape[2:]))
    result = None
    try:
        monkeypatch.setattr(
            "quantem.gpu._maped.mps._automatic_region_frames",
            lambda shape: 1024,
        )
        source.append(raw)
        decoded = source.decode_scan_range_device(1019, 1031)
        try:
            np.testing.assert_array_equal(
                decoded.to_numpy(), values.reshape(-1, *shape[2:])[1019:1031]
            )
        finally:
            decoded.release()

        loaded = io.FourDSTEMData(
            source,
            {
                "working_shape": shape,
                "source_shape": shape,
                "pixel_mask": (~valid).astype(np.uint8),
                "representation": "ans",
                "lossless_exact": True,
            },
        )
        shifts = torch.zeros((1, 2), dtype=torch.float32, device="mps")
        result = merge_to_scaled_h5(
            [loaded], shifts, shifts, tmp_path / "merged_master.h5"
        )
        expected = values.copy()
        expected[:, :, ~valid] = 0
        expected[0] = 0
        expected[-1] = 0
        expected[:, 0] = 0
        expected[:, -1] = 0
        report = result.metadata["precision"]
        assert report["storage"] == "scaled_uint16"
        assert report["range_scope"] == "complete merged output"
        assert result.metadata["maped_merge"]["backend"] == "mps"
        assert result.metadata["maped_merge"]["region_frames"] == 1024
        summary = result.metadata["maped_summary"]
        assert summary["mean_bright_field"]["divisor"] == np.prod(shape[2:])
        assert summary["mean_bright_field"]["alignment_role"] == "real_space"
        assert summary["intensity_normalization"] == "none"
        assert result.metadata["maped_merge"]["real_space_shifts_row_column"] == [
            [0.0, 0.0]
        ]
        reopened_metadata = io.inspect(tmp_path / "merged_master.h5").metadata
        assert reopened_metadata["maped_summary"] == summary
        for index in (0, 401, 1100, np.prod(shape[:2]) - 1):
            np.testing.assert_allclose(
                result.data.frame(index),
                expected.reshape(-1, *shape[2:])[index],
                rtol=0,
                atol=report["scale"],
            )
    finally:
        raw.release()
        source.release()
        if result is not None:
            result.close()
