"""Metal ANS and bounded MAPED parity against a NumPy count fixture."""

import h5py
import hdf5plugin
import numpy as np
import pytest

pytest.importorskip("Metal")
torch = pytest.importorskip("torch")

from quantem.gpu import io
from quantem.gpu._maped.mps import _automatic_region_frames, _merge_regions
from quantem.gpu.io.backends.mps._streamed import MPSStreamedCounts
from quantem.gpu.io.backends.mps.precision import upload
from quantem.gpu.maped import merge


def _median_corrected(raw, pixel_mask):
    expected = raw.copy()
    height, width = pixel_mask.shape
    for row, column in np.argwhere(pixel_mask != 0):
        neighbors = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = row + dr, column + dc
                if (
                    (dr or dc)
                    and 0 <= rr < height
                    and 0 <= cc < width
                    and pixel_mask[rr, cc] == 0
                ):
                    neighbors.append(raw[..., rr, cc])
        expected[..., row, column] = np.median(
            np.stack(neighbors, axis=-1), axis=-1
        ).astype(raw.dtype)
    return expected


def test_mps_region_planner_uses_bounded_scan_rows():
    assert _automatic_region_frames((512, 512, 192, 192)) == 4096
    large = _automatic_region_frames((512, 512, 512, 512))
    assert 512 <= large <= 4096
    assert large % 512 == 0


def test_mps_region_merge_handles_shifted_regions_before_source_rows():
    """A positive row shift may put a complete early region above the source."""
    shape = (8, 4, 2, 2)
    values = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
    source = MPSStreamedCounts(shape, np.uint16)
    raw = upload(values.reshape(-1, *shape[2:]))
    try:
        source.append(raw)
        regions = list(
            _merge_regions(
                [source],
                np.asarray([[4.0, 0.0]], np.float32),
                np.zeros((1, 2), np.float32),
                scans_per_region=4,
            )
        )
        assert len(regions) == 8
        np.testing.assert_array_equal(regions[0][1].get(), 0)
    finally:
        for _, region in locals().get("regions", []):
            region.release()
        raw.release()
        source.release()


def test_h5_ans_defaults_to_gpu_median_hot_pixel_correction(tmp_path):
    dtype = np.uint16
    raw = (np.arange(2 * 5 * 6 * 8).reshape(2, 5, 6, 8) * 7 % 251).astype(dtype)
    mask = np.zeros((6, 8), np.uint8)
    mask[0, 0] = 16
    mask[2, 3] = 20
    raw[..., mask != 0] = np.iinfo(dtype).max
    expected = _median_corrected(raw, mask)
    path = tmp_path / "hot-pixels.h5"
    with h5py.File(path, "w") as handle:
        data = handle.require_group("entry/data")
        data.create_dataset(
            "data_000001",
            data=raw.reshape(-1, 6, 8),
            chunks=(1, 6, 8),
            **hdf5plugin.Bitshuffle(nelems=0, cname="lz4"),
        )
        detector = handle.require_group("entry/instrument/detector")
        detector_specific = detector.require_group("detectorSpecific")
        detector_specific["ntrigger"] = 10
        detector_specific["y_pixels_in_detector"] = 6
        detector_specific["x_pixels_in_detector"] = 8
        detector_specific["pixel_mask"] = mask

    loaded = io.load(
        path,
        backend="mps",
        scan_shape=(2, 5),
        apply_mask=False,
        verbose=False,
    )
    try:
        decoded = loaded.data.decode_scan_range_device(0, 10)
        try:
            np.testing.assert_array_equal(
                decoded.to_numpy().reshape(raw.shape), expected
            )
        finally:
            decoded.release()
        correction = loaded.metadata["hot_pixel_correction"]
        assert correction["method"] == "median"
        assert correction["pixel_count"] == 2
        assert correction["coordinates_row_column"] == [[0, 0], [2, 3]]
        assert correction["applied"] is True
        mean_dp = loaded.data.mean_dp_device()
        detector_mean = loaded.data.detector_mean_device()
        try:
            np.testing.assert_allclose(
                mean_dp.to_numpy(), expected.mean(axis=(0, 1)), rtol=0, atol=1e-5
            )
            np.testing.assert_allclose(
                detector_mean.to_numpy(), expected.mean(axis=(2, 3)), rtol=0, atol=1e-5
            )
        finally:
            mean_dp.release()
            detector_mean.release()
    finally:
        loaded.close()


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
        result = merge(
            [loaded], shifts, shifts, save_to=tmp_path / "merged_master.h5"
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
        np.testing.assert_allclose(
            result.read(scan_region=(1, 3, 150, 154)).cpu().numpy(),
            expected[1:3, 150:154],
            rtol=0,
            atol=report["scale"],
        )
    finally:
        raw.release()
        source.release()
        if result is not None:
            result.close()


def test_resident_read_matches_numpy_region():
    """The same bounded read contract restores an MPS Torch tensor."""
    shape = (5, 6, 4, 7)
    values = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape) % 251
    source = MPSStreamedCounts(shape, np.uint16)
    raw = upload(values.reshape(-1, *shape[2:]))
    try:
        source.append(raw)
        loaded = io.FourDSTEMData(
            source,
            {
                "working_shape": shape,
                "working_dtype": "uint16",
                "representation": "ans",
            },
        )
        observed = loaded.read(
            scan_region=(1, 4, 2, 6),
            detector_region=(1, 4, 3, 7),
        )
        assert observed.device.type == "mps"
        np.testing.assert_array_equal(
            observed.cpu().numpy(), values[1:4, 2:6, 1:4, 3:7]
        )
    finally:
        raw.release()
        source.release()
