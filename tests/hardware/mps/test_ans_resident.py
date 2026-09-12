"""Metal tests for corrected ANS residency and bounded public reads."""

import h5py
import hdf5plugin
import numpy as np
import pytest

pytest.importorskip("Metal")
pytest.importorskip("torch")

from quantem.gpu import io
from quantem.gpu.io.backends.mps._streamed import MPSStreamedCounts
from quantem.gpu.io.backends.mps.precision import upload


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


def test_h5_ans_defaults_to_gpu_median_hot_pixel_correction(tmp_path):
    raw = (np.arange(2 * 5 * 6 * 8).reshape(2, 5, 6, 8) * 7 % 251).astype(
        np.uint16
    )
    mask = np.zeros((6, 8), np.uint8)
    mask[0, 0] = 16
    mask[2, 3] = 20
    raw[..., mask != 0] = np.iinfo(np.uint16).max
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
    finally:
        loaded.close()


def test_resident_read_matches_numpy_region():
    """The public read contract restores requested MPS rows and columns."""
    shape = (5, 6, 4, 7)
    values = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape) % 251
    valid = np.ones(shape[2:], dtype=bool)
    valid[2, 4] = False
    source = MPSStreamedCounts(shape, np.uint16, valid)
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
        expected = values[1:4, 2:6, 1:4, 3:7].copy()
        expected[..., 1, 1] = 0
        np.testing.assert_array_equal(observed.cpu().numpy(), expected)
    finally:
        raw.release()
        source.release()
