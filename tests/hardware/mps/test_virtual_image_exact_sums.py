"""Metal virtual-image sums of saturated frames stay exact past 2^31 and 2^32.

``MetalVirtualImage`` once summed uint8/uint16 counts in 32 bits: masked sums
in int32 per frame, the detector total in int32 atomics, and selected-frame
means in uint32 per pixel. A saturated uint16 frame passes 2^31 above 32,768
pixels and a pixel summed over 65,538 saturated frames passes 2^32, so these
wrapped. The expected values are exact integer sums from NumPy.
"""

import numpy as np
import pytest

from tests.hardware.mps.test_virtual_image_ownership import _chunks, _release

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(
    not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
    reason="needs an Apple GPU",
)


@pytest.mark.parametrize(
    "dtype, frames, chunk_frames",
    # uint16 sums 70,000 frames in one chunk: past 2^32 per pixel inside one
    # mean pass. uint8 sums 8.5 million frames in two chunks of four-pixel blocks.
    [(np.uint16, 70_000, 70_000), (np.uint8, 8_500_000, 4_250_000)],
)
def test_detector_total_and_mean_of_saturated_frames(dtype, frames, chunk_frames):
    from quantem.gpu import detector
    from quantem.gpu.resident.mps.frames import ChunkedFrames

    maximum = int(np.iinfo(dtype).max)
    chunks = _chunks(np.full((frames, 1, 4), maximum, dtype=dtype), chunk_frames)
    try:
        chunked = ChunkedFrames(chunks)
        exact = np.full((1, 4), frames * maximum, dtype=np.uint64)
        assert int(exact[0, 0]) > 2**31
        np.testing.assert_array_equal(chunked.vi.detector_sum_exact(), exact)
        np.testing.assert_array_equal(chunked.detector_sum, exact)
        session = detector.prepare(chunked)
        np.testing.assert_array_equal(session.mean_dp(), (exact / frames).astype(np.float32))
        if dtype == np.uint16:
            mean = np.empty(4, dtype=np.float32)
            np.divide(exact.reshape(-1), float(frames), out=mean, casting="unsafe")
            np.testing.assert_array_equal(session.reduce_frames(np.arange(frames), "mean"), mean.reshape(1, 4))
    finally:
        _release(chunks)


def test_masked_sums_of_saturated_frames():
    from quantem.gpu import detector
    from quantem.gpu.resident.mps.frames import ChunkedFrames

    # Fewer than 96 detector rows: no binned sidecar, every sum reads native counts.
    shape = (64, 1024)
    values = np.full((2, *shape), np.iinfo(np.uint16).max, dtype=np.uint16)
    chunks = _chunks(values, 2)
    try:
        session = detector.prepare(ChunkedFrames(chunks))
        rows = np.indices(shape)[0]
        noisy = np.random.default_rng(5).random(shape) < 0.6
        # 64 row spans (span kernel), the full detector, a dense noisy mask (total
        # minus complement) and its sparse complement (dense-mask kernel).
        for mask in (rows < 48, np.ones(shape, dtype=bool), noisy, ~noisy):
            exact = values[:, mask].sum(axis=1, dtype=np.uint64).reshape(session.scan_shape)
            np.testing.assert_array_equal(session.masked_sum_exact(mask), exact)
            np.testing.assert_array_equal(session.masked_sum(mask), exact.astype(np.float32))
        assert int(values[:, noisy].sum(dtype=np.uint64)) // 2 > 2**31
    finally:
        _release(chunks)


def _center_of_mass_reference(values, mask):
    """CUDA's definition: exact integer moments over the exact total, divided in float64, 0 for an empty pattern."""
    working = values.astype(np.uint64) * mask
    rows, cols = np.indices(mask.shape, dtype=np.uint64)
    total = working.sum(axis=(1, 2)).astype(np.float64)
    centers = []
    for weights in (cols, rows):
        moment = (working * weights).sum(axis=(1, 2)).astype(np.float64)
        centers.append(np.divide(moment, total, out=np.zeros_like(total), where=total > 0).astype(np.float32))
    return centers


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32])
def test_center_of_mass_divides_exact_moments_like_cuda(dtype):
    from quantem.gpu.resident.mps.frames import ChunkedFrames

    shape = (40, 52)
    values = np.random.default_rng(13).integers(0, 250, (300, *shape)).astype(dtype)
    values[7] = 0
    chunks = _chunks(values, 128)
    try:
        frames = ChunkedFrames(chunks)
        rows, cols = np.indices(shape)
        disk = np.hypot(rows - 19.5, cols - 25.5) <= 15
        for mask in (None, disk):
            expected = _center_of_mass_reference(values, np.ones(shape, bool) if mask is None else disk)
            com_col, com_row = frames.vi.center_of_mass(mask)
            np.testing.assert_array_equal(com_col, expected[0])
            np.testing.assert_array_equal(com_row, expected[1])
        assert com_col[7] == com_row[7] == 0
    finally:
        _release(chunks)


@pytest.mark.parametrize("runner", ["chunked", "torch-mps"])
def test_dense_mean_pattern_and_frame_reductions(runner):
    """Totals above 2^24: the mean divides in float64 and rounds once; integer sums stay exact uint64."""
    from quantem.gpu import detector
    from quantem.gpu.resident.mps.frames import ChunkedFrames

    counts = np.random.default_rng(81).integers(52_000, 65_536, (324, 6, 8)).astype(np.uint16)
    chunks = _chunks(counts, 200) if runner == "chunked" else []
    try:
        data = ChunkedFrames(chunks) if runner == "chunked" else torch.from_numpy(counts.reshape(18, 18, 6, 8)).to("mps")
        session = detector.prepare(data)
        total = counts.sum(axis=0, dtype=np.uint64)
        expected = (total / 324).astype(np.float32)
        assert np.any(total.astype(np.float32) / np.float32(324) != expected)
        np.testing.assert_array_equal(session.mean_dp(), expected)
        indices = np.arange(10, 310)
        selected = counts[indices].sum(axis=0, dtype=np.uint64)
        np.testing.assert_array_equal(session.reduce_frames(indices, "mean"), (selected / 300).astype(np.float32))
        result = session.reduce_frames(indices, "sum")
        assert result.dtype == np.uint64
        np.testing.assert_array_equal(result, selected)
    finally:
        _release(chunks)
