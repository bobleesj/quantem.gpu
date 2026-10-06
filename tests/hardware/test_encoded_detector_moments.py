"""Encoded CUDA and MPS acquisitions give exact detector totals, moments and centres of mass.

Both backends are compared with the same NumPy reference of the stored counts,
so a pass on each machine means CUDA and MPS return identical values.
"""

import os

import numpy as np
import pytest

from quantem.gpu import detector, dpc, io
from tests.hardware.test_screening_products import write_master

SCAN_SHAPE = (32, 32)
DETECTOR_SHAPE = (24, 28)
FLAGGED = ([2, 17], [3, 25])


def _backend():
    backend = os.environ.get("QEM_TEST_BACKEND")
    if backend not in {"cuda", "mps"}:
        pytest.skip("Set QEM_TEST_BACKEND=cuda or mps on physical hardware.")
    return backend


def _counts(seed):
    """Poisson counts of a wandering bright disk, an empty pattern and one full-scale count."""
    rng = np.random.default_rng(seed)
    scan_rows, scan_cols = np.indices(SCAN_SHAPE)
    rows, cols = np.indices(DETECTOR_SHAPE)
    shift_row = 1.5 * np.sin(scan_rows / 4.0)[..., None, None]
    shift_col = 1.2 * np.cos(scan_cols / 5.0)[..., None, None]
    disk = np.hypot(rows - 11.4 - shift_row, cols - 13.7 - shift_col) < 6.5
    counts = rng.poisson(np.where(disk, 60.0, 1.5)).astype(np.uint16)
    counts[3, 4] = 0
    counts[5, 6, 10, 12] = np.iinfo(np.uint16).max
    counts[..., FLAGGED[0], FLAGGED[1]] = np.iinfo(np.uint16).max
    return counts


def _load(tmp_path, counts, backend):
    """Load flagged pixels uncorrected, so the residents must exclude them."""
    pixel_mask = np.zeros(counts.shape[2:], dtype=np.uint32)
    pixel_mask[FLAGGED] = 1
    tmp_path.mkdir(parents=True, exist_ok=True)
    master = write_master(tmp_path, counts, pixel_mask)
    return io.load(master, backend=backend, hot_pixel_correction="none")


def _valid(shape=DETECTOR_SHAPE):
    valid = np.ones(shape, dtype=np.uint64)
    valid[FLAGGED] = 0
    return valid


def _reference_center_of_mass(counts, weights):
    """CUDA's definition: exact integer moments over the exact total, 0 for an empty pattern."""
    working = counts.astype(np.uint64) * weights
    rows, cols = np.indices(counts.shape[2:], dtype=np.uint64)
    denominator = np.maximum(working.sum(axis=(2, 3)).astype(np.float64), 1.0)
    com_row = (working * rows).sum(axis=(2, 3)).astype(np.float64) / denominator
    com_col = (working * cols).sum(axis=(2, 3)).astype(np.float64) / denominator
    return com_row.astype(np.float32), com_col.astype(np.float32)


def test_center_of_mass_total_and_moments_are_exact(tmp_path):
    backend = _backend()
    counts = _counts(11)
    valid = _valid()
    mask = np.zeros(DETECTOR_SHAPE, dtype=bool)
    mask[4:20, 6:22] = True
    rows = np.indices(DETECTOR_SHAPE)[0]
    with _load(tmp_path, counts, backend) as loaded:
        session = detector.prepare(loaded)
        working = counts.astype(np.uint64) * valid
        np.testing.assert_array_equal(session.detector_total(), working.sum(axis=(0, 1)))
        # 2000 + row exceeds one exact Metal pass for uint16 counts (weight 1024), so MPS splits it.
        for weights in (rows, 2000 + rows):
            np.testing.assert_array_equal(
                session.weighted_sum_exact(weights),
                (working * weights.astype(np.uint64)).sum(axis=(2, 3)),
            )
        for weights, selection in ((valid, None), (valid * mask, mask)):
            expected_row, expected_col = _reference_center_of_mass(counts, weights)
            com_row, com_col = session.center_of_mass(selection)
            assert com_row.dtype == com_col.dtype == np.float32
            np.testing.assert_array_equal(com_row, expected_row)
            np.testing.assert_array_equal(com_col, expected_col)
        assert com_row[3, 4] == com_col[3, 4] == 0.0
        expected_row, expected_col = _reference_center_of_mass(counts, valid)
        dpc_row, dpc_col = dpc.center_of_mass(loaded)
        np.testing.assert_array_equal(dpc_row, expected_row - float(np.mean(expected_row)))
        np.testing.assert_array_equal(dpc_col, expected_col - float(np.mean(expected_col)))


def test_weighted_sums_of_saturated_wide_frames_are_exact(tmp_path):
    """Saturated patterns on a detector wider than 1024 pixels: 32 pixels of count x column pass 2^31.

    The CUDA residual decoder adds 32 pixels' weight x count in int32; MPS splits
    weights into digits that keep each pass below 2^31, and CUDA must do the same.
    """
    backend = _backend()
    shape = (24, 1100)
    counts = np.random.default_rng(31).poisson(3.0, (8, 8, *shape)).astype(np.uint16)
    counts[:2] = np.iinfo(np.uint16).max
    counts[..., FLAGGED[0], FLAGGED[1]] = np.iinfo(np.uint16).max
    valid = _valid(shape)
    rows, cols = np.indices(shape)
    assert 32 * int(np.iinfo(np.uint16).max) * int(cols.max()) > 2**31
    with _load(tmp_path, counts, backend) as loaded:
        session = detector.prepare(loaded)
        working = counts.astype(np.uint64) * valid
        for weights in (cols, 2000 + rows, cols * 1_000_000):
            np.testing.assert_array_equal(
                session.weighted_sum_exact(weights),
                (working * weights.astype(np.uint64)).sum(axis=(2, 3)),
            )
        expected_row, expected_col = _reference_center_of_mass(counts, valid)
        com_row, com_col = session.center_of_mass()
        np.testing.assert_array_equal(com_row, expected_row)
        np.testing.assert_array_equal(com_col, expected_col)


def test_frame_reductions_exclude_flagged_pixels(tmp_path):
    """Selected-frame sums, maxima and means count flagged pixels as zero, like every other product."""
    backend = _backend()
    counts = _counts(41)
    indices = [0, 5, 37, 517, 1023]
    with _load(tmp_path, counts, backend) as loaded:
        session = detector.prepare(loaded)
        selected = counts.reshape(-1, *DETECTOR_SHAPE)[indices].astype(np.uint64) * _valid()
        total = selected.sum(axis=0)
        np.testing.assert_array_equal(session.reduce_frames_exact(indices), total)
        np.testing.assert_array_equal(session.reduce_frames_max(indices), selected.max(axis=0))
        # Integer counts: exact uint64 sum and maximum on CUDA and MPS alike; the mean divides in float64.
        for mode, expected in (("sum", total), ("max", selected.max(axis=0))):
            result = session.reduce_frames(indices, mode)
            assert result.dtype == np.uint64
            np.testing.assert_array_equal(result, expected)
        np.testing.assert_array_equal(session.reduce_frames(indices, "mean"), (total / len(indices)).astype(np.float32))


def test_frame_mean_divides_the_exact_total_once(tmp_path):
    """Totals above 2^24 round once: float64(total) / n to float32, on CUDA and MPS alike."""
    backend = _backend()
    counts = np.random.default_rng(71).integers(50_000, 65_536, (18, 18, *DETECTOR_SHAPE)).astype(np.uint16)
    indices = np.arange(10, 310)
    with _load(tmp_path, counts, backend) as loaded:
        session = detector.prepare(loaded)
        total = (counts.reshape(-1, *DETECTOR_SHAPE)[indices].astype(np.uint64) * _valid()).sum(axis=0)
        assert int(total.max()) > 2**24
        expected = (total.astype(np.float64) / len(indices)).astype(np.float32)
        assert np.any(total.astype(np.float32) / np.float32(len(indices)) != expected)
        np.testing.assert_array_equal(session.reduce_frames(indices, "mean"), expected)


def test_mean_pattern_divides_the_exact_total_once(tmp_path):
    """Detector totals above 2^24: the mean pattern is float64(total) / n rounded once, on CUDA and MPS."""
    backend = _backend()
    counts = np.random.default_rng(83).integers(52_000, 65_536, (18, 18, *DETECTOR_SHAPE)).astype(np.uint16)
    with _load(tmp_path, counts, backend) as loaded:
        total = (counts.astype(np.uint64) * _valid()).sum(axis=(0, 1))
        expected = (total / 324).astype(np.float32)
        assert np.any(total.astype(np.float32) / np.float32(324) != expected)
        np.testing.assert_array_equal(detector.prepare(loaded).mean_dp(), expected)


def test_sessions_report_the_flagged_detector_pixels(tmp_path):
    """A viewer builds its masks from the session's validity, so its own sums agree with exact products."""
    backend = _backend()
    loaded = [_load(tmp_path / f"acquisition-{index}", _counts(61 + index), backend) for index in range(2)]
    try:
        valid = _valid().astype(bool)
        alone = detector.prepare(loaded[0])
        reported = alone.detector_validity
        np.testing.assert_array_equal(reported, valid)
        reported[...] = False
        np.testing.assert_array_equal(alone.detector_validity, valid)
        np.testing.assert_array_equal(detector.prepare(loaded).detector_validity, np.stack([valid, valid]))
        assert detector.prepare(np.ones((2, 2, *DETECTOR_SHAPE), np.uint16)).detector_validity is None
    finally:
        for item in loaded:
            item.close()


def test_series_answers_equal_each_acquisition_alone(tmp_path):
    backend = _backend()
    counts = [_counts(21), _counts(22)]
    loaded = [_load(tmp_path / f"acquisition-{index}", values, backend) for index, values in enumerate(counts)]
    try:
        series = detector.prepare(loaded)
        alone = [detector.prepare(item) for item in loaded]
        masks = np.stack([
            detector.detector_mask((11.4, 13.7), 0.0, 6.5, DETECTOR_SHAPE),
            detector.detector_mask((11.4, 13.7), 6.5, np.inf, DETECTOR_SHAPE),
        ])
        assert series.series_shape == (2,)
        for index in (0, 517, 1023):
            np.testing.assert_array_equal(series.frame(index), np.stack([item.frame(index) for item in alone]))
        np.testing.assert_array_equal(
            series.masked_sum_exact(masks[0]), np.stack([item.masked_sum_exact(masks[0]) for item in alone])
        )
        np.testing.assert_array_equal(
            series.masked_sums_exact(masks), np.stack([item.masked_sums_exact(masks) for item in alone], axis=1)
        )
        np.testing.assert_array_equal(series.mean_dp(), np.stack([item.mean_dp() for item in alone]))
        np.testing.assert_array_equal(series.detector_total(), np.stack([item.detector_total() for item in alone]))
        np.testing.assert_array_equal(
            series.detector_total()[1], (counts[1].astype(np.uint64) * _valid()).sum(axis=(0, 1))
        )
    finally:
        for item in loaded:
            item.close()
