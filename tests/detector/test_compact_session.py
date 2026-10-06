"""Public detector workflow and unchanged host defaults, without CUDA allocation."""

import numpy as np

from quantem.gpu import detector


class _DeviceArray(np.ndarray):
    """Host stand-in that records the API's explicit device-to-host boundary."""

    downloads = 0

    def get(self):
        type(self).downloads += 1
        return np.asarray(self).copy()


def test_existing_numpy_workflow_keeps_shape_and_precision():
    """Single-acquisition NumPy callers retain float32 and exact uint64 outputs."""
    counts = np.arange(3 * 4 * 5 * 6, dtype=np.uint16).reshape(3, 4, 5, 6)
    counts[0, 0] = 65535
    session = detector.prepare(counts)
    mask = np.ones((5, 6), bool)
    image = session.masked_sum(mask)
    exact = session.masked_sum_exact(mask)
    assert image.dtype == np.float32 and image.shape == (3, 4)
    assert exact.dtype == np.uint64 and exact.shape == (3, 4)
    assert session.series_shape == ()
    np.testing.assert_array_equal(exact, counts.sum(axis=(-2, -1), dtype=np.uint64))
    np.testing.assert_array_equal(session.frame(0), counts[0, 0])


def test_streamed_host_patterns_use_the_native_decoder():
    """Reopened count streams expose the same owned point DP on host and device."""
    from quantem.gpu.detector.cuda.streamed_series import StreamedSeriesCompute

    source = StreamedSeriesCompute.__new__(StreamedSeriesCompute)
    source.n_frames = 12
    values = np.arange(30, dtype=np.uint16).reshape(5, 6).view(_DeviceArray)
    source.frame_native = lambda index: values
    _DeviceArray.downloads = 0
    from quantem.gpu.detector.session import DetectorSession

    session = DetectorSession.__new__(DetectorSession)
    session._backend = source
    actual = session.frame(7)
    assert _DeviceArray.downloads == 1
    np.testing.assert_array_equal(actual, values)
    actual.fill(0)
    assert values[-1, -1] == 29


def test_fractional_annuli_keep_float64_inclusive_boundaries():
    """Translated, boundary, empty and equal-radius masks match Euclidean math."""
    rows, cols = np.indices((192, 192), dtype=np.float64)
    for row, col, inner, outer in (
        (99.125, 92.375, 37.13, 88.37),
        (-3.125, 190.75, 39.13, 81.37),
        (96.25, 96.25, 0, 0),
        (0, 0, 1, 1),
        (96, 96, 0, 300),
    ):
        distance = np.hypot(rows - row, cols - col)
        expected = (distance >= inner) & (distance <= outer)
        actual = detector.detector_mask(
            (row, col), inner, outer, (192, 192), dtype=np.float64
        )
        np.testing.assert_array_equal(actual, expected)
