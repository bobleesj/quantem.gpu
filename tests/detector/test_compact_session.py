"""Public detector workflow and unchanged host defaults, without CUDA allocation."""

import numpy as np

from quantem.gpu import detector, io
from quantem.gpu._compact.source import CompactSeries
from quantem.gpu.io.load import LoadResult


class _DeviceArray(np.ndarray):
    """Host stand-in that records the API's explicit device-to-host boundary."""

    downloads = 0

    def get(self):
        type(self).downloads += 1
        return np.asarray(self).copy()


class _FixtureSeries(CompactSeries):
    """Only the query implementation is replaced by a small independent oracle."""

    shape = (2, 3, 4, 5, 6)
    scan_shape = (3, 4)
    det_shape = (5, 6)
    series_shape = (2,)
    n_frames = 12

    def __init__(self):
        self.counts = np.arange(np.prod(self.shape), dtype=np.uint16).reshape(
            self.shape
        )
        self.counts[0, 0, 0, 0, 0] = 65535
        self.valid = np.ones(self.det_shape, bool)
        self.valid[1, 2] = False
        self.last = {}
        self.requests = 0

    def masked_sum_native(self, mask, *, out=None):
        self.requests += 1
        expected = (self.counts * (mask & self.valid)).sum(
            axis=(-2, -1), dtype=np.uint32
        )
        if out is not None:
            out[...] = expected
            return out
        return expected.view(_DeviceArray)

    def frame_native(self, index, *, out=None):
        expected = self.counts.reshape(2, 12, 5, 6)[:, index].copy()
        if out is not None:
            out[...] = expected
            return out
        return expected.view(_DeviceArray)


def test_complete_series_native_workflow_without_host_handoff(monkeypatch):
    """One mask returns both complete acquisitions, and out is caller-owned."""
    source = _FixtureSeries()
    session = detector.prepare(LoadResult(source, {}))
    mask = np.ones((5, 6), bool)
    _DeviceArray.downloads = 0
    images = session.masked_sum(mask, output="native")
    snapshot = images.copy()
    destination = np.empty(source.shape[:3], np.uint32).view(_DeviceArray)
    shifted = mask.copy()
    shifted[:, :2] = False
    result = session.masked_sum_exact(shifted, output="native", out=destination)
    patterns = session.frame(7, output="native")
    assert _DeviceArray.downloads == 0
    assert source.requests == 2
    assert session.series_shape == (2,)
    assert result is destination
    np.testing.assert_array_equal(images, snapshot)
    np.testing.assert_array_equal(patterns, source.counts.reshape(2, 12, 5, 6)[:, 7])
    np.testing.assert_array_equal(
        result,
        (source.counts * (shifted & source.valid)).sum(axis=(-2, -1), dtype=np.uint32),
    )


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


def test_prepared_folder_uses_existing_io_load(tmp_path, monkeypatch):
    """The normal load API selects the compact loader without generic H5 decoding."""
    (tmp_path / "checkpoint.json").write_text(
        '{"format":"compact-prepared-series-v1"}'
    )
    import sys
    from types import SimpleNamespace

    source = _FixtureSeries()
    source.load_seconds = 0.25
    source.load_timing = {"stream_seconds": 0.20}
    calls = []

    def load_prepared(path, *, device):
        calls.append((path, device))
        return source

    monkeypatch.setitem(
        sys.modules,
        "quantem.gpu._compact.load",
        SimpleNamespace(load=load_prepared),
    )
    loaded = io.load(tmp_path, backend="cuda", device=1, verbose=False)
    assert calls == [(tmp_path, 1)]
    assert loaded.data is source
    assert loaded.metadata["series_shape"] == (2,)
    assert detector.prepare(loaded).series_shape == (2,)


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


def test_mutating_completed_outputs_cannot_change_next_detector(monkeypatch):
    """The real output-ownership path protects incremental state from callers."""
    import sys
    from contextlib import nullcontext
    from types import SimpleNamespace

    source = CompactSeries(0)
    source.shape = (2, 2, 2, 2, 2)
    source.det_shape = (2, 2)
    source.start = object()
    source.ready = SimpleNamespace(record=lambda: None, synchronize=lambda: None)
    # Replace only the accelerator operations and reduction, retaining the real
    # source's native result ownership, mutation and previous-image machinery.
    monkeypatch.setitem(
        sys.modules,
        "cupy",
        SimpleNamespace(
            empty_like=np.empty_like,
            copyto=np.copyto,
            cuda=SimpleNamespace(
                Device=lambda _: nullcontext(), get_elapsed_time=lambda *args: 0.0
            ),
        ),
    )
    source._destination = lambda out, shape, dtype: (
        np.empty(shape, dtype) if out is None else out
    )

    def integrate(mask):
        value = int((mask * np.array([[3, 5], [7, 11]])).sum())
        if source.previous is None:
            source.output.fill(value)
        else:
            source.output[...] = (
                source.output.astype(np.int64) + value - source.previous_value
            )
        source.previous = mask.copy()
        source.previous_value = value
        return source.output

    source.update = integrate
    first = source.masked_sum_native(np.ones((2, 2), bool))
    first.fill(0)
    second = source.masked_sum_native(np.array([[1, 1], [0, 1]], bool))
    np.testing.assert_array_equal(second, np.full((2, 2, 2), 19, np.uint32))
    second.fill(123)
    returned = source.masked_sum_native(np.ones((2, 2), bool), out=first)
    assert returned is first
    np.testing.assert_array_equal(returned, np.full((2, 2, 2), 26, np.uint32))
