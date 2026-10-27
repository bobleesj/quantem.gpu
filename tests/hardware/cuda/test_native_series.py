"""Exact joint native queries across dense and independently encoded sources."""

import numpy as np
import pytest
from quantem.gpu import io, detector


def test_uint64_series_sum_never_truncates_high_counts():
    cp = pytest.importorskip("cupy")
    if cp.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("CUDA device required")
    from quantem.gpu.resident.cuda.counts import StreamedCounts

    source = StreamedCounts((1, 1, 257, 257), "uint16")
    source.append(cp.full((1, 257, 257), 65535, cp.uint16))
    session = detector.prepare([source, source])
    value = session.masked_sum(np.ones((257, 257), bool), output="native")
    assert value.dtype == cp.uint64
    np.testing.assert_array_equal(
        value.get(), np.full((2, 1, 1), 257 * 257 * 65535, np.uint64)
    )


def test_uint32_series_display_rounding_preserves_exact_native_sums():
    """Display sums round once at float32 ties; native count sums stay exact."""
    cp = pytest.importorskip("cupy")
    if cp.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("CUDA device required")
    from quantem.gpu.resident.cuda.counts import StreamedCounts

    totals = np.array(
        [0, 1, 2**24 - 1, 2**24 + 1, 2**24 + 3,
         2**31 - 65, 2**31 + 129, 192 * 192 * 65535],
        dtype=np.uint32,
    )
    counts = np.zeros((8, 192 * 192), dtype=np.uint16)
    for row, total in zip(counts, totals, strict=True):
        full, remainder = divmod(int(total), 65535)
        row[:full] = 65535
        if remainder:
            row[full] = remainder
    source = StreamedCounts((2, 4, 192, 192), "uint16")
    source.append(cp.asarray(counts.reshape(8, 192, 192)))
    session = detector.prepare([source, source])
    mask = np.ones((192, 192), dtype=bool)
    native = session.masked_sum(mask, output="native")
    assert native.dtype == cp.uint32
    expected = np.broadcast_to(totals.reshape(2, 4), (2, 2, 4))
    np.testing.assert_array_equal(native.get(), expected)
    displayed = session.masked_sum(mask)
    np.testing.assert_array_equal(displayed.view(np.uint32), expected.astype(np.float32).view(np.uint32))
    session.masked_sum(np.zeros_like(mask))
    np.testing.assert_array_equal(displayed.view(np.uint32), expected.astype(np.float32).view(np.uint32))
    np.testing.assert_array_equal(session.masked_sum_exact(mask), expected)


def test_series_of_dense_arrays_is_refused():
    """A series joins acquisitions that io.load encoded; dense arrays are queried one at a time."""
    cp = pytest.importorskip("cupy")
    if cp.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("CUDA device required")
    data = cp.zeros((2, 2, 4, 4), cp.uint16)
    with pytest.raises(TypeError, match="streamed count sources"):
        detector.prepare([data, data])


def test_original_h5_joint_queries_preserve_raw_counts(tmp_path):
    import h5py

    cp = pytest.importorskip("cupy")
    if cp.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("CUDA device required")
    raw = (np.arange(32 * 6, dtype=np.uint16) % 12).reshape(32, 6)
    raw[:, 2] = 65535
    file = tmp_path / "original.h5"
    with h5py.File(file, "w") as handle:
        handle["entry/data/data"] = raw.reshape(4, 8, 2, 3)
    loaded = io.load(file, backend="cuda", apply_mask=False)
    session = detector.prepare([loaded, loaded])
    for index in (0, 7, 31):
        np.testing.assert_array_equal(
            session.frame(index, output="native").get(),
            np.broadcast_to(raw[index].reshape(2, 3), (2, 2, 3)),
        )
    np.testing.assert_array_equal(
        session.masked_sum(np.ones((2, 3), bool), output="native").get(),
        np.broadcast_to(raw.sum(-1).reshape(4, 8), (2, 4, 8)),
    )
