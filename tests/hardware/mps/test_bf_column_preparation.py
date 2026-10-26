"""Exact BF columns read as stored and converted on the GPU prepare the same G(q, k), bit for bit."""

import platform

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(platform.system() != "Darwin", reason="Requires physical Apple GPU")
pytest.importorskip("Metal")
pytest.importorskip("mlx.core")

SCAN = (64, 64)
DETECTOR = (12, 13)
CENTER = (5.4, 6.1)
RADIUS = 4.3


def _selection():
    from quantem.gpu.direct_ptycho.brightfield import BrightfieldDisk

    rows, cols = np.nonzero(np.hypot(*np.mgrid[:DETECTOR[0], :DETECTOR[1]] - np.reshape(CENTER, (2, 1, 1))) <= RADIUS)
    return BrightfieldDisk(rows=rows.astype(np.int32), cols=cols.astype(np.int32), bright_field_center_px=CENTER,
                           bright_field_radius_px=RADIUS, bright_field_edge_radius_px=RADIUS,
                           detector_shape=DETECTOR)


def _counts(dtype):
    rng = np.random.default_rng(11)
    high = int(np.iinfo(dtype).max) + 1
    return rng.integers(0, high, (*SCAN, *DETECTOR)).astype(dtype)


def _exported(tmp_path, counts, selection, **kwargs):
    from quantem.gpu.direct_ptycho.mps.frames import MpsBfColumnFrames

    columns = counts.reshape(-1, *DETECTOR)[:, selection.rows, selection.cols].T
    path = tmp_path / f"columns.{counts.dtype.name}"
    np.ascontiguousarray(columns).tofile(path)
    return MpsBfColumnFrames(path, selection=selection, scan_shape=SCAN, dtype=counts.dtype, **kwargs), columns


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_columns_read_as_stored(tmp_path, monkeypatch, dtype) -> None:
    """Any order of requested pixels reads exactly the stored columns, in reads of one or several columns."""
    from quantem.gpu.direct_ptycho.mps import frames as frames_module

    selection = _selection()
    counts = _counts(dtype)
    frames, columns = _exported(tmp_path, counts, selection, dc_value=1.0)
    order = np.concatenate([np.arange(3, 9), [0], np.arange(selection.size - 1, 12, -1), [2, 1]])
    # three columns per read, so runs split across reads and threads
    monkeypatch.setattr(frames_module, "COLUMN_READ_BYTES", 3 * SCAN[0] * SCAN[1] * np.dtype(dtype).itemsize)
    out = np.empty((order.size, SCAN[0] * SCAN[1]), dtype)
    assert frames.columns_into(selection.rows[order], selection.cols[order], out) is out
    np.testing.assert_array_equal(out, columns[order])
    as_float = np.empty(out.shape, np.float32)
    frames.columns_float32_into(selection.rows[order], selection.cols[order], as_float)
    np.testing.assert_array_equal(as_float, columns[order].astype(np.float32))
    with pytest.raises(ValueError, match="C-contiguous"):
        frames.columns_into(selection.rows[order], selection.cols[order], out.astype(np.float32))


def test_detector_sum_is_exact(tmp_path, monkeypatch) -> None:
    """Exact columns without a stored sum or DC value sum every stored count in uint64 (here past float32's 2**24)."""
    from quantem.gpu.direct_ptycho.mps import frames as frames_module

    selection = _selection()
    counts = _counts(np.uint16)
    monkeypatch.setattr(frames_module, "COLUMN_READ_BYTES", 2 * SCAN[0] * SCAN[1] * 2)
    frames, columns = _exported(tmp_path, counts, selection)
    expected = np.zeros(DETECTOR, np.uint64)
    expected[selection.rows, selection.cols] = columns.sum(axis=1, dtype=np.uint64)
    assert expected.max() > 2**24
    assert frames.detector_sum.dtype == np.uint64
    np.testing.assert_array_equal(frames.detector_sum, expected)


@pytest.mark.parametrize("dc_value_override", [None, 2.5e5 + 0j])
def test_preparation_keeps_the_host_float32_transform(tmp_path, dc_value_override) -> None:
    """G(q, k) of exact columns is, bit for bit, the transform of the host float32 stack preparation used to build.

    The reference converts each chunk's stored columns to a C-contiguous float32 stack on the host, as preparation
    did before converting on the GPU, and transforms it with the same chunks. Chunks of five pixels, with the aperture
    closed on some, cover the compaction; without a DC value the chunks' host DC sums give the same DC value too.
    """
    import mlx.core as mx

    from quantem.gpu.direct_ptycho.mps.prepared import prepare_selection

    selection = _selection()
    counts = _counts(np.uint16)
    exported, columns = _exported(tmp_path, counts, selection, dc_value=1.0)
    prepared = prepare_selection(
        exported, scan_shape=SCAN, selection=selection, voltage_kV=300.0, semiangle_mrad=10.5,
        scan_sampling=(0.5, 0.5), detector_sampling_mrad=(3.0, 3.0), scan_detector_rotation_deg=7.0, chunk_bf=5,
        compact_inactive=True, dc_value_override=dc_value_override,
    )
    active = np.zeros(selection.size, bool)
    active[prepared.bf_storage_indices_np] = True
    assert 0 < active.sum() < selection.size
    spectra, dc_sums = [], []
    for start in range(0, selection.size, 5):
        chunk = np.arange(start, min(start + 5, selection.size))
        if dc_value_override is None:
            stack = columns[chunk].astype(np.float32)
            dc_sums.append(stack.sum(axis=1, dtype=np.float32))
            stack = stack[active[chunk]]
        else:
            stack = columns[chunk[active[chunk]]].astype(np.float32)
        if stack.shape[0]:
            spectra.append(np.asarray(mx.fft.rfft2(mx.array(stack.reshape(-1, *SCAN)))))
    np.testing.assert_array_equal(np.asarray(prepared.g_qk), np.concatenate(spectra))
    if dc_value_override is None:
        assert prepared.dc_value == complex(np.concatenate(dc_sums).astype(np.complex64).mean())
