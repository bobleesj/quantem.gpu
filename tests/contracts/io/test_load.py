from types import SimpleNamespace

import numpy as np
import pytest

from quantem.gpu.device import cuda_runtime as memory_module


def test_get_metadata_reports_detector_source_dtype(tmp_path) -> None:
    """Metadata inspection records the stored count dtype before load transforms."""
    import h5py

    from quantem.gpu.formats.hdf5.master import get_metadata

    master = tmp_path / "fixture.h5"
    with h5py.File(master, "w") as h5_file:
        data = h5_file.create_dataset(
            "entry/data/data",
            shape=(4, 6, 6),
            dtype=np.uint16,
        )
        data.attrs["scan_shape"] = (2, 2)
        data.attrs["det_shape"] = (6, 6)

    metadata = get_metadata(str(master))

    assert metadata["source_dtype"] == "uint16"
    assert metadata["detector_shape"] == (6, 6)


def test_load_rejects_unknown_hot_pixel_correction() -> None:
    from importlib import import_module

    load_module = import_module("quantem.gpu.io.load")
    with pytest.raises(
        ValueError,
        match="hot_pixel_correction must be 'median', 'zero', or 'none'",
    ):
        load_module.load(
            "scan_master.h5",
            hot_pixel_correction="interpolate",
            verbose=False,
        )


def test_get_libc_returns_none_when_posix_fadvise_is_unavailable(monkeypatch) -> None:
    """macOS libc exists but does not expose Linux posix_fadvise."""
    import ctypes
    import ctypes.util

    monkeypatch.setattr(ctypes.util, "find_library", lambda _name: "libc.dylib")
    monkeypatch.setattr(ctypes, "CDLL", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(memory_module, "_LIBC", None)

    assert memory_module._get_libc() is None
    assert memory_module._LIBC is False


def test_apply_scan_shape_supports_serpentine_order() -> None:
    """Full flat scans can be unflattened with odd scan rows reversed."""
    from importlib import import_module
    selection = import_module("quantem.gpu.io.selection")

    data = np.arange(12, dtype=np.uint16).reshape(6, 1, 2)

    result = selection._apply_scan_shape(
        data,
        explicit=(2, 3),
        meta={},
        scan_order="serpentine",
    )

    expected = np.asarray(
        [
            [[[0, 1]], [[2, 3]], [[4, 5]]],
            [[[10, 11]], [[8, 9]], [[6, 7]]],
        ],
        dtype=np.uint16,
    )
    np.testing.assert_array_equal(result, expected)


def test_load_rejects_unknown_scan_order() -> None:
    """Unknown flattened scan order names should fail before any IO starts."""
    from importlib import import_module
    selection = import_module("quantem.gpu.io.selection")

    with pytest.raises(ValueError, match="scan_order must be"):
        selection._normalize_scan_order("zigzag")


def _numpy_resampled_scan_crop_reference(
    data: np.ndarray,
    *,
    source_scan_region: tuple[int, int, int, int],
    target_scan_region: tuple[int, int, int, int],
    scan_shift_row_col: tuple[float, float],
) -> np.ndarray:
    """Small NumPy reference for CUDA scan-space bilinear resampling."""
    row_start, row_stop, col_start, col_stop = target_scan_region
    source_row_start, _source_row_stop, source_col_start, _source_col_stop = (
        source_scan_region
    )
    shift_row, shift_col = scan_shift_row_col
    out = np.empty(
        (
            row_stop - row_start,
            col_stop - col_start,
            data.shape[-2],
            data.shape[-1],
        ),
        dtype=np.float32,
    )
    for out_row in range(out.shape[0]):
        src_row = row_start + out_row + shift_row - source_row_start
        src_row = np.clip(src_row, 0.0, max(float(data.shape[0]) - 1.001, 0.0))
        r0 = int(np.floor(src_row))
        r1 = min(r0 + 1, data.shape[0] - 1)
        wr = np.float32(src_row - r0)
        for out_col in range(out.shape[1]):
            src_col = col_start + out_col + shift_col - source_col_start
            src_col = np.clip(src_col, 0.0, max(float(data.shape[1]) - 1.001, 0.0))
            c0 = int(np.floor(src_col))
            c1 = min(c0 + 1, data.shape[1] - 1)
            wc = np.float32(src_col - c0)
            out[out_row, out_col] = (
                (1.0 - wr)
                * ((1.0 - wc) * data[r0, c0] + wc * data[r0, c1])
                + wr * ((1.0 - wc) * data[r1, c0] + wc * data[r1, c1])
            )
    return out


def test_resample_scan_crop_matches_numpy_reference() -> None:
    """The public resident-array resampler should match explicit bilinear math."""
    cp = pytest.importorskip("cupy")
    try:
        device_count = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError as error:
        pytest.skip(f"CUDA resampling requires an available runtime: {error}")
    if device_count == 0:
        pytest.skip("CUDA resampling requires a visible CUDA device.")
    from quantem.gpu.io.resample import resample_scan_crop

    data_np = np.arange(5 * 6 * 2 * 3, dtype=np.uint16).reshape(5, 6, 2, 3)
    source_region = (10, 15, 20, 26)
    target_region = (11, 14, 21, 25)
    shift = (0.35, -0.20)

    got = resample_scan_crop(
        cp.asarray(data_np),
        source_scan_region=source_region,
        target_scan_region=target_region,
        scan_shift_row_col=shift,
    )
    cp.cuda.get_current_stream().synchronize()
    expected = _numpy_resampled_scan_crop_reference(
        data_np.astype(np.float32),
        source_scan_region=source_region,
        target_scan_region=target_region,
        scan_shift_row_col=shift,
    )

    np.testing.assert_allclose(cp.asnumpy(got), expected, rtol=1.0e-6, atol=5.0e-5)
    assert got.dtype == cp.float32

    strided_np = data_np[:, :, 1:2, :]
    strided_got = resample_scan_crop(
        cp.asarray(data_np)[:, :, 1:2, :],
        source_scan_region=source_region,
        target_scan_region=target_region,
        scan_shift_row_col=shift,
    )
    cp.cuda.get_current_stream().synchronize()
    strided_expected = _numpy_resampled_scan_crop_reference(
        strided_np.astype(np.float32),
        source_scan_region=source_region,
        target_scan_region=target_region,
        scan_shift_row_col=shift,
    )
    np.testing.assert_allclose(
        cp.asnumpy(strided_got),
        strided_expected,
        rtol=1.0e-6,
        atol=5.0e-5,
    )


@pytest.mark.parametrize(
    "selector",
    [
        {"scan_region": (0, 1, 0, 1)},
    ],
)
def test_selective_scan_loading_rejects_cpu_backend(monkeypatch, selector) -> None:
    """The public CPU loader must not pretend to implement selective GPU IO."""
    from importlib import import_module
    load_module = import_module("quantem.gpu.io.load")

    monkeypatch.setattr("quantem.gpu.device.select.resolve_backend", lambda _backend: "cpu")

    with pytest.raises(RuntimeError, match="CUDA and MPS"):
        load_module.load(
            "scan_master.h5", representation="dense",
            backend="cpu",
            scan_shape=(1, 1),
            verbose=False,
            **selector,
        )


def test_load_region_keyword_is_not_supported() -> None:
    from importlib import import_module
    load_module = import_module("quantem.gpu.io.load")

    with pytest.raises(TypeError, match="unexpected keyword"):
        load_module.load(
            "scan_master.h5", representation="dense",
            region=(0, 1, 0, 1),
            verbose=False,
        )


def test_pinned_buffer_release_prunes_redundant_smaller_size(monkeypatch) -> None:
    small_array = np.zeros(80, dtype=np.uint8)
    large_array = np.zeros(100, dtype=np.uint8)
    small = {"arr": small_array, "size": 80, "free": True, "addr": 1}
    large = {"arr": large_array, "size": 100, "free": False, "addr": 2}
    released = []
    monkeypatch.setattr(memory_module, "_PINNED_BUFS", [small, large])

    def unregister(entry) -> bool:
        released.append(entry["size"])
        return True

    monkeypatch.setattr(memory_module, "_unregister_pinned_entry", unregister)

    memory_module._release_pinned(large_array[:90])

    assert memory_module._PINNED_BUFS == [large]
    assert large["free"] is True
    assert released == [80]
    assert small == {}


def test_large_pinned_registration_rounds_to_bounded_reuse_class() -> None:
    from quantem.gpu.device.cuda_runtime import _pinned_registration_size

    mebibyte = 1024 * 1024

    assert _pinned_registration_size(63 * mebibyte) == 63 * mebibyte
    assert _pinned_registration_size(64 * mebibyte) == 64 * mebibyte
    assert _pinned_registration_size(345 * mebibyte + 700_000) == 348 * mebibyte
    assert _pinned_registration_size(347 * mebibyte + 900_000) == 348 * mebibyte


@pytest.mark.parametrize("nbytes", [0, -1])
def test_pinned_registration_requires_positive_size(nbytes: int) -> None:
    from quantem.gpu.device.cuda_runtime import _pinned_registration_size

    with pytest.raises(ValueError, match="must be positive"):
        _pinned_registration_size(nbytes)


def test_pinned_buffer_release_keeps_distinct_size_classes(monkeypatch) -> None:
    small_array = np.zeros(40, dtype=np.uint8)
    large_array = np.zeros(100, dtype=np.uint8)
    small = {"arr": small_array, "size": 40, "free": True, "addr": 1}
    large = {"arr": large_array, "size": 100, "free": False, "addr": 2}
    monkeypatch.setattr(memory_module, "_PINNED_BUFS", [small, large])
    monkeypatch.setattr(memory_module, "_unregister_pinned_entry", lambda _entry: True)

    memory_module._release_pinned(large_array[:90])

    assert memory_module._PINNED_BUFS == [small, large]
    assert all(entry["free"] for entry in memory_module._PINNED_BUFS)


def test_pinned_buffer_release_prunes_newly_released_smaller_buffer(
    monkeypatch,
) -> None:
    small_array = np.zeros(80, dtype=np.uint8)
    large_array = np.zeros(100, dtype=np.uint8)
    small = {"arr": small_array, "size": 80, "free": False, "addr": 1}
    large = {"arr": large_array, "size": 100, "free": True, "addr": 2}
    released = []
    monkeypatch.setattr(memory_module, "_PINNED_BUFS", [small, large])

    def unregister(entry) -> bool:
        released.append(entry["size"])
        return True

    monkeypatch.setattr(memory_module, "_unregister_pinned_entry", unregister)

    memory_module._release_pinned(small_array)

    assert memory_module._PINNED_BUFS == [large]
    assert released == [80]
    assert small == {}


def test_pinned_buffer_release_defers_pruning_until_pipeline_drains(
    monkeypatch,
) -> None:
    small_array = np.zeros(80, dtype=np.uint8)
    large_array = np.zeros(100, dtype=np.uint8)
    small = {"arr": small_array, "size": 80, "free": False, "addr": 1}
    large = {"arr": large_array, "size": 100, "free": True, "addr": 2}
    released = []
    monkeypatch.setattr(memory_module, "_PINNED_BUFS", [small, large])

    def unregister(entry) -> bool:
        released.append(entry["size"])
        return True

    monkeypatch.setattr(memory_module, "_unregister_pinned_entry", unregister)

    memory_module._release_pinned(small_array, prune=False)

    assert memory_module._PINNED_BUFS == [small, large]
    assert all(entry["free"] for entry in memory_module._PINNED_BUFS)
    assert released == []

    memory_module._prune_pinned_free()

    assert memory_module._PINNED_BUFS == [large]
    assert released == [80]
    assert small == {}


def test_pinned_buffer_pipeline_prune_retains_requested_reuse_slots(
    monkeypatch,
) -> None:
    arrays = [np.zeros(size, dtype=np.uint8) for size in (80, 90, 95, 100, 105)]
    entries = [
        {"arr": array, "size": int(array.size), "free": True, "addr": index}
        for index, array in enumerate(arrays, start=1)
    ]
    released = []
    monkeypatch.setattr(memory_module, "_PINNED_BUFS", entries.copy())

    def unregister(entry) -> bool:
        released.append(entry["size"])
        return True

    monkeypatch.setattr(memory_module, "_unregister_pinned_entry", unregister)

    memory_module._prune_pinned_free(retain_per_size_class=4)

    assert [entry["size"] for entry in memory_module._PINNED_BUFS] == [90, 95, 100, 105]
    assert released == [80]


def test_decompress_prepared_releases_staging_buffer_on_success(monkeypatch) -> None:
    from importlib import import_module

    decode_module = import_module("quantem.gpu.io.hdf5.cuda.decode")
    read_buffer = np.zeros(4, dtype=np.uint8)
    result = object()
    released = []
    monkeypatch.setattr(
        decode_module,
        "_decode_prepared",
        lambda *_args, **_kwargs: result,
    )
    monkeypatch.setattr(decode_module, "_release_pinned", released.append)

    assert decode_module.decompress_prepared({"read_buffer": read_buffer}) is result
    assert len(released) == 1
    assert released[0] is read_buffer


def test_decompress_prepared_synchronizes_before_failure_release(monkeypatch) -> None:
    from importlib import import_module

    decode_module = import_module("quantem.gpu.io.hdf5.cuda.decode")
    read_buffer = np.zeros(4, dtype=np.uint8)
    events = []

    def fail_decode(*_args, **_kwargs):
        events.append("decode")
        raise RuntimeError("decode failed")

    class FakeDevice:
        def synchronize(self) -> None:
            events.append("synchronize")

    fake_cupy = SimpleNamespace(cuda=SimpleNamespace(Device=lambda: FakeDevice()))
    monkeypatch.setattr(decode_module, "cp", fake_cupy)
    monkeypatch.setattr(decode_module, "_decode_prepared", fail_decode)
    monkeypatch.setattr(
        decode_module,
        "_release_pinned",
        lambda buffer: events.append(("release", buffer)),
    )

    with pytest.raises(RuntimeError, match="decode failed"):
        decode_module.decompress_prepared({"read_buffer": read_buffer})

    assert events[:2] == ["decode", "synchronize"]
    assert events[2][0] == "release"
    assert events[2][1] is read_buffer


def test_contiguous_frame_read_plan_preserves_order_and_coalesces() -> None:
    """The fast sequential planner must retain exact chunk destinations."""

    from quantem.gpu.formats.hdf5.reads import _contiguous_frame_read_plan

    source_infos = [
        {
            "chunk_infos": [(100, 10), (115, 10), (1_000, 5)],
            "n_frames": 3,
        },
        {
            "chunk_infos": [(200, 12), (216, 8)],
            "n_frames": 2,
        },
    ]
    result = _contiguous_frame_read_plan(
        source_infos,
        [0, 3, 5],
        np.arange(1, 5, dtype=np.int64),
        max_gap_bytes=5,
    )

    assert result is not None
    offsets, sizes, plan, total_bytes = result
    np.testing.assert_array_equal(offsets, [0, 10, 15, 31])
    np.testing.assert_array_equal(sizes, [10, 5, 12, 8])
    assert plan == {
        0: [(115, 0, 10), (1_000, 10, 5)],
        1: [(200, 15, 24)],
    }
    assert total_bytes == 39


def test_contiguous_frame_read_plan_defers_selector_semantics() -> None:
    """Arbitrary order, duplicates, and physical reordering use the safe path."""

    from quantem.gpu.formats.hdf5.reads import _contiguous_frame_read_plan

    source_infos = [{"chunk_infos": [(100, 10), (90, 10)], "n_frames": 2}]
    assert (
        _contiguous_frame_read_plan(
            source_infos,
            [0, 2],
            np.array([0, 1]),
            max_gap_bytes=0,
        )
        is None
    )
    assert (
        _contiguous_frame_read_plan(
            source_infos,
            [0, 2],
            np.array([0, 0]),
            max_gap_bytes=0,
        )
        is None
    )


def test_frame_source_disk_cache_round_trips_compact_chunk_index(tmp_path) -> None:
    """Prepared source indexes retain exact uint64 byte offsets and sizes."""

    from quantem.gpu.formats.hdf5.frames import (
        _load_frame_source_disk_cache,
        _write_frame_source_disk_cache,
    )

    cache_path = tmp_path / "source-index.pkl"
    signature = {"source": "immutable-test-source"}
    source_infos = [
        {
            "path": "data.h5",
            "dataset_path": "/entry/data/data",
            "n_frames": 2,
            "frame_shape": (192, 192),
            "dtype": np.dtype(np.uint16),
            "chunk_infos": [(2**40 + 7, 12_345), (2**40 + 20_000, 13_579)],
        }
    ]
    _write_frame_source_disk_cache(
        str(cache_path),
        signature=signature,
        source_infos=source_infos,
    )

    loaded = _load_frame_source_disk_cache(
        str(cache_path),
        signature=signature,
    )

    assert loaded is not None
    loaded_infos = loaded
    assert loaded_infos[0]["dtype"] == np.dtype(np.uint16)
    assert loaded_infos[0]["frame_shape"] == (192, 192)
    assert loaded_infos[0]["chunk_infos"].dtype == np.dtype(np.uint64)
    np.testing.assert_array_equal(
        loaded_infos[0]["chunk_infos"],
        np.asarray(source_infos[0]["chunk_infos"], dtype=np.uint64),
    )
