import numpy as np
import pytest

from quantem.gpu.device import cuda_runtime as memory_module


@pytest.mark.parametrize("apply_mask", [False, True])
def test_cpu_reference_records_source_and_actual_mask_policy(tmp_path, apply_mask):
    """Reference loading reports the default median fill accurately without changing other counts.

    The mask once meant zeroing; it now applies io.load's default ``hot_pixel_correction="median"``, as
    every GPU load does. Every neighbor holds 17, so the fill is 17.
    """
    import h5py

    from quantem.gpu import io

    counts = np.full((2, 3, 8, 8), 17, dtype=np.uint16)
    master = tmp_path / "source_master.h5"
    io.save(master, counts, backend="cpu", verbose=False)
    mask = np.zeros((8, 8), dtype=np.uint32)
    mask[2, 4] = 16
    with h5py.File(master, "r+") as handle:
        handle.create_dataset("entry/instrument/detector/detectorSpecific/pixel_mask", data=mask)
    with io.load(master, backend="cpu", apply_mask=apply_mask, verbose=False) as loaded:
        assert loaded.metadata["source_path"] == str(master)
        correction = loaded.metadata["hot_pixel_correction"]
        assert correction["backend"] == "cpu"
        assert correction["applied"] is apply_mask
        assert correction["method"] == ("median" if apply_mask else "none")
        assert correction["stage"] == ("cpu_load_before_dense_storage" if apply_mask else "none")
        expected = counts.copy()
        if apply_mask:
            assert correction["coordinates_row_col"] == [[2, 4]]
        np.testing.assert_array_equal(loaded.read(), expected)


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
        data.attrs["detector_shape"] = (6, 6)

    metadata = get_metadata(str(master))

    assert metadata["source_dtype"] == "uint16"
    assert metadata["detector_shape"] == (6, 6)


def test_qem_of_a_master_holds_no_hdf5_layout_attributes(tmp_path) -> None:
    """The layout attributes of a master's data group are not acquisition metadata and do not reach a saved ``.qem``.

    ``io.save`` marks ``entry/data`` of a master as ``NXdata``, names its default dataset in ``signal`` and lists
    the frame range of each data file in ``data_file_frame_ranges``. The reader copied all three into the dataset
    metadata, and every ``.qem`` made from such a master stored them in its header: keys the format does not
    list and no reader uses.
    """
    import h5py

    from quantem.gpu import io
    from quantem.gpu.formats.hdf5.master import get_metadata
    from quantem.gpu.formats.qem.validation import read_envelope

    counts = np.random.default_rng(1).poisson(3, (2, 3, 8, 8)).astype(np.uint16)
    master = tmp_path / "scan_master.h5"
    io.save(master, counts, backend="cpu", verbose=False)
    layout = {"NX_class", "signal", "data_file_frame_ranges"}
    with h5py.File(master, "r") as handle:
        assert layout <= set(handle["entry/data"].attrs)
    assert not layout & set(get_metadata(str(master)))
    with io.load(master, backend="cpu", verbose=False) as loaded:
        io.save(tmp_path / "scan.qem", loaded, backend="cpu", verbose=False)
    assert not layout & set(read_envelope(tmp_path / "scan.qem")[0]["metadata"])


def test_load_names_master_calibration_attributes_it_does_not_read(tmp_path) -> None:
    """A master written by an earlier version loads with one warning naming the calibration it holds and is not read.

    Such a master keeps the detector step under ``detector_sampling`` with ``detector_sampling_unit`` and may hold
    one number as the scan step. Both loaded as uncalibrated axes without a word; the attributes stayed in the
    metadata, where the scan step raised when the sampling was read and the detector pair when a ``.qem`` was saved.
    The message is a warning, so a quiet load (``verbose=False``, as ``io.convert`` loads) gives it too.
    """
    import warnings

    import h5py

    from quantem.gpu import io

    counts = np.random.default_rng(1).poisson(3, (2, 3, 8, 8)).astype(np.uint16)
    master = tmp_path / "scan_master.h5"
    io.save(master, counts, backend="cpu", verbose=False, metadata={"detector_sampling_inv_A": [0.02, 0.03]})
    # a master of this version: nothing to say, on a quiet or a verbose load
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for verbose in (False, True):
            with io.load(master, backend="cpu", verbose=verbose) as current:
                assert current.sampling == (None, None, 0.02, 0.03)
    with h5py.File(master, "r+") as handle:
        handle.attrs["scan_sampling_A"] = 0.264
        handle.attrs["detector_sampling"] = [0.2, 0.3]
        handle.attrs["detector_sampling_unit"] = "mrad"
    for verbose in (False, True):
        with pytest.warns(UserWarning, match="Calibration not read") as caught, io.load(master, backend="cpu", verbose=verbose) as loaded:
            # nothing is converted: the scan axes are uncalibrated, and the pair this version reads is kept
            assert loaded.sampling == (None, None, 0.02, 0.03)
            assert not {"scan_sampling_A", "detector_sampling", "detector_sampling_unit"} & set(loaded.metadata)
            if verbose:
                io.save(tmp_path / "copy.qem", loaded, backend="cpu", verbose=False)
        assert len(caught) == 1
        for name in ("scan_master.h5", "scan_sampling_A (not a [row, col] pair)", "detector_sampling,", "detector_sampling_unit"):
            assert name in str(caught[0].message)
    with io.load(tmp_path / "copy.qem", backend="cpu", verbose=False) as copy:
        assert copy.sampling == (None, None, 0.02, 0.03)
        np.testing.assert_array_equal(copy.data, counts)


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


def test_pinned_buffer_release_defers_pruning_to_the_next_pruning_release(
    monkeypatch,
) -> None:
    small_array = np.zeros(80, dtype=np.uint8)
    large_array = np.zeros(100, dtype=np.uint8)
    small = {"arr": small_array, "size": 80, "free": False, "addr": 1}
    large = {"arr": large_array, "size": 100, "free": False, "addr": 2}
    released = []
    monkeypatch.setattr(memory_module, "_PINNED_BUFS", [small, large])

    def unregister(entry) -> bool:
        released.append(entry["size"])
        return True

    monkeypatch.setattr(memory_module, "_unregister_pinned_entry", unregister)

    memory_module._release_pinned(small_array, prune=False)

    assert memory_module._PINNED_BUFS == [small, large]
    assert small["free"] and released == []

    memory_module._release_pinned(large_array)

    assert memory_module._PINNED_BUFS == [large]
    assert released == [80]
    assert small == {}


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
