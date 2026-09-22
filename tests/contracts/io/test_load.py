from __future__ import annotations

import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from quantem.gpu.io import _memory as memory_module


def _mock_mps_backend(monkeypatch) -> None:
    """Bypass host detection for unit tests with a fake MPS decoder."""
    monkeypatch.setattr(
        "quantem.gpu.io.backends.resolve_backend",
        lambda _backend: "mps",
    )


def test_mps_dataset_path_u8_declares_clipping_before_detector_bin(monkeypatch) -> None:
    """The generic MPS branch must not hide uint8 intent until after binning."""
    from importlib import import_module

    load_module = import_module("quantem.gpu.io.load")
    mps_package = import_module("quantem.gpu.io.backends.mps")
    decoder = types.ModuleType("quantem.gpu.io.backends.mps.dense")
    monkeypatch.setitem(sys.modules, decoder.__name__, decoder)
    monkeypatch.setattr(mps_package, "dense", decoder, raising=False)
    calls = {}

    class FakeMPSTensor:
        dtype = "torch.uint8"

        def to(self, device):
            calls["device"] = device
            return self

    torch_module = types.ModuleType("torch")
    torch_module.from_numpy = lambda array: FakeMPSTensor()
    monkeypatch.setitem(sys.modules, "torch", torch_module)

    monkeypatch.setattr(
        load_module,
        "get_metadata",
        lambda path: {
            "detector_shape": (2, 2),
            "source_dtype": "uint16",
        },
    )
    monkeypatch.setattr(load_module, "read_pixel_mask", lambda path: None)

    def fake_load_master(path, **kwargs):
        calls.update(kwargs)
        return np.full((1, 1, 1), 255, dtype=np.uint8)

    monkeypatch.setattr(decoder, "load_master", fake_load_master, raising=False)
    data, metadata = load_module._load_view(
        "fixture.h5",
        "mps",
        dataset_path="entry/data/data",
        scan_shape=(1, 1),
        det_bin=2,
        output_dtype=np.uint8,
        verbose=False,
    )

    assert str(data.dtype) == "torch.uint8"
    assert calls["output_dtype"] == np.dtype(np.uint8)
    assert calls["device"] == "mps"
    assert metadata["source_detector_shape"] == (2, 2)
    assert metadata["detector_shape"] == (1, 1)
    assert metadata["det_bin"] == 2
    assert metadata["source_dtype"] == "uint16"
    assert metadata["dtype"] == "uint8"


def test_cpu_detector_bin_metadata_preserves_source_and_working_geometry(
    monkeypatch,
) -> None:
    """CPU detector binning reports the source separately from returned pixels."""
    from importlib import import_module

    load_module = import_module("quantem.gpu.io.load")
    reference = import_module("quantem.gpu.io.backends.cpu.dense")
    source = np.arange(4 * 4 * 4, dtype=np.uint16).reshape(4, 4, 4)
    expected = source.reshape(4, 2, 2, 2, 2).sum(axis=(2, 4), dtype=np.uint64)
    expected = expected.astype(np.uint16)

    monkeypatch.setattr(
        load_module,
        "get_metadata",
        lambda path: {
            "scan_shape": (2, 2),
            "detector_shape": (4, 4),
        },
    )
    monkeypatch.setattr(load_module, "read_pixel_mask", lambda path: None)
    monkeypatch.setattr(
        reference,
        "load_master",
        lambda path, **kwargs: expected.copy(),
    )

    result = load_module._load_view(
        "fixture.h5",
        "cpu",
        scan_shape=(2, 2),
        det_bin=2,
        verbose=False,
    )

    np.testing.assert_array_equal(result.data, expected.reshape(2, 2, 2, 2))
    assert result.metadata["source_detector_shape"] == (4, 4)
    assert result.metadata["detector_shape"] == (2, 2)
    assert result.metadata["det_bin"] == 2
    assert result.metadata["source_dtype"] == "uint16"
    assert result.metadata["dtype"] == "uint16"


def test_get_metadata_reports_detector_source_dtype(tmp_path) -> None:
    """Metadata inspection records the stored count dtype before load transforms."""
    import h5py

    from quantem.gpu.io.load import get_metadata

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


def test_mps_output_dtype_u4_does_not_alias_to_uint32() -> None:
    """MPS dtype normalization must not pass public 'u4' to np.dtype."""
    source = Path("src/quantem/gpu/io/backends/mps/dense.py").read_text()

    assert 'token in {"u4", "uint4"}' in source
    assert "not NumPy's four-byte '<u4' dtype" in source


def test_mps_multi_dataset_loader_threads_output_dtype(monkeypatch) -> None:
    """C1: lazy MPS browse loads, expect requested uint8 dtype to reach load."""
    from importlib import import_module
    load_module = import_module("quantem.gpu.io.load")
    series_module = import_module("quantem.gpu.io.backends.mps.series")
    from quantem.gpu.detector.backends.mps import kernels as mps_compute

    calls = []

    def fake_load(path, **kwargs):
        calls.append({"path": path, "kwargs": kwargs})
        return SimpleNamespace(row_prefix=False, metadata={}), {}

    class FakeChunkedFrames:
        def __init__(self, data, *, row_prefix=False):
            self.data = data
            self.row_prefix = row_prefix

    class FakeMultiChunkedFrames:
        def __init__(self, datasets, *, n_total, names):
            self.datasets = list(datasets)
            self.n_total = n_total
            self.names = names
            self.n_ready = len(datasets)
            self.on_ready = None

    monkeypatch.setattr(load_module, "load", fake_load)
    monkeypatch.setattr(mps_compute, "ChunkedFrames", FakeChunkedFrames)
    monkeypatch.setattr(mps_compute, "MultiChunkedFrames", FakeMultiChunkedFrames)

    lazy = series_module.load_mps_datasets(
        ["tilt_0_master.h5", "tilt_1_master.h5"],
        det_bin=4,
        output_dtype=np.uint8,
        verbose=False,
    )

    assert lazy.det_bin == 4
    assert calls[0]["path"] == "tilt_0_master.h5"
    assert calls[0]["kwargs"]["backend"] == "mps"
    assert calls[0]["kwargs"]["det_bin"] == 4
    assert calls[0]["kwargs"]["dtype"] is np.uint8


def test_get_libc_returns_none_when_posix_fadvise_is_unavailable(monkeypatch) -> None:
    """macOS libc exists but does not expose Linux posix_fadvise."""
    import ctypes
    import ctypes.util
    from importlib import import_module
    load_module = import_module("quantem.gpu.io.load")

    monkeypatch.setattr(ctypes.util, "find_library", lambda _name: "libc.dylib")
    monkeypatch.setattr(ctypes, "CDLL", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(memory_module, "_LIBC", None)

    assert load_module._get_libc() is None
    assert memory_module._LIBC is False


def test_apply_scan_shape_supports_serpentine_order() -> None:
    """Full flat scans can be unflattened with odd scan rows reversed."""
    from importlib import import_module
    load_module = import_module("quantem.gpu.io.load")

    data = np.arange(12, dtype=np.uint16).reshape(6, 1, 2)

    result = load_module._apply_scan_shape(
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


def test_scan_region_frame_indices_support_serpentine_order() -> None:
    """Serpentine ROI indices should be returned in visual row/column order."""
    from importlib import import_module
    load_module = import_module("quantem.gpu.io.load")

    indices = load_module._scan_region_frame_indices(
        (1, 3, 2, 5),
        (5, 6),
        scan_order="snake",
    )

    np.testing.assert_array_equal(
        indices,
        np.asarray([9, 8, 7, 14, 15, 16], dtype=np.int64),
    )


def test_load_rejects_unknown_scan_order() -> None:
    """Unknown flattened scan order names should fail before any IO starts."""
    from importlib import import_module
    load_module = import_module("quantem.gpu.io.load")

    with pytest.raises(ValueError, match="scan_order must be"):
        load_module._normalize_scan_order("zigzag")


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
    from quantem.gpu.io.load import resample_scan_crop

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


def test_scan_indices_rowcol_supports_serpentine_order() -> None:
    """Sparse scan positions map logical row/col to physical HDF5 frames."""
    from importlib import import_module
    load_module = import_module("quantem.gpu.io.load")

    positions = np.asarray(
        [
            [1, 2],
            [0, 5],
            [1, 4],
            [1, 2],
        ],
        dtype=np.int64,
    )

    frame_indices, logical_positions = load_module._normalize_scan_indices(
        positions,
        (5, 6),
        scan_order="serpentine",
    )

    np.testing.assert_array_equal(
        frame_indices,
        np.asarray([9, 5, 7, 9], dtype=np.int64),
    )
    np.testing.assert_array_equal(logical_positions, positions)


def test_load_scan_indices_reads_sorted_unique_and_restores_order(
    tmp_path,
    monkeypatch,
) -> None:
    """Random sparse IO should coalesce disk reads but return stochastic order."""
    from importlib import import_module
    load_module = import_module("quantem.gpu.io.load")
    _mock_mps_backend(monkeypatch)

    master = tmp_path / "scan_master.h5"
    master.write_bytes(b"placeholder")
    calls = {}

    monkeypatch.setattr(load_module, "get_metadata", lambda _path: {"scan_shape": (5, 6)})
    monkeypatch.setattr(load_module, "_discover_chunk_names", lambda _path: ["data_000001"])

    def fake_prepare(filepath, chunk_names, frame_indices, apply_mask=True):
        calls["filepath"] = filepath
        calls["chunk_names"] = chunk_names
        calls["frame_indices"] = frame_indices.copy()
        calls["apply_mask"] = apply_mask
        return {
            "selected_frame_indices": frame_indices.copy(),
            "pixel_mask": np.zeros((1, 1), dtype=np.uint8),
            "dtype": np.dtype(np.uint16),
            "total_compressed_bytes": 4096,
            "read_span_count": 2,
            "read_gap_bytes": 128,
            "prepare_timing_s": {"compressed_pread": 0.001},
        }

    def fake_mps_decode(prepared, **kwargs):
        calls["mps_kwargs"] = kwargs
        values = prepared["selected_frame_indices"].astype(np.uint16)
        return values.reshape(-1, 1, 1)

    monkeypatch.setattr(load_module, "_prepare_master_frames", fake_prepare)
    monkeypatch.setitem(
        sys.modules,
        "quantem.gpu.io.backends.mps.dense",
        types.SimpleNamespace(load_prepared_frames=fake_mps_decode),
    )

    result = load_module.load_scan_indices(
        str(master),
        np.asarray([[2, 5], [1, 0], [2, 5], [0, 3]], dtype=np.int64),
        backend="mps",
        verbose=False,
    )

    np.testing.assert_array_equal(
        calls["frame_indices"],
        np.asarray([3, 6, 17], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        result.data[:, 0, 0],
        np.asarray([17, 6, 17, 3], dtype=np.uint16),
    )
    assert result.metadata["unique_frame_count"] == 3
    assert result.metadata["duplicate_frame_count"] == 1
    assert result.metadata["read_order"] == "sorted_unique_hdf5_frame_indices"
    assert result.metadata["total_compressed_bytes"] == 4096
    assert result.metadata["read_span_count"] == 2
    assert result.metadata["read_gap_bytes"] == 128
    assert result.metadata["prepare_timing_s"] == {"compressed_pread": 0.001}


def test_load_scan_indices_multi_file_accepts_per_file_batches(
    tmp_path,
    monkeypatch,
) -> None:
    """Multi-master sparse IO should support different random positions per file."""
    from importlib import import_module
    load_module = import_module("quantem.gpu.io.load")
    _mock_mps_backend(monkeypatch)

    masters = [tmp_path / "a_master.h5", tmp_path / "b_master.h5"]
    for master in masters:
        master.write_bytes(b"placeholder")

    monkeypatch.setattr(load_module, "get_metadata", lambda _path: {"scan_shape": (4, 4)})
    monkeypatch.setattr(load_module, "_discover_chunk_names", lambda _path: ["data_000001"])

    def fake_prepare(filepath, chunk_names, frame_indices, apply_mask=True):
        offset = 100 if filepath.endswith("b_master.h5") else 0
        return {
            "selected_frame_indices": frame_indices.copy(),
            "offset": offset,
            "pixel_mask": None,
            "dtype": np.dtype(np.uint16),
        }

    def fake_mps_decode(prepared, **kwargs):
        values = prepared["selected_frame_indices"].astype(np.uint16)
        values = values + np.uint16(prepared["offset"])
        return values.reshape(-1, 1, 1)

    monkeypatch.setattr(load_module, "_prepare_master_frames", fake_prepare)
    monkeypatch.setitem(
        sys.modules,
        "quantem.gpu.io.backends.mps.dense",
        types.SimpleNamespace(load_prepared_frames=fake_mps_decode),
    )

    result = load_module.load_scan_indices(
        [str(p) for p in masters],
        np.asarray(
            [
                [5, 0, 5],
                [4, 1, 3],
            ],
            dtype=np.int64,
        ),
        backend="mps",
        verbose=False,
    )

    assert result.data.shape == (2, 3, 1, 1)
    np.testing.assert_array_equal(result.data[0, :, 0, 0], [5, 0, 5])
    np.testing.assert_array_equal(result.data[1, :, 0, 0], [104, 101, 103])
    assert result.metadata["positions_per_file"] == [3, 3]
    assert result.metadata["unique_frame_count_per_file"] == [2, 3]


def test_random_scan_indices_are_reproducible_and_per_file() -> None:
    """Random scan sampling should look like a deterministic DataLoader sampler."""
    from importlib import import_module
    load_module = import_module("quantem.gpu.io.load")

    one = load_module.random_scan_indices(4, (4, 4), seed=123)
    again = load_module.random_scan_indices(4, (4, 4), seed=123)
    per_file = load_module.random_scan_indices(4, (4, 4), n_files=3, seed=123)
    positions = load_module.random_scan_indices(
        4,
        (4, 4),
        seed=123,
        return_positions=True,
    )

    np.testing.assert_array_equal(one, again)
    assert one.shape == (4,)
    assert per_file.shape == (3, 4)
    assert positions.shape == (4, 2)
    assert np.all(one >= 0)
    assert np.all(one < 16)
    assert len({int(v) for v in one}) == 4
    assert not np.array_equal(per_file[0], per_file[1])
    np.testing.assert_array_equal(positions[:, 0] * 4 + positions[:, 1], one)


def test_random_scan_indices_rejects_oversampling_without_replacement() -> None:
    """Without replacement, random sampling should fail before any IO starts."""
    from importlib import import_module
    load_module = import_module("quantem.gpu.io.load")

    with pytest.raises(ValueError, match="Cannot sample"):
        load_module.random_scan_indices(17, (4, 4), replace=False)


def test_drift_scan_positions_preserves_integer_and_fractional_vectors() -> None:
    """Dense drift fields produce float probe positions without resampling."""
    from importlib import import_module

    load_module = import_module("quantem.gpu.io.load")
    nominal = np.asarray([[1, 2], [3, 4]], dtype=np.int64)
    drift = np.zeros((2, 6, 7, 2), dtype=np.float32)
    drift[0, 1, 2] = [1, -2]
    drift[0, 3, 4] = [1, -2]
    drift[1, 1, 2] = [0.4, 0.6]
    drift[1, 3, 4] = [0.4, 0.6]

    got = load_module._drift_scan_positions(nominal, drift, scan_shape=(6, 7))

    assert got.shape == (2, 2, 2)
    assert got.dtype == np.float32
    np.testing.assert_allclose(
        got,
        np.asarray(
            [
                [[2.0, 0.0], [4.0, 2.0]],
                [[1.4, 2.6], [3.4, 4.6]],
            ],
            dtype=np.float32,
        ),
    )


def test_drift_batch_example_has_forty_frames_and_shared_positions() -> None:
    """The 40-frame joint example keeps one position set and float drift."""
    from importlib import import_module

    load_module = import_module("quantem.gpu.io.load")
    positions = np.asarray([[r, r] for r in range(10)], dtype=np.int64)
    vectors = np.zeros((40, 16, 16, 2), dtype=np.float32)
    vectors[..., 0] = np.arange(40, dtype=np.float32)[:, None, None] * 0.25
    vectors[..., 1] = -np.arange(40, dtype=np.float32)[:, None, None] * 0.125

    corrected = load_module._drift_scan_positions(
        positions, vectors, scan_shape=(16, 16)
    )

    assert corrected.shape == (40, 10, 2)
    assert corrected.dtype == np.float32
    np.testing.assert_allclose(corrected[0], positions)
    np.testing.assert_allclose(corrected[4, 3], [4.0, 2.5])
    np.testing.assert_allclose(corrected[39, 9], [18.75, 4.125])


def test_drift_scan_positions_rejects_wrong_field_size() -> None:
    """Every source field must match the logical scan dimensions exactly."""
    from importlib import import_module

    load_module = import_module("quantem.gpu.io.load")
    with pytest.raises(ValueError, match="does not match scan_shape"):
        load_module._drift_scan_positions(
            np.asarray([[1, 1]], dtype=np.int64),
            np.zeros((1, 4, 4, 2), dtype=np.float32),
            scan_shape=(5, 5),
        )


def test_sparse_prep_workers_default_to_single_reader() -> None:
    """Sparse HDF5 prep should not assume more readers are faster."""
    from importlib import import_module
    load_module = import_module("quantem.gpu.io.load")

    assert load_module._normalize_prep_workers(None, n_files=40) == 1
    assert load_module._normalize_prep_workers(8, n_files=40) == 8
    assert load_module._normalize_prep_workers(100, n_files=40) == 40

    with pytest.raises(ValueError, match="positive integer"):
        load_module._normalize_prep_workers(0, n_files=40)


def test_load_with_scan_region_rejects_slice_and_range_forms() -> None:
    """Keep the public crop API simple: one flat row/column bounds tuple."""
    from importlib import import_module
    load_module = import_module("quantem.gpu.io.load")

    with pytest.raises(TypeError, match="scan_region must be"):
        load_module._normalize_scan_region((slice(0, 1), range(1)), (5, 6))


def test_load_with_detector_region_rejects_invalid_bounds() -> None:
    """Detector-region bounds use explicit detector row/column intervals."""
    from importlib import import_module
    load_module = import_module("quantem.gpu.io.load")

    with pytest.raises(TypeError, match="detector_region must be"):
        load_module._normalize_detector_region((slice(0, 1), 2, 0, 1), (5, 6))
    with pytest.raises(ValueError, match="detector row region"):
        load_module._normalize_detector_region((-1, 2, 0, 1), (5, 6))
    with pytest.raises(ValueError, match="detector column region"):
        load_module._normalize_detector_region((0, 2, 3, 7), (5, 6))


@pytest.mark.parametrize(
    "selector",
    [
        {"scan_region": (0, 1, 0, 1)},
        {"scan_indices": [0]},
        {"random_positions": 1, "seed": 7},
    ],
)
def test_selective_scan_loading_rejects_cpu_backend(monkeypatch, selector) -> None:
    """The public CPU loader must not pretend to implement selective GPU IO."""
    from importlib import import_module
    load_module = import_module("quantem.gpu.io.load")

    monkeypatch.setattr("quantem.gpu.io.backends.resolve_backend", lambda _backend: "cpu")

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


def test_torch_detector_bin_sum_matches_numpy_reference() -> None:
    torch = pytest.importorskip("torch")
    from quantem.gpu.io.load import bin

    data_np = np.arange(2 * 3 * 4 * 4, dtype=np.uint16).reshape(2, 3, 4, 4)
    data_torch = torch.as_tensor(data_np)

    out = bin(data_torch, factor=2, axes="detector", reduction="sum")

    expected = data_np.reshape(2, 3, 2, 2, 2, 2).sum(axis=(3, 5), dtype=np.uint64)
    np.testing.assert_array_equal(out.numpy(), expected.astype(np.int64))


def test_cuda_detector_bin_default_widens_before_exact_sum() -> None:
    from importlib import import_module

    load_module = import_module("quantem.gpu.io.load")

    dtype, narrow = load_module._default_decoded_output_dtype(
        np.uint16,
        auto_narrow=True,
        detector_bin=2,
    )

    assert dtype == np.dtype(np.uint32)
    assert narrow is False


def test_cupy_bin_does_not_require_torch(monkeypatch) -> None:
    import builtins
    from importlib import import_module

    load_module = import_module("quantem.gpu.io.load")
    original_import = builtins.__import__

    def import_without_torch(name, *args, **kwargs):
        if name == "torch":
            raise ModuleNotFoundError("torch is intentionally absent")
        return original_import(name, *args, **kwargs)

    fake_cupy = SimpleNamespace(
        ndarray=np.ndarray,
        float32=np.float32,
        integer=np.integer,
        issubdtype=np.issubdtype,
        uint32=np.uint32,
        zeros=np.zeros,
    )
    monkeypatch.setattr(builtins, "__import__", import_without_torch)
    monkeypatch.setattr(load_module, "cp", fake_cupy)
    source = np.arange(3 * 5 * 2 * 2, dtype=np.uint16).reshape(3, 5, 2, 2)

    result = load_module.bin(
        source,
        factor=2,
        axes="scan",
        reduction="sum",
        edge="partial",
    )

    padded = np.zeros((4, 6, 2, 2), dtype=np.uint16)
    padded[:3, :5] = source
    expected = padded.reshape(2, 2, 3, 2, 2, 2).sum(
        axis=(1, 3), dtype=np.uint32
    )
    np.testing.assert_array_equal(result, expected)


def test_torch_scan_bin_partial_keeps_incomplete_edges_exactly() -> None:
    torch = pytest.importorskip("torch")
    from quantem.gpu.io.load import bin

    data_np = np.arange(3 * 5 * 2 * 2, dtype=np.uint16).reshape(3, 5, 2, 2)
    out = bin(
        torch.as_tensor(data_np),
        factor=2,
        axes="scan",
        reduction="sum",
        edge="partial",
    )

    padded = np.zeros((4, 6, 2, 2), dtype=np.uint16)
    padded[:3, :5] = data_np
    expected = padded.reshape(2, 2, 3, 2, 2, 2).sum(
        axis=(1, 3), dtype=np.uint64
    )
    np.testing.assert_array_equal(out.numpy(), expected.astype(np.int64))


def test_pinned_buffer_release_prunes_redundant_smaller_size(monkeypatch) -> None:
    from importlib import import_module

    load_module = import_module("quantem.gpu.io.load")
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

    load_module._release_pinned(large_array[:90])

    assert memory_module._PINNED_BUFS == [large]
    assert large["free"] is True
    assert released == [80]
    assert small == {}


def test_large_pinned_registration_rounds_to_bounded_reuse_class() -> None:
    from quantem.gpu.io.load import _pinned_registration_size

    mebibyte = 1024 * 1024

    assert _pinned_registration_size(63 * mebibyte) == 63 * mebibyte
    assert _pinned_registration_size(64 * mebibyte) == 64 * mebibyte
    assert _pinned_registration_size(345 * mebibyte + 700_000) == 348 * mebibyte
    assert _pinned_registration_size(347 * mebibyte + 900_000) == 348 * mebibyte


@pytest.mark.parametrize("nbytes", [0, -1])
def test_pinned_registration_requires_positive_size(nbytes: int) -> None:
    from quantem.gpu.io.load import _pinned_registration_size

    with pytest.raises(ValueError, match="must be positive"):
        _pinned_registration_size(nbytes)


def test_pinned_buffer_release_keeps_distinct_size_classes(monkeypatch) -> None:
    from importlib import import_module

    load_module = import_module("quantem.gpu.io.load")
    small_array = np.zeros(40, dtype=np.uint8)
    large_array = np.zeros(100, dtype=np.uint8)
    small = {"arr": small_array, "size": 40, "free": True, "addr": 1}
    large = {"arr": large_array, "size": 100, "free": False, "addr": 2}
    monkeypatch.setattr(memory_module, "_PINNED_BUFS", [small, large])
    monkeypatch.setattr(memory_module, "_unregister_pinned_entry", lambda _entry: True)

    load_module._release_pinned(large_array[:90])

    assert memory_module._PINNED_BUFS == [small, large]
    assert all(entry["free"] for entry in memory_module._PINNED_BUFS)


def test_pinned_buffer_release_prunes_newly_released_smaller_buffer(
    monkeypatch,
) -> None:
    from importlib import import_module

    load_module = import_module("quantem.gpu.io.load")
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

    load_module._release_pinned(small_array)

    assert memory_module._PINNED_BUFS == [large]
    assert released == [80]
    assert small == {}


def test_pinned_buffer_release_defers_pruning_until_pipeline_drains(
    monkeypatch,
) -> None:
    from importlib import import_module

    load_module = import_module("quantem.gpu.io.load")
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

    load_module._release_pinned(small_array, prune=False)

    assert memory_module._PINNED_BUFS == [small, large]
    assert all(entry["free"] for entry in memory_module._PINNED_BUFS)
    assert released == []

    load_module._prune_pinned_free()

    assert memory_module._PINNED_BUFS == [large]
    assert released == [80]
    assert small == {}


def test_pinned_buffer_pipeline_prune_retains_requested_reuse_slots(
    monkeypatch,
) -> None:
    from importlib import import_module

    load_module = import_module("quantem.gpu.io.load")
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

    load_module._prune_pinned_free(retain_per_size_class=4)

    assert [entry["size"] for entry in memory_module._PINNED_BUFS] == [90, 95, 100, 105]
    assert released == [80]


def test_prepare_master_releases_pinned_buffer_after_preparation_error(
    monkeypatch,
    tmp_path: Path,
) -> None:
    h5py = pytest.importorskip("h5py")
    from importlib import import_module

    load_module = import_module("quantem.gpu.io.load")
    master = tmp_path / "scan_master.h5"
    with h5py.File(master, "w") as handle:
        handle.create_dataset(
            "entry/data/data",
            data=np.zeros((1, 2, 2), dtype=np.uint16),
        )

    allocated = np.zeros(master.stat().st_size, dtype=np.uint8)
    released = []
    monkeypatch.setattr(load_module, "_alloc_pinned_fast", lambda _size: allocated)
    monkeypatch.setattr(load_module, "_release_pinned", released.append)
    monkeypatch.setattr(
        load_module.np,
        "cumsum",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("index failed")),
    )

    with pytest.raises(RuntimeError, match="index failed"):
        load_module._prepare_master(str(master), ["data"])

    assert len(released) == 1
    assert released[0] is allocated


def test_decompress_prepared_releases_staging_buffer_on_success(monkeypatch) -> None:
    from importlib import import_module

    load_module = import_module("quantem.gpu.io.load")
    read_buffer = np.zeros(4, dtype=np.uint8)
    result = object()
    released = []
    monkeypatch.setattr(
        load_module,
        "_decompress_prepared_impl",
        lambda *_args, **_kwargs: result,
    )
    monkeypatch.setattr(load_module, "_release_pinned", released.append)

    assert load_module._decompress_prepared({"read_buffer": read_buffer}) is result
    assert len(released) == 1
    assert released[0] is read_buffer


def test_decompress_prepared_synchronizes_before_failure_release(monkeypatch) -> None:
    from importlib import import_module

    load_module = import_module("quantem.gpu.io.load")
    read_buffer = np.zeros(4, dtype=np.uint8)
    events = []

    def fail_decode(*_args, **_kwargs):
        events.append("decode")
        raise RuntimeError("decode failed")

    class FakeDevice:
        def synchronize(self) -> None:
            events.append("synchronize")

    fake_cupy = SimpleNamespace(cuda=SimpleNamespace(Device=lambda: FakeDevice()))
    monkeypatch.setattr(load_module, "cp", fake_cupy)
    monkeypatch.setattr(load_module, "_decompress_prepared_impl", fail_decode)
    monkeypatch.setattr(
        load_module,
        "_release_pinned",
        lambda buffer: events.append(("release", buffer)),
    )

    with pytest.raises(RuntimeError, match="decode failed"):
        load_module._decompress_prepared({"read_buffer": read_buffer})

    assert events[:2] == ["decode", "synchronize"]
    assert events[2][0] == "release"
    assert events[2][1] is read_buffer


def test_load_many_parallel_propagates_reader_failure(monkeypatch) -> None:
    from importlib import import_module

    load_module = import_module("quantem.gpu.io.load")
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace())
    monkeypatch.setattr(load_module, "disk_of", lambda _path: "nvme0")
    monkeypatch.setattr(load_module, "_discover_chunk_names", lambda _path: ["data"])
    monkeypatch.setattr(
        load_module,
        "_prepare_master",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("read failed")),
    )

    with pytest.raises(OSError, match="read failed"):
        load_module._load_many_parallel(["broken_master.h5"])


def test_load_many_parallel_releases_queued_buffers_after_decode_failure(
    monkeypatch,
) -> None:
    from importlib import import_module

    load_module = import_module("quantem.gpu.io.load")
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace())
    monkeypatch.setattr(load_module, "disk_of", lambda _path: "nvme0")
    monkeypatch.setattr(load_module, "_discover_chunk_names", lambda _path: ["data"])

    buffers = {
        "a_master.h5": np.zeros(1, dtype=np.uint8),
        "b_master.h5": np.zeros(2, dtype=np.uint8),
    }
    both_prepared = threading.Event()
    prepared_count = 0
    prepared_lock = threading.Lock()
    released = []

    def prepare(path, *_args, **_kwargs):
        nonlocal prepared_count
        with prepared_lock:
            prepared_count += 1
            if prepared_count == 2:
                both_prepared.set()
        return {"read_buffer": buffers[path]}

    def fail_decode(prepared, **_kwargs):
        assert both_prepared.wait(timeout=1.0)
        load_module._release_pinned(prepared["read_buffer"])
        raise RuntimeError("decode failed")

    monkeypatch.setattr(load_module, "_prepare_master", prepare)
    monkeypatch.setattr(load_module, "_decompress_prepared", fail_decode)
    monkeypatch.setattr(load_module, "_release_pinned", released.append)

    with pytest.raises(RuntimeError, match="decode failed"):
        load_module._load_many_parallel(list(buffers))

    assert {id(buffer) for buffer in released} == {
        id(buffer) for buffer in buffers.values()
    }


def test_load_many_parallel_rejects_empty_gpu_list() -> None:
    from importlib import import_module

    load_module = import_module("quantem.gpu.io.load")
    with pytest.raises(ValueError, match="at least one CUDA device"):
        load_module._load_many_parallel(["scan_master.h5"], gpus=[])


def test_source_aligned_ranges_group_complete_shards_without_splitting() -> None:
    """Sequential screening should group shards and split only oversized ones."""

    from quantem.gpu.io.load import _source_aligned_frame_ranges

    assert _source_aligned_frame_ranges(
        [0, 10_000, 20_000, 30_000, 40_000, 50_000, 55_000],
        frame_count=55_000,
        max_batch_frames=42_000,
    ) == ((0, 40_000), (40_000, 55_000))
    assert _source_aligned_frame_ranges(
        [0, 50_000, 55_000],
        frame_count=55_000,
        max_batch_frames=42_000,
    ) == ((0, 42_000), (42_000, 50_000), (50_000, 55_000))


def test_source_aligned_ranges_fail_closed_on_incomplete_sources() -> None:
    """The private range planner may never silently omit scan positions."""

    from quantem.gpu.io.load import _source_aligned_frame_ranges

    with pytest.raises(ValueError, match="but 11 are required"):
        _source_aligned_frame_ranges(
            [0, 10],
            frame_count=11,
            max_batch_frames=4,
        )


def test_contiguous_frame_read_plan_preserves_order_and_coalesces() -> None:
    """The fast sequential planner must retain exact chunk destinations."""

    from quantem.gpu.io.load import _contiguous_frame_read_plan

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

    from quantem.gpu.io.load import _contiguous_frame_read_plan

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

    from quantem.gpu.io.load import (
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
        pixel_mask=None,
    )

    loaded = _load_frame_source_disk_cache(
        str(cache_path),
        signature=signature,
    )

    assert loaded is not None
    loaded_infos, loaded_mask = loaded
    assert loaded_mask is None
    assert loaded_infos[0]["dtype"] == np.dtype(np.uint16)
    assert loaded_infos[0]["frame_shape"] == (192, 192)
    assert loaded_infos[0]["chunk_infos"].dtype == np.dtype(np.uint64)
    np.testing.assert_array_equal(
        loaded_infos[0]["chunk_infos"],
        np.asarray(source_infos[0]["chunk_infos"], dtype=np.uint64),
    )
