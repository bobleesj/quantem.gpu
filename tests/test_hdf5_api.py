from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest


def test_load_stacked_u8_routes_to_direct_output_dtype(monkeypatch) -> None:
    """Public dtype='u8' must reach stacked list loads before materializing U16."""
    from quantem.gpu.io import hdf5

    calls = {}

    def fake_load_impl(filepath, *args, **kwargs):
        calls["filepath"] = filepath
        calls["kwargs"] = kwargs
        return hdf5.LoadResult(
            np.zeros((2, 1, 1, 1, 1), dtype=np.uint8),
            {"file_names": ["a", "b"]},
        )

    monkeypatch.setattr(hdf5, "_load_impl", fake_load_impl)

    hdf5.load(["a_master.h5", "b_master.h5"], dtype="u8", verbose=False)

    assert calls["filepath"] == ["a_master.h5", "b_master.h5"]
    assert calls["kwargs"]["output_dtype"] is np.uint8


def test_load_u8_does_not_override_explicit_output_dtype(monkeypatch) -> None:
    """Explicit lower-level output_dtype remains authoritative."""
    from quantem.gpu.io import hdf5

    calls = {}

    def fake_load_impl(filepath, *args, **kwargs):
        calls["kwargs"] = kwargs
        return hdf5.LoadResult(
            np.zeros((1, 1, 1), dtype=np.float16),
            {},
        )

    monkeypatch.setattr(hdf5, "_load_impl", fake_load_impl)

    hdf5.load("a_master.h5", dtype="u8", output_dtype=np.float16, verbose=False)

    assert calls["kwargs"]["output_dtype"] is np.float16


def test_load_uint32_routes_to_native_uint32_output_dtype(monkeypatch) -> None:
    """Public dtype='uint32' should request 4-byte detector counts."""
    from quantem.gpu.io import hdf5

    calls = {}

    def fake_load_impl(filepath, *args, **kwargs):
        calls["kwargs"] = kwargs
        return hdf5.LoadResult(
            np.zeros((1, 1, 1), dtype=np.uint32),
            {},
        )

    monkeypatch.setattr(hdf5, "_load_impl", fake_load_impl)

    hdf5.load("a_master.h5", dtype="uint32", verbose=False)

    assert calls["kwargs"]["output_dtype"] is np.uint32


def test_load_u32_routes_to_parallel_gpu_output_dtype(monkeypatch) -> None:
    """Public dtype='u32' should reach the multi-GPU/list load path."""
    from quantem.gpu.io import hdf5

    calls = {}

    def fake_load_many_parallel(paths, **kwargs):
        calls["paths"] = paths
        calls["kwargs"] = kwargs
        return [
            hdf5.LoadResult(np.zeros((1, 1, 1), dtype=np.uint32), {}),
            hdf5.LoadResult(np.zeros((1, 1, 1), dtype=np.uint32), {}),
        ]

    monkeypatch.setattr(hdf5, "_load_many_parallel", fake_load_many_parallel)

    hdf5.load(["a_master.h5", "b_master.h5"], dtype="u32", gpus=[0, 1], verbose=False)

    assert calls["paths"] == ["a_master.h5", "b_master.h5"]
    assert calls["kwargs"]["output_dtype"] is np.uint32


def test_load_u4_routes_to_packed_four_bit_output_dtype(monkeypatch) -> None:
    """Public dtype='u4' must not silently mean NumPy's four-byte uint32."""
    from quantem.gpu.io import hdf5
    from quantem.gpu.uint4 import pack_uint4_numpy

    calls = {}

    def fake_load_impl(filepath, *args, **kwargs):
        calls["filepath"] = filepath
        calls["kwargs"] = kwargs
        return hdf5.LoadResult(
            pack_uint4_numpy(np.zeros((1, 1, 1), dtype=np.uint8)),
            {},
        )

    monkeypatch.setattr(hdf5, "_load_impl", fake_load_impl)

    hdf5.load("a_master.h5", dtype="u4", verbose=False)

    assert calls["filepath"] == "a_master.h5"
    assert calls["kwargs"]["output_dtype"] == "uint4"


def test_load_u4_rejects_conflicting_output_dtype() -> None:
    """A packed uint4 request must not be combined with a different cast."""
    from quantem.gpu.io import hdf5

    with pytest.raises(ValueError, match="dtype='u4'"):
        hdf5.load(
            "a_master.h5",
            dtype="u4",
            output_dtype=np.uint8,
            verbose=False,
        )


def test_mps_output_dtype_u4_does_not_alias_to_uint32() -> None:
    """MPS dtype normalization must not pass public 'u4' to np.dtype."""
    source = Path("src/quantem/gpu/io/backends/mps.py").read_text()

    assert 'token in {"u4", "uint4"}' in source
    assert "not NumPy's four-byte '<u4' dtype" in source


def test_mps_multi_dataset_loader_threads_output_dtype(monkeypatch) -> None:
    """C1: lazy MPS browse loads, expect requested uint8 dtype to reach load."""
    from quantem.gpu.io import hdf5, mps_multi
    import quantem.gpu.detector.compute.mps.kernels as mps_compute

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

    monkeypatch.setattr(hdf5, "load", fake_load)
    monkeypatch.setattr(mps_compute, "ChunkedFrames", FakeChunkedFrames)
    monkeypatch.setattr(mps_compute, "MultiChunkedFrames", FakeMultiChunkedFrames)

    lazy = mps_multi.load_mps_datasets(
        ["tilt_0_master.h5", "tilt_1_master.h5"],
        det_bin=4,
        output_dtype=np.uint8,
        verbose=False,
    )

    assert lazy.det_bin == 4
    assert calls[0]["path"] == "tilt_0_master.h5"
    assert calls[0]["kwargs"]["backend"] == "mps"
    assert calls[0]["kwargs"]["det_bin"] == 4
    assert calls[0]["kwargs"]["output_dtype"] is np.uint8


def test_get_libc_returns_none_when_posix_fadvise_is_unavailable(monkeypatch) -> None:
    """macOS libc exists but does not expose Linux posix_fadvise."""
    import ctypes
    import ctypes.util

    from quantem.gpu.io import hdf5

    monkeypatch.setattr(ctypes.util, "find_library", lambda _name: "libc.dylib")
    monkeypatch.setattr(ctypes, "CDLL", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(hdf5, "_LIBC", None)

    assert hdf5._get_libc() is None
    assert hdf5._LIBC is False


def test_apply_scan_shape_supports_serpentine_order() -> None:
    """Full flat scans can be unflattened with odd scan rows reversed."""
    from quantem.gpu.io import hdf5

    data = np.arange(12, dtype=np.uint16).reshape(6, 1, 2)

    result = hdf5._apply_scan_shape(
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
    from quantem.gpu.io import hdf5

    indices = hdf5._scan_region_frame_indices(
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
    from quantem.gpu.io import hdf5

    with pytest.raises(ValueError, match="scan_order must be"):
        hdf5._normalize_scan_order("zigzag")


def test_load_with_scan_region_maps_scan_roi_to_flat_frames(tmp_path, monkeypatch) -> None:
    """Region loading should request only the flattened scan frames in row-major order."""
    from quantem.gpu.io import hdf5

    master = tmp_path / "scan_master.h5"
    master.write_bytes(b"placeholder")
    calls = {}

    monkeypatch.setattr(hdf5, "get_metadata", lambda _path: {"scan_shape": (5, 6)})
    monkeypatch.setattr(hdf5, "_discover_chunk_names", lambda _path: ["data_000001"])

    def fake_prepare(filepath, chunk_names, frame_indices, apply_mask=True):
        calls["filepath"] = filepath
        calls["chunk_names"] = chunk_names
        calls["frame_indices"] = frame_indices.copy()
        calls["apply_mask"] = apply_mask
        return {"pixel_mask": None, "dtype": np.dtype(np.uint16)}

    def fake_mps_decode(prepared, **kwargs):
        calls["mps_kwargs"] = kwargs
        return np.arange(6 * 2 * 2, dtype=np.uint16).reshape(6, 2, 2)

    monkeypatch.setattr(hdf5, "_prepare_master_frames", fake_prepare)
    monkeypatch.setitem(
        sys.modules,
        "quantem.gpu.io.backends.mps",
        types.SimpleNamespace(load_prepared_frames=fake_mps_decode),
    )

    result = hdf5.load(
        str(master),
        scan_region=(1, 3, 2, 5),
        backend="mps",
        verbose=False,
        det_bin=1,
    )

    np.testing.assert_array_equal(
        calls["frame_indices"],
        np.asarray([8, 9, 10, 14, 15, 16], dtype=np.int64),
    )
    assert result.data.shape == (2, 3, 2, 2)
    assert result.metadata["full_scan_shape"] == (5, 6)
    assert result.metadata["scan_shape"] == (2, 3)
    assert result.metadata["scan_order"] == "row-major"
    assert result.metadata["scan_region"] == {
        "row_start": 1,
        "row_stop": 3,
        "col_start": 2,
        "col_stop": 5,
        "shape": [2, 3],
    }


def test_load_with_scan_region_maps_serpentine_roi_to_flat_frames(tmp_path, monkeypatch) -> None:
    """Serpentine crop-first IO should read frames in corrected scan order."""
    from quantem.gpu.io import hdf5

    master = tmp_path / "scan_master.h5"
    master.write_bytes(b"placeholder")
    calls = {}

    monkeypatch.setattr(hdf5, "get_metadata", lambda _path: {"scan_shape": (5, 6)})
    monkeypatch.setattr(hdf5, "_discover_chunk_names", lambda _path: ["data_000001"])

    def fake_prepare(filepath, chunk_names, frame_indices, apply_mask=True):
        calls["frame_indices"] = frame_indices.copy()
        return {"pixel_mask": None, "dtype": np.dtype(np.uint16)}

    def fake_mps_decode(prepared, **kwargs):
        calls["mps_kwargs"] = kwargs
        return np.arange(6 * 2 * 2, dtype=np.uint16).reshape(6, 2, 2)

    monkeypatch.setattr(hdf5, "_prepare_master_frames", fake_prepare)
    monkeypatch.setitem(
        sys.modules,
        "quantem.gpu.io.backends.mps",
        types.SimpleNamespace(load_prepared_frames=fake_mps_decode),
    )

    result = hdf5.load(
        str(master),
        scan_region=(1, 3, 2, 5),
        backend="mps",
        scan_order="serpentine",
        verbose=False,
    )

    np.testing.assert_array_equal(
        calls["frame_indices"],
        np.asarray([9, 8, 7, 14, 15, 16], dtype=np.int64),
    )
    assert result.data.shape == (2, 3, 2, 2)
    assert result.metadata["scan_order"] == "serpentine"


def test_scan_region_crop_is_private_implementation_behind_load(monkeypatch) -> None:
    """The crop-first API is load(path, scan_region=...), not a second public verb."""
    from quantem.gpu.io import hdf5

    calls = {}

    def fake_resolve_backend(backend):
        calls["backend"] = backend
        return "cuda"

    def fake_load_scan_crop(filepath, scan_region, **kwargs):
        calls["filepath"] = filepath
        calls["scan_region"] = scan_region
        calls["kwargs"] = kwargs
        return hdf5.LoadResult(
            np.zeros((2, 3, 4, 4), dtype=np.uint8),
            {"scan_region": scan_region},
        )

    monkeypatch.setattr(hdf5, "_load_scan_crop_impl", fake_load_scan_crop)
    monkeypatch.setattr("quantem.gpu.io.backends.resolve_backend", fake_resolve_backend)

    result = hdf5.load(
        "scan_master.h5",
        scan_region=(1, 3, 2, 5),
        backend="auto",
        det_bin=2,
        dtype="u8",
        verbose=False,
    )

    assert calls["backend"] == "auto"
    assert calls["filepath"] == "scan_master.h5"
    assert calls["scan_region"] == (1, 3, 2, 5)
    assert calls["kwargs"] == {
        "scan_shape": None,
        "scan_order": "row-major",
        "det_bin": 2,
        "apply_mask": True,
        "verbose": False,
        "auto_narrow": True,
        "output_dtype": np.uint8,
        "target_scan_region": None,
        "scan_shift_row_col": None,
        "scan_resample_dtype": np.float32,
        "detector_region": None,
        "backend": "cuda",
    }
    assert result.data.dtype == np.uint8


def test_load_with_scan_region_detector_region_options_route_through_load(monkeypatch) -> None:
    """Detector cropping remains an option on load(), not a new load_* API."""
    from quantem.gpu.io import hdf5

    calls = {}

    def fake_resolve_backend(backend):
        calls["backend"] = backend
        return "cuda"

    def fake_load_scan_crop(filepath, scan_region, **kwargs):
        calls["filepath"] = filepath
        calls["scan_region"] = scan_region
        calls["kwargs"] = kwargs
        return hdf5.LoadResult(
            np.zeros((2, 3, 2, 4), dtype=np.uint8),
            {"detector_region": kwargs["detector_region"]},
        )

    monkeypatch.setattr(hdf5, "_load_scan_crop_impl", fake_load_scan_crop)
    monkeypatch.setattr("quantem.gpu.io.backends.resolve_backend", fake_resolve_backend)

    result = hdf5.load(
        "scan_master.h5",
        scan_region=(1, 3, 2, 5),
        detector_region=(4, 6, 0, 4),
        backend="auto",
        dtype="u8",
        verbose=False,
    )

    assert calls["backend"] == "auto"
    assert calls["filepath"] == "scan_master.h5"
    assert calls["scan_region"] == (1, 3, 2, 5)
    assert calls["kwargs"]["detector_region"] == (4, 6, 0, 4)
    assert calls["kwargs"]["output_dtype"] is np.uint8
    assert result.metadata["detector_region"] == (4, 6, 0, 4)


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
    from quantem.gpu.io import resample_scan_crop

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


def test_load_with_scan_region_resampling_options_route_through_load(monkeypatch) -> None:
    """Drift-resampled crop loading remains an option on load(), not a new API."""
    from quantem.gpu.io import hdf5

    calls = {}

    def fake_resolve_backend(backend):
        calls["backend"] = backend
        return "cuda"

    def fake_load_scan_crop_series(filepath, scan_region, **kwargs):
        calls["filepath"] = filepath
        calls["scan_region"] = scan_region
        calls["kwargs"] = kwargs
        return hdf5.LoadResult(
            np.zeros((2, 3, 4, 1, 1), dtype=np.float32),
            {"scan_region": kwargs["target_scan_region"]},
        )

    shifts = np.asarray([[0.25, -0.50], [1.25, 0.75]], dtype=np.float32)
    monkeypatch.setattr(hdf5, "_load_scan_crop_series_impl", fake_load_scan_crop_series)
    monkeypatch.setattr("quantem.gpu.io.backends.resolve_backend", fake_resolve_backend)

    result = hdf5.load(
        ["a_master.h5", "b_master.h5"],
        scan_region=[(0, 4, 0, 5), (1, 5, 2, 7)],
        target_scan_region=(1, 4, 1, 5),
        scan_shift_row_col=shifts,
        scan_resample_dtype=np.float32,
        scan_shape=(8, 9),
        backend="auto",
        dtype="u8",
        verbose=False,
        stack=True,
    )

    assert calls["backend"] == "auto"
    assert calls["filepath"] == ["a_master.h5", "b_master.h5"]
    assert calls["scan_region"] == [(0, 4, 0, 5), (1, 5, 2, 7)]
    assert calls["kwargs"]["target_scan_region"] == (1, 4, 1, 5)
    np.testing.assert_array_equal(calls["kwargs"]["scan_shift_row_col"], shifts)
    assert calls["kwargs"]["detector_region"] is None
    assert calls["kwargs"]["output_dtype"] is np.uint8
    assert result.data.dtype == np.float32


def test_load_with_scan_region_resampling_requires_shift(monkeypatch) -> None:
    """Alignment options must be complete so the coordinate frame is explicit."""
    from quantem.gpu.io import hdf5

    monkeypatch.setattr("quantem.gpu.io.backends.resolve_backend", lambda _backend: "cuda")

    with pytest.raises(ValueError, match="must be passed together"):
        hdf5.load(
            "scan_master.h5",
            scan_region=(0, 2, 0, 2),
            target_scan_region=(0, 1, 0, 1),
            backend="auto",
            verbose=False,
        )


def test_load_with_scan_region_resampling_rejects_mps_backend(monkeypatch) -> None:
    """Aligned scan-region output is CUDA-only until a Metal sampler exists."""
    from quantem.gpu.io import hdf5

    monkeypatch.setattr("quantem.gpu.io.backends.resolve_backend", lambda _backend: "mps")

    with pytest.raises(RuntimeError, match="backend='cuda'"):
        hdf5.load(
            "scan_master.h5",
            scan_region=(0, 2, 0, 2),
            target_scan_region=(0, 1, 0, 1),
            scan_shift_row_col=(0.0, 0.0),
            backend="mps",
            verbose=False,
        )


def test_load_series_resampling_rejects_wrong_shift_count(monkeypatch) -> None:
    """A drift-corrected time series needs one shift per loaded master."""
    from quantem.gpu.io import hdf5

    monkeypatch.setattr("quantem.gpu.io.backends.resolve_backend", lambda _backend: "cuda")

    with pytest.raises(TypeError, match=r"\(n_files, 2\)"):
        hdf5.load(
            ["a_master.h5", "b_master.h5"],
            scan_region=[(0, 2, 0, 2), (1, 3, 1, 3)],
            target_scan_region=(0, 2, 0, 2),
            scan_shift_row_col=np.asarray([[0.0, 0.0]], dtype=np.float32),
            backend="auto",
            scan_shape=(4, 4),
            verbose=False,
        )


def test_load_series_accepts_per_file_scan_regions(
    tmp_path,
    monkeypatch,
) -> None:
    """A time series can load drift-aware source crops without a union rectangle."""
    from quantem.gpu.io import hdf5

    masters = [tmp_path / "a_master.h5", tmp_path / "b_master.h5"]
    for master in masters:
        master.write_bytes(b"placeholder")

    calls = []
    monkeypatch.setattr(hdf5, "get_metadata", lambda _path: {"scan_shape": (5, 6)})
    monkeypatch.setattr(hdf5, "_discover_chunk_names", lambda _path: ["data_000001"])

    def fake_prepare(filepath, chunk_names, frame_indices, apply_mask=True):
        calls.append((Path(filepath).name, frame_indices.copy()))
        return {
            "selected_frame_indices": frame_indices.copy(),
            "pixel_mask": None,
            "dtype": np.dtype(np.uint16),
        }

    def fake_mps_decode(prepared, **kwargs):
        return prepared["selected_frame_indices"].astype(np.uint16).reshape(-1, 1, 1)

    monkeypatch.setattr(hdf5, "_prepare_master_frames", fake_prepare)
    monkeypatch.setitem(
        sys.modules,
        "quantem.gpu.io.backends.mps",
        types.SimpleNamespace(load_prepared_frames=fake_mps_decode),
    )

    result = hdf5.load(
        [str(path) for path in masters],
        scan_region=[(0, 2, 0, 2), (2, 4, 3, 5)],
        backend="mps",
        scan_shape=(5, 6),
        verbose=False,
        stack=True,
    )

    assert [name for name, _ in calls] == ["a_master.h5", "b_master.h5"]
    np.testing.assert_array_equal(calls[0][1], np.asarray([0, 1, 6, 7], dtype=np.int64))
    np.testing.assert_array_equal(calls[1][1], np.asarray([15, 16, 21, 22], dtype=np.int64))
    assert result.data.shape == (2, 2, 2, 1, 1)
    np.testing.assert_array_equal(result.data[0, :, :, 0, 0], [[0, 1], [6, 7]])
    np.testing.assert_array_equal(result.data[1, :, :, 0, 0], [[15, 16], [21, 22]])
    assert result.metadata["scan_region_mode"] == "per_file"
    assert result.metadata["scan_regions"][0]["row_start"] == 0
    assert result.metadata["scan_regions"][1]["col_start"] == 3


def test_load_series_per_file_scan_regions_support_variable_shapes_with_stack_false(
    tmp_path,
    monkeypatch,
) -> None:
    """Per-file drift crops may clip at scan edges, so stack=False keeps them exact."""
    from quantem.gpu.io import hdf5

    masters = [tmp_path / "a_master.h5", tmp_path / "b_master.h5"]
    for master in masters:
        master.write_bytes(b"placeholder")

    monkeypatch.setattr(hdf5, "get_metadata", lambda _path: {"scan_shape": (5, 6)})
    monkeypatch.setattr(hdf5, "_discover_chunk_names", lambda _path: ["data_000001"])

    def fake_prepare(filepath, chunk_names, frame_indices, apply_mask=True):
        return {
            "selected_frame_indices": frame_indices.copy(),
            "pixel_mask": None,
            "dtype": np.dtype(np.uint16),
        }

    def fake_mps_decode(prepared, **kwargs):
        return prepared["selected_frame_indices"].astype(np.uint16).reshape(-1, 1, 1)

    monkeypatch.setattr(hdf5, "_prepare_master_frames", fake_prepare)
    monkeypatch.setitem(
        sys.modules,
        "quantem.gpu.io.backends.mps",
        types.SimpleNamespace(load_prepared_frames=fake_mps_decode),
    )

    result = hdf5.load(
        [str(path) for path in masters],
        scan_region=[(0, 2, 0, 2), (2, 5, 3, 5)],
        backend="mps",
        scan_shape=(5, 6),
        verbose=False,
        stack=False,
    )

    assert [tuple(array.shape) for array in result.data] == [(2, 2, 1, 1), (3, 2, 1, 1)]
    assert result.metadata["scan_region_mode"] == "per_file"
    assert result.metadata["scan_shape"] == (2, 2)


def test_full_scan_region_routes_to_full_loader(monkeypatch) -> None:
    """A full scan_region should not force the sparse crop loader."""
    from quantem.gpu.io import hdf5

    calls = {}

    def fake_resolve_backend(backend):
        calls["backend"] = backend
        return "cuda"

    def fake_load_impl(filepath, **kwargs):
        calls["filepath"] = filepath
        calls["kwargs"] = kwargs
        return hdf5.LoadResult(
            np.zeros((4, 5, 2, 2), dtype=np.uint8),
            {"scan_shape": (4, 5)},
        )

    def fail_load_scan_crop(*args, **kwargs):
        raise AssertionError("full scan region must use the full loader")

    monkeypatch.setattr(hdf5, "_load_impl", fake_load_impl)
    monkeypatch.setattr(hdf5, "_load_scan_crop_impl", fail_load_scan_crop)
    monkeypatch.setattr("quantem.gpu.io.backends.resolve_backend", fake_resolve_backend)

    result = hdf5.load(
        "scan_master.h5",
        scan_region=(0, 4, 0, 5),
        scan_shape=(4, 5),
        backend="auto",
        det_bin=2,
        dtype="u8",
        verbose=False,
    )

    assert calls["backend"] == "auto"
    assert calls["filepath"] == "scan_master.h5"
    assert calls["kwargs"] == {
        "scan_shape": (4, 5),
        "scan_order": "row-major",
        "det_bin": 2,
        "apply_mask": True,
        "verbose": False,
        "auto_narrow": True,
        "output_dtype": np.uint8,
        "backend": "cuda",
    }
    assert result.data.shape == (4, 5, 2, 2)


def test_full_scan_region_with_detector_region_keeps_crop_loader(monkeypatch) -> None:
    """A full scan source still needs the crop path when detector rows are selected."""
    from quantem.gpu.io import hdf5

    calls = {}

    def fake_resolve_backend(backend):
        calls["backend"] = backend
        return "cuda"

    def fail_load_impl(*args, **kwargs):
        raise AssertionError("detector_region must not bypass the scan crop loader")

    def fake_load_scan_crop(filepath, scan_region, **kwargs):
        calls["filepath"] = filepath
        calls["scan_region"] = scan_region
        calls["kwargs"] = kwargs
        return hdf5.LoadResult(
            np.zeros((4, 5, 2, 3), dtype=np.uint8),
            {"detector_region": kwargs["detector_region"]},
        )

    monkeypatch.setattr(hdf5, "_load_impl", fail_load_impl)
    monkeypatch.setattr(hdf5, "_load_scan_crop_impl", fake_load_scan_crop)
    monkeypatch.setattr("quantem.gpu.io.backends.resolve_backend", fake_resolve_backend)

    result = hdf5.load(
        "scan_master.h5",
        scan_region=(0, 4, 0, 5),
        detector_region=(1, 3, 2, 5),
        scan_shape=(4, 5),
        backend="auto",
        dtype="u8",
        verbose=False,
    )

    assert calls["backend"] == "auto"
    assert calls["filepath"] == "scan_master.h5"
    assert calls["scan_region"] == (0, 4, 0, 5)
    assert calls["kwargs"]["detector_region"] == (1, 3, 2, 5)
    assert calls["kwargs"]["output_dtype"] is np.uint8
    assert result.data.shape == (4, 5, 2, 3)


def test_full_scan_region_with_resampling_keeps_crop_loader(monkeypatch) -> None:
    """A full source read still needs the crop path when drift resampling is requested."""
    from quantem.gpu.io import hdf5

    calls = {}

    def fake_resolve_backend(backend):
        calls["backend"] = backend
        return "cuda"

    def fail_load_impl(*args, **kwargs):
        raise AssertionError("resampling must not bypass the scan crop loader")

    def fake_load_scan_crop(filepath, scan_region, **kwargs):
        calls["filepath"] = filepath
        calls["scan_region"] = scan_region
        calls["kwargs"] = kwargs
        return hdf5.LoadResult(
            np.zeros((4, 5, 2, 2), dtype=np.float32),
            {"scan_resampling": {"mode": "bilinear"}},
        )

    monkeypatch.setattr(hdf5, "_load_impl", fail_load_impl)
    monkeypatch.setattr(hdf5, "_load_scan_crop_impl", fake_load_scan_crop)
    monkeypatch.setattr("quantem.gpu.io.backends.resolve_backend", fake_resolve_backend)

    result = hdf5.load(
        "scan_master.h5",
        scan_region=(0, 4, 0, 5),
        target_scan_region=(0, 4, 0, 5),
        scan_shift_row_col=(0.25, -0.5),
        scan_shape=(4, 5),
        backend="auto",
        dtype="u8",
        verbose=False,
    )

    assert calls["backend"] == "auto"
    assert calls["filepath"] == "scan_master.h5"
    assert calls["scan_region"] == (0, 4, 0, 5)
    assert calls["kwargs"]["target_scan_region"] == (0, 4, 0, 5)
    assert calls["kwargs"]["scan_shift_row_col"] == (0.25, -0.5)
    assert calls["kwargs"]["output_dtype"] is np.uint8
    assert result.data.dtype == np.float32


def test_scan_indices_rowcol_supports_serpentine_order() -> None:
    """Sparse scan positions map logical row/col to physical HDF5 frames."""
    from quantem.gpu.io import hdf5

    positions = np.asarray(
        [
            [1, 2],
            [0, 5],
            [1, 4],
            [1, 2],
        ],
        dtype=np.int64,
    )

    frame_indices, logical_positions = hdf5._normalize_scan_indices(
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
    from quantem.gpu.io import hdf5

    master = tmp_path / "scan_master.h5"
    master.write_bytes(b"placeholder")
    calls = {}

    monkeypatch.setattr(hdf5, "get_metadata", lambda _path: {"scan_shape": (5, 6)})
    monkeypatch.setattr(hdf5, "_discover_chunk_names", lambda _path: ["data_000001"])

    def fake_prepare(filepath, chunk_names, frame_indices, apply_mask=True):
        calls["filepath"] = filepath
        calls["chunk_names"] = chunk_names
        calls["frame_indices"] = frame_indices.copy()
        calls["apply_mask"] = apply_mask
        return {
            "selected_frame_indices": frame_indices.copy(),
            "pixel_mask": np.zeros((1, 1), dtype=np.uint8),
            "dtype": np.dtype(np.uint16),
        }

    def fake_mps_decode(prepared, **kwargs):
        calls["mps_kwargs"] = kwargs
        values = prepared["selected_frame_indices"].astype(np.uint16)
        return values.reshape(-1, 1, 1)

    monkeypatch.setattr(hdf5, "_prepare_master_frames", fake_prepare)
    monkeypatch.setitem(
        sys.modules,
        "quantem.gpu.io.backends.mps",
        types.SimpleNamespace(load_prepared_frames=fake_mps_decode),
    )

    result = hdf5.load_scan_indices(
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


def test_load_scan_indices_multi_file_accepts_per_file_batches(
    tmp_path,
    monkeypatch,
) -> None:
    """Multi-master sparse IO should support different random positions per file."""
    from quantem.gpu.io import hdf5

    masters = [tmp_path / "a_master.h5", tmp_path / "b_master.h5"]
    for master in masters:
        master.write_bytes(b"placeholder")

    monkeypatch.setattr(hdf5, "get_metadata", lambda _path: {"scan_shape": (4, 4)})
    monkeypatch.setattr(hdf5, "_discover_chunk_names", lambda _path: ["data_000001"])

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

    monkeypatch.setattr(hdf5, "_prepare_master_frames", fake_prepare)
    monkeypatch.setitem(
        sys.modules,
        "quantem.gpu.io.backends.mps",
        types.SimpleNamespace(load_prepared_frames=fake_mps_decode),
    )

    result = hdf5.load_scan_indices(
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


def test_load_scan_indices_is_available_through_load(monkeypatch) -> None:
    """The friendly sparse API is load(path, scan_indices=...)."""
    from quantem.gpu.io import hdf5

    calls = {}

    def fake_resolve_backend(backend):
        calls["backend"] = backend
        return "cuda"

    def fake_load_scan_indices(filepath, scan_indices, **kwargs):
        calls["filepath"] = filepath
        calls["scan_indices"] = scan_indices
        calls["kwargs"] = kwargs
        return hdf5.LoadResult(
            np.zeros((4, 2, 2), dtype=np.uint8),
            {"scan_indices": scan_indices},
        )

    monkeypatch.setattr(hdf5, "load_scan_indices", fake_load_scan_indices)
    monkeypatch.setattr("quantem.gpu.io.backends.resolve_backend", fake_resolve_backend)

    result = hdf5.load(
        "scan_master.h5",
        scan_indices=[8, 4, 8, 2],
        backend="auto",
        det_bin=2,
        dtype="u8",
        verbose=False,
    )

    assert calls["backend"] == "auto"
    assert calls["filepath"] == "scan_master.h5"
    assert calls["scan_indices"] == [8, 4, 8, 2]
    assert calls["kwargs"] == {
        "scan_shape": None,
        "scan_order": "row-major",
        "index_mode": "scan",
        "det_bin": 2,
        "apply_mask": True,
        "verbose": False,
        "auto_narrow": True,
        "output_dtype": np.uint8,
        "backend": "cuda",
        "stack": True,
        "prep_workers": None,
    }
    assert result.data.dtype == np.uint8


def test_random_scan_indices_are_reproducible_and_per_file() -> None:
    """Random scan sampling should look like a deterministic DataLoader sampler."""
    from quantem.gpu.io import hdf5

    one = hdf5.random_scan_indices(4, (4, 4), seed=123)
    again = hdf5.random_scan_indices(4, (4, 4), seed=123)
    per_file = hdf5.random_scan_indices(4, (4, 4), n_files=3, seed=123)
    positions = hdf5.random_scan_indices(
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
    assert len(set(int(v) for v in one)) == 4
    assert not np.array_equal(per_file[0], per_file[1])
    np.testing.assert_array_equal(positions[:, 0] * 4 + positions[:, 1], one)


def test_random_scan_indices_rejects_oversampling_without_replacement() -> None:
    """Without replacement, random sampling should fail before any IO starts."""
    from quantem.gpu.io import hdf5

    with pytest.raises(ValueError, match="Cannot sample"):
        hdf5.random_scan_indices(17, (4, 4), replace=False)


def test_sparse_prep_workers_default_to_single_reader() -> None:
    """Sparse HDF5 prep should not assume more readers are faster."""
    from quantem.gpu.io import hdf5

    assert hdf5._normalize_prep_workers(None, n_files=40) == 1
    assert hdf5._normalize_prep_workers(8, n_files=40) == 8
    assert hdf5._normalize_prep_workers(100, n_files=40) == 40

    with pytest.raises(ValueError, match="positive integer"):
        hdf5._normalize_prep_workers(0, n_files=40)


def test_load_random_positions_is_available_through_load(monkeypatch) -> None:
    """The easy stochastic API is load(path, random_positions=...)."""
    from quantem.gpu.io import hdf5

    calls = {}

    def fake_resolve_backend(backend):
        calls["backend"] = backend
        return "cuda"

    def fake_load_scan_indices(filepath, scan_indices, **kwargs):
        calls["filepath"] = filepath
        calls["scan_indices"] = np.asarray(scan_indices)
        calls["kwargs"] = kwargs
        return hdf5.LoadResult(
            np.zeros((2, 4, 2, 2), dtype=np.uint8),
            {},
        )

    monkeypatch.setattr(hdf5, "load_scan_indices", fake_load_scan_indices)
    monkeypatch.setattr("quantem.gpu.io.backends.resolve_backend", fake_resolve_backend)

    result = hdf5.load(
        ["a_master.h5", "b_master.h5"],
        random_positions=4,
        seed=123,
        scan_shape=(4, 4),
        backend="auto",
        dtype="u8",
        verbose=False,
        prep_workers=3,
    )

    expected = hdf5.random_scan_indices(4, (4, 4), n_files=2, seed=123)
    assert calls["backend"] == "auto"
    assert calls["filepath"] == ["a_master.h5", "b_master.h5"]
    np.testing.assert_array_equal(calls["scan_indices"], expected)
    assert calls["kwargs"]["prep_workers"] == 3
    assert calls["kwargs"]["output_dtype"] is np.uint8
    assert result.metadata["sample"] == {
        "mode": "random_positions",
        "positions_per_file": 4,
        "scan_shape": [4, 4],
        "seed": 123,
        "replace": False,
        "same_random_positions": False,
        "n_files": 2,
        "index_space": "logical_row_major_scan",
    }


def test_load_rejects_mixed_sparse_modes() -> None:
    """Callers should choose either explicit or generated scan positions."""
    from quantem.gpu.io import hdf5

    with pytest.raises(ValueError, match="Pass only one"):
        hdf5.load(
            "scan_master.h5",
            scan_indices=[1, 2],
            random_positions=2,
            scan_shape=(4, 4),
        )


def test_load_random_positions_rejects_hdf5_index_mode() -> None:
    """Generated random positions are always logical scan positions."""
    from quantem.gpu.io import hdf5

    with pytest.raises(ValueError, match="random_positions generates logical"):
        hdf5.load(
            "scan_master.h5",
            random_positions=2,
            scan_shape=(4, 4),
            index_mode="hdf5",
        )


def test_load_with_scan_region_routes_mps_to_sparse_decoder(tmp_path, monkeypatch) -> None:
    """MPS crop-first IO should use quantem.gpu's sparse Metal decode path."""
    from quantem.gpu.io import hdf5

    master = tmp_path / "scan_master.h5"
    master.write_bytes(b"placeholder")
    calls = {}

    monkeypatch.setattr(hdf5, "get_metadata", lambda _path: {"scan_shape": (4, 4)})
    monkeypatch.setattr(hdf5, "_discover_chunk_names", lambda _path: ["data_000001"])

    def fake_prepare(filepath, chunk_names, frame_indices, apply_mask=True):
        calls["frame_indices"] = frame_indices.copy()
        return {
            "pixel_mask": np.zeros((2, 2), dtype=np.uint8),
            "dtype": np.dtype(np.uint16),
        }

    def fake_mps_decode(prepared, **kwargs):
        calls["prepared"] = prepared
        calls["mps_kwargs"] = kwargs
        return np.arange(4 * 2 * 2, dtype=np.uint16).reshape(4, 2, 2)

    monkeypatch.setattr(hdf5, "_prepare_master_frames", fake_prepare)
    monkeypatch.setitem(
        sys.modules,
        "quantem.gpu.io.backends.mps",
        types.SimpleNamespace(load_prepared_frames=fake_mps_decode),
    )

    result = hdf5.load(
        str(master),
        scan_region=(1, 3, 1, 3),
        backend="mps",
        scan_order="serpentine",
        verbose=False,
    )

    np.testing.assert_array_equal(
        calls["frame_indices"],
        np.asarray([6, 5, 9, 10], dtype=np.int64),
    )
    assert calls["mps_kwargs"]["det_bin"] == 1
    assert calls["mps_kwargs"]["pixel_mask"].shape == (2, 2)
    assert result.data.shape == (2, 2, 2, 2)
    assert result.metadata["backend"] == "mps"
    assert result.metadata["scan_order"] == "serpentine"


def test_load_with_scan_and_detector_region_crops_decoded_detector(
    tmp_path,
    monkeypatch,
) -> None:
    """Detector-region output should slice decoded detector rows/columns exactly."""
    from quantem.gpu.io import hdf5

    master = tmp_path / "scan_master.h5"
    master.write_bytes(b"placeholder")

    monkeypatch.setattr(hdf5, "get_metadata", lambda _path: {"scan_shape": (4, 4)})
    monkeypatch.setattr(hdf5, "_discover_chunk_names", lambda _path: ["data_000001"])

    def fake_prepare(filepath, chunk_names, frame_indices, apply_mask=True):
        return {
            "pixel_mask": None,
            "dtype": np.dtype(np.uint16),
        }

    decoded = np.arange(4 * 3 * 4, dtype=np.uint16).reshape(4, 3, 4)

    def fake_mps_decode(prepared, **kwargs):
        return decoded.copy()

    monkeypatch.setattr(hdf5, "_prepare_master_frames", fake_prepare)
    monkeypatch.setitem(
        sys.modules,
        "quantem.gpu.io.backends.mps",
        types.SimpleNamespace(load_prepared_frames=fake_mps_decode),
    )

    result = hdf5.load(
        str(master),
        scan_region=(1, 3, 1, 3),
        detector_region=(1, 3, 1, 4),
        backend="mps",
        verbose=False,
    )

    expected = decoded.reshape(2, 2, 3, 4)[..., 1:3, 1:4]
    np.testing.assert_array_equal(result.data, expected)
    assert result.data.shape == (2, 2, 2, 3)
    assert result.metadata["decoded_detector_shape"] == (3, 4)
    assert result.metadata["output_detector_shape"] == (2, 3)
    assert result.metadata["detector_region"] == {
        "row_start": 1,
        "row_stop": 3,
        "col_start": 1,
        "col_stop": 4,
        "shape": [2, 3],
    }


def test_mps_multi_dataset_loader_is_owned_by_quantem_gpu(monkeypatch) -> None:
    """MPS list loads should dispatch to quantem.gpu.io, not widget IO."""
    from quantem.gpu.io import hdf5
    from quantem.gpu.io import mps_multi

    calls = {}

    monkeypatch.setattr("quantem.gpu.io.backends.resolve_backend", lambda _backend: "mps")

    def fake_load_mps_datasets(filepath, **kwargs):
        calls["filepath"] = filepath
        calls["kwargs"] = kwargs
        return "lazy-mps-handle"

    monkeypatch.setattr(mps_multi, "load_mps_datasets", fake_load_mps_datasets)

    result = hdf5.load(
        ["a_master.h5", "b_master.h5"],
        backend="mps",
        det_bin=4,
        verbose=False,
    )

    assert result == "lazy-mps-handle"
    assert calls["filepath"] == ["a_master.h5", "b_master.h5"]
    assert calls["kwargs"]["det_bin"] == 4
    assert calls["kwargs"]["verbose"] is False


def test_load_with_scan_region_rejects_slice_and_range_forms() -> None:
    """Keep the public crop API simple: one flat row/column bounds tuple."""
    from quantem.gpu.io import hdf5

    with pytest.raises(TypeError, match="scan_region must be"):
        hdf5._normalize_scan_region((slice(0, 1), range(0, 1)), (5, 6))


def test_load_with_detector_region_rejects_invalid_bounds() -> None:
    """Detector-region bounds use explicit detector row/column intervals."""
    from quantem.gpu.io import hdf5

    with pytest.raises(TypeError, match="detector_region must be"):
        hdf5._normalize_detector_region((slice(0, 1), 2, 0, 1), (5, 6))
    with pytest.raises(ValueError, match="detector row region"):
        hdf5._normalize_detector_region((-1, 2, 0, 1), (5, 6))
    with pytest.raises(ValueError, match="detector column region"):
        hdf5._normalize_detector_region((0, 2, 3, 7), (5, 6))


def test_load_with_detector_region_rejects_uint4(monkeypatch) -> None:
    """Packed uint4 columns cannot be sliced with physical detector-column bounds."""
    from quantem.gpu.io import hdf5

    monkeypatch.setattr("quantem.gpu.io.backends.resolve_backend", lambda _backend: "cuda")

    with pytest.raises(ValueError, match="detector_region=.*dtype='u4'"):
        hdf5.load(
            "scan_master.h5",
            scan_region=(0, 1, 0, 1),
            detector_region=(0, 1, 0, 1),
            dtype="u4",
            backend="auto",
            verbose=False,
        )


def test_load_with_scan_region_rejects_cpu_backend(monkeypatch) -> None:
    """Crop-first loading should fail honestly when no accelerated backend exists."""
    from quantem.gpu.io import hdf5

    monkeypatch.setattr("quantem.gpu.io.backends.resolve_backend", lambda _backend: "cpu")

    with pytest.raises(RuntimeError, match="CUDA and MPS"):
        hdf5.load(
            "scan_master.h5",
            scan_region=(0, 1, 0, 1),
            backend="cpu",
            verbose=False,
        )


def test_load_region_keyword_is_not_supported() -> None:
    from quantem.gpu.io import hdf5

    with pytest.raises(TypeError, match="region="):
        hdf5.load(
            "scan_master.h5",
            region=(0, 1, 0, 1),
            verbose=False,
        )


def test_torch_detector_bin_sum_matches_numpy_reference() -> None:
    torch = pytest.importorskip("torch")
    from quantem.gpu.io import bin

    data_np = np.arange(2 * 3 * 4 * 4, dtype=np.uint16).reshape(2, 3, 4, 4)
    data_torch = torch.as_tensor(data_np)

    out = bin(data_torch, factor=2, axes="detector", reduction="sum")

    expected = data_np.reshape(2, 3, 2, 2, 2, 2).sum(axis=(3, 5), dtype=np.uint64)
    np.testing.assert_array_equal(out.numpy(), expected.astype(np.int64))
