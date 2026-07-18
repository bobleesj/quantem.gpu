from __future__ import annotations

import sys
import types

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


def test_load_scan_region_maps_scan_roi_to_flat_frames(tmp_path, monkeypatch) -> None:
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
        return {"pixel_mask": None}

    def fake_decompress(prepared, **kwargs):
        calls["decompress_kwargs"] = kwargs
        return np.arange(6 * 2 * 2, dtype=np.uint16).reshape(6, 2, 2)

    monkeypatch.setattr(hdf5, "_prepare_master_frames", fake_prepare)
    monkeypatch.setattr(hdf5, "_decompress_prepared", fake_decompress)

    result = hdf5.load_scan_region(
        str(master),
        scan_region=(1, 3, 2, 5),
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


def test_load_scan_region_maps_serpentine_roi_to_flat_frames(tmp_path, monkeypatch) -> None:
    """Serpentine crop-first IO should read frames in corrected scan order."""
    from quantem.gpu.io import hdf5

    master = tmp_path / "scan_master.h5"
    master.write_bytes(b"placeholder")
    calls = {}

    monkeypatch.setattr(hdf5, "get_metadata", lambda _path: {"scan_shape": (5, 6)})
    monkeypatch.setattr(hdf5, "_discover_chunk_names", lambda _path: ["data_000001"])

    def fake_prepare(filepath, chunk_names, frame_indices, apply_mask=True):
        calls["frame_indices"] = frame_indices.copy()
        return {"pixel_mask": None}

    def fake_decompress(prepared, **kwargs):
        calls["decompress_kwargs"] = kwargs
        return np.arange(6 * 2 * 2, dtype=np.uint16).reshape(6, 2, 2)

    monkeypatch.setattr(hdf5, "_prepare_master_frames", fake_prepare)
    monkeypatch.setattr(hdf5, "_decompress_prepared", fake_decompress)

    result = hdf5.load_scan_region(
        str(master),
        scan_region=(1, 3, 2, 5),
        scan_order="serpentine",
        verbose=False,
    )

    np.testing.assert_array_equal(
        calls["frame_indices"],
        np.asarray([9, 8, 7, 14, 15, 16], dtype=np.int64),
    )
    assert result.data.shape == (2, 3, 2, 2)
    assert result.metadata["scan_order"] == "serpentine"


def test_load_scan_region_is_available_through_load(monkeypatch) -> None:
    """The friendly crop-first API is load(path, scan_region=...), not a second verb."""
    from quantem.gpu.io import hdf5

    calls = {}

    def fake_resolve_backend(backend):
        calls["backend"] = backend
        return "cuda"

    def fake_load_scan_region(filepath, scan_region, **kwargs):
        calls["filepath"] = filepath
        calls["scan_region"] = scan_region
        calls["kwargs"] = kwargs
        return hdf5.LoadResult(
            np.zeros((2, 3, 4, 4), dtype=np.uint8),
            {"scan_region": scan_region},
        )

    monkeypatch.setattr(hdf5, "load_scan_region", fake_load_scan_region)
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
        "backend": "cuda",
    }
    assert result.data.dtype == np.uint8


def test_load_scan_region_routes_mps_to_sparse_decoder(tmp_path, monkeypatch) -> None:
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


def test_load_scan_region_rejects_slice_and_range_forms() -> None:
    """Keep the public crop API simple: one flat row/column bounds tuple."""
    from quantem.gpu.io import hdf5

    with pytest.raises(TypeError, match="scan_region must be"):
        hdf5._normalize_scan_region((slice(0, 1), range(0, 1)), (5, 6))


def test_load_scan_region_rejects_cpu_backend(monkeypatch) -> None:
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
