from __future__ import annotations

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
    assert result.metadata["scan_region"] == {
        "row_start": 1,
        "row_stop": 3,
        "col_start": 2,
        "col_stop": 5,
        "shape": [2, 3],
    }


def test_torch_detector_bin_sum_matches_numpy_reference() -> None:
    torch = pytest.importorskip("torch")
    from quantem.gpu.io import bin

    data_np = np.arange(2 * 3 * 4 * 4, dtype=np.uint16).reshape(2, 3, 4, 4)
    data_torch = torch.as_tensor(data_np)

    out = bin(data_torch, factor=2, axes="detector", reduction="sum")

    expected = data_np.reshape(2, 3, 2, 2, 2, 2).sum(axis=(3, 5), dtype=np.uint64)
    np.testing.assert_array_equal(out.numpy(), expected.astype(np.int64))
