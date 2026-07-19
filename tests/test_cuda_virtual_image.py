from __future__ import annotations

import numpy as np
import pytest


def _cupy_with_device():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CUDA device is not available.")
    except cp.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CUDA device is not available: {exc}")
    return cp


def _mask(det_shape: tuple[int, int], radius: float, *, invert: bool = False) -> np.ndarray:
    row = np.arange(det_shape[0], dtype=np.float32)[:, None]
    col = np.arange(det_shape[1], dtype=np.float32)[None, :]
    center = ((det_shape[0] - 1) / 2.0, (det_shape[1] - 1) / 2.0)
    disk = (row - center[0]) ** 2 + (col - center[1]) ** 2 <= radius**2
    return ~disk if invert else disk


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_cuda_masked_sum_matches_cupy_selected_sum(dtype) -> None:
    cp = _cupy_with_device()
    from quantem.gpu.compute.cuda import cuda_masked_sum

    rng = np.random.default_rng(31)
    data_np = rng.integers(0, 200, size=(5, 6, 12, 10), dtype=dtype)
    data = cp.asarray(data_np)
    det_mask = _mask((12, 10), 3.0)

    got = cuda_masked_sum(data, det_mask)
    expected = (
        data.reshape(-1, 12 * 10)[:, cp.asarray(det_mask.reshape(-1))]
        .sum(axis=1, dtype=cp.uint64)
        .astype(cp.float32)
        .reshape(5, 6)
    )

    cp.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_cuda_sum_all_matches_cupy_row_sum(dtype) -> None:
    cp = _cupy_with_device()
    from quantem.gpu.compute.cuda import cuda_sum_all_uint64

    rng = np.random.default_rng(33)
    data_np = rng.integers(0, 200, size=(4, 5, 13, 11), dtype=dtype)
    data = cp.asarray(data_np)

    got = cuda_sum_all_uint64(data)
    expected = data.reshape(-1, 13 * 11).sum(axis=1, dtype=cp.uint64).reshape(4, 5)

    cp.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_cuda_center_of_mass_matches_reference(dtype) -> None:
    cp = _cupy_with_device()
    from quantem.gpu.compute.cuda import cuda_center_of_mass

    rng = np.random.default_rng(35)
    data_np = rng.integers(0, 200, size=(4, 5, 13, 11), dtype=dtype)
    data = cp.asarray(data_np)
    rows = cp.arange(13, dtype=cp.float64)[:, None]
    cols = cp.arange(11, dtype=cp.float64)[None, :]
    total = cp.maximum(data.sum(axis=(2, 3), dtype=cp.float64), 1e-10)
    expected_row = ((data * rows).sum(axis=(2, 3), dtype=cp.float64) / total).astype(
        cp.float32
    )
    expected_col = ((data * cols).sum(axis=(2, 3), dtype=cp.float64) / total).astype(
        cp.float32
    )

    got_row, got_col = cuda_center_of_mass(data)

    cp.testing.assert_allclose(got_row, expected_row, rtol=0, atol=1e-6)
    cp.testing.assert_allclose(got_col, expected_col, rtol=0, atol=1e-6)


def test_cuda_center_of_mass_masked_matches_reference() -> None:
    cp = _cupy_with_device()
    from quantem.gpu.compute.cuda import cuda_center_of_mass

    rng = np.random.default_rng(39)
    data_np = rng.integers(0, 60000, size=(3, 4, 10, 12), dtype=np.uint16)
    data = cp.asarray(data_np)
    det_mask = _mask((10, 12), 3.0)
    mask = cp.asarray(det_mask)
    rows = cp.arange(10, dtype=cp.float64)[:, None]
    cols = cp.arange(12, dtype=cp.float64)[None, :]
    masked = data * mask
    total = cp.maximum(masked.sum(axis=(2, 3), dtype=cp.float64), 1e-10)
    expected_row = ((masked * rows).sum(axis=(2, 3), dtype=cp.float64) / total).astype(
        cp.float32
    )
    expected_col = ((masked * cols).sum(axis=(2, 3), dtype=cp.float64) / total).astype(
        cp.float32
    )

    got_row, got_col = cuda_center_of_mass(data, det_mask)

    cp.testing.assert_allclose(got_row, expected_row, rtol=0, atol=1e-6)
    cp.testing.assert_allclose(got_col, expected_col, rtol=0, atol=1e-6)


def test_cuda_dense_mask_uses_integer_complement_and_matches_cupy() -> None:
    cp = _cupy_with_device()
    from quantem.gpu.compute.cuda import cuda_masked_sum

    rng = np.random.default_rng(37)
    data_np = rng.integers(0, 60000, size=(4, 4, 16, 16), dtype=np.uint16)
    data = cp.asarray(data_np)
    det_mask = _mask((16, 16), 3.5, invert=True)

    got = cuda_masked_sum(data, det_mask)
    expected = (
        data.reshape(-1, 16 * 16)[:, cp.asarray(det_mask.reshape(-1))]
        .sum(axis=1, dtype=cp.uint64)
        .astype(cp.float32)
        .reshape(4, 4)
    )

    cp.testing.assert_array_equal(got, expected)


def test_cuda_virtual_image_kernel_source_uses_warp_and_fused_dense_path() -> None:
    from quantem.gpu.compute.cuda import _CUDA_VI_CODE

    assert "__shfl_down_sync" in _CUDA_VI_CODE
    assert "selected_sum_f32_u16_16f" in _CUDA_VI_CODE
    assert "selected_sum_from_total_f32_u16_16f" in _CUDA_VI_CODE
    assert "total_sum_u16_4f" in _CUDA_VI_CODE
    assert "center_of_mass_full_u16_4f" in _CUDA_VI_CODE
    assert "center_of_mass_selected_u16_4f" in _CUDA_VI_CODE


def test_cupy_compute_backend_dispatches_to_cuda_kernel_backend() -> None:
    cp = _cupy_with_device()
    from quantem.gpu.compute.backends import CudaKernelCompute, compute_backend

    data = cp.ones((4, 4, 12, 12), dtype=cp.uint16)
    backend = compute_backend(data)

    assert isinstance(backend, CudaKernelCompute)

    sparse_mask = _mask((12, 12), 2.0)
    sparse = backend.masked_sum(sparse_mask)
    np.testing.assert_array_equal(sparse, np.full((4, 4), sparse_mask.sum(), np.float32))
    assert backend._total_cache_uint64 is None
    assert len(backend._mask_index_cache) == 1

    dense_mask = ~sparse_mask
    dense = backend.masked_sum(dense_mask)
    np.testing.assert_array_equal(dense, np.full((4, 4), dense_mask.sum(), np.float32))
    assert backend._total_cache_uint64 is not None
    assert len(backend._mask_index_cache) == 1


def test_cuda_compute_backend_caches_full_center_of_mass() -> None:
    cp = _cupy_with_device()
    from quantem.gpu.compute.backends import CudaKernelCompute, compute_backend

    rng = np.random.default_rng(41)
    data_np = rng.integers(0, 200, size=(4, 5, 13, 11), dtype=np.uint16)
    data = cp.asarray(data_np)
    rows = cp.arange(13, dtype=cp.float64)[:, None]
    cols = cp.arange(11, dtype=cp.float64)[None, :]
    total = cp.maximum(data.sum(axis=(2, 3), dtype=cp.float64), 1e-10)
    expected_row = cp.asnumpy(
        ((data * rows).sum(axis=(2, 3), dtype=cp.float64) / total).astype(cp.float32)
    )
    expected_col = cp.asnumpy(
        ((data * cols).sum(axis=(2, 3), dtype=cp.float64) / total).astype(cp.float32)
    )
    backend = compute_backend(data)
    assert isinstance(backend, CudaKernelCompute)
    got_col, got_row = backend.center_of_mass()
    cached_col, cached_row = backend.center_of_mass()

    assert cached_col is got_col
    assert cached_row is got_row
    np.testing.assert_allclose(got_row.reshape(4, 5), expected_row, rtol=0, atol=1e-6)
    np.testing.assert_allclose(got_col.reshape(4, 5), expected_col, rtol=0, atol=1e-6)
