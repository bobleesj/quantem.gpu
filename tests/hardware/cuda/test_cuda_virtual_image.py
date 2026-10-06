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


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32])
def test_cuda_masked_sum_matches_cupy_selected_sum(dtype) -> None:
    cp = _cupy_with_device()
    from quantem.gpu.detector.cuda.dense import CudaKernelCompute

    rng = np.random.default_rng(31)
    data_np = rng.integers(0, 200, size=(5, 6, 12, 10), dtype=dtype)
    data = cp.asarray(data_np)
    det_mask = _mask((12, 10), 3.0)

    got = CudaKernelCompute(data).masked_sum(det_mask)
    expected = (
        data.reshape(-1, 12 * 10)[:, cp.asarray(det_mask.reshape(-1))]
        .sum(axis=1, dtype=cp.uint64)
        .astype(cp.float32)
        .reshape(5, 6)
    )

    cp.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32])
def test_cuda_sum_all_matches_cupy_row_sum(dtype) -> None:
    cp = _cupy_with_device()
    from quantem.gpu.detector.cuda.dense import CudaKernelCompute

    rng = np.random.default_rng(33)
    data_np = rng.integers(0, 200, size=(4, 5, 13, 11), dtype=dtype)
    data = cp.asarray(data_np)

    got = CudaKernelCompute(data).masked_sum_exact(np.ones((13, 11), dtype=bool))
    expected = data.reshape(-1, 13 * 11).sum(axis=1, dtype=cp.uint64).reshape(4, 5)

    cp.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32])
def test_cuda_center_of_mass_matches_reference(dtype) -> None:
    cp = _cupy_with_device()
    from quantem.gpu.detector.cuda.dense import CudaKernelCompute

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

    got_col, got_row = CudaKernelCompute(data).center_of_mass()
    got_row, got_col = got_row.reshape(4, 5), got_col.reshape(4, 5)

    cp.testing.assert_allclose(got_row, expected_row, rtol=0, atol=1e-6)
    cp.testing.assert_allclose(got_col, expected_col, rtol=0, atol=1e-6)


def test_cuda_center_of_mass_masked_matches_reference() -> None:
    cp = _cupy_with_device()
    from quantem.gpu.detector.cuda.dense import CudaKernelCompute

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

    got_col, got_row = CudaKernelCompute(data).center_of_mass(det_mask)
    got_row, got_col = got_row.reshape(3, 4), got_col.reshape(3, 4)

    cp.testing.assert_allclose(got_row, expected_row, rtol=0, atol=1e-6)
    cp.testing.assert_allclose(got_col, expected_col, rtol=0, atol=1e-6)


def test_cuda_dense_mask_uses_integer_complement_and_matches_cupy() -> None:
    cp = _cupy_with_device()
    from quantem.gpu.detector.cuda.dense import CudaKernelCompute

    rng = np.random.default_rng(37)
    data_np = rng.integers(0, 60000, size=(4, 4, 16, 16), dtype=np.uint16)
    data = cp.asarray(data_np)
    det_mask = _mask((16, 16), 3.5, invert=True)

    got = CudaKernelCompute(data).masked_sum(det_mask)
    expected = (
        data.reshape(-1, 16 * 16)[:, cp.asarray(det_mask.reshape(-1))]
        .sum(axis=1, dtype=cp.uint64)
        .astype(cp.float32)
        .reshape(4, 4)
    )

    cp.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize(
    "dtype, det_shape",
    [(np.uint16, (400, 400)), (np.uint8, (5804, 5804))],
)
@pytest.mark.slow
def test_cuda_float_masked_sum_of_saturated_frames_exceeds_32_bits(dtype, det_shape) -> None:
    """A saturated frame sums past 2^32 through either kernel: the selected pixels or total minus complement."""
    cp = _cupy_with_device()
    from quantem.gpu import detector

    maximum = int(np.iinfo(dtype).max)
    pixels = det_shape[0] * det_shape[1]
    # Fewest pixels whose saturated sum crosses 2^32; still under half the detector.
    crossing = 2**32 // maximum + 1
    assert crossing * maximum > 2**32 and crossing < pixels // 2
    data = cp.full((1, 2, *det_shape), maximum, dtype=dtype)
    session = detector.prepare(data)
    for selected in (crossing, pixels - crossing):
        mask = np.zeros(pixels, dtype=bool)
        mask[:selected] = True
        expected = np.full((1, 2), selected * maximum, dtype=np.uint64)
        np.testing.assert_array_equal(session.masked_sum_exact(mask.reshape(det_shape)), expected)
        np.testing.assert_array_equal(session.masked_sum(mask.reshape(det_shape)), expected.astype(np.float32))


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32])
def test_cuda_center_of_mass_divides_exact_moments(dtype) -> None:
    """The definition MPS shares: exact integer moments over the exact total in float64, 0 when empty."""
    cp = _cupy_with_device()
    from quantem.gpu.detector.cuda.dense import CudaKernelCompute

    shape = (40, 52)
    values = np.random.default_rng(13).integers(0, 250, (300, *shape)).astype(dtype)
    values[7] = 0
    rows, cols = np.indices(shape, dtype=np.uint64)
    disk = np.hypot(rows - 19.5, cols - 25.5) <= 15
    compute = CudaKernelCompute(cp.asarray(values))
    for mask in (None, disk):
        working = values.astype(np.uint64) * (np.ones(shape, bool) if mask is None else disk)
        total = working.sum(axis=(1, 2)).astype(np.float64)
        got = compute.center_of_mass(mask)
        for center, weights in zip(got, (cols, rows)):
            moment = (working * weights).sum(axis=(1, 2)).astype(np.float64)
            expected = np.divide(moment, total, out=np.zeros_like(total), where=total > 0).astype(np.float32)
            np.testing.assert_array_equal(center, expected)


@pytest.mark.parametrize("runner", ["cupy", "torch-cuda"])
def test_cuda_dense_mean_pattern_and_frame_reductions(runner) -> None:
    """Totals above 2^24: the mean divides in float64 and rounds once; integer sum and max stay exact uint64."""
    cp = _cupy_with_device()
    from quantem.gpu import detector

    counts = np.random.default_rng(81).integers(52_000, 65_536, (18, 18, 6, 7)).astype(np.uint16)
    data = cp.asarray(counts) if runner == "cupy" else pytest.importorskip("torch").from_numpy(counts).cuda()
    session = detector.prepare(data)
    total = counts.sum(axis=(0, 1), dtype=np.uint64)
    expected = (total / 324).astype(np.float32)
    assert np.any(total.astype(np.float32) / np.float32(324) != expected)
    np.testing.assert_array_equal(session.mean_dp(), expected)
    indices = np.arange(10, 310)
    selected = counts.reshape(-1, 6, 7)[indices]
    np.testing.assert_array_equal(
        session.reduce_frames(indices, "mean"), (selected.sum(axis=0, dtype=np.uint64) / 300).astype(np.float32)
    )
    for mode, expected_counts in (("sum", selected.sum(axis=0, dtype=np.uint64)), ("max", selected.max(axis=0))):
        result = session.reduce_frames(indices, mode)
        assert result.dtype == np.uint64
        np.testing.assert_array_equal(result, expected_counts)


def test_cuda_frame_mean_divides_the_exact_total_once() -> None:
    """Totals above 2^24 round once: float64(total) / n to float32, as on MPS."""
    cp = _cupy_with_device()
    from quantem.gpu import detector

    counts = np.random.default_rng(73).integers(50_000, 65_536, (20, 16, 6, 7)).astype(np.uint16)
    indices = np.arange(10, 310)
    total = counts.reshape(-1, 6, 7)[indices].sum(axis=0, dtype=np.uint64)
    expected = (total.astype(np.float64) / len(indices)).astype(np.float32)
    assert np.any(total.astype(np.float32) / np.float32(len(indices)) != expected)
    np.testing.assert_array_equal(detector.prepare(cp.asarray(counts)).reduce_frames(indices, "mean"), expected)


def test_cuda_virtual_image_kernel_source_uses_warp_and_fused_dense_path() -> None:
    from pathlib import Path

    from quantem.gpu.detector.cuda import dense

    _CUDA_VI_CODE = Path(dense.__file__).with_name("kernels").joinpath("virtual_image.cu").read_text()

    assert "__shfl_down_sync" in _CUDA_VI_CODE
    assert "selected_sum_f32_u16_16f" in _CUDA_VI_CODE
    assert "selected_sum_f32_u32_16f" in _CUDA_VI_CODE
    assert "selected_sum_u64_u16_16f" in _CUDA_VI_CODE
    assert "selected_sum_u64_u32_16f" in _CUDA_VI_CODE
    assert "selected_sum_from_total_f32_u16_16f" in _CUDA_VI_CODE
    assert "selected_sum_from_total_f32_u32_16f" in _CUDA_VI_CODE
    assert "total_sum_u16_4f" in _CUDA_VI_CODE
    assert "total_sum_u32_4f" in _CUDA_VI_CODE
    assert "center_of_mass_full_u16_4f" in _CUDA_VI_CODE
    assert "center_of_mass_full_u32_4f" in _CUDA_VI_CODE
    assert "center_of_mass_selected_u16_4f" in _CUDA_VI_CODE
    assert "center_of_mass_selected_u32_4f" in _CUDA_VI_CODE
    assert "selected_frame_sum_u64_u16" in _CUDA_VI_CODE
    assert "selected_frame_sum_u64_u32" in _CUDA_VI_CODE
    assert "selected_frame_max_u32_u16" in _CUDA_VI_CODE
    assert "selected_frame_max_u32_u32" in _CUDA_VI_CODE


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32])
def test_cuda_selected_frame_sum_matches_exact_reference(dtype) -> None:
    cp = _cupy_with_device()
    from quantem.gpu.detector.cuda.dense import CudaKernelCompute

    rng = np.random.default_rng(47)
    data_np = rng.integers(0, 200, size=(7, 6, 13, 11), dtype=dtype)
    data = cp.asarray(data_np)
    indices = np.asarray([0, 3, 8, 17, 31, 41], dtype=np.int32)

    got = CudaKernelCompute(data).reduce_frames_exact(indices)
    expected = data.reshape(-1, 13, 11)[indices].sum(axis=0, dtype=cp.uint64)

    cp.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32])
def test_cuda_selected_frame_max_matches_exact_reference(dtype) -> None:
    cp = _cupy_with_device()
    from quantem.gpu.detector.cuda.dense import CudaKernelCompute

    rng = np.random.default_rng(71)
    data_np = rng.integers(0, 200, size=(7, 6, 13, 11), dtype=dtype)
    data = cp.asarray(data_np)
    indices = np.asarray([0, 3, 8, 17, 31, 41], dtype=np.int32)

    got = CudaKernelCompute(data).reduce_frames_max(indices)
    expected = data.reshape(-1, 13, 11)[indices].max(axis=0).astype(cp.uint32)

    cp.testing.assert_array_equal(got, expected)


def test_cupy_compute_backend_dispatches_to_cuda_kernel_backend() -> None:
    cp = _cupy_with_device()
    from quantem.gpu.detector.cuda.dense import CudaKernelCompute
    from quantem.gpu.detector.session import resolve_backend

    data = cp.ones((4, 4, 12, 12), dtype=cp.uint16)
    backend = resolve_backend(data)

    assert isinstance(backend, CudaKernelCompute)

    sparse_mask = _mask((12, 12), 2.0)
    sparse = backend.masked_sum(sparse_mask)
    np.testing.assert_array_equal(sparse, np.full((4, 4), sparse_mask.sum(), np.float32))
    assert backend._totals is None
    assert len(backend._mask_indices) == 1

    dense_mask = ~sparse_mask
    dense = backend.masked_sum(dense_mask)
    np.testing.assert_array_equal(dense, np.full((4, 4), dense_mask.sum(), np.float32))
    assert backend._totals is not None
    assert len(backend._mask_indices) == 1


def test_cuda_exact_masked_sum_preserves_counts_above_float32_limit() -> None:
    cp = _cupy_with_device()
    from quantem.gpu import detector

    data = cp.full((2, 3, 20, 20), 100_000, dtype=cp.uint32)
    mask = np.ones((20, 20), dtype=bool)

    result = detector.prepare(data).masked_sum_exact(mask)

    assert result.dtype == np.uint64
    np.testing.assert_array_equal(result, np.full((2, 3), 40_000_000, np.uint64))


def test_cuda_compute_backend_caches_full_center_of_mass() -> None:
    cp = _cupy_with_device()
    from quantem.gpu.detector.cuda.dense import CudaKernelCompute
    from quantem.gpu.detector.session import resolve_backend

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
    backend = resolve_backend(data)
    assert isinstance(backend, CudaKernelCompute)
    got_col, got_row = backend.center_of_mass()
    cached_col, cached_row = backend.center_of_mass()

    assert cached_col is got_col
    assert cached_row is got_row
    np.testing.assert_allclose(got_row.reshape(4, 5), expected_row, rtol=0, atol=1e-6)
    np.testing.assert_allclose(got_col.reshape(4, 5), expected_col, rtol=0, atol=1e-6)
