from __future__ import annotations

import os

import pytest


def _real_dtype() -> str:
    return os.environ.get("QUANTEM_GPU_REAL_DTYPE", "uint8")


@pytest.mark.skipif(
    not os.environ.get("QUANTEM_GPU_REAL_MASTER"),
    reason="set QUANTEM_GPU_REAL_MASTER to a real compressed master for sparse IO",
)
def test_real_master_scan_indices_match_crop_slice() -> None:
    """Compare sparse scan-index decode with a crop-slice reference."""
    import cupy as cp
    import numpy as np

    from quantem.gpu.io import load

    master = os.environ["QUANTEM_GPU_REAL_MASTER"]
    if not os.path.exists(master):
        pytest.skip(f"master not available: {master}")

    region = tuple(
        int(value)
        for value in os.environ.get("QUANTEM_GPU_REAL_REGION", "0,8,0,8").split(",")
    )
    if len(region) != 4:
        raise ValueError("QUANTEM_GPU_REAL_REGION must be r0,r1,c0,c1")

    scan_shape = tuple(
        int(value)
        for value in os.environ.get("QUANTEM_GPU_REAL_SCAN_SHAPE", "512,512").split(",")
    )
    if len(scan_shape) != 2:
        raise ValueError("QUANTEM_GPU_REAL_SCAN_SHAPE must be rows,cols")

    r0, r1, c0, c1 = region
    positions = np.asarray(
        [
            [r0, c0],
            [r1 - 1, c1 - 1],
            [r0, c1 - 1],
            [r0, c0],
        ],
        dtype=np.int64,
    )
    crop = load(
        master,
        scan_region=region,
        backend="cuda",
        scan_shape=scan_shape,
        verbose=False,
        dtype=_real_dtype(),
    )
    sparse = load(
        master,
        scan_indices=positions,
        backend="cuda",
        scan_shape=scan_shape,
        verbose=False,
        dtype=_real_dtype(),
    )
    ref = crop.data[positions[:, 0] - r0, positions[:, 1] - c0]

    assert tuple(sparse.data.shape) == tuple(ref.shape)
    assert bool(cp.array_equal(sparse.data, ref).get())
    assert sparse.metadata["unique_frame_count"] == 3


@pytest.mark.skipif(
    not os.environ.get("QUANTEM_GPU_REAL_MASTER"),
    reason="set QUANTEM_GPU_REAL_MASTER to a real compressed master for random IO",
)
def test_real_master_random_positions_loads_sparse_batch() -> None:
    """Smoke the one-line random-position sparse HDF5 loader on real data."""
    import cupy as cp

    from quantem.gpu.io import load

    master = os.environ["QUANTEM_GPU_REAL_MASTER"]
    if not os.path.exists(master):
        pytest.skip(f"master not available: {master}")

    scan_shape = tuple(
        int(value)
        for value in os.environ.get("QUANTEM_GPU_REAL_SCAN_SHAPE", "512,512").split(",")
    )
    result = load(
        master,
        random_positions=32,
        seed=123,
        backend="cuda",
        scan_shape=scan_shape,
        verbose=False,
        dtype=_real_dtype(),
    )

    assert tuple(result.data.shape[:1]) == (32,)
    assert result.metadata["sample"]["mode"] == "random_positions"
    assert result.metadata["sample"]["seed"] == 123
    assert result.metadata["unique_frame_count"] == 32
    assert bool(cp.isfinite(result.data).all().get())


@pytest.mark.skipif(
    not os.environ.get("QUANTEM_GPU_MPS_REAL_MASTER"),
    reason=(
        "set QUANTEM_GPU_MPS_REAL_MASTER on Apple Silicon for real MPS "
        "sparse det-bin agreement"
    ),
)
def test_real_mps_sparse_det_bin_matches_no_bin_sum() -> None:
    """MPS random sparse det_bin=2 must equal exact 2x2 sums from no-bin data."""
    Metal = pytest.importorskip("Metal")
    import numpy as np

    from quantem.gpu.io import load, random_scan_indices

    if Metal.MTLCreateSystemDefaultDevice() is None:
        pytest.skip("Metal device is not available")

    master = os.environ["QUANTEM_GPU_MPS_REAL_MASTER"]
    if not os.path.exists(master):
        pytest.skip(f"master not available: {master}")

    scan_shape = tuple(
        int(value)
        for value in os.environ.get("QUANTEM_GPU_MPS_REAL_SCAN_SHAPE", "256,256").split(",")
    )
    if len(scan_shape) != 2:
        raise ValueError("QUANTEM_GPU_MPS_REAL_SCAN_SHAPE must be rows,cols")
    n_positions = int(os.environ.get("QUANTEM_GPU_MPS_REAL_POSITIONS", "512"))
    indices = random_scan_indices(n_positions, scan_shape, seed=42)

    no_bin = load(
        master,
        scan_indices=indices,
        backend="mps",
        scan_shape=scan_shape,
        det_bin=1,
        verbose=False,
    )
    det_bin2 = load(
        master,
        scan_indices=indices,
        backend="mps",
        scan_shape=scan_shape,
        det_bin=2,
        verbose=False,
    )

    no_bin_arr = np.asarray(no_bin.data)
    det_bin2_arr = np.asarray(det_bin2.data)
    assert no_bin_arr.ndim == 3
    assert tuple(det_bin2_arr.shape) == (
        n_positions,
        no_bin_arr.shape[1] // 2,
        no_bin_arr.shape[2] // 2,
    )
    reference = no_bin_arr.reshape(
        n_positions,
        det_bin2_arr.shape[1],
        2,
        det_bin2_arr.shape[2],
        2,
    ).sum(axis=(2, 4), dtype=np.uint64)

    np.testing.assert_array_equal(det_bin2_arr, reference.astype(det_bin2_arr.dtype))
    assert int(det_bin2_arr.sum()) > 0
    assert det_bin2.metadata["backend"] == "mps"
    assert det_bin2.metadata["det_bin"] == 2
    assert det_bin2.metadata["unique_frame_count"] == n_positions
