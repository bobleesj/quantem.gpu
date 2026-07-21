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
