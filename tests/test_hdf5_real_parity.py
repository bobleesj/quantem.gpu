from __future__ import annotations

import os

import pytest


def _checksum(arr):
    import cupy as cp

    flat = arr.ravel()
    return (
        int(flat.sum(dtype=cp.uint64).get()),
        int(flat.max().get()),
        int(flat.min().get()),
        int(arr.size),
        tuple(int(x) for x in arr.shape),
        str(arr.dtype),
    )


@pytest.mark.skipif(
    not os.environ.get("QUANTEM_GPU_PARITY_MASTER"),
    reason="set QUANTEM_GPU_PARITY_MASTER to a real Arina master for old/new parity",
)
def test_real_master_matches_legacy_widget_checksum() -> None:
    """Compare quantem.gpu against the pre-migration widget loader on real data."""
    import cupy as cp

    import quantem.gpu.io.hdf5 as gpu_hdf5
    import quantem.widget.io.hdf5 as widget_hdf5

    master = os.environ["QUANTEM_GPU_PARITY_MASTER"]
    if not os.path.exists(master):
        pytest.skip(f"master not available: {master}")
    if cp.cuda.runtime.memGetInfo()[0] / 1e9 < 8.0:
        pytest.skip("not enough free VRAM for real-master parity")

    old = widget_hdf5.load(master, verbose=False, backend="cuda")
    old_ck = _checksum(old.data)
    del old
    cp.get_default_memory_pool().free_all_blocks()

    new = gpu_hdf5.load(master, verbose=False, backend="cuda")
    new_ck = _checksum(new.data)
    del new
    cp.get_default_memory_pool().free_all_blocks()

    assert new_ck == old_ck


@pytest.mark.skipif(
    not os.environ.get("QUANTEM_GPU_PARITY_MASTER"),
    reason="set QUANTEM_GPU_PARITY_MASTER to a real Arina master for crop parity",
)
def test_real_master_crop_first_matches_full_slice() -> None:
    """Compare crop-first sparse decode with full-load slicing on real data."""
    import cupy as cp

    from quantem.gpu.io import load

    master = os.environ["QUANTEM_GPU_PARITY_MASTER"]
    if not os.path.exists(master):
        pytest.skip(f"master not available: {master}")
    if cp.cuda.runtime.memGetInfo()[0] / 1e9 < 24.0:
        pytest.skip("not enough free VRAM for full-master crop parity")

    region = tuple(
        int(value)
        for value in os.environ.get("QUANTEM_GPU_PARITY_REGION", "0,32,0,32").split(",")
    )
    if len(region) != 4:
        raise ValueError("QUANTEM_GPU_PARITY_REGION must be r0,r1,c0,c1")

    crop = load(master, scan_region=region, backend="cuda", verbose=False)
    full = load(master, backend="cuda", verbose=False)
    ref = cp.ascontiguousarray(full.data[region[0]:region[1], region[2]:region[3]])

    assert tuple(crop.data.shape) == tuple(ref.shape)
    assert bool(cp.array_equal(crop.data, ref).get())


@pytest.mark.skipif(
    not os.environ.get("QUANTEM_GPU_MPS_PARITY_MASTER"),
    reason="set QUANTEM_GPU_MPS_PARITY_MASTER on Apple Silicon for MPS crop parity",
)
def test_real_master_mps_crop_first_matches_full_chunked_slice() -> None:
    """Compare MPS crop-first sparse decode with full chunked MPS slicing."""
    import numpy as np

    from quantem.gpu.io import load

    master = os.environ["QUANTEM_GPU_MPS_PARITY_MASTER"]
    if not os.path.exists(master):
        pytest.skip(f"master not available: {master}")
    region = tuple(
        int(value)
        for value in os.environ.get("QUANTEM_GPU_PARITY_REGION", "0,32,0,32").split(",")
    )
    if len(region) != 4:
        raise ValueError("QUANTEM_GPU_PARITY_REGION must be r0,r1,c0,c1")

    full = load(
        master,
        backend="mps",
        verbose=False,
        skip_mps_memory_check=True,
    )
    crop = load(master, scan_region=region, backend="mps", verbose=False)

    starts = np.cumsum([0] + [int(n) for n in full.data.metadata["chunk_n_frames"]])
    scan_c = int(full.data.scan_shape[1])
    r0, r1, c0, c1 = region
    frames = []
    for row in range(r0, r1):
        for col in range(c0, c1):
            frame_index = row * scan_c + col
            chunk_index = int(np.searchsorted(starts, frame_index, side="right") - 1)
            frames.append(
                np.asarray(full.data.chunks[chunk_index][frame_index - starts[chunk_index]])
            )
    ref = np.stack(frames, axis=0).reshape(r1 - r0, c1 - c0, *crop.data.shape[-2:])

    assert tuple(crop.data.shape) == tuple(ref.shape)
    assert np.array_equal(np.asarray(crop.data), ref)
