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
