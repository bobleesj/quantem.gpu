"""Dropped Torch MPS tensors and MLX arrays go back to the Metal driver, and the reported gigabytes are the driver's."""

import pytest

torch = pytest.importorskip("torch")
mx = pytest.importorskip("mlx.core")

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs an Apple GPU")


def test_release_cached_memory_returns_dropped_torch_and_mlx_blocks(capsys):
    """A 1 GiB Torch MPS tensor held by a reference cycle and a 1 GiB MLX array are both returned.

    After ``del`` the Metal driver still holds both: the tensor is alive in the
    cycle and the array's buffer sits in MLX's cache. Torch's tensor counter
    sees only the tensor, so over the call it falls by 1 GiB while 2 GiB go
    back. The call must collect the cycle, empty both caches, and return the
    drop of the Metal driver's total, which is exactly the 2 GiB allocated.
    """
    from quantem.gpu import device

    block_bytes = 1 << 30
    # first use of each allocator leaves a few kilobytes of driver state that is not a cached block
    warm_t = torch.ones(1, dtype=torch.uint8, device="mps")
    warm = mx.ones((1,), dtype=mx.uint8)
    mx.eval(warm)
    del warm_t, warm
    device.release_cached_memory()
    driver_start = torch.mps.driver_allocated_memory()
    tensor_t = torch.ones(block_bytes, dtype=torch.uint8, device="mps")
    torch.mps.synchronize()
    cycle = [tensor_t]
    cycle.append(cycle)
    array = mx.ones((block_bytes,), dtype=mx.uint8)
    mx.eval(array)
    assert torch.mps.driver_allocated_memory() - driver_start == 2 * block_bytes
    del tensor_t, cycle, array
    assert torch.mps.driver_allocated_memory() - driver_start == 2 * block_bytes
    released_gb = device.release_cached_memory(verbose=True)
    assert released_gb == 2 * block_bytes / 1e9
    assert torch.mps.driver_allocated_memory() == driver_start
    assert capsys.readouterr().out == f"MPS: released 2.1 GB, {driver_start / 1e9:.1f} GB in use\n"
