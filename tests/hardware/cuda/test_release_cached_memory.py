"""Dropped Torch and CuPy arrays and cached FFT plans go back to the driver, and the reported gigabytes are this process's."""

import os
import re
import shutil
import subprocess

import pytest

cp = pytest.importorskip("cupy")
torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or shutil.which("nvidia-smi") is None, reason="needs a CUDA GPU and nvidia-smi"
)


def _driver_bytes_of_this_process() -> int:
    """GPU memory the driver attributes to this process, as ``nvidia-smi`` lists it (MiB resolution).

    ``cudaMemGetInfo`` cannot serve as the reference: it counts every process
    on the GPU, and other jobs moved it by gigabytes during runs of this test.
    """
    listing = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout
    rows = [line.split(", ") for line in listing.splitlines()]
    return sum(int(used_mib) << 20 for pid, used_mib in rows if int(pid) == os.getpid())


def test_release_cached_memory_returns_dropped_torch_and_cupy_blocks(capsys):
    """A 1 GiB Torch tensor held by a reference cycle and a 1 GiB CuPy array are both returned.

    After ``del`` the driver still holds both for this process: the tensor is
    alive in the cycle and the array's block sits in CuPy's pool (2 GiB, or
    16 MiB more when Torch allocates in expandable segments). The call must
    collect the cycle, empty both caches and return exactly what the driver
    held, and the driver's memory for this process must be back at its start.
    """
    from quantem.gpu import device

    block_bytes = 1 << 30
    # the driver context a first allocation opens (550 MiB) is not a cached block: open it before the first reading
    warm_t, warm = torch.empty(1, dtype=torch.uint8, device="cuda:0"), cp.empty(1, dtype=cp.uint8)
    del warm_t, warm
    device.release_cached_memory()  # blocks cached by earlier tests would be counted
    driver_start = _driver_bytes_of_this_process()
    tensor_t = torch.empty(block_bytes, dtype=torch.uint8, device="cuda:0")
    cycle = [tensor_t]
    cycle.append(cycle)
    array = cp.empty(block_bytes, dtype=cp.uint8)
    del tensor_t, cycle, array
    held_bytes = _driver_bytes_of_this_process() - driver_start
    assert held_bytes >= 2 * block_bytes
    released_gb = device.release_cached_memory(verbose=True)
    assert released_gb == held_bytes / 1e9
    assert _driver_bytes_of_this_process() == driver_start
    first_line = capsys.readouterr().out.splitlines()[0]  # one line per GPU the test session has opened
    assert re.fullmatch(r"GPU 0: released 2\.[12] GB, \d+\.\d GB in use, \d+\.\d GB free", first_line)


def test_release_cached_memory_returns_the_work_area_of_cached_fft_plans():
    """The plan of a 500 x 500 FFT keeps its work area in CuPy's pool after the arrays are dropped.

    Emptying the pool alone leaves that block, because the plan cache still
    refers to it. The call must drop the cached plans first.
    """
    from quantem.gpu import device

    device.release_cached_memory()
    pool = cp.get_default_memory_pool()
    pool_start = pool.total_bytes()
    batch = cp.zeros((64, 500, 500), dtype=cp.complex64)
    spectrum = cp.fft.fft2(batch)
    del batch, spectrum
    pool.free_all_blocks()
    plan_bytes = pool.total_bytes() - pool_start
    assert plan_bytes > 0
    assert device.release_cached_memory() == plan_bytes / 1e9
    assert pool.total_bytes() == pool_start
