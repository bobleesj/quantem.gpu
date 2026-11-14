"""Rehearse a small GPU on a large one: ``small_gpu.cap(3700)`` before any data is loaded.

A test tool, never a setting. After the call this process sees and can take only ``megabytes`` of device memory,
whatever other jobs on the GPU do:

- ``cp.cuda.runtime.memGetInfo()`` and ``torch.cuda.mem_get_info()`` answer ``megabytes`` minus what this process
  holds (the CUDA context, the CuPy pool with its cached blocks, allocations made outside the pool with
  ``cp.cuda.using_allocator()``, PyTorch's reserved memory), as a card of that size would;
- the CuPy pool, those outside allocations and PyTorch refuse an allocation beyond it with their usual out-of-memory errors.

``megabytes`` is the free memory a fresh process sees before its CUDA context exists (an RTX A400 with nothing
else on it: about 3700 to 3900). ``context_megabytes`` is the context's share, 560 measured on an RTX PRO 6000;
a smaller card's context is smaller, so the rehearsal is on the strict side.

Not modelled: memory CUDA takes outside both allocators (a kernel's local-memory reservation, about 400 MiB for
the fit's random-search kernel on a 188-SM card), and page rounding of single allocations.
"""
import threading
import time
import weakref


def cap(megabytes: int, context_megabytes: int = 560):
    import cupy as cp
    import torch

    budget, context = megabytes << 20, context_megabytes << 20
    pool = cp.get_default_memory_pool()
    real_total = cp.cuda.runtime.memGetInfo()[1]
    peak = [0]

    raw = [0]  # bytes allocated outside the pool (cp.cuda.using_allocator(): the encoded scan's blocks)
    real_using_allocator = cp.cuda.using_allocator

    class Counted(cp.cuda.Memory):
        """A device allocation outside the pool, counted until it is freed."""

    def give_back(nbytes: int) -> None:
        raw[0] -= nbytes

    def exact(nbytes: int):
        if context + pool.total_bytes() + torch.cuda.memory_reserved() + raw[0] + nbytes > budget:
            raise cp.cuda.runtime.CUDARuntimeError(2)  # CUDA_ERROR_OUT_OF_MEMORY, as a full card answers
        memory = Counted(nbytes)
        raw[0] += nbytes
        weakref.finalize(memory, give_back, nbytes)
        return cp.cuda.MemoryPointer(memory, 0)

    def using_allocator(allocator=None):
        return real_using_allocator(exact if allocator is None else allocator)

    cp.cuda.using_allocator = using_allocator

    def held() -> int:
        return context + pool.total_bytes() + torch.cuda.memory_reserved() + raw[0]

    def enforce() -> int:
        """Set both allocators' limits from what the other holds; the bytes still free."""
        now = held()
        peak[0] = max(peak[0], now)
        pool.set_limit(size=max(1 << 20, budget - context - torch.cuda.memory_reserved() - raw[0]))
        torch.cuda.set_per_process_memory_fraction(
            min(1.0, max(1 << 20, budget - context - pool.total_bytes() - raw[0]) / real_total))
        return max(0, budget - now)

    def info(*_args, **_kwargs):
        return enforce(), budget

    cp.cuda.runtime.memGetInfo = info
    torch.cuda.mem_get_info = info
    enforce()

    def keep():
        while True:
            enforce()
            time.sleep(0.01)

    threading.Thread(target=keep, name="small-gpu", daemon=True).start()
    global raw_megabytes
    raw_megabytes = lambda: raw[0] >> 20  # held outside the pool now
    return lambda: peak[0] >> 20  # the most this process held, in MiB
