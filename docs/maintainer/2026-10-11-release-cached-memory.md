# Releasing cached GPU memory - 2026-10-11

## Question

Two functions returned cached GPU memory to the driver:
`quantem.gpu.device.release_cached_memory()` and quantem.widget's
`free_gpu()`. Neither did everything the other did. What does one function in
quantem.gpu have to do so that dropped arrays go back to the driver on CUDA
and on Metal, which GPUs should it visit, which number should it return, and
what does it cost?

## Setup

- One NVIDIA RTX PRO 6000 Blackwell (96 GB), driver 580.173.02, one visible
  GPU, Python 3.12.3, PyTorch 2.14.1 (CUDA 13.0), CuPy 14.2.0. Other jobs held
  4 to 49 GB of the GPU and ran at up to 100% utilization during the
  measurements.
- One Apple silicon laptop (M5 Max, 128 GB), macOS 26.4, Python 3.13.13,
  PyTorch 2.14.1, MLX 0.32.3, pyobjc-framework-Metal 12.2.2.
- Revisions: base `39ad87c` (rc16); the change is the commit before the one
  that adds this page.
- Scenario: start both allocators, allocate a 1 GiB Torch tensor and put it in
  a list that contains itself (a reference cycle), allocate a 1 GiB CuPy array
  (CUDA) or MLX array (Metal), `del` all three names, call the function. The
  memory the driver holds for the process is read before and after: the
  per-process value of `nvidia-smi --query-compute-apps` on CUDA,
  `torch.mps.driver_allocated_memory()` on Metal.

## What each allocator keeps

Apple silicon, one 1 GiB allocation at a time, in MiB:

| Step | Metal driver total | Torch tensor counter | MLX cache |
|---|---:|---:|---:|
| start | 1.4 | 0.0 | 0.0 |
| Torch tensor alive | 1025.5 | 1024.0 | 0.0 |
| tensor dropped | 1025.5 | 0.0 | 0.0 |
| `torch.mps.empty_cache()` | 1.5 | 0.0 | 0.0 |
| MLX array alive | 1025.5 | 0.0 | 0.0 |
| MLX array dropped | 1025.5 | 0.0 | 1024.0 |
| `torch.mps.empty_cache()` | 1025.5 | 0.0 | 1024.0 |
| `mx.clear_cache()` | 1.5 | 0.0 | 0.0 |

`torch.mps.driver_allocated_memory()` equals Metal's `currentAllocatedSize`
for the process at every step, so it counts MLX's buffers as well as Torch's.
Torch's tensor counter reads zero as soon as the tensor is dropped, while the
driver still holds the block. Torch cannot return MLX's cache. The Metal
residents hand their buffers straight back to the driver when they are
collected, so the driver is read before the garbage collection.

CUDA, CuPy's FFT plan cache: after `cp.fft.fft2` of a (256, 500, 500)
complex64 batch, with the arrays dropped and the pool emptied, the pool still
held 0.477 GiB, the work area of the cached plan. Clearing the plan cache and
emptying the pool again left 0 bytes. The next `fft2` took 3.0 ms instead of
2.2 ms. `PlanCache.clear()` takes 88 ms even on an empty cache (it runs a
garbage collection), so the function calls it only when a plan is cached.

## Which GPUs to visit

CuPy's pool counts and frees the blocks of its current GPU only, so each GPU
has to be made current in turn. `cupy.cuda.Device(0).use()` alone, in a
process that has not used the GPU, opens a driver context of 550 MiB in
0.11 s, which stays until the process exits. Visiting every visible GPU would
therefore open the GPUs the process never used, and raise when one of them
cannot be opened.

`torch._C._cuda_hasPrimaryContext(index)` reports whether the process holds a
context on a GPU. It read False in a fresh process and True after CuPy had
opened the GPU, took 0.28 ms, and opened nothing. It is not public Torch API;
neither Torch nor CuPy has a public call for the question. The function visits
the GPUs for which it is True: a GPU without a context holds none of the
process's blocks. In a fresh process the call returned 0.0 and left no context
(73 to 87 ms, the garbage collection).

## The scenario before and after

| Backend | Function | Held after `del` | Returned value | Still held after the call |
|---|---|---:|---|---:|
| CUDA | rc16 `release_cached_memory()` | 2.000 GiB | `None` | 1.000 GiB |
| CUDA | this change | 2.000 GiB | `2.147483648` | 0.000 GiB |
| Metal | rc16 `release_cached_memory()` | 2.000 GiB | `None` | 2.000 GiB |
| Metal | quantem.widget rc42 `free_gpu()` | 2.000 GiB | `1.073741824` | 1.000 GiB |
| Metal | this change | 2.000 GiB | `2.147483648` | 0.000 GiB |

Each row was run three times on CUDA and twice on Metal with the same values
(the widget row once). On CUDA rc16 left the tensor of the reference cycle. On
Metal it left both blocks: the tensor was alive in the cycle, and MLX's cache
was not touched. The widget's function collected the cycle and left MLX's
cache. Over the call on Metal, Torch's tensor counter fell by 1.000 GiB while
the driver got 2.000 GiB back; a result computed from the counter would be
half.

With 3 GiB of Torch tensors and 1 GiB of MLX arrays dropped on Metal, the call
printed `MPS: released 4.3 GB, 0.0 GB in use` and returned `4.294967296`. The
tensor counter did not change over that call.

## Which number to return on CUDA

`cudaMemGetInfo` (`torch.cuda.mem_get_info`) counts every process on the GPU.
A first version returned its drop over the call. For the scenario's 2.147 GB,
on the GPU shared with other jobs, it read:

| Run | `cudaMemGetInfo` drop over the call |
|---|---:|
| no other allocation during the call (several runs) | 2.147483648 GB |
| another test suite running | 2.158034944 GB |
| another test suite running | 2.147418112 GB |
| another job at 92% utilization | 1.864368128 GB |
| another job at 92% utilization | 2.023751680 GB |
| another job at 92% utilization | -13.241417728 GB |

The function returns what this process handed back instead: the drop in the
bytes Torch's allocator has reserved plus the drop in the bytes of CuPy's
pool, per GPU. Both counts include cached blocks, and the per-process memory
`nvidia-smi` lists fell by the same 2048 MiB in every run. The CUDA test
passed every run with an exact comparison while the other jobs ran.
`verbose=True` prints the driver's totals for the GPU beside it, for example
`GPU 0: released 2.1 GB, 32.0 GB in use, 69.9 GB free`. On Metal the driver's
total is already per process.

## Cost

| Step, CUDA, in a process that imported quantem.gpu | Time |
|---|---:|
| `gc.collect()` (344 000 tracked objects) | 74 to 89 ms |
| whole call, 2 GiB to return, no cached FFT plan (3 runs) | 82 to 101 ms |
| whole call with a cached FFT plan (two garbage collections) | 156 to 158 ms |
| rc16 function, 1 GiB to return (6 runs) | 0.4 to 0.7 ms |
| CuPy pool of one GPU and Torch's cache only | 0.6 ms |

Garbage collection is the cost of the call. `parallax.run` returned its blocks
with this function once per run; it now empties its own GPU's CuPy pool and
Torch's cache in place, the same two allocator calls as before.
`scripts/benchmark_ssb.py --stages parallax --repeats 7` on one 256 x 256 scan
with a 192 x 192 uint16 detector, the two revisions run alternately, twice
each:

| Revision | Median warm `parallax.run` | Output digests |
|---|---:|---|
| base | 0.060 s, 0.060 s | image `8aa57a3354d2032e`, density `27d3e3d3af67b134`, shifts `24fcb83ac86e3118` |
| this change | 0.062 s, 0.061 s | the same three digests, the same aberrations and rotation |

## Conclusion

`device.release_cached_memory(verbose=False)` collects garbage and empties
Torch's cache; on every GPU the process has opened it drops CuPy's cached FFT
plans and empties CuPy's pool; it empties CuPy's pinned pool; on Apple silicon
it empties the Torch MPS and MLX caches. It returns what the process handed
back to the driver in units of 1e9 bytes. The hardware tests
(`tests/hardware/cuda/test_release_cached_memory.py`,
`tests/hardware/mps/test_release_cached_memory.py`) require the returned value
to equal what the driver held for the scenario exactly and the driver's
per-process memory to return to its start.

Limits: arrays something still refers to stay, including the locals of a
failed call that the traceback of the last error keeps alive. FFT plans cached
by other threads stay. With a CPU-only Torch beside CuPy the CUDA path does
not run. A second GPU was not available for this record, so the path over
several GPUs is covered by a simulated test only
(`tests/contracts/test_device.py`).

## Rejected ideas

- Returning the drop of `cudaMemGetInfo`: it includes other processes (table
  above).
- Reporting from Torch's MPS tensor counter: it excludes Torch's own cached
  blocks and MLX's.
- Visiting every visible GPU, as `free_gpu()` did: it opens the GPUs the
  process never used (550 MiB each).
- Clearing the FFT plan cache on every call: 88 ms each time, plans or not.
- Calling the function from `parallax.run`: 80 ms of garbage collection in a
  run that takes 60 ms on a 256 x 256 scan.
