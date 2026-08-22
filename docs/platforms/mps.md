# Python MPS

Python MPS provides the shared Python contracts on Apple Silicon through
MLX/PyObjC/Metal and chunk-backed unified-memory representations.

## Dispatch and implementation layers

| Layer | Python MPS/Metal source | Implementation responsibility |
|---|---|---|
| Device selection | `src/quantem/gpu/device/backend.py` | require macOS Metal/PyObjC or an available Torch MPS device |
| IO orchestration | `src/quantem/gpu/io/load.py` | source planning, metadata, policy-free crop/bin/dtype contract |
| MPS decode adapter | `src/quantem/gpu/io/backends/mps/decoder.py` | map compressed chunks and submit Metal decode work |
| Decode shader | `src/quantem/gpu/io/backends/mps/kernels/bslz4.msl` | bitshuffle/LZ4 reconstruction and typed output |
| Chunk series | `src/quantem/gpu/io/backends/mps/series.py` | preserve source lifetime without full duplication |
| Detector adapter | `src/quantem/gpu/detector/compute/mps/kernels.py` | chunk-backed frame and reduction interface |
| Detector shader | `src/quantem/gpu/detector/compute/mps/metal/reductions.msl` | exact sums and detector moments |
| DPC | `src/quantem/gpu/dpc/compute/mps/backend.py` | MPS CoM/DPC primitives under the shared workflow |
| SSB | `src/quantem/gpu/ssb/compute/mps` | MLX preparation, size-specific kernels, exact objective, optimizer |

The IO call path is:

```text
io.load(..., backend="mps")
  → backend validation
  → source and chunk planning
  → MPS decoder + bslz4.msl
  → chunk-backed/resident LoadResult + provenance
```

Python owns validation and typed results. Metal owns full-volume decode and
reductions. MLX owns the current Python MPS FFT/reconstruction path. Those are
implementation layers of one MPS runtime, not separate public workflows.

```python
from quantem.gpu import io

loaded = io.load("scan_master.h5", backend="mps", dtype="u16", det_bin=1)

try:
    # Use loaded.data while its zero-copy Metal-backed chunks are live.
    first_chunk = loaded.data.chunks[0]
finally:
    # Release caller-owned direct Metal buffers after the final consumer.
    loaded.data.free()
```

## Execution and memory model

CPU and GPU share physical memory, but redundant arrays and synchronization are
still expensive. Large detector data may remain chunk-backed; device kernels
consume those chunks without materializing a second full host array. Resource
plans include mapped source, decoded destination, scratch slots, reduction/FFT
buffers, process reserve, memory pressure, and swap—not compressed file size.

Optimize queue overlap, reusable `MTLBuffer` storage, prepared pipelines, and
fused decode/conversion/bin/reduction while preserving exact counts. A unified
memory mapping is not an H2D copy, so profiling should report page-in and GPU
access honestly rather than inventing “upload” time.

`MPSChunked4DSTEM` has explicit lifetime ownership. Its NumPy views are backed
by buffers created directly with Metal/PyObjC; deleting the Python wrapper or
calling `clear_mps_cache()` does not release caller-owned output buffers. Call
`loaded.data.free()` only after the final reader has finished. A benchmark that
repeats `io.load` without this release accumulates roughly one resident payload
per repetition and measures memory pressure rather than steady-state loader
speed.

For the full `512x512x192x192 uint16` plan, the resident payload is exactly
19,327,352,832 bytes (18.00 GiB). On the 2026-08-22 Phil audit, the sampled
Metal-driver high-water mark was 21,276,016,640 bytes (19.814835 GiB), while
maximum process RSS was only 734,314,496 bytes. The smaller RSS is not the GPU
memory footprint: direct Metal allocations are outside the complete scope of
that process counter.

The retained binned timings use an identity-bound source audit whose maximum
count is 53. That proves bin2, bin4, and bin8 exact sums fit `uint16` for this
fixture (maximum possible sums 212, 848, and 3,392). Do not infer that every
`uint16` source can retain `uint16` after detector summation; use a wider dtype
or fail closed unless a complete range audit proves the requested result fits.

## Build and focused checks

```bash
python -m pip install -e ".[mps,dev]"
PYTHONPATH=src python -m pytest -q \
  tests/test_device.py \
  tests/test_mps_chunk_dispatch.py \
  tests/test_products_parity.py \
  tests/test_ssb_mps_close.py
```

Metal-dependent skips on a non-Mac host are structure checks only. Physical
MPS signoff records the Mac model, chip, memory, OS, source/cache condition,
and exact command.

## Profiling

Record physical Mac model/chip/GPU cores, unified memory, source/cache and
process state, critical-path wall time, and command-buffer GPU intervals. For
memory, record all of these separately:

- exact logical resident payload from shape and dtype;
- sampled Metal-driver allocation at the high-water mark;
- Metal-driver allocation after output release;
- process RSS/footprint; and
- whole-system pressure and swap.

`torch.mps.current_allocated_memory()` can remain zero for these direct
Metal/PyObjC buffers. Use `torch.mps.driver_allocated_memory()` as the
accelerator high-water signal and keep RSS as a separate diagnostic. Instruments
Metal System Trace is useful when available; kernel timestamps and
wall-to-first-product remain required.

## Acceptance

The backend preserves `I[R_r,R_c,k_r,k_c]` and
`(row, column) ≡ (r, c)`. Unsafe plans fail before allocation or return a
typed cost estimate to the caller; they never crop the scan. Automatic detector
binning is a visible client policy and records original/output detector shapes,
dtypes, factor, memory estimate, and reason.
