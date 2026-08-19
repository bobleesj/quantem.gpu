# Python MPS

Python MPS provides the shared Python contracts on Apple Silicon through
MLX/PyObjC/Metal and chunk-backed unified-memory representations.

## Source map

| Operation | MPS/Metal source |
|---|---|
| HDF5 decode | `src/quantem/gpu/io/backends/mps` |
| BF/DF/ADF and detector moments | `src/quantem/gpu/detector/compute/mps` |
| DPC | `src/quantem/gpu/dpc/compute/mps` |
| SSB | `src/quantem/gpu/ssb/compute/mps` |

```python
from quantem.gpu import io

loaded = io.load("scan_master.h5", backend="mps", dtype="u16", det_bin=1)
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

## Profiling

Record physical Mac model/chip/GPU cores, unified memory, pressure/swap,
process peak, source/cache state, critical-path wall time, and command-buffer
GPU intervals. Instruments Metal System Trace is useful when available; kernel
timestamps and wall-to-first-product remain required.

## Acceptance

The backend preserves `I[r_y,r_x,q_y,q_x]` and
`(row, column) ≡ (y, x)`. Unsafe plans fail before allocation or return a
typed cost estimate to the caller; they never crop the scan. Automatic detector
binning is a visible client policy and records original/output detector shapes,
dtypes, factor, memory estimate, and reason.
