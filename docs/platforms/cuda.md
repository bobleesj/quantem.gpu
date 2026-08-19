# CUDA

CUDA is the primary runtime for NVIDIA workstations and servers. Start with the
scientific operation, then implement its adapter and kernels without exposing
launch details through the public API.

## Source map

| Operation | CUDA source |
|---|---|
| HDF5 read/decode | `src/quantem/gpu/io/backends/cuda` |
| BF/DF/ADF and detector moments | `src/quantem/gpu/detector/compute/cuda` |
| DPC | `src/quantem/gpu/dpc/compute/cuda` |
| SSB | `src/quantem/gpu/ssb/compute/cuda` |
| Remote service | `src/quantem/gpu/remote` |

```python
from quantem.gpu import detector, io

loaded = io.load("scan_master.h5", backend="cuda", dtype="u16", det_bin=1)
bright_field = detector.bf(loaded.data)
```

## Execution and memory model

Keep compressed input, decoded counts, masks, reduction outputs, and FFT
intermediates device-resident across compatible operations. Use persistent
streams, file descriptors, pinned buffers, compiled kernels, and prepared
geometry where measured. Source-shard-aligned double/triple buffering can
overlap storage read, host preparation, H2D, decode, and reduction.

Dedicated VRAM admission includes decoder scratch, destination arrays,
reduction/FFT workspaces, allocator reserve, and other active processes.
Compressed bytes do not predict fit. Record both process allocated/reserved
bytes and total-card occupancy.

## Profiling

Use synchronized wall intervals plus CUDA events or Nsight/CUPTI. Attribute
index planning, `pread`, pinned allocation, H2D, decode, reductions,
synchronization, and small result copies. Do not sum overlapping stream times
and call the result wall time.

## Acceptance

CUDA results preserve `I[r_y,r_x,q_y,q_x]` and
`(row, column) ≡ (y, x)`. Integer decode/reduction paths are byte-exact against
the frozen reference. Floating products use the operation-specific parity
metric. Record GPU model, driver/runtime, source shape/dtype, crop/bin plan,
memory, cache state, and exact source/kernel revisions.
