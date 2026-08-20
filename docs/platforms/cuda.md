# CUDA

CUDA is the primary runtime for NVIDIA workstations and servers. Start with the
scientific operation, then implement its adapter and kernels without exposing
launch details through the public API.

## Dispatch and implementation layers

| Layer | CUDA source | Implementation responsibility |
|---|---|---|
| Device selection | `src/quantem/gpu/device/backend.py` | probe CuPy/runtime and reject unavailable CUDA explicitly |
| IO contract | `src/quantem/gpu/io/backends/protocol.py` | validate `cuda`, `mps`, or explicit `cpu` |
| Load orchestration | `src/quantem/gpu/io/load.py` | source discovery, frame planning, metadata, crop/bin/dtype provenance |
| CUDA decode | `src/quantem/gpu/io/backends/cuda/decoder.py` | bitshuffle/LZ4 decode into CuPy-resident arrays |
| Detector dispatch | `src/quantem/gpu/detector/workflow.py` and `compute/backends.py` | select resident CUDA reducers and normalize small results |
| Detector kernels | `src/quantem/gpu/detector/compute/cuda/kernels.py` | masks, exact sums, selected frames, and moments |
| DPC | `src/quantem/gpu/dpc/compute/cuda/backend.py` | CUDA CoM/DPC primitives under the shared workflow |
| SSB | `src/quantem/gpu/ssb/compute/cuda` | prepared geometry, size-specific FFT kernels, objective, optimizer |
| Display | `src/quantem/gpu/display/cuda.py` | resident display statistics and transformations |

The ordinary IO call path is:

```text
io.load(..., backend="cuda")
  → io.backends.protocol.resolve_backend
  → io.load source/index/read planning
  → io.backends.cuda.decoder
  → CuPy-resident LoadResult + provenance
```

Detector and reconstruction calls dispatch from their public workflow to the
CUDA adapter only after inspecting the resident data type. Backend-specific
classes and RawKernel launch shapes do not appear in the public API.

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

Synchronization is an observation boundary, not a default after every kernel.
Independent read, H2D, decode, and reduction work may overlap on streams. A
host readback is allowed for a requested small result or parity artifact, not
for an intermediate full detector volume.

## Build and focused checks

```bash
python -m pip install -e ".[cuda,dev]"
PYTHONPATH=src python -m pytest -q \
  tests/test_device.py \
  tests/test_cuda_virtual_image.py \
  tests/test_products_parity.py
```

Real-data and SSB gates are environment-qualified and run only with their
recorded source fixture. A skipped CUDA test is not CUDA evidence.

## Profiling

Use synchronized wall intervals plus CUDA events or Nsight/CUPTI. Attribute
index planning, `pread`, pinned allocation, H2D, decode, reductions,
synchronization, and small result copies. Do not sum overlapping stream times
and call the result wall time.

## Acceptance

CUDA results preserve `I[R_r,R_c,k_r,k_c]` and
`(row, column) ≡ (r, c)`. Integer decode/reduction paths are byte-exact against
the frozen reference. Floating products use the operation-specific parity
metric. Record GPU model, driver/runtime, source shape/dtype, crop/bin plan,
memory, cache state, and exact source/kernel revisions.

The minimum review bundle contains a CPU-reference comparison, rectangular
row/column case, dtype/overflow case, full requested crop/bin plan, synchronized
wall and device timings, process and total-card memory, and the exact command.

### 6 GiB VRAM release floor

The minimum CUDA device class is **6 GiB of dedicated VRAM**. A configuration
receives ✓ only when the complete load, decode, reduction, and first-product
pipeline fits inside that physical limit, including destination arrays,
decoder and reduction scratch, CuPy allocation reserve, products, and the
concurrent process baseline.

For the full `512x512` scan and `192x192` source detector, native detector bin 1
requires an 18.00 GiB `uint16` resident payload and exact detector bin 2 requires
a 9.00 GiB `uint32` payload, so both are **No** for this floor. Detector bins 4
and 8 have 2.25 GiB and 0.5625 GiB resident payloads and are candidates, but
remain **Pending** until a physical 6 GiB run—or a clearly labeled capped
pre-check followed by physical signoff—retains total-card peak and parity.
Measurements on a larger Blackwell GPU do not by themselves prove this gate.
