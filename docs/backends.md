# Backends

The canonical source-tree and cross-language test organization is documented
in [Backend layout and parity contract](maintainer/backend-layout-and-parity.md).
Backend implementations remain private behind one domain-level scientific API;
folder moves require the machine-readable parity gates in
`tests/parity/backend_matrix.json`.

`quantem.gpu` supports three backend names:

| Backend | Purpose | Notes |
|---|---|---|
| `cuda` | NVIDIA GPU IO, decompression, reductions, and SSB reference paths | Uses CuPy/CUDA kernels where available. |
| `mps` | Apple Silicon Metal/MLX paths | Used for chunk-backed loading, BF/DF/DPC images, and SSB fit/reconstruction/preview paths. |
| `cpu` | Test/reference implementation | Available only when explicitly requested by a test or reference comparison. |

Check the selected backend:

```python
import quantem.gpu as qgpu

backend = qgpu.device.detect()
print(backend)
```

Request a backend explicitly when you need honest failure:

```python
qgpu.device.resolve("cuda")  # raises if CUDA is unavailable
qgpu.device.resolve("mps")   # raises if MPS is unavailable
```

Use `backend="auto"` for normal scripts and `backend="cuda"` or
`backend="mps"` in parity/performance tests.

WebGPU is a browser runtime, not a Python device backend. Reusable browser
compute sources live beside their scientific domains and are bundled by
`quantem.widget` for anywidget and exported-HTML use. SSB-specific WebGPU
implementation files live under `quantem.gpu.ssb.compute.webgpu`. Browser performance claims must log a real
adapter; SwiftShader or another software adapter is only a smoke test.

The Show4DSTEM command-line entry points live in `quantem.widget`, while
`quantem.gpu` owns the reusable load/decode/reduction kernels:

```bash
quantem show4dstem /data/session --backend mps --count 7 --bin 1 --dtype u8
quantem show4dstem /data/session --backend cuda --count 7 --devices 0,1 --bin 1 --dtype u8
quantem show4dstem /data/session --backend webgpu --html --count 7 --bin 1 --dtype u8
```

CUDA can use explicit `--devices 0,1` placement through the widget CLI. WebGPU
does not expose CUDA-style multi-GPU placement; the browser chooses one hardware
adapter for the page and the widget must not pretend a browser page is using two
CUDA GPUs.

## Backend coverage

CUDA and MPS are the native production backends. CPU is test/reference only
and is never selected as a silent scientific fallback.

Status terms: `Done` means implemented with real-data parity and performance
evidence; `Partial` means source exists but the full signoff matrix is not
complete; `Gap` means the backend does not implement that capability yet.

| Capability | CUDA | MPS | WebGPU | CPU | Notes |
|---|---|---|---|---|---|
| Device report and explicit selection | Done | Done | NA | Done | WebGPU adapter selection is browser-side. |
| HDF5 metadata, readiness, discovery | Done | Done | Done | Done | Keep one shared API for widget/live callers. |
| Full HDF5 bitshuffle/LZ4 load/decompress | Done | Done | Done | Reference | CUDA kernels, Metal chunk-backed loaders, and WebGPU WGSL decode are implemented for `uint8`/`uint16`/`uint32` detector sources. `load(..., dtype='uint32')` or `dtype='u32'` requests native four-byte unsigned output. `load(..., dtype='u4')` means true packed 4-bit counts (`0..15`), not NumPy `<u4`; CUDA returns a packed two-counts-per-byte array after an exact range audit. MPS/WebGPU HDF5 packed `u4` output is a named gap and raises honestly. WebGPU strict full-stack no-bin `1024x1024x192x192` browse is rejected as a memory-policy path; use product-first, crop, or explicit bin. |
| `load(..., scan_region=...)` crop-first IO | Done | Done | Done | Reference | WebGPU uses frame-window slicing before upload/decode. |
| Detector bin during load, min-memory | Done | Done | Done | Reference | WebGPU has explicit count-preserving `detBin` source support; full `512x512x192x192` `detBin=2/4/8` headed parity is exact on a real NVIDIA WebGPU adapter, including native non-low8 `uint16` `detBin=2`. |
| BF/DF/ADF resident kernels | Done | Done | Done | Reference | CUDA RawKernel, MPS Metal, and WebGPU WGSL selected reducers are implemented for `uint8`/`uint16`/`uint32` resident data; CUDA also has packed `uint4` selected/dense reducers and CoM kernels. |
| Dense DF/ADF strategy | Done | Done | Done | Reference | Dense masks use cached `total - complement` where cheaper. |
| CoM/DPC resident kernels | Done | Done | Done | Reference | WebGPU row/col DPC has full no-bin headed signoff on real hardware. |
| Cached BF/DF/CoM/rotation products | Done | Done | Product-first Done / cache-read Done | Cache-read Done | CUDA and MPS build the raw-HDF5 product cache; WebGPU owns browser selected-block product caches and can read prepared cache products. |
| iDPC | Done | Done | Done | Reference | WebGPU fixed-rotation iDPC is implemented with paired DPC buffers and a dual-real FFT; parity is float32 FFT tolerance, not bit-exact. |
| Ptychographic SSB preview | Done | Done | Partial | Reference | WebGPU SSB source lives under `quantem.gpu.ssb.compute.webgpu` and is bundled by the widget; the full browser matrix is not complete. |
| Ptychographic SSB fit/reconstruction | Done | Done | Partial | Not target | MPS supports current parity shapes; large exact phase/loss is still slower than CUDA. |
| Native Browser FFT (`MetalImageFFT.logMagnitude`) | NA | Done | NA | Reference | Native Swift/Metal product for already-transferred 2D BF/ADF/custom images. 512×512 must stay inside 120 Hz when warm. Not a Python MPS path. |
| GIF/MP4 movie rendering | Done | Done | NA | Fallback | CUDA/NVENC and Metal/VideoToolbox paths live here; widget owns UI buttons. |
| Browser source ownership | Done | Done | Done | NA | Reusable TypeScript/WGSL source lives beside each scientific domain. |

The rule for new heavy work is: implement the compute or IO path in
`quantem.gpu`, then let widget/live call it.

## Current measured summary

The complete table, including date, exact revision, hardware, shape/dtype,
cache state, load/bin plan, benchmark definition, calibration, and parity, is
the [verified benchmark results](performance/results.md). The headline rows
below link directly to those stable evidence IDs.

| Evidence ID | Path | Result | Interpretation |
|---|---|---:|---|
| CUDA-512-LOAD | Warm CUDA full-stack `512` load | `0.450 s` p50 | Warm source, not first encounter. |
| M2-AIR-BIN4-E2E | Physical 8 GB M2 Air application load | `1.985-2.043 s` wall p50 | First process, full scan, explicit exact-sum detector bin 4; not true cold and not no-bin. |
| WEBGPU-512-FULL | Prepared local-file WebGPU full-stack `512` load | `0.772 s` p50 | 946-cycle prepared-source soak; not cold. |
| PRODUCT-CACHE-REOPEN | Saved calibration-product reopen | `6.8-8.0 ms` | Derived cache reopen; never presented as raw-source load. |
| WEBGPU-DPC-512 | Warm GPU-resident DPC row/column/iDPC display | `14.9/13.2/13.2 ms` p50 | Display timing after load; not end-to-end source load. |

The retained July diagnostics and rejected experiments remain available in the
[optimization ledger](maintainer/backend-optimization-matrix.md), with their
original values and status labels.

## Adding a backend kernel

For agents and maintainers, a new optimized path is not complete until the
source, tests, documentation, and measured evidence land together.

| Kernel family | CUDA source | MPS source | WebGPU source | Required gate |
|---|---|---|---|---|
| HDF5 bitshuffle/LZ4 decode | `quantem.gpu.io.backends.cuda` | `quantem.gpu.io.backends.mps` | `quantem.gpu.io.backends.webgpu` | Corrected-frame checksum parity and load-stage timing. |
| BF/DF/ADF masked sums | `quantem.gpu.detector.compute.cuda` / `detector` | `quantem.gpu.detector.compute.mps` | `quantem.gpu.detector.compute.webgpu` / `local-h5.ts` | Exact integer product parity and first/warm interaction timing. |
| CoM/DPC | `quantem.gpu.dpc.compute.cuda` | `quantem.gpu.dpc.compute.mps` | `quantem.gpu.dpc.compute.webgpu` | Row/col CoM and centered DPC parity within `1e-5`. |
| Display colormap/histogram/log/FFT | `quantem.gpu.display.cuda` | `MetalDisplayKernels` | `quantem.gpu.display.webgpu` | Exact uint8 RGBA and 256-bin counts for linear/signed-log float32 fixtures; FFT agreement within the stated float precision. |
| SSB object, phase, loss | `quantem.gpu.ssb.compute.cuda` | `quantem.gpu.ssb.compute.mps` | `quantem.gpu.ssb.compute.webgpu` | Same complete BF disk, aberrations, float32/complex64 parity, and interactive redraw timing. |
| Movie rendering | `quantem.gpu.movie.cuda` | `quantem.gpu.movie.mps` | NA | Frame parity and encoded movie smoke tests. |
