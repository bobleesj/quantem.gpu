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
compute sources live beside their scientific domains and are packaged for
browser clients. SSB-specific WebGPU implementation files live under
`quantem.gpu.ssb.compute.webgpu`. Browser performance claims must log a real
adapter; SwiftShader or another software adapter is only a smoke test.

CUDA service placement can select several devices for independent resident
datasets. Their VRAM is not combined to make one dataset fit. WebGPU does not
expose CUDA-style multi-GPU placement; the browser selects one adapter for a
page.

## Backend coverage

CUDA and MPS are the native production backends. CPU is test/reference only
and is never selected as a silent scientific fallback.

Status terms: `Done` means implemented with real-data parity and performance
evidence; `Partial` means source exists but the full signoff matrix is not
complete; `Gap` means the backend does not implement that capability yet.

| Capability | CUDA | MPS | WebGPU | CPU | Notes |
|---|---|---|---|---|---|
| Device report and explicit selection | Done | Done | NA | Done | WebGPU adapter selection is browser-side. |
| HDF5 metadata, readiness, discovery | Done | Done | Done | Done | Keep one shared API for all clients. |
| Full HDF5 bitshuffle/LZ4 load/decompress | Done | Done | Done | Reference | CUDA and WebGPU retain native `uint8`/`uint16` source paths. Python MPS has signed-off native `uint16`/`uint32` source decode plus explicit audited or saturating `uint8` output; native-`uint8` bitshuffle-source decode remains unqualified. `load(..., dtype='uint32')` or `dtype='u32'` requests native four-byte unsigned output. `load(..., dtype='u4')` means true packed 4-bit counts (`0..15`), not NumPy `<u4`; CUDA returns a packed two-counts-per-byte array after an exact range audit. MPS/WebGPU HDF5 packed `u4` output is a named gap and raises honestly. WebGPU strict full-stack no-bin `1024x1024x192x192` browse is rejected as a memory-policy path; use product-first, crop, or explicit bin. |
| `load(..., scan_region=...)` crop-first IO | Done | Done | Done | Reference | WebGPU uses frame-window slicing before upload/decode. |
| Detector bin during load, min-memory | Done | Done | Done | Reference | WebGPU has explicit count-preserving `detBin` source support; full `512x512x192x192` `detBin=2/4/8` headed parity is exact on a real NVIDIA WebGPU adapter, including native non-low8 `uint16` `detBin=2`. |
| BF/DF/ADF resident kernels | Done | Done | Done | Reference | CUDA RawKernel, MPS Metal, and WebGPU WGSL selected reducers are implemented for `uint8`/`uint16`/`uint32` resident data; CUDA also has packed `uint4` selected/dense reducers and CoM kernels. |
| Dense DF/ADF strategy | Done | Done | Done | Reference | Dense masks use cached `total - complement` where cheaper. |
| CoM/DPC resident kernels | Done | Partial | Done | Reference | Detector-bin-4 CUDA/MPS CoM passes the frozen gate. The public MPS native-detector interaction sidecar is detector-bin-2 and is not full-resolution parity. WebGPU row/col DPC has full no-bin headed signoff on real hardware. |
| Cached detector/DPC products | Done | Done | Product-first Done / cache-read Done | Cache-read Done | CUDA and MPS build the raw-HDF5 product cache. The exact `uint16` CUDA path additionally exposes exact `uint64` total/ABF/ADF maps; other paths report those optional fields as unavailable. WebGPU owns browser selected-block product caches and can read prepared cache products. |
| iDPC | Done | Partial | Done | Reference | Current CUDA/MPS detector-bin-4 iDPC exceeds the frozen `1e-5` cross-backend gate; native-detector MPS sidecar iDPC is also blocked. WebGPU fixed-rotation iDPC uses paired DPC buffers and a dual-real FFT with an explicit float32 tolerance. |
| Ptychographic SSB preview | Done | Done | Partial | Reference | Python MPS and the separate native `MetalSSBKernels` product have parity-qualified implementations; WebGPU source lives under `quantem.gpu.ssb.compute.webgpu`, but its full browser matrix is not complete. |
| Ptychographic SSB fit/reconstruction | Done | Done | Partial | Not target | Python MPS supports its current parity shapes. Native Swift/Metal supports exact 512×512 reconstruction, phase-variance loss, and deterministic 200-trial TPE plus Nelder–Mead fitting; other native scan sizes remain gaps. |
| Native Browser FFT (`MetalImageFFT.logMagnitude`) | NA | Done | NA | Reference | Native Swift/Metal product for already-transferred 2D BF/ADF/custom images. 512×512 must stay inside 120 Hz when warm. Not a Python MPS path. |
| GIF/MP4 movie rendering | Done | Done | NA | Fallback | CUDA/NVENC and Metal/VideoToolbox paths live here; presentation controls remain client-owned. |
| Browser source ownership | Done | Done | Done | NA | Reusable TypeScript/WGSL source lives beside each scientific domain. |

The rule for new heavy work is: implement the compute or IO path in
`quantem.gpu`, then let clients call the shared contract.

## Benchmark ownership

This page owns capability and source-boundary status only. Numerical results are
not copied here:

- [Implementation overview](dashboard.md) is the current human-facing speed,
  memory, feature, and parity dashboard.
- [Verified benchmark results](performance/results.md) is the authoritative
  provenance ledger.
- [Optimization ledger](maintainer/backend-optimization-matrix.md) preserves
  accepted and rejected experiments.

Keeping the backend map timing-free prevents a prepared-index reopen, resident
kernel, historical campaign, or application first product from drifting into a
misleading source-load comparison.

## Adding a backend kernel

For agents and maintainers, a new optimized path is not complete until the
source, tests, documentation, and measured evidence land together.

| Kernel family | CUDA source | MPS source | WebGPU source | Required gate |
|---|---|---|---|---|
| HDF5 bitshuffle/LZ4 decode | `quantem.gpu.io.backends.cuda` | `quantem.gpu.io.backends.mps` | `quantem.gpu.io.backends.webgpu` | Corrected-frame checksum parity and load-stage timing. |
| BF/DF/ADF masked sums | `quantem.gpu.detector.compute.cuda` / `detector` | `quantem.gpu.detector.compute.mps` | `quantem.gpu.detector.compute.webgpu` / `local-h5.ts` | Exact integer product parity and first/warm interaction timing. |
| CoM/DPC | `quantem.gpu.dpc.compute.cuda` | `quantem.gpu.dpc.compute.mps` | `quantem.gpu.dpc.compute.webgpu` | Row/col CoM and centered DPC parity within `1e-5`. |
| Display colormap/histogram/log/FFT | `quantem.gpu.display.cuda` | `MetalDisplayKernels` | `quantem.gpu.display.webgpu` | Exact uint8 RGBA and 256-bin counts for linear/signed-log float32 fixtures; FFT agreement within the stated float precision. |
| SSB object, phase, loss | `quantem.gpu.ssb.compute.cuda` | `quantem.gpu.ssb.compute.mps`; native `MetalSSBKernels` | `quantem.gpu.ssb.compute.webgpu` | Same complete BF disk, aberrations, float32/complex64 parity, and interactive redraw timing. |
| Movie rendering | `quantem.gpu.movie.cuda` | `quantem.gpu.movie.mps` | NA | Frame parity and encoded movie smoke tests. |
