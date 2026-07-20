# Backends

`quantem.gpu` supports three backend names:

| Backend | Purpose | Notes |
|---|---|---|
| `cuda` | NVIDIA GPU IO, decompression, reductions, and SSB reference paths | Uses CuPy/CUDA kernels where available. |
| `mps` | Apple Silicon Metal/MLX paths | Used for MacBook chunk-backed loading, BF/DF/DPC images, and MPS SSB preview/free-fit paths. |
| `cpu` | Portable reference fallback | Useful for metadata, availability, and parity checks; not the target for heavy workflows. |

Check the selected backend:

```python
import quantem.gpu as qgpu

report = qgpu.device_report()
print(report.selected)
print(report.cuda_available, report.cuda_device_count, report.cuda_error)
print(report.mps_available, report.mps_error)
```

Request a backend explicitly when you need honest failure:

```python
qgpu.select_device("cuda")  # raises if CUDA is unavailable
qgpu.select_device("mps")   # raises if MPS is unavailable
```

Use `backend="auto"` for normal scripts and `backend="cuda"` or
`backend="mps"` in parity/performance tests.

WebGPU is a browser runtime, not a Python device backend. Reusable browser
compute sources live in `quantem.gpu.webgpu` and are bundled by
`quantem.widget` for anywidget and exported-HTML use. Browser performance
claims must log a real adapter; SwiftShader or another software adapter is only
a smoke test.

## Backend coverage

CUDA and MPS are the primary production backends. CPU exists for reference,
availability, and small fallback workflows.

Status terms: `Done` means implemented with real-data parity and performance
evidence; `Partial` means source exists but the full signoff matrix is not
complete; `Gap` means the backend does not implement that capability yet.

| Capability | CUDA | MPS | WebGPU | CPU | Notes |
|---|---|---|---|---|---|
| Device report and explicit selection | Done | Done | NA | Done | WebGPU adapter selection is browser-side. |
| HDF5 metadata, readiness, discovery | Done | Done | Done | Done | Keep one shared API for widget/live callers. |
| Full HDF5 bitshuffle/LZ4 load/decompress | Done | Done | Done | Reference | CUDA kernels, Metal chunk-backed loaders, and WebGPU WGSL decode are implemented. |
| `load(..., scan_region=...)` crop-first IO | Done | Done | Done | Reference | WebGPU uses frame-window slicing before upload/decode. |
| Detector bin during load, min-memory | Done | Done | Gap | Reference | WebGPU detector-bin load remains a real gap; never hide binning. |
| BF/DF/ADF resident kernels | Done | Done | Done | Reference | CUDA RawKernel, MPS Metal, and WebGPU WGSL selected reducers are implemented. |
| Dense DF/ADF strategy | Done | Done | Done | Reference | Dense masks use cached `total - complement` where cheaper. |
| CoM/DPC resident kernels | Done | Done | Partial | Reference | WebGPU source exists but still needs broader full no-bin FPS signoff. |
| iDPC | Done | Done | Partial | Reference | Browser integration is narrower than CUDA/MPS. |
| Ptychographic SSB preview/object steering | Done | Done | Partial | Reference | WebGPU source is shipped and widget-bundled; the full browser matrix is not complete. |
| Ptychographic SSB optimizer/free-fit | Done | Done | Partial | Not target | MPS supports current parity shapes; large exact phase/loss is still slower than CUDA. |
| GIF/MP4 movie rendering | Done | Done | NA | Fallback | CUDA/NVENC and Metal/VideoToolbox paths live here; widget owns UI buttons. |
| Browser source ownership | Done | Done | Done | NA | Reusable TypeScript/WGSL source is shipped in `quantem.gpu.webgpu`. |

The rule for new heavy work is: implement the compute or IO path in
`quantem.gpu`, then let widget/live call it.

## Current measured summary

These numbers are public-safe summaries. They do not include raw local paths or
project-specific dataset names. The full-stack rows use `512x512x192x192` HDF5
evidence. CUDA reference timing was measured on an NVIDIA RTX PRO 6000
Blackwell GPU. WebGPU timing used real Chrome WebGPU on Apple Metal.

| Path | Backend / hardware | Shape | Median | Parity / notes |
|---|---|---:|---:|---|
| HDF5 load/decompress | CUDA, RTX PRO 6000 Blackwell | full `512` | `450 ms` | 946-run warm reference; resident stack `9.66 GB`. |
| Local HDF5 full-stack load | WebGPU, Chrome Apple Metal | full `512` | `772 ms` | Corrected-frame checksum parity versus CUDA. |
| Local HDF5 scan crop | WebGPU, Chrome Apple Metal | true `256` crop | `338 ms` | Corrected-frame checksum parity versus CUDA. |
| Product-first BF selected-block sidecar | WebGPU, Chrome Apple Metal | true `256`, BF radius `30` | `210 ms` | Product max/mean abs error `0` versus CUDA. |
| Product-first BF selected-block sidecar | WebGPU, Chrome Apple Metal | full `512`, BF radius `30` | `378 ms` | Product max/mean abs error `0` versus CUDA. |
| Product-first BF selected-block sidecar | WebGPU, Chrome Apple Metal | `1024` repeat-stress, BF radius `30` | `1170 ms` | Four repeats of real `512` evidence; not true 1024 acquisition signoff. |
| Visible Show4DSTEM interaction | WebGPU, Chrome Apple Metal | full `512` local HDF5 | full load `933 ms`; drag `0.5-0.9 ms` | GPU-resident warm BF/ADF/DPC display interactions. |

## Adding a backend kernel

For agents and maintainers, a new optimized path is not complete until the
source, tests, documentation, and measured evidence land together.

| Kernel family | CUDA source | MPS source | WebGPU source | Required gate |
|---|---|---|---|---|
| HDF5 bitshuffle/LZ4 decode | `quantem.gpu.io.backends.cuda` | `quantem.gpu.io.backends.mps` | `quantem.gpu.webgpu.bslz4` and `local-h5.ts` | Corrected-frame checksum parity and load-stage timing. |
| BF/DF/ADF masked sums | `quantem.gpu.compute.cuda` / `detector` | `quantem.gpu.compute.mps` | `quantem.gpu.webgpu.compute` / `local-h5.ts` | Exact integer product parity and first/warm interaction timing. |
| CoM/DPC | `quantem.gpu.compute.cuda` / `dpc` | `quantem.gpu.compute.mps` / `dpc` | `quantem.gpu.webgpu.compute` | Row/col CoM and centered DPC parity within `1e-5`. |
| SSB object, phase, loss | `quantem.gpu.ssb.cuda` | `quantem.gpu.ssb.mps` | `quantem.gpu.webgpu.showptycho-ssb` | Same BF policy, same aberrations, phase/loss parity, and interactive redraw timing. |
| Movie rendering | `quantem.gpu.movie.cuda` | `quantem.gpu.movie.mps` | NA | Frame parity and encoded movie smoke tests. |
