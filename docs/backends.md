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
| Detector bin during load, min-memory | Done | Done | Done | Reference | WebGPU has explicit count-preserving `detBin` source support; full `512x512x192x192` `detBin=2/4/8` headed parity is exact on a real NVIDIA WebGPU adapter, including native non-low8 `uint16` `detBin=2`. |
| BF/DF/ADF resident kernels | Done | Done | Done | Reference | CUDA RawKernel, MPS Metal, and WebGPU WGSL selected reducers are implemented. |
| Dense DF/ADF strategy | Done | Done | Done | Reference | Dense masks use cached `total - complement` where cheaper. |
| CoM/DPC resident kernels | Done | Done | Done | Reference | WebGPU row/col DPC has full no-bin headed signoff on real hardware. |
| Cached BF/DF/CoM/rotation products | Done | Cache-read Done | Cache-read Done | Cache-read Done | CUDA builds the full raw-HDF5 product cache; all backends can read an existing cache for instant UI launch. |
| iDPC | Done | Done | Done | Reference | WebGPU fixed-rotation iDPC is implemented with paired DPC buffers and a dual-real FFT; parity is float32 FFT tolerance, not bit-exact. |
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
Blackwell GPU. WebGPU timing used real Chrome WebGPU on Apple Metal or NVIDIA
Blackwell as listed.

| Path | Backend / hardware | Shape | Median | Parity / notes |
|---|---|---:|---:|---|
| HDF5 load/decompress | CUDA, RTX PRO 6000 Blackwell | full `512` | `450 ms` | 946-run warm reference; resident stack `9.66 GB`. |
| HDF5 load/decompress | CUDA, RTX PRO 6000 Blackwell | true full `1024` | `4.704 s` | Real acquisition, no bin/crop, `uint16` output, selected corrected frames bit-exact, resident stack `77.31 GB`. |
| Stochastic HDF5 ptycho mini-batch | CUDA, RTX PRO 6000 Blackwell | 40 masters x 1000 global random positions, detector `192` | cold `8.90 s` at `prep_workers=1`; warm `1.0-1.6 s` | Native `uint16`, no detector bin, output `2.95 GB`; `prep_workers=8` was slower in cold and warm tests, so tune per storage path. |
| BF/DF/CoM/rotation cache build | CUDA, RTX PRO 6000 Blackwell | true full `1024`, detector `192`, 12 GB cap | `12.31 s` first build; `11.76 s` streaming raw HDF5; `18 ms` rotation search | Native `uint16`, no detector bin, chunk resident under budget, cache output `16.93 MB`; BF/DF/CoM use custom CUDA kernels. This is a cache-build step, not the screen launch path. |
| BF/DF/CoM/rotation cache hit | Any backend-facing caller | true full `1024`, detector `192` products | `6.8-8.0 ms` local cache read | Reads the small `.npz` product sidecar and is the path intended for sub-`0.5 s` UI launch. Cache hits do not initialize CUDA. |
| HDF5 load/decompress | MPS, Apple Metal | true full `1024` | `4.617 s` | Real acquisition, no bin/crop, chunk-backed `uint16` output, selected corrected frames bit-exact, resident stack `77.31 GB`. |
| Local HDF5 full-stack load | WebGPU, Chrome Apple Metal | full `512` | `772 ms` | Corrected-frame checksum parity versus CUDA. |
| Local HDF5 detector-bin load | WebGPU, Chrome NVIDIA Blackwell | full `512` and true `256` crop, `detBin=2/4/8` | full `1199/1212/1106 ms`; crop p95 `798/813/775 ms` | Corrected-frame checksum parity exact versus zero-bad-before-bin reference; crop medians `774/755/733 ms`; native non-low8 `uint16` `detBin=2` also exact at `2651 ms`. |
| Local HDF5 scan crop | WebGPU, Chrome Apple Metal | true `256` crop | `338 ms` | Corrected-frame checksum parity versus CUDA. |
| Product-first BF selected-block sidecar | WebGPU, Chrome Apple Metal | true `256`, BF radius `30` | `210 ms` | Product max/mean abs error `0` versus CUDA. |
| Product-first BF selected-block sidecar | WebGPU, Chrome Apple Metal | full `512`, BF radius `30` | `378 ms` | Product max/mean abs error `0` versus CUDA. |
| Product-first BF selected-block sidecar | WebGPU, Chrome NVIDIA Blackwell | true `1024`, BF radius `30` | `4.92 s` wall; `1.56 s` product stage | True real-acquisition product-first BF signoff; selected compressed payload `6.88 GB`, output `4.19 MB`, max/mean abs error `0` versus an independent Python reference. This is not full-stack no-bin browse/load signoff. |
| Product-first BF selected-block sidecar | WebGPU, Chrome Apple Metal | `1024` repeat-stress, BF radius `30` | `1170 ms` | Four repeats of real `512` evidence; not true 1024 acquisition signoff. |
| Visible Show4DSTEM interaction | WebGPU, Chrome Apple Metal | full `512` local HDF5 | full load `933 ms`; drag `0.5-0.9 ms` | GPU-resident warm BF/ADF/DPC display interactions. |
| DPC/iDPC display | WebGPU, Chrome NVIDIA Blackwell | full `512` no-bin | DPC row/col/iDPC display medians `14.9/13.2/13.2 ms` | FFT command batching keeps iDPC in the 30 FPS budget by median; full recompute medians `13.7/19.3/22.7 ms`; corrected-frame parity passed; DPC max abs error `7.63e-6`; iDPC mean abs error `4.70e-6`, max `3.05e-5` from float32 FFT order; idle RAF `60 FPS`. Local-file timing harness runs reject URL fallback with `--require-local-profile`. |

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
