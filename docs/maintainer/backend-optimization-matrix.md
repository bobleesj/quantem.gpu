# Backend Optimization Matrix

This page is the single checklist for accelerated Show4DSTEM and SSB work
across CUDA, MPS, and WebGPU. It summarizes what is optimized today, what is
only source-present, and what still needs real user-facing agreement and FPS
signoff.

Do not count a speedup if it changes the microscope evidence: no hidden scan
crop, detector binning, BF reduction, saved derived cache, or CPU fallback.

## Frame Budgets

| Target | Budget | Use case |
| --- | ---: | --- |
| Smooth drag | `<=16.7 ms` | 60 FPS ROI dragging and scrub feedback. |
| Interactive | `<=33.3 ms` | 30 FPS BF/DF/ADF/DPC and SSB live steering. |
| Reviewable | `<=100 ms` | 10 FPS large or exact paths that are still usable. |
| Slow path | `>100 ms` | Needs redesign before calling it interactive. |

Report cold setup separately from warm interaction. For browser workflows, split
file parse, decompression, upload, compute, readback, colormap, and canvas
present. One end-to-end number is useful only after those stages are known.

## Memory Footprint

Raw 4D-STEM footprint for native `192x192` detector data:

| Scan shape | Raw `uint8` | Raw `uint16` | One float32 image |
| --- | ---: | ---: | ---: |
| `512x512x192x192` | `9.66 GB` / `9.0 GiB` | `19.33 GB` / `18.0 GiB` | `1.05 MB` |
| `1024x1024x192x192` | `38.65 GB` / `36.0 GiB` | `77.31 GB` / `72.0 GiB` | `4.19 MB` |

SSB Hermitian `G_qk` footprint, complex64 half-plane:

| Active BF | `128x128` | `256x256` | `512x512` | `1024x1024` |
| ---: | ---: | ---: | ---: | ---: |
| `~2827` BF, radius-30 style | `0.19 GB` | `0.75 GB` | `2.98 GB` | `11.88 GB` |
| `~8809` BF, full-BF style | `0.59 GB` | `2.33 GB` | `9.27 GB` | `37.02 GB` |
| `~9070` BF, fitted full field | `0.60 GB` | `2.40 GB` | `9.55 GB` | `38.12 GB` |

Full-plane SSB `G_qk` is about twice the Hermitian footprint and is not a
public runtime mode.

## Product Coverage

| Product path | CUDA | MPS | WebGPU | Current gap |
| --- | --- | --- | --- | --- |
| HDF5 load/decompress | CUDA bitshuffle/LZ4 kernels, device arrays, scan-region load. | Metal bitshuffle/LZ4, chunk-backed unified-memory data, scan-region load. | Browser HDF5/chunk reader and WGSL decode sources. | WebGPU cold parse/decode/upload needs a full `512` and `1024` timing table with a real adapter. |
| BF/ADF sparse masks | RawKernel selected-pixel reducer with warp-shuffle reductions. | Metal selected-pixel reducer on chunk-backed `uint8`/`uint16`. | `maskedSumBuffer` GPU-resident selected-index reducer. | MPS/WebGPU need broader full-size real-data agreement and repeated drag FPS signoff. |
| Dense DF | Cached full-detector total minus complement. | Cached total minus complement. | Cached full-detector total minus complement source path. | WebGPU needs explicit dense-DF timing and memory behavior at `512` and `1024`. |
| CoM/DPC | Fused CUDA moment reducer, backend cache. | Raw Metal `com_u8`/`com_u16`, chunk-backed dispatch. | `maskedCoMBuffer` and `maskedDpcBuffer` source paths exist. | WebGPU DPC needs headed agreement/perf signoff and no readback in the display path. |
| iDPC | Uses shared DPC phase reconstruction after CoM. | Same API over MPS CoM outputs. | Needs browser integration policy. | Decide whether iDPC stays CPU-side after GPU CoM or gets a browser solver. |
| SSB object redraw | Optimized native kernels for `128/256/512/1024`; `512` and object review are strong. | Implemented; real `512` object-wave steering is usable after warm-up. | ShowPtycho SSB WGSL source supports `128/256/512/1024`. | MPS/WebGPU need broader same-BF real-data signoff. |
| SSB exact phase/loss | `512` real full-BF meets about 30 FPS; `1024` remains slow. | `128` real-time, `256` near, `512/1024` not real-time. | Source exists; not yet a CUDA-level matrix. | Extend 12-cell matrix with real WebGPU adapter and MPS full-BF runs. |

## Current Measured Checkpoints

| Workflow | Backend | Shape / BF policy | Result | Status |
| --- | --- | --- | --- | --- |
| BF virtual image | CUDA | full `512x512x192x192` real data | `4.96 ms -> 1.35 ms`, max abs error `0` | Strong. |
| ADF virtual image | CUDA | full `512x512x192x192` real data | `16.16 ms -> 3.86 ms`, max abs error `0` | Strong. |
| Dense DF virtual image | CUDA | full `512x512x192x192` real data | `62.64 ms -> 1.84 ms`, max abs error `0` | Strong. |
| CoM/DPC | CUDA | full `512x512x192x192` real data | `200.42 ms -> 12.39 ms`, max abs error `0` | Strong. |
| Seven-panel BF/ADF/DF grid | CUDA | detector-bin2 seven-panel real workflow | BF `0.57 ms/panel`, ADF `1.37 ms/panel`, DF `0.57 ms/panel`, max abs error `0` | Strong for current grid policy. |
| MPS no-bin load + VI + CoM smoke | MPS | full `512x512x192x192` chunk-backed local master | load about `1 s`, masked-sum about `2 ms`, CoM about `0.17-0.20 s` | Smoke passed; needs formal agreement/perf matrix. |
| Show4DSTEM WebGPU headed stress | WebGPU | `128x128` scan, `96x96` detector, real adapter | mount/decode about `1.94 s`; BF warm `2.7-3.7 ms`; DPC warm floor `6-10 ms`; idle RAF 60 FPS | Promising small case; not a full `512` signoff. |
| Real crop product agreement gate | CUDA | opt-in local HDF5 crop, BF radius `30`, `32x32` scan | BF/ADF/DF exact; full and BF-masked CoM within `1e-5` | New regression gate; scale region when GPU is free. |
| SSB exact phase/loss | CUDA | real `512x512` full-BF field | mean about `32.5 ms`, p50 about `32.2 ms`, p95 about `33.3 ms` | Meets 30 FPS with small p95 margin. |
| SSB exact phase/loss | CUDA | synthetic `1024x1024`, full-BF style | about `198 ms`, `5 FPS` | Not interactive yet. |
| SSB exact phase/loss | MPS | synthetic full-BF style | `128` passes; `256` near; `512` around `5-6 FPS`; `1024` around `1 FPS` | Needs deeper MPS topology for large exact phase/loss. |

## Required Agreement Gates

Every new optimization should add or update one of these gates before claiming
speed:

| Gate | Reference | Required metrics |
| --- | --- | --- |
| CUDA BF/ADF/DF | Previous CuPy or NumPy selected-pixel sum on the same resident data. | max abs error, dtype, mask pixel count, timing before/after, peak temp memory. |
| CUDA CoM/DPC | Previous CoM implementation on the same mask and scan. | row/col max abs error, centered DPC error, timing before/after, cache behavior. |
| MPS BF/ADF/DF | NumPy or CUDA reference from the same loaded evidence. | max abs error, first-click timing, warm repeated timing, unified-memory footprint. |
| MPS CoM/DPC | CUDA or NumPy reference with the same detector mask and coordinate convention. | row/col max abs error, DPC component error, first-click timing, cached repeat timing. |
| Real HDF5 crop products | Independent NumPy reference from the exact loaded crop. | BF/ADF/DF bit-exact raw-count sums; full and masked CoM error at `1e-5`; CUDA and MPS backends via environment variables. |
| WebGPU BF/ADF/DF | Python `quantem.gpu` reference arrays exported with public-safe labels. | browser adapter, max/mean/p99 error, warm compute ms, display ms, no SwiftShader timing claims. |
| WebGPU CoM/DPC | Python CUDA/MPS reference arrays with same mask and zero-mean policy. | row/col error, centered component error, compute/readback/display split. |
| SSB object | Corrected-object reference at same BF, aberrations, scan size, and precision. | object complex error or phase/amplitude image error, warm redraw FPS. |
| SSB phase/loss | Existing exact phase/loss path; never object-wave identity as reference. | phase mean/p99/max, scalar loss delta, optimizer/free-fit agreement. |

## Next Optimization Queue

1. WebGPU Show4DSTEM full-`512` report: load/decode/upload/compute/display split
   for BF, ADF, dense DF, CoMx, CoMy, CoM magnitude, and iDPC where available.
2. WebGPU full-`512` agreement harness: compare browser outputs against
   `quantem.gpu` CUDA/MPS references with public-safe fixture labels and image
   difference maps.
3. WebGPU generated-HDF5 browser gate: cover `h5_url` fetch, HDF5 parse,
   bitshuffle/LZ4 decode, bad-pixel masks, BF/DF/ADF, and DPC without relying
   on local fixture paths.
4. WebGPU dense-DF cache signoff: verify `total - complement` behavior on a
   headed real adapter so large annular/dense masks do not scan most detector
   pixels on every drag.
5. WebGPU timing telemetry: attach adapter info, resident bytes, readback
   bytes, and per-stage timings; use GPU timestamp queries when available.
6. WebGPU DPC display path: prove `maskedDpcBuffer` feeds display without a
   GPU-to-CPU-to-GPU bounce, or wire the missing buffer adoption path.
7. MPS full-`512` VI/DPC report: BF/ADF/DF/CoM/DPC first-click and warm-repeat
   timings with the same masks as CUDA references.
8. MPS IO signoff: no-bin chunked, uint8 output, uint32 narrowing, fast
   sidecar decode, row-prefix, binned load, crop-first, and expected
   memory-guard failures.
9. MPS `1024x1024x192x192 uint8` policy test: shape-only support exists; the
   next step is a real allocation/performance plan on a high-memory Mac.
10. SSB MPS `512/1024` exact phase/loss topology: optimize row/column FFT work;
   chunk-size-only changes are not enough.
11. SSB WebGPU matrix: run `128/256/512/1024` object, phase, and phase+loss
   against CUDA references on a real WebGPU adapter.
