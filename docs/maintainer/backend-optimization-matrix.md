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
| HDF5 load/decompress | CUDA bitshuffle/LZ4 kernels, device arrays, scan-region load. | Metal bitshuffle/LZ4, chunk-backed unified-memory data, scan-region load. | Browser HDF5/chunk reader, local-file acquisition, selected-block sidecars, and WGSL decode sources. | WebGPU full `512` local-file parity and selected-block `256/512/1024` product gates are signed off; true `1024` acquisition and a faster cooperative decoder remain. |
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
| HDF5 load/decompress | CUDA | full `512x512x192x192`, `uint8` browse output | cold first load about `2.8-3.0 s`; refreshed warm steady load median `0.443 s`, range `0.427-0.460 s`; resident stack `9.66 GB` | Meets single-dataset warm target; cold includes process/device/cache startup. |
| Seven-master HDF5 load/decompress | CUDA | seven full `512x512x192x192` masters, `uint8` browse output | explicit warmup then measured loads sum to `2.90 s`; per-master range `0.38-0.47 s` | Meets `3-4 s` steady seven-dataset target on idle GPU. |
| Seven-panel BF/ADF/DF grid | CUDA | detector-bin2 seven-panel real workflow | BF `0.57 ms/panel`, ADF `1.37 ms/panel`, DF `0.57 ms/panel`, max abs error `0` | Strong for current grid policy. |
| MPS no-bin load + VI + CoM smoke | MPS | full `512x512x192x192` chunk-backed local master | load about `1 s`, masked-sum about `2 ms`, CoM about `0.17-0.20 s` | Smoke passed; formal crop-product agreement now covers the masked CoM fallback. |
| HDF5 load/decompress | MPS | full `512x512x192x192`, `uint8` zero-copy chunked output on Apple GPU | current-source public load returns `MPSChunked4DSTEM`; refreshed single-master median `0.577 s`, range `0.550-0.593 s`, resident `9.66 GB`; previous independent seven-load median `0.69 s`, sum `6.54 s` with two outliers; retained-scratch earlier pass about `4.22 s` | Near CUDA steady-state for one load; retained seven-load target is close, independent cold-ish repeats still show Apple memory-pressure variance. |
| Show4DSTEM WebGPU headed stress | WebGPU | `128x128` scan, `96x96` detector, real adapter | mount/decode about `1.94 s`; BF warm `2.7-3.7 ms`; DPC warm floor `6-10 ms`; idle RAF 60 FPS | Promising small case; not a full `512` signoff. |
| HDF5 load/decompress | WebGPU | full `512x512x192x192`, real Chrome adapter, `uint8` browse output from `uint16` source | best URL total `5.78-5.86 s`; split run: fetch wait `2.37 s`, parse `0.26 s`, decode+upload `3.14 s` (`1.60 s` CPU staging/upload, `1.53 s` GPU queue wait), compressed fetch `3.2 GB`, resident decoded stack `9.7 GB` | Functional full-`512` signoff; URL acquisition is slower than local-file acquisition. |
| HDF5 load/decompress | WebGPU on Apple GPU | full `512x512x192x192`, real Chrome `apple metal-3`, served over localhost tunnel | full-stack total `9.15 s`: fetch wait `5.19 s`, parse `0.28 s`, decode+upload `3.68 s`, GPU wait `3.24 s`; same evidence and `uint8` browse output | Hardware path is valid, but tunneled HTTP acquisition dominates; use local-file acquisition for real local review. |
| HDF5 local-file Show4DSTEM production path | WebGPU on Apple GPU | full `512x512x192x192`, browser local files, worker read, optional `QH5IDX01` block-index sidecars, same WGSL bslz4 decode | previous general `uint16 -> uint8` path: representative `3.78 s`, seven independent loads `26.44 s`; count-audited lossless low8 frame-cooperative default (`wg32`, full-load file group `8`, staging pipeline, full-load worker count `8`): frame-index worker-parse path page profile `0.756 s` / wall `0.807 s`; 946-cycle soak median `0.772 s`, range `0.726-0.879 s`, latest visible widget full load `0.933 s`, parse `0.004 s`, read wait about `0.250 s`, upload bucket about `0.289 s`, GPU wait about `0.553 s`; metadata sidecars add `0.019 GB` for the full dataset; decoded resident stack `9.7 GB`, real adapter `apple metal-3` | Strong browser checksum parity versus CUDA and about a `4.9-5.2x` production-path speedup. Still short of the `0.5 s` strict full-stack target because WebGPU must upload about `3.17 GB` compressed bytes and materialize the `9.7 GB` stack. |
| HDF5 local-file scan-region full-stack path | WebGPU on Apple GPU | true `256x256x192x192` crop from full `512x512x192x192` local evidence, `uint8` browse output | crop-aware frame-window decode with data-file prefilter and optional block-index sidecars: 946-cycle soak median `0.338 s`, range `0.316-0.464 s`; selected compressed decode `0.81 GB`, local HDF5 reads `1.70 GB`, metadata sidecars `0.010 GB` for touched files; exact corrected-frame checksums versus CUDA crop reference; crop default remains worker count `2`, group `4`, decode batch `8` | Strong crop-first full-stack parity and now faster than the CUDA warm crop reference previously measured around `0.46 s` by page profile. This is a real crop load, not full decode plus post-crop. |
| Product-first BF from HDF5 | WebGPU on Apple GPU | true `256x256` scan crop from full `512x512x192x192` real local evidence, BF radius `30` | 3-repeat median wall `0.78 s`, product stage `0.21 s`, selected compressed bytes `381.7 MB`, max/mean abs error `0` versus CUDA crop-first reference | Strong crop-region parity. This is not a prefix; row-major scan-region mapping is explicit. |
| Product-first BF from HDF5 | WebGPU on Apple GPU | full `512x512x192x192` real local evidence, BF radius `30` | 3-repeat median wall `1.03 s`, product stage `0.51 s`, selected compressed bytes `1.52 GB`, max/mean abs error `0` versus CUDA reference | Strong product-first parity without materializing the `9.7 GB` decoded stack. Still reads the whole native HDF5 files before packing selected blocks. |
| Product-first BF selected-block sidecar | WebGPU on Apple GPU | true `256x256` crop from full `512x512x192x192` evidence, BF radius `30`, production local-H5 API | auto pixel/direct-float/staging-pipeline kernel: 946-cycle soak median `0.210 s`, range `0.185-0.246 s`, product stage about `0.100 s`, selected payload `0.382 GB`; max/mean abs error `0` versus CUDA reference | Strong crop parity. The crop path uses sidecar span filtering before full file reads and auto-batches small row-window specs. |
| Product-first BF selected-block sidecar | WebGPU on Apple GPU | full `512x512x192x192` exact derived bslz4 block streams, BF radius `30`, production local-H5 API | auto grouped-mask/direct-float/staging-pipeline kernel: 946-cycle soak median `0.378 s`, range `0.358-0.473 s`, visible widget product-first run `0.307-0.336 s`, selected payload `1.52 GB`; max/mean abs error `0` versus CUDA reference | Current best. Hits the CUDA-like single-product target for this BF product by storing exact selected detector-block streams instead of reading unrelated HDF5 detector blocks. |
| Product-first BF selected-block sidecar | WebGPU on Apple GPU | `1024x1024` repeat-stress gate, BF radius `30`, four repeats of real `512` evidence, production local-H5 API | auto grouped-mask/direct-float/staging-pipeline kernel with compact shared memory for large scans: 946-cycle soak median `1.170 s`, range `1.142-1.631 s`, product stage about `0.595 s`, selected payload `6.08 GB`, max/mean abs error `0` | Dispatch/output scaling gate only. Not a true 1024 real-acquisition signoff. |
| WebGPU load/product soak | WebGPU on Apple GPU plus CUDA reference on RTX PRO 6000 Blackwell | full `512`, true `256` crop, and `1024` repeat-stress selected-block products | 946 cycles produced 5676 timing rows. Five rows had transient Chrome/CDP socket or timeout harness failures; successful parity rows had no numeric mismatch. | Use this as the current stability baseline for WebGPU IO/product changes. It is not a substitute for true 1024 acquisition signoff. |
| Product-first BF from HDF5 | WebGPU on Apple GPU | `1024x1024` repeat-stress gate, BF radius `30`, four repeats of real `512` evidence | 3-repeat median wall `4.08 s`, product stage `1.80 s`, selected compressed bytes `6.08 GB`, max/mean abs error `0` versus repeated CUDA reference | Dispatch/output scaling gate only. Not a true 1024 real-acquisition signoff. |
| HDF5 load/decompress experiment | WebGPU | full `512x512x192x192`, lossless-valid-pixel `uint8` sidecar source | sidecar generation `155 s`; compressed size `3.07 GB`; best decode/upload `3.18 s` at batch 8, but total stayed `6.17 s`; default batch total `5.70 s` | Not adopted as a default speed path; high `uint16` bit planes were already cheap. |
| BF virtual image | WebGPU | full `512x512x192x192`, GPU-resident display path | first BF r30 click about `35-44 ms`; warm repeats about `7-13 ms` | Warm path is interactive; first click still includes setup/cache work. |
| Dense DF virtual image | WebGPU | full `512x512x192x192`, dense annular mask | first click about `195-199 ms`; warm repeats about `20-41 ms` after total cache | Uses total-minus-complement path, but complement reducer still needs CUDA-level tuning. |
| DPC/CoM | WebGPU | full `512x512x192x192`, BF mask, GPU-resident display plus validation readback | launch warm-cache entries about `55-58 ms`; buffer+readback probe about `167 ms` | Functional; not yet 30-60 FPS for full no-bin DPC. |
| Real HDF5 crop-first equality gate | CUDA | full `512x512x192x192` load plus `128x128` crop-first region | old/new full checksum passed; crop-first data exactly matches full-load slice | Strong IO/decompress parity gate on real data. |
| Real HDF5 crop-first equality gate | MPS | full `512x512x192x192` chunked load plus `128x128` crop-first region | crop-first data exactly matches full chunked slice on Apple GPU | Strong MPS IO/decompress parity gate on real data. |
| Real crop product agreement gate | CUDA | opt-in local HDF5 crop, BF radius `30`, `128x128` scan | BF/ADF/DF exact; full and BF-masked CoM within `1e-5`; wall `6:10` with independent reference | Strong product correctness gate; wall time is reference-heavy, not a clean benchmark. |
| Real crop product agreement gate | MPS | opt-in local HDF5 crop, BF radius `30`, `128x128` scan | BF/ADF/DF exact; full and BF-masked CoM within `1e-5`; initially exposed masked-CoM fallback bug, fixed and passed | Strong product correctness gate on Apple GPU. |
| WebGPU corrected-frame checksum gate | WebGPU on Apple GPU | full `512x512x192x192`, first/middle/last detector frames after bad-pixel correction | selected-frame `sum/min/max/n` exactly matches CUDA for all three scan indices | Strong browser HDF5 parse/decode/chunk-order/dtype/bad-pixel parity gate without reading the full 9.7 GB stack back to CPU. |
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
| WebGPU HDF5 load/decode | Python CUDA/MPS corrected-load reference from the same local evidence. | selected corrected frame checksum, frame order, dtype, bad-pixel count, adapter, load split, no private path in artifacts. |
| SSB object | Corrected-object reference at same BF, aberrations, scan size, and precision. | object complex error or phase/amplitude image error, warm redraw FPS. |
| SSB phase/loss | Existing exact phase/loss path; never object-wave identity as reference. | phase mean/p99/max, scalar loss delta, optimizer/free-fit agreement. |

## Next Optimization Queue

1. WebGPU product-first HDF5 evidence layout: exact BF r30 products now pass
   CUDA parity for true `256`, full `512`, and a `1024` repeat-stress gate. The
   selected-block sidecar path proves the `0.5 s` full-`512` target is reachable
   when the browser reads only exact detector-block evidence. The production
   local-H5 masked-sum API now discovers sidecars, checks selected-block
   coverage, filters crop spans before full reads, and falls back honestly when
   a dragged ROI needs detector blocks not present in the sidecar. Next
   production work is maintaining the sidecar cache writer/invalidation policy
   and wiring the fast product path into the Show4DSTEM UI product loop.
2. WebGPU compressed-byte upload and decoder floor: the count-audited low8
   frame-cooperative kernel plus staging pipeline cut the full local-file path
   to a `0.725 s` page profile in the latest full-`512` block-index run. The
   selected-block full-`512` BF product path reaches `0.370 s` page total /
   `0.209 s` product stage with exact parity in the latest fresh Chrome run.
   The remaining full-stack gap is materializing the whole `9.7 GB` browse cube;
   the product path shows why selected evidence is the right interactive route.
3. WebGPU pipelined selected-block acquisition: extend the staging-pipeline
   idea to browser range/local cache management, with a bounded staging/raw
   buffer ring and explicit peak-memory reporting.
4. WebGPU Show4DSTEM full-`512` report: load/decode/upload/compute/display split
   for BF, ADF, dense DF, CoMx, CoMy, CoM magnitude, and iDPC where available.
5. WebGPU bitshuffle/LZ4 kernel redesign: the frame-cooperative low8 path is now
   the default for count-audited lossless `uint8` browse loads. The next jump
   needs lower atomic/shared-memory pressure or a different browser payload
   contract, not more fetch batching.
6. WebGPU full-`512` agreement harness: compare browser outputs against
   `quantem.gpu` CUDA/MPS references with public-safe fixture labels and image
   difference maps.
7. WebGPU generated-HDF5 browser gate: cover `h5_url` fetch, HDF5 parse,
   bitshuffle/LZ4 decode, bad-pixel masks, BF/DF/ADF, and DPC without relying
   on local fixture paths.
8. WebGPU dense-DF cache signoff: verify `total - complement` behavior on a
   headed real adapter so large annular/dense masks do not scan most detector
   pixels on every drag.
9. WebGPU timing telemetry follow-up: source now reports adapter info,
   resident bytes, fetch/parse/upload/build/GPU-wait splits, and timestamp-query
   availability; next step is optional GPU timestamp pass timing.
10. WebGPU DPC display path: prove `maskedDpcBuffer` feeds display without a
   GPU-to-CPU-to-GPU bounce, or wire the missing buffer adoption path.
11. MPS full-`512` VI/DPC report: BF/ADF/DF/CoM/DPC first-click and warm-repeat
   timings with the same masks as CUDA references.
12. MPS IO signoff: no-bin chunked, uint8 output, uint32 narrowing, fast
   sidecar decode, row-prefix, binned load, crop-first, and expected
   memory-guard failures.
13. MPS `1024x1024x192x192 uint8` policy test: shape-only support exists; the
   next step is a real allocation/performance plan on a high-memory Mac.
14. SSB MPS `512/1024` exact phase/loss topology: optimize row/column FFT work;
   chunk-size-only changes are not enough.
15. SSB WebGPU matrix: run `128/256/512/1024` object, phase, and phase+loss
   against CUDA references on a real WebGPU adapter.

## Failed / Bounded Hypotheses

| Hypothesis | Result | Decision |
| --- | --- | --- |
| Increase WebGPU HDF5 decode batch from `4` to `8`. | Kernel/decode stage can improve (`3.46 s` on `uint16`, `3.18 s` on `uint8` sidecar), but total wall time worsened to about `6.2 s` because fetch wait increased. | Keep default batch `4`; leave `window.__QT_H5_DECODE_BATCH` for profiling. |
| Increase data-file fetch window from `8` to `16/24`. | Higher windows reduced some fetch waits in isolated runs but caused master/data contention or decode variance; total stayed worse than default. | Keep default fetch window `8`; leave `window.__QT_H5_FETCH_WINDOW` for profiling. |
| Fetch/parse HDF5 master before data files. | Master timing dropped to about `50 ms`, but total wall time worsened because data fetch no longer overlapped the metadata read. | Do not reorder as default. |
| Embed HDF5 bad-pixel metadata and skip browser master fetch. | Removes `masterFetchMs`, but default total remains about `6.0 s` because the old master fetch was mostly overlapped with data fetch. Still avoids one browser HDF5 parse and one local file fetch. | Keep for exported local H5-source fixtures; it is simpler and not a hidden evidence change. |
| Production local-file worker/group/decodeBatch sweep. | Real Chrome `apple metal-3` runs with `0/4/8` workers and group/batch `4/8` stayed in the `3.83-4.25 s` range after the corrected-frame checksum gate; GPU wait remained about `3.18-3.23 s`. Earlier runs stayed in the `3.74-4.12 s` range. | Keep default workers/group/batch `4/4/4`; focus on the WGSL decoder instead of more browser read orchestration. |
| Module Blob worker for local HDF5 reads. | `new Worker(blobUrl, { type: "module" })` failed under the standalone `file://` Show4DSTEM export and forced URL fallback. | Use a classic inline Blob worker for local-file acquisition. |
| Fused `uint8` source sidecar. | Valid browser path, but full-source compressed bytes remain about `3.1 GB` and total load did not beat the production `uint16 -> uint8` path. | Keep fused `uint8` source support for correctness/compatibility; do not promote sidecars as the main load-speed fix. |
| Low8-only bslz4 decode. | Count audit after bad-pixel correction showed max unmasked count `57` and zero unmasked pixels above `255`, so low8 is lossless for the tested browse output. Block-cooperative low8 reduced full local-file load to about `1.27 s`; frame-cooperative low8 reduced it to about `1.12 s` and seven-load profile-sum to `7.88 s`, all with corrected-frame checksum parity. | Promote frame-cooperative low8 for count-audited `h5_uint8_lossless=True` exports. Keep it disabled for general `uint16` data unless the audit passes. |
| HDF5 frame-index manifest only. | The metadata-only frame-offset manifest preserved parity and reduced parse time, but full-stack local load improved only about `1%` because file read, compressed upload, and WGSL decode dominate. | Keep the manifest as a useful metadata cache and as a prerequisite for future selected-block/range layouts; do not claim it is the load-time fix by itself. |
| Product-first masked-sum HDF5 decode. | Exact BF r30 parity now passes for a true `256` crop, full `512`, and a `1024` repeat-stress gate. Full `512` reaches about `1.03 s` wall and `0.51 s` product stage, but it still reads whole native HDF5 files and packs `1.52 GB` of selected compressed blocks. | Promote as the correct virtual-image product direction. The next speed jump requires selected-block acquisition/layout, not more scan cropping or BF reduction. |
| Selected-block sidecar layout. | A sidecar preserving exact native bslz4 streams for selected detector blocks cut full `512` BF r30 wall time from about `1.03 s` to median `0.528 s`, with exact CUDA parity; seven repeats summed to `3.716 s`. | Promote as the browser fast-load contract for product-first VI/DPC/SSB evidence, provided metadata records the detector-block coverage and the UI falls back when the mask asks for missing blocks. |
| Sidecar crop prefilter plus auto product batching. | Moving sidecar span filtering before full reads and auto-batching small row-window specs cut the true `256x256` crop page total to about `0.253 s` and product stage to about `0.141 s`, with exact CUDA parity before the later direct-float/pipeline work. | Promote for scan-region products. Keep full `512`/`1024` at batch `1`, where larger batches increased upload pressure. |
| Pixel reducer workgroup variants. | Forced `pixel-wg64/128/256` all preserved exact parity on `256`, `512`, and `1024` repeat-stress runs, but `wg64` was fastest in every case. Current representative page totals for `wg64/128/256`: `256` crop `0.292/0.315/0.363 s`, `512` full `0.470/0.551/0.689 s`, `1024` repeat-stress `1.536/1.835/2.461 s`. | Keep wider pixel variants as profiling knobs only. The production default is now the grouped-mask `wg64` kernel below. |
| Grouped-mask popcount reducer. | Re-tested after the current browser/source sync, grouped-mask `wg64` preserved exact parity and was fastest for full `512` and `1024` repeat-stress before the later direct-float/pipeline work. It was not fastest for the true `256` crop after auto-batching; pixel `wg64` won there. | Use automatic routing: pixel `wg64` for `<=256x256` product outputs, grouped-mask `wg64` above that. Leave `window.__QT_BSLZ4_MASKED_SUM_GROUPMASK=true/false` for diagnostics. |
| Direct-float selected-block output. | Writing float32 product pixels directly from the masked-sum decoder removed the separate u32 output buffer and conversion pass while preserving exact parity. It improved full `512` selected-block page total from about `0.468 s` to about `0.447 s`; `256` needed the pixel route to avoid a group-mask regression. | Keep direct-float output as the production selected-block product path. The sums are below float32's exact integer range for the tested BF/DF masks. |
| Selected-block staging pipeline. | Preparing group `N+1` while group `N` is in flight preserved exact parity and cut page totals to `0.219 s` for true `256`, `0.389 s` for full `512`, and `1.206 s` for the `1024` repeat-stress gate after the large-scan compact-shared update. GPU wait still dominates large scans, but CPU staging/submit overhead is better hidden. | Promote staging pipeline by default for selected-block product-first kernels. Leave `window.__QT_BSLZ4_MASKED_SUM_PIPELINE=false` for profiling. |
| Large selected-block compact shared memory. | Shrinking grouped-mask shared-memory clearing from full `uint16` block size to low8 byte size regressed `256`/`512`, but improved the `1024` repeat-stress gate from about `0.688 s` to `0.632 s` product stage with exact parity. Compact `wg128`/`wg256` were slower (`0.963 s` / `1.570 s` product stage). | Promote only for scans above `512x512` with `wg64`; keep the old shared-memory layout for `256` and full `512`. Leave `window.__QT_BSLZ4_MASKED_SUM_COMPACT_SHARED=true/false` for diagnostics. |
| Block-parallel selected-block grouped-mask prototype. | Splitting each selected detector block into separate workgroups and accumulating with u32 atomics preserved exact parity, but slowed the `1024` repeat-stress product stage to about `0.670 s` versus `0.632 s` for compact grouped-mask. | Removed from production source. The extra atomics/conversion pass did not pay for three selected BF blocks. |
| Serial grouped-mask selected-block decoder. | One-lane LZ4 decode followed by the same grouped-mask popcount preserved exact parity, but slowed full `512` to `0.666 s` page / `0.504 s` product stage and the `1024` repeat-stress gate to `2.372 s` page / `1.755 s` product stage. | Removed from production source. The cooperative LZ4 copy still beats serial decode despite atomics and token-loop barriers. |
| Scratch selected-group sidecar. | A mask-specific sidecar storing only the exact low-bitplane byte groups reduced the full `512` BF payload from `1.52 GB` to `0.847 GB` and preserved exact parity, but the fresh browser upload path dominated: staging-parallel route `0.427 s` page / `0.330 s` product, mapped route `0.430 s` page / `0.341 s` product, both slower than the retained selected-block path (`0.370 s` page / `0.209 s` product). | Do not promote as a package format yet. Revisit only with a reusable upload pool, browser persistent GPU cache, or a broader sidecar policy that still supports interactive mask changes. |
| Full-stack WebGPU frame-coop workgroup size. | After fixing the template so `window.__BSLZ4_FRAME_WG` changes the actual `@workgroup_size`, full `512` local-file HDF5 checksum parity passed for `wg8/16/32/64/128`. Current representative profile times: `wg8` `1.127 s`, `wg16` `0.958 s`, `wg32` `0.859 s`, `wg64` `0.992 s`, `wg128` `1.179 s`. | Promote `wg32` as the full-stack low8 default. The old `wg128` "speedup" was invalid because only the loop stride changed; the corrected `wg128` is parity-clean but slower. |
| Full-stack WebGPU file-group scheduler. | With `wg32`, full `512` local-file HDF5 checksum parity stayed exact. File group `8` repeated at about `0.849-0.851 s` page profile, versus `group=4` around `0.834-0.850 s` in paired runs but with higher decode bookkeeping; `group=12/16` increased page profile to about `0.88-0.93 s`. More than two read workers did not reduce total time. True `256` crop loads stayed faster with `group=4` (`0.397 s`) than `group=8` (`0.433 s`). | Promote size-aware low8 local-file grouping: `groupSize=8` for full loads and `groupSize=4` for small crop loads. Keep the two-worker default. |
| Full-stack WebGPU decode batch. | At file group `8`, `decodeBatch=2` preserved parity and lowered page profile slightly (`0.828-0.834 s` versus `0.834-0.850 s`), but median wall time rose (`1.090 s` versus `1.052 s`) and upload pressure increased. `decodeBatch=4/8` regressed. | Keep full-stack `decodeBatch=1`; leave `window.__QT_H5_DECODE_BATCH` for profiling. |
| Full-stack WebGPU direct frame-index parser. | Replacing per-frame `number[]` block metadata with one preallocated typed-array pass preserved parity and reduced parse time modestly (`~0.17 s` to `~0.164-0.165 s`). | Keep. It removes allocation churn but does not change the main upload/decode floor. |
| Full-stack frame-index worker parse. | Moving frame-index bslz4 metadata construction into the local-file worker preserved checksum parity and, with full-load worker count `8`, improved full `512` page profile from about `0.856 s` to `0.756 s` and wall from about `1.025 s` to `0.807 s`. The `256` crop stayed exact with worker count `2` and page profile about `0.362 s`. | Promote: use worker count `8` for manifest-backed full low8 loads, keep smaller worker count for scan-region crops. |
| Full-stack HDF5 block-index sidecar. | A `QH5IDX01` metadata sidecar stores only deterministic bslz4 block offsets/lengths, not detector pixels or compressed payloads. It reduced full `512` parse time from about `0.244 s` to `0.005 s` and kept exact CUDA checksum parity; final retained full-load profile was `0.725 s` page / `0.799 s` wall. The true `256` crop stayed exact at `0.344 s` page / `0.513 s` wall. | Promote as an optional metadata cache because it removes repeated HDF5 block-header walking. Do not claim it solves the strict `0.5 s` full-stack target; parse was mostly overlapped with upload/decode. |
| Full-stack compressed sidecar with low-plane decode. | A compressed-payload sidecar that stored normal low8 streams passed exact CUDA checksum parity but was not faster (`0.736 s` page / `0.797 s` wall) than the retained block-index path. A count-audited low6 variant also passed parity but was slower (`0.777-1.019 s` page across `wg16/32/64/128` and `fpw1/2`). | Removed from production source. The native bslz4 stream remains effectively upload-bound, and trimming decoded bit planes did not reduce compressed bytes enough to pay for another cache format. |
| Full-stack WebGPU subgroup token parser. | A scratch V-R shader removed redundant token parsing with subgroup broadcasts and looked fast (`0.55-0.57 s` page profile; GPU wait about `0.22 s`), but corrected-frame checksums were all zero on real Chrome `apple metal-3`. | Removed from production source. Do not promote any subgroup-token parser until it passes the corrected-frame checksum gate. |
| Full-stack WebGPU worker count 4 refresh. | Older retained artifacts had `workerCount=4` near the best observed full-stack profile, but a fresh paused A/B on the current bundle gave default `0.785 s` page profile versus `workers=4` `0.796 s`, both with exact parity. | Keep the current default worker policy; do not flip back to 4 without a new paired win. |
| Full-stack WebGPU branchless bit-transpose expression. | Replacing the low8 bit-transpose loop with direct bit expressions preserved parity but slowed the default `wg64` profile to about `1.004 s` in a paired run. | Reverted; the browser compiler handles the compact loop better on Apple WebGPU. |
| Full-stack WebGPU combined staging upload. | A single aligned staging/raw buffer per group preserved parity, but did not beat normal staging at the current default (`0.860 s` versus `0.856 s` page profile in a paired run). | Do not promote. Keep normal per-spec staging as the default. |
| Full-stack WebGPU packed-word shared low8 decoder. | Replacing packed shared-memory byte atomics with owner-writes-full-word shared memory preserved corrected-frame checksum parity, but slowed full `512` profile time to about `1.104 s`; GPU wait rose to about `0.815 s`. | Do not promote. Removing atomics alone is not enough; the extra word assembly and shared-memory traffic outweighed the savings on Apple WebGPU. |
| Full-stack block-cooperative low8 decoder template. | The off-default block-cooperative template had an unreplaced shared-memory size placeholder that produced all-zero output in a control run. After fixing `__SH_WORDS__`, it passed exact checksum parity but was slower (`0.895 s` page profile) than the frame-cooperative default. | Keep the template fix for correctness of the profiling knob, but keep frame-cooperative `wg32` as default. |
| Skip zero-literal token barriers. | About `47%` of sampled LZ4 tokens had zero literals, but a guarded skip-barrier variant preserved parity and slowed the block-index full run to about `0.772 s` page profile. | Removed from production source; fewer barriers did not beat the compiler/GPU behavior of the uniform barrier path. |
| Pack decoded output chunks into larger group buffers. | Binding decoded chunks as offsets into about four larger output buffers preserved checksum parity but slowed the block-index full run to about `0.744 s` page profile. | Removed from production source; fewer output buffers did not offset allocation/zeroing/binding costs. |
| Full-stack scan-region frame-window crop. | Adding crop-aware bslz4 frame-window slicing and frame-index data-file prefilter made a true `256x256x192x192` browser full-stack crop exact against CUDA and reduced profile time to about `0.39-0.40 s`, versus CUDA warm crop median about `0.46 s`. | Promote for lower-level WebGPU local-H5 crop loads. UI wiring needs a matching scan-shape state path before using cropped stacks in Show4DSTEM display. |
| Merge cropped frame windows into fewer full-stack specs. | Merging the `256` crop from `262` row-window specs down to `4` specs reduced upload/GPU wait, but added about `53 ms` of CPU packing and slowed profile time from about `0.389 s` to about `0.441 s`. | Removed from production source. The faster default keeps row-window specs and relies on decode batching/staging pipeline. |
| Truncate native bslz4 uploads to the low8 prefix. | Full token audit found the low8 prefix is `3.124 GB` of `3.138 GB` compressed bytes (`99.55%`). High bitplanes are not the upload bottleneck in the native compressed layout. | Do not implement prefix-trim packing for native HDF5; it adds CPU token parsing with almost no byte savings. |
| `queue.writeBuffer` compressed upload. | Full-load attempt did not finish in the normal profiling window and was not competitive with staging-buffer upload. | Keep the staging uploader as the default and leave writeBuffer as an off-by-default experiment only. |
| Chunked `queue.writeBuffer` compressed upload. | Splitting writes into `64 MiB` pieces fixed the timeout and preserved checksum parity, but a full local-file run took about `1.82 s`; upload rose to about `0.82 s` and GPU wait to about `0.93 s`. | Keep staging upload as default. Chunked writeBuffer remains a profiling switch only. |
| Single-lane token parser for low8 WGSL decode. | The first non-uniform-barrier version looked fast but produced all-zero frames, so it was invalid. A fixed `256`-token uniform-loop version passed checksum parity but slowed to about `1.55 s` because the extra barriers dominate. | Do not promote. A future parser must preserve uniform barriers without adding a per-token metadata barrier. |
| Multiple frames per WebGPU workgroup. | `2/4/8` frames per workgroup passed checksum parity after fixing the frame-count uniform. A paired seven-load repeat showed `fpw=2` median about `1.19 s`, essentially tied with `fpw=1` and not a robust win; larger values were slower. | Keep default `fpw=1`. Leave the harness/runtime knob for profiling only. |
| Aligned 32-bit word copies inside the WGSL LZ4 decode loop. | Browser validated and rendered, but repeat full runs were not a stable speedup and sometimes slower. | Reverted; next kernel attempt should be cooperative LZ4 copy, not extra scalar branches. |
| Experimental parallel LZ4 WGSL kernel. | Full `512` browser run regressed to about `7.62 s` total; GPU wait rose to about `4.89 s` versus about `1.5 s` for the serial fused kernel. | Leave `window.__BSLZ4_PARALLEL` off by default; do not promote until parity and speed are both proven. |
| Direct mapped WebGPU compressed-byte upload. | CPU upload bucket dropped (`~0.8 s` versus `~2.2 s` in one paired run), but GPU wait and fetch wait rose and total worsened (`~6.36 s` direct versus `~5.96 s` staging). A concurrent direct-decode attempt did not finish in the normal full-load window. | Removed as production code; keep the staging uploader. |
| Product-first WebGPU radial-profile HDF5 decoder for BF/DF/ADF. | On real Chrome `apple metal-3`, full `512` radial-profile load over the same localhost tunnel took `9.07 s`, essentially tied with the full-stack path (`9.15 s`). It reduced output bytes but did not reduce fetch cost and increased/rebalanced decode work (`3.90 s` radial decode vs `3.68 s` full-stack). | Removed from production source; pursue local-worker acquisition, pipelined decode, and cooperative LZ4 instead. |
| MPS depth/read-ahead/LZ4-y/hazard sweeps. | Default depth-2/read-ahead remained best or tied: retained-scratch medians stayed around `0.60-0.62 s`; untracked Metal hazard mode did not improve the seven-load sum. | Keep the simpler default MPS scheduler and scratch reuse. |
