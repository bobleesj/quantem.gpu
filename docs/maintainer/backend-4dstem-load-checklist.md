# Backend 4D-STEM Load/Decode Checklist

Handoff checklist for accelerated Show4DSTEM HDF5 load, decode, and product
paths across **CUDA**, **MPS**, and **WebGPU**. Pair this with
`backend-optimization-matrix.md`; that matrix is the measured evidence log and
this page is the pass/fail rollup.

Rule: do not count a capability as done if it changes the microscope evidence.
No hidden scan crop, detector bin, BF reduction, saved derived float cache, or
CPU fallback can masquerade as an accelerated backend. Parity is exact integer
match versus the CUDA reference on real data, except float CoM/DPC values where
the acceptance tolerance is `<=1e-5`.

## Legend

| Mark | Meaning |
| --- | --- |
| Done | Implemented and signed off with real-data parity plus performance evidence. |
| Partial | Implemented or source-present, but not signed off. |
| Gap | Not implemented; real backend gap. |
| NA | Not applicable to this backend. |

## Capability Matrix

| # | Capability | CUDA | MPS | WebGPU | Notes / gap |
| --- | --- | :---: | :---: | :---: | --- |
| 1 | **uint8 source load/decode** | Done | Done | Done | All three decode bitshuffle/LZ4 `uint8` browse evidence. WebGPU low8 is allowed only after count audit proves it is lossless after bad-pixel correction. |
| 1 | **uint16 source load/decode** | Done | Done | Done | CUDA/MPS preserve native integer evidence. WebGPU decodes `uint16` source and may pack browse output to lossless `uint8` only when audited; native `uint16` masked-sum remains source-supported. |
| 1 | **uint32 source load/decode** | Partial | Partial | Gap | CUDA/MPS have partial/source plumbing but no real detector signoff. WGSL has 32-bit bitshuffle pieces, but no real acquisition path or parity gate. |
| 2 | **Detector bin, min-memory streaming** | Done | Done | Gap | CUDA and MPS can bin during load without materializing the full no-bin stack. WebGPU has no load-time detector-bin path yet; any bin must be explicit in UI/API. |
| 3 | **Scan-region crop load, true crop** | Done | Done | Done | CUDA/MPS crop during HDF5 load. WebGPU slices frame windows and prefilters data files before upload/decode. |
| 4 | **Product-first region, BF/DF/ADF without full stack** | Done | Done | Done | CUDA/MPS use backend kernels over resident/chunked data. WebGPU selected-block sidecars compute exact product evidence without the full decoded browse stack. |
| 5 | **128 scan load** | Done | Done | Done | CUDA/MPS crop equality gates pass; WebGPU headed stress passes on real hardware. |
| 5 | **256 scan load** | Done | Done | Done | WebGPU true crop and selected-block product gates are exact versus CUDA and faster than the previous warm CUDA crop baseline. |
| 5 | **512 scan load** | Done | Done | Done | CUDA warm steady load meets the target; MPS is near CUDA; WebGPU local-file block-index path is exact but still above the strict full-stack target. |
| 5 | **1024 scan load** | Partial | Partial | Partial | Only repeat-stress gates exist today. No backend is signed off on a real `1024x1024x192x192` acquisition. |
| 6 | **Minimum memory footprint / chunking** | Done | Done | Partial | CUDA streams/binning into device arrays. MPS has chunk-backed unified memory. WebGPU product-first avoids the full stack, but full-browse still materializes the decoded cube. |
| 7 | **Speed comparable to CUDA** | Done | Done | Partial | CUDA is the reference. MPS is near CUDA for one full load. WebGPU product/crop paths are CUDA-like; full-browse is still short of the strict `0.5 s` target. |

## Signed-Off Evidence Map

Every `Done` cell above must be represented here. Evidence rows are public-safe:
they summarize shape, parity, stage split, and footprint without raw file paths.

| Capability | Backends covered | Evidence anchor in matrix | Parity gate | Performance / footprint evidence |
| --- | --- | --- | --- | --- |
| uint8 source load/decode | CUDA, MPS, WebGPU | `HDF5 load/decompress`, `HDF5 local-file Show4DSTEM production path`, `WebGPU corrected-frame checksum gate` | Corrected-frame integer checksums match CUDA; count audit permits WebGPU low8 browse. | CUDA warm full load median `0.450 s` across 946 runs on an RTX PRO 6000 Blackwell GPU; MPS median `0.577 s`; WebGPU block-index median `0.772 s` across 946 real Chrome rows; decoded full browse stack is `9.7 GB`. |
| uint16 source load/decode | CUDA, MPS, WebGPU | `HDF5 load/decompress`, `MPS no-bin load + VI + CoM smoke`, WebGPU source contract | CUDA/MPS preserve integer load; WebGPU source supports `uint16` decode and product kernels. | Native `uint16` raw footprint for full `512` is `19.33 GB`; WebGPU browse may pack only audited low8 output. |
| Detector bin, min-memory streaming | CUDA, MPS | `Seven-panel BF/ADF/DF grid`, MPS fused bin rows | Product parity is exact for CUDA detector-bin workflows; MPS fused bin path is source-backed and covered by chunk tests. | CUDA bin2 reduces resident full-stack bytes from `19.33 GB` to `4.83 GB`; MPS exposes fused bin sidecar paths. |
| Scan-region crop load, true crop | CUDA, MPS, WebGPU | `Real HDF5 crop-first equality gate`, `HDF5 local-file scan-region full-stack path` | Crop-first arrays/checksums match the corresponding full-load slice. | WebGPU true `256x256` crop page profile median `0.338 s` over the 946-cycle soak, with reduced compressed decode/read volume. |
| Product-first region, BF/DF/ADF without full stack | CUDA, MPS, WebGPU | `BF virtual image`, `Dense DF virtual image`, `Real crop product agreement gate`, `Product-first BF selected-block sidecar` | BF/ADF/DF integer sums exact; CoM within `1e-5`; WebGPU product max/mean abs error `0` versus CUDA. | CUDA full `512` BF/ADF/DF kernels are millisecond-scale; WebGPU selected-block BF medians were `0.210 s` for true `256`, `0.378 s` for full `512`, and `1.170 s` for `1024` repeat-stress without materializing the full stack. |
| 128 scan load | CUDA, MPS, WebGPU | `Real HDF5 crop-first equality gate`, `Show4DSTEM WebGPU headed stress` | CUDA/MPS crop equality; WebGPU real-adapter headed smoke with product interaction. | WebGPU `128x128` headed stress shows warm BF in a few milliseconds and idle RAF at 60 FPS. |
| 256 scan load | CUDA, MPS, WebGPU | `HDF5 local-file scan-region full-stack path`, `Product-first BF selected-block sidecar` | WebGPU crop checksums and product parity are exact versus CUDA. | WebGPU selected-block true `256x256` crop page total median `0.210 s`, range `0.185-0.246 s`; full-stack crop page profile median `0.338 s`. |
| 512 scan load | CUDA, MPS, WebGPU | `HDF5 load/decompress`, `HDF5 local-file Show4DSTEM production path`, `Product-first BF selected-block sidecar` | CUDA/MPS full-load gates pass; WebGPU corrected-frame checksum parity passes on first/middle/last frames. | CUDA warm median `0.450 s`; refreshed MPS range `0.550-0.593 s`; WebGPU full-browse median `0.772 s` and product-first BF median `0.378 s`. |
| Minimum memory footprint / chunking | CUDA, MPS | Memory footprint table, CUDA detector-bin rows, MPS chunk-backed rows | Chunked/bin paths preserve the requested explicit evidence policy. | CUDA and MPS avoid unnecessary full no-bin materialization when crop/bin is requested; WebGPU full-browse is not signed off for this row. |
| Speed comparable to CUDA | CUDA, MPS | `HDF5 load/decompress`, `Seven-master HDF5 load/decompress`, MPS load rows | Same evidence policy and dtype; no hidden crop/bin. | CUDA is reference; refreshed MPS one-load median is about `0.58 s`, within the current near-CUDA band. WebGPU remains `Partial` for full-browse. |

## Current WebGPU Execution Queue

These are the current follow-up targets after the 946-cycle browser soak. Each
item must produce a JSON artifact with adapter, parity, stage split, and
footprint fields before it can update a `Done` cell.

| Priority | Target | Required result |
| ---: | --- | --- |
| 1 | Full-stack WebGPU `512x512x192x192` load/decode | Current median is `0.772 s`; next win must close the remaining gap to the strict `0.5 s` full-stack target while preserving checksum parity. |
| 2 | WebGPU `256x256` crop load/decode | Current median is `0.338 s`; preserve exact crop checksum and keep crop-first IO below CUDA warm crop timing. |
| 3 | WebGPU selected-block product for `256`, `512`, and `1024` repeat-stress | Current medians are `0.210 s`, `0.378 s`, and `1.170 s`; preserve `max_abs=0` and keep the automatic route as the default. |
| 4 | WebGPU detector-bin load path | Implement explicit `det_bin` during load/product, never silent binning. Parity must compare against CUDA explicit-bin reference. |
| 5 | Full no-bin WebGPU DPC | Move from `Partial` to `Done` only after row/col CoM parity, display-path timing, and no-readback interaction evidence are recorded. |
| 6 | True `1024x1024x192x192` acquisition | Replace repeat-stress with a real acquisition only after CUDA reference load/parity succeeds. |

## Acceptance Gate Per Capability

1. **Real data only.** Synthetic tests can guard source shape, but they do not
   make a `Done` cell.
2. **Exact parity.** Corrected-frame `sum/min/max/n` integer-exact versus CUDA;
   float CoM/DPC within `1e-5`. Record max/mean abs error for products.
3. **Adapter honesty.** WebGPU timing must log the adapter and reject software
   adapters such as SwiftShader.
4. **Stage split.** Report fetch/read, parse, pack, upload, decode/GPU wait,
   compute, readback, and display separately where the path has those stages.
5. **No hidden reduction.** If bin/crop/subsample/selected blocks are active,
   they are explicit in the API/report and reflected in the shape/footprint.
6. **Footprint stated.** Report resident bytes, compressed upload/read bytes,
   sidecar bytes, and peak transient when available.
7. **Default path named.** A profiling flag does not count as shipped unless
   the production default selects it automatically and tests cover the default.

## Rejected Or Bounded Hypotheses

Do not retry these without a new reason that changes the bottleneck:

- `decodeBatch` larger than the retained defaults. Some kernels ran faster, but
  total wall time worsened from fetch/upload pressure.
- Fetch window `16/24` and broad worker/group/batch sweeps. They increased
  contention or variance; the useful wins came from decoder/layout changes.
- Full-source `uint8` compressed sidecar as the main speed path. It kept about
  the same multi-GB compressed payload and did not beat the retained local-file
  route. This does **not** reject the retained audited low8 frame decoder.
- Compressed-payload low6 sidecar. It preserved parity in a count-audited run
  but was slower than the block-index path, so low6 code is not shipped.
- High-bitplane prefix trimming for native HDF5. The useful low8 prefix was
  still about `99.55%` of the native compressed bytes.
- `queue.writeBuffer`, chunked `writeBuffer`, combined staging buffers, packed
  shared-memory low8, zero-literal barrier skip, and packed decoded-output group
  buffers. Each preserved or partially preserved parity in a profiling run but
  regressed the full-stack profile.
- A subgroup-token full-stack low8 decoder. It reduced apparent GPU wait in a
  scratch shader, but failed the corrected-frame checksum gate with all-zero
  output, so it is not shipped.

## Privacy Rule

Public docs and JSON artifacts may contain anonymized labels such as
`master-1`, shape, dtype, backend, adapter, and timing. They must not contain raw local file paths or collaborator/project-specific dataset names.
