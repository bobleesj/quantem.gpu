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
| 2 | **Detector bin, min-memory streaming** | Done | Done | Done | CUDA and MPS can bin during load without materializing the full no-bin stack. WebGPU has explicit count-preserving `detBin` source support in the local-H5 loader; full-512 and true crop-256 `detBin=2/4/8` headed parity passed on a real NVIDIA WebGPU adapter with exact corrected-frame checksums, including native non-low8 `uint16` `detBin=2`. |
| 3 | **Scan-region crop load, true crop** | Done | Done | Done | CUDA/MPS crop during HDF5 load. WebGPU slices frame windows and prefilters data files before upload/decode. |
| 4 | **Product-first region, BF/DF/ADF without full stack** | Done | Done | Done | CUDA/MPS use backend kernels over resident/chunked data. WebGPU selected-block sidecars compute exact product evidence without the full decoded browse stack. |
| 5 | **128 scan load** | Done | Done | Done | CUDA/MPS crop equality gates pass; WebGPU headed stress passes on real hardware. |
| 5 | **256 scan load** | Done | Done | Done | WebGPU true crop and selected-block product gates are exact versus CUDA and faster than the previous warm CUDA crop baseline. |
| 5 | **512 scan load** | Done | Done | Done | CUDA warm steady load meets the target; MPS is near CUDA; WebGPU local-file block-index path is exact but still above the strict full-stack target. |
| 5 | **1024 scan load** | Done | Done | Partial | CUDA and MPS have true real-acquisition no-bin `1024x1024x192x192` reference loads with bit-exact selected-frame parity. WebGPU has true-acquisition product-first BF signoff, but full-stack no-bin browser scan load still needs signoff; repeat-stress gates do not count. |
| 6 | **Minimum memory footprint / chunking** | Done | Done | Partial | CUDA streams/binning into device arrays. MPS has chunk-backed unified memory. WebGPU product-first avoids the full stack, but full-browse still materializes the decoded cube. |
| 7 | **Speed comparable to CUDA** | Done | Done | Partial | CUDA is the reference. MPS is near CUDA for one full load. WebGPU product/crop paths are CUDA-like and full no-bin browser iDPC now clears 30 FPS by median; full-browse is still short of the strict CUDA-like target. |

## Signed-Off Evidence Map

Every `Done` cell above must be represented here. Evidence rows are public-safe:
they summarize shape, parity, stage split, and footprint without raw file paths.

| Capability | Backends covered | Evidence anchor in matrix | Parity gate | Performance / footprint evidence |
| --- | --- | --- | --- | --- |
| uint8 source load/decode | CUDA, MPS, WebGPU | `HDF5 load/decompress`, `HDF5 local-file Show4DSTEM production path`, `WebGPU corrected-frame checksum gate` | Corrected-frame integer checksums match CUDA; count audit permits WebGPU low8 browse. | CUDA warm full load median `0.450 s` across 946 runs on an RTX PRO 6000 Blackwell GPU; MPS median `0.577 s`; WebGPU block-index median `0.772 s` across 946 real Chrome rows; decoded full browse stack is `9.7 GB`. |
| uint16 source load/decode | CUDA, MPS, WebGPU | `HDF5 load/decompress`, `MPS no-bin load + VI + CoM smoke`, WebGPU source contract | CUDA/MPS preserve integer load; WebGPU source supports `uint16` decode and product kernels. | Native `uint16` raw footprint for full `512` is `19.33 GB`; WebGPU browse may pack only audited low8 output. |
| Detector bin, min-memory streaming | CUDA, MPS, WebGPU | `Seven-panel BF/ADF/DF grid`, MPS fused bin rows, `WebGPU detector-bin local-file load` | Product parity is exact for CUDA detector-bin workflows; MPS fused bin path is source-backed and covered by chunk tests; WebGPU full-512 and true crop-256 `detBin=2/4/8` corrected-frame checksums are exact against the zero-bad-before-bin reference. | CUDA bin2 reduces resident full-stack bytes from `19.33 GB` to `4.83 GB`; MPS exposes fused bin sidecar paths; WebGPU full-512 count-audited low8 page profiles were `1.199/1.212/1.106 s` for `detBin=2/4/8` on NVIDIA Blackwell WebGPU. True crop-256 repeated medians were `0.774/0.755/0.733 s` with p95 `0.798/0.813/0.775 s` for `detBin=2/4/8`; native non-low8 `uint16` `detBin=2` was exact at `2.651 s`. |
| Scan-region crop load, true crop | CUDA, MPS, WebGPU | `Real HDF5 crop-first equality gate`, `HDF5 local-file scan-region full-stack path` | Crop-first arrays/checksums match the corresponding full-load slice. | WebGPU true `256x256` crop page profile median `0.338 s` over the 946-cycle soak, with reduced compressed decode/read volume. |
| Product-first region, BF/DF/ADF without full stack | CUDA, MPS, WebGPU | `BF virtual image`, `Dense DF virtual image`, `Real crop product agreement gate`, `Product-first BF selected-block sidecar` | BF/ADF/DF integer sums exact; CoM within `1e-5`; WebGPU product max/mean abs error `0` versus CUDA. | CUDA full `512` BF/ADF/DF kernels are millisecond-scale; WebGPU selected-block BF medians were `0.210 s` for true `256`, `0.378 s` for full `512`, `1.170 s` for `1024` repeat-stress, and `4.92 s` wall / `1.56 s` product stage for true real-acquisition `1024`, without materializing the full stack. |
| 128 scan load | CUDA, MPS, WebGPU | `Real HDF5 crop-first equality gate`, `Show4DSTEM WebGPU headed stress` | CUDA/MPS crop equality; WebGPU real-adapter headed smoke with product interaction. | WebGPU `128x128` headed stress shows warm BF in a few milliseconds and idle RAF at 60 FPS. |
| 256 scan load | CUDA, MPS, WebGPU | `HDF5 local-file scan-region full-stack path`, `Product-first BF selected-block sidecar` | WebGPU crop checksums and product parity are exact versus CUDA. | WebGPU selected-block true `256x256` crop page total median `0.210 s`, range `0.185-0.246 s`; full-stack crop page profile median `0.338 s`. |
| 512 scan load | CUDA, MPS, WebGPU | `HDF5 load/decompress`, `HDF5 local-file Show4DSTEM production path`, `Product-first BF selected-block sidecar` | CUDA/MPS full-load gates pass; WebGPU corrected-frame checksum parity passes on first/middle/last frames. | CUDA warm median `0.450 s`; refreshed MPS range `0.550-0.593 s`; WebGPU full-browse median `0.772 s` and product-first BF median `0.378 s`. |
| 1024 scan load | CUDA, MPS | `HDF5 load/decompress` | Real-acquisition no-bin selected corrected frames are bit-exact against direct HDF5. | `1024x1024x192x192 uint16`, `77.31 GB` resident stack; CUDA `4.704 s` on an RTX PRO 6000 Blackwell GPU, MPS chunk-backed `4.617 s` on Apple Metal. |
| Full no-bin DPC/iDPC | WebGPU | `DPC/CoM/iDPC` | Headed real-adapter load parity passed. Browser DPC row/col uses GPU-resident display buffers with validation readback; iDPC uses paired DPC buffers plus a dual-real FFT and matches the Python reference within float32 FFT tolerance. | Full `512x512x192x192` no-bin after FFT command batching: DPC row/col/iDPC display medians `14.9/13.2/13.2 ms`; recompute medians `13.7/19.3/22.7 ms`; DPC max abs error `7.63e-6`; iDPC mean abs error `4.70e-6`, max `3.05e-5`; idle RAF `60 FPS` on NVIDIA Blackwell WebGPU. Local-file timing reruns must use `--require-local-profile` to reject URL fallback. |
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
| 3 | WebGPU selected-block product for `256`, `512`, and true `1024` | Current medians are `0.210 s`, `0.378 s`, and `4.92 s` wall for true real-acquisition `1024`; preserve `max_abs=0` and keep the automatic route as the default. |
| 4 | WebGPU detector-bin load path | Full-512 plus true crop-256 `detBin=2/4/8` are signed off with repeated/p95 evidence. Keep presets explicit about binning and continue p95 refreshes after decoder or upload changes. |
| 5 | Browser iDPC optimization | WebGPU iDPC is implemented and signed off against the Python reference at float32 FFT tolerance. Median display/recompute now clears 30 FPS; next target is p95/outlier tightening and stricter max-error analysis if the Python reference moves to a float32 FFT baseline. |
| 6 | True `1024x1024x192x192` acquisition on WebGPU | Product-first BF selected-block true-acquisition signoff is done with exact parity. Full-stack no-bin browser browse/load still needs either enough free WebGPU VRAM for a true full-stack run or an explicit documented memory-policy rejection. |

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
