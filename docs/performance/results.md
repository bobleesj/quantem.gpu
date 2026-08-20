# Verified benchmark results

This page is the concise provenance index for numerical performance claims in
the public documentation. It does not turn historical diagnostics into current
release promises. Each row names the original measurement revision, cache
state, load plan, benchmark definition, and scientific agreement gate.

“First process” means a process-isolated first source encounter for which the
operating-system storage cache was not forcibly evicted. It is not called cold.
“Prepared” means an index, sidecar, or derived product cache already existed.

## Current three-host full-scan campaign

- **Date tested:** 2026-08-19
- **Measured revision:** `8c47a466d573f74e425faff611939a17fa6efbf2`
- **Source tree:** `c3094dcfeb7adc2e8268031678202fcb8517a2a0`
- **Fixture:** `real-512x512x192x192-u16-bslz4-27shard`, 28 HDF5 files,
  3,169,920,193 compressed bytes, aggregate file-manifest SHA-256
  `741e7bcf13ffd77bcacfeeabc0b7edb7b427448273ceba2a166426b8f73f509a`.
All 28 hashes match on Phil, Rodman, and MJGOAT.

Every row below uses the complete `512x512` scan, no scan crop, scan bin 1,
native `uint16` source counts, and the explicit detector bin shown. No
operating-system cache purge or reboot was performed, so no result is labeled
cold. MJGOAT ran with `CUDA_VISIBLE_DEVICES=0`; GPU 0 initially had only the
Sunshine process using 952 MiB and reported 1% utilization.

### Warm raw-source load

Wall time is the public `quantem.gpu.io.load` call through synchronized backend
completion. Detector-bin-1 uses seven sustained repetitions; the other rows
use five or seven as shown. Resident payload is not peak process memory.

| Platform | Detector bin | Output detector | Resident dtype | Repetitions | p50 | p95 | Maximum | Resident payload | State | Device tested | Date tested |
|---|---:|---:|---|---:|---:|---:|---:|---:|---|---|---|
| **CUDA** | 1 | `192x192` | `uint16` | 7 | **0.588 s** | **0.674 s** | **0.695 s** | **18.00 GiB** | Warm source | NVIDIA RTX PRO 6000 Blackwell, GPU 0 | 2026-08-19 |
| **CUDA** | 2 | `96x96` | `uint32` | 5 | **0.626 s** | **0.682 s** | **0.693 s** | **9.00 GiB** | Warm source | NVIDIA RTX PRO 6000 Blackwell, GPU 0 | 2026-08-19 |
| **CUDA** | 4 | `48x48` | `uint32` | 7 | **0.553 s** | **0.660 s** | **0.688 s** | **2.25 GiB** | Warm source | NVIDIA RTX PRO 6000 Blackwell, GPU 0 | 2026-08-19 |
| **CUDA** | 8 | `24x24` | `uint32` | 5 | **0.599 s** | **0.607 s** | **0.607 s** | **0.5625 GiB** | Warm source | NVIDIA RTX PRO 6000 Blackwell, GPU 0 | 2026-08-19 |
| **Python MPS** | 1 | `192x192` | `uint16` | 7 | **2.164 s** | **2.214 s** | **2.218 s** | **18.00 GiB** | Warm source | Apple M5 Max (`Mac17,6`, 40-core GPU, 128 GB) | 2026-08-19 |
| **Python MPS** | 2 | `96x96` | `uint16` | 5 | **0.691 s** | **0.713 s** | **0.719 s** | **4.50 GiB** | Warm source | Apple M5 Max (`Mac17,6`, 40-core GPU, 128 GB) | 2026-08-19 |
| **Python MPS** | 4 | `48x48` | `uint16` | 7 | **0.586 s** | **0.589 s** | **0.590 s** | **1.125 GiB** | Warm source | Apple M5 Max (`Mac17,6`, 40-core GPU, 128 GB) | 2026-08-19 |
| **Python MPS** | 8 | `24x24` | `uint16` | 5 | **0.575 s** | **0.576 s** | **0.576 s** | **0.28125 GiB** | Warm source | Apple M5 Max (`Mac17,6`, 40-core GPU, 128 GB) | 2026-08-19 |
| **Python MPS** | 1 | `192x192` | `uint16` | — | — | — | — | **18.00 GiB** | Blocked before decode | Apple M5 MacBook Pro (`Mac17,2`, 10-core GPU, 24 GB) | 2026-08-19 |
| **Python MPS** | 2 | `96x96` | `uint16` | 5 | **2.224 s** | **2.382 s** | **2.386 s** | **4.50 GiB** | Warm source | Apple M5 MacBook Pro (`Mac17,2`, 10-core GPU, 24 GB) | 2026-08-19 |
| **Python MPS** | 4 | `48x48` | `uint16` | 7 | **1.695 s** | **1.834 s** | **1.838 s** | **1.125 GiB** | Warm source | Apple M5 MacBook Pro (`Mac17,2`, 10-core GPU, 24 GB) | 2026-08-19 |
| **Python MPS** | 8 | `24x24` | `uint16` | 5 | **1.580 s** | **1.586 s** | **1.588 s** | **0.28125 GiB** | Warm source | Apple M5 MacBook Pro (`Mac17,2`, 10-core GPU, 24 GB) | 2026-08-19 |

The 24 GB M5 memory guard is a successful policy result, not a failed load. A
native-detector allocation would materialize 18.00 GiB against a conservative
12.4 GiB limit derived from the 17.8 GiB recommended Metal working set. The
loader stopped before allocation and recommended detector bin 2; it did not
silently change the requested plan.

For CUDA, the CuPy pool used/reserved after the seven-repetition native row was
18.00/36.00 GiB. Detector-bin-2, detector-bin-4, and detector-bin-8 used/reserved
9.00/18.00 GiB, 2.25/4.50 GiB, and 0.5625/1.129 GiB respectively. These are
post-run allocator observations, not total-card peak samples. Apple rows retain
logical Metal payload plus process logs; process RSS alone does not include the
whole unified-memory payload and is not promoted as peak memory.

### First campaign encounter plus scientific products

These are single first campaign encounters, not medians and not cold. The wall
boundary is synchronized public load plus mean diffraction, exact
total/BF/DF, CoM row/column, and fixed-orientation iDPC. Evidence-NPZ
serialization is excluded from the retained total.

| Platform | Detector bin | Load | Products | Load + products | Memory observation | Scientific state | Device tested | Date tested |
|---|---:|---:|---:|---:|---|---|---|---|
| **CUDA** | 4 | **1.991 s** | **36.435 ms** | **2.027 s** | Process RSS max **3.42 GiB**; CuPy used/reserved **2.25/2.26 GiB** | Integer/CoM pass; iDPC block | NVIDIA RTX PRO 6000 Blackwell, GPU 0 | 2026-08-19 |
| **Python MPS** | 4 | **1.881 s** | **101.286 ms** | **1.982 s** | Peak memory footprint **2.93 GiB** | Phil/Rodman exact; CUDA iDPC block | Apple M5 Max (`Mac17,6`, 40-core GPU, 128 GB) | 2026-08-19 |
| **Python MPS** | 4 | **2.662 s** | **112.817 ms** | **2.775 s** | Peak memory footprint **2.93 GiB** | Phil/Rodman exact; CUDA iDPC block | Apple M5 MacBook Pro (`Mac17,2`, 10-core GPU, 24 GB) | 2026-08-19 |

The measured product-stage decomposition is:

| Platform | Mean diffraction | Mask fit | Total exact | BF exact | DF exact | CoM | iDPC | Device tested | Date tested |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| **CUDA** | **6.102 ms** | **0.155 ms** | **8.706 ms** | **1.637 ms** | **1.452 ms** | **2.531 ms** | **15.852 ms** | NVIDIA RTX PRO 6000 Blackwell, GPU 0 | 2026-08-19 |
| **Python MPS** | **76.788 ms** | **0.235 ms** | **4.021 ms** | **2.245 ms** | **2.310 ms** | **5.258 ms** | **10.429 ms** | Apple M5 Max (`Mac17,6`, 40-core GPU) | 2026-08-19 |
| **Python MPS** | **59.063 ms** | **0.399 ms** | **14.459 ms** | **6.757 ms** | **7.131 ms** | **14.681 ms** | **10.327 ms** | Apple M5 MacBook Pro (`Mac17,2`, 10-core GPU) | 2026-08-19 |

### Native Swift/Metal release diagnostics

The same source revision was rebuilt in release mode. Catalog/index timings are
metadata preparation, not detector decode; prepared index reopen is not a
source load.

| Operation | Statistic | Apple M5 Max | Apple M5 | Shape/state | Date tested |
|---|---|---:|---:|---|---|
| Native HDF5 catalog only | Single run | **12.397 ms** | **20.198 ms** | Same 28-file fixture | 2026-08-19 |
| Native HDF5 first index build | Single run | **1.370 s** | **1.239 s** | New cache directory | 2026-08-19 |
| Native HDF5 prepared index reopen | p50 | **3.669 ms** | **2.375 ms** | Seven process-isolated reopens | 2026-08-19 |
| Native HDF5 prepared index reopen | p95 | **3.865 ms** | **2.651 ms** | Seven process-isolated reopens | 2026-08-19 |
| Float32 FFT | First execution | **7.566 ms** | **7.001 ms** | `512x512` | 2026-08-19 |
| Float32 FFT | Warm p50 | **0.213 ms** | **0.551 ms** | 20 warm transforms | 2026-08-19 |
| Float32 FFT | Warm p95 | **0.391 ms** | **0.865 ms** | 20 warm transforms | 2026-08-19 |
| UInt32 statistics | First execution | **3.679 ms** | **2.141 ms** | `512x512` | 2026-08-19 |
| UInt32 statistics | Warm p50 | **0.296 ms** | **0.727 ms** | 20 warm analyses | 2026-08-19 |
| UInt32 statistics | Warm p95 | **0.408 ms** | **0.975 ms** | 20 warm analyses | 2026-08-19 |

The display benchmark retained exact range `0:4095` and histogram sum
`262144`. Its linear-render GPU median was `0.0143 ms` on the M5 Max and
`0.0471 ms` on the M5. These resident 2D timings are not wall-to-wall 4D-STEM
load times.

### Numerical adjudication

The frozen gates are byte equality for integer products, `rtol=0`,
`atol=1e-5` for CoM, and `rtol=1e-5`, `atol=1e-5` for iDPC.

| Plan | Product | Comparison | Maximum absolute error | Gate |
|---|---|---|---:|---|
| Detector bin 4 | Mean diffraction, total, BF, DF | Phil vs Rodman vs CUDA | **0** | **Pass** |
| Detector bin 4 | CoM row/column | Phil vs Rodman | **0** | **Pass** |
| Detector bin 4 | CoM row/column | Phil vs CUDA | **1.91e-6** | **Pass** |
| Detector bin 4 | iDPC | Phil vs Rodman | **0** | **Pass** |
| Detector bin 4 | iDPC | Phil vs CUDA | **2.84e-5** | **Block** |
| Detector bin 1 | Mean diffraction, total, BF, DF | Public MPS vs CUDA | **0** | **Pass** |
| Detector bin 1 | CoM row/column | Public MPS bin-2 sidecar vs CUDA | **0.508 px** | **Block** |
| Detector bin 1 | iDPC | Public MPS bin-2 sidecar vs CUDA | **1.16e-2** | **Block** |
| Detector bin 1 | CoM row/column | Direct full-resolution Metal diagnostic vs CUDA | **7.63e-6** | **Pass** |
| Detector bin 1 | iDPC | Direct full-resolution Metal diagnostic vs CUDA | **7.58e-5** | **Block** |

The native-detector public MPS session can automatically reuse a detector-bin-2
interaction sidecar. That is suitable for responsive interaction only when its
changed detector sampling is explicit; it is not full-resolution parity. A
diagnostic bypass showed that the existing full-resolution Metal CoM kernel
takes `83.3 ms` on the M5 Max and passes the CoM gate. This private diagnostic
does not itself change the public contract, and the iDPC gate remains blocked.

Detector-bin-2 and detector-bin-8 rows are timing evidence only because this
campaign did not retain equivalent real-data product arrays for those plans.
No result inherits detector-bin-4 parity.

### Focused checks and environment gaps

- Phil Python: **132 passed, 10 skipped** in 2.73 s.
- Rodman Python: **132 passed, 10 skipped** in 5.24 s.
- MJGOAT GPU 0 Python/CUDA: **148 passed, 2 skipped** in 8.98 s.
- Phil Swift/Metal: **66 tests, 5 skipped, 0 failures**.
- Rodman release and debug production products: **build pass**.
- Rodman Swift tests: **blocked before execution** because the installed
  CommandLineTools toolchain has no XCTest module; this is an environment gap,
  not a passing test row.

All benchmark commands placed the isolated exact source first. This matters on
Phil because its default Python environment otherwise resolves a stale editable
checkout from a mounted path.

## Historical native CUDA and MPS IO

| ID | Date and exact revision | Hardware | Source and load plan | Cache state and benchmark definition | Result | Scientific and calibration provenance |
|---|---|---|---|---|---|---|
| CUDA-512-LOAD | 2026-07-20, [`b61572e4`](https://github.com/bobleesj/quantem.gpu/commit/b61572e47058e4b3fa48835541f667d46b762cf0) | NVIDIA RTX PRO 6000 Blackwell | `512x512x192x192`, source `uint16`, full scan, no crop/bin, audited `uint8` browse output | Warm source; 946-run load/decompress median | `0.450 s`; decoded resident `9.66 GB` | Corrected selected-frame integer checksums. No detector calibration enters load timing. |
| CUDA-1024-LOAD | 2026-07-20, [`cee0ba5c`](https://github.com/bobleesj/quantem.gpu/commit/cee0ba5ca3725b03054ecf5e6a14e304bb93d4ed) | NVIDIA RTX PRO 6000 Blackwell | `1024x1024x192x192`, `uint16`, full scan and detector, no crop/bin | First observed source; one end-to-end load/decompress wall measurement, storage cache not controlled | `4.704 s`; decoded resident `77.31 GB` | Selected corrected frames bit-exact to direct HDF5 (`max_abs=0`, `sum_abs=0`). No detector calibration enters load timing. |
| CUDA-STOCHASTIC-IO | 2026-07-28, [`1c5dd03b`](https://github.com/bobleesj/quantem.gpu/commit/1c5dd03b3ba60b98417449e55a18f0e41a58536b) | NVIDIA RTX PRO 6000 Blackwell | 40 sources, 1000 global random positions per source, detector `192x192`, native `uint16`, no detector bin | First-process reader sweep `1/2/4/8`; storage eviction method not retained. Warm-source repeats reported separately. | first process `8.90/8.98/9.47/9.97 s`; warm `1.0-1.6 s`; GPU decode sum `0.05-0.11 s`; output `2.95 GB` | Global stochastic order and raw counts preserved. No calibration mask is applied. Historical diagnostic: no newer equivalent benchmark was retained. |
| CUDA-CAL-BUILD | 2026-07-28, [`1c5dd03b`](https://github.com/bobleesj/quantem.gpu/commit/1c5dd03b3ba60b98417449e55a18f0e41a58536b) | NVIDIA RTX PRO 6000 Blackwell, allocator capped at 12 GB | `1024x1024x192x192`, native `uint16`, no crop/bin, bounded scan-row chunks | First product-cache build; source-cache state not retained. Wall includes raw HDF5 stream and calibration reductions. | build `12.31 s`; raw stream `11.76 s`; reductions `8.68 ms/chunk`; rotation `0.018-0.022 s`; cache `16.93 MB` | Mean DP, BF/DF, CoM, and rotation are built from the full scan. BF radius/rotation matched the reference; their fitted values were not retained in public evidence. Not a launch/cache-hit time. |
| MPS-CAL-BUILD | 2026-07-21, [`6c8ca5d0`](https://github.com/bobleesj/quantem.gpu/commit/6c8ca5d0a66bf78d88d6310fba5a1b9a2ea50326) | Apple Metal GPU; Mac model was not retained | `512x512x192x192`, native `uint16`, no crop/bin, 64-row chunks | First product-cache build; source-cache state not retained. Wall includes raw HDF5 stream and Metal reductions. | build `3.96 s`; raw stream `3.95 s`; chunk load p50 `316.5 ms`; chunk reduce p50 `29.3 ms`; rotation `2.6 ms` | Mean DP/BF/DF bit-exact to CUDA; CoM row/column max error `7.63e-6`; BF radius and rotation matched, but fitted values were not retained. Historical diagnostic because the exact Mac model is missing. |
| PRODUCT-CACHE-REOPEN | 2026-07-20, [`628214a8`](https://github.com/bobleesj/quantem.gpu/commit/628214a857963aa0e9684c41b9f388d83946d1c5) | Backend-neutral local filesystem; host model not retained | Saved BF/DF/CoM/rotation products for a full `1024` scan; raw detector volume is not reopened | Saved-result reopen, five repeats | `8.0/7.2/7.1/6.8/6.8 ms` | Reopens the persisted calibration products; it is never represented as raw HDF5 load or cache build. Historical diagnostic because host/storage identity is missing. |
| MPS-1024-LOAD | 2026-07-20, [`cee0ba5c`](https://github.com/bobleesj/quantem.gpu/commit/cee0ba5ca3725b03054ecf5e6a14e304bb93d4ed) | Apple Metal GPU; Mac model was not retained | `1024x1024x192x192`, chunk-backed `uint16`, full scan and detector, no crop/bin | First observed source; one end-to-end load/decompress wall measurement, storage cache not controlled | `4.617 s`; decoded resident `77.31 GB` | Selected corrected frames bit-exact to direct HDF5 (`max_abs=0`, `sum_abs=0`). Historical diagnostic because the exact Mac model is missing. |
| M2-AIR-BIN4-E2E | 2026-08-18, measured candidate `2c047160`; published integration [`e662d7fe`](https://github.com/bobleesj/quantem.gpu/commit/e662d7feebf78e7c1513276651d0be55a555cb40) | Physical 8 GB `Mac14,2` M2 MacBook Air | `512x512x192x192` `uint16`; full `512x512` scan, no crop, scan bin 1, explicit exact-sum detector bin 4 to `48x48` | First process / first observed source, seven alternating process-isolated loads per fixture; OS storage cache was not forcibly evicted. Wall is action to complete first product; Metal is the fused decode/bin interval. | Fixture A wall p50/p95/max `1.985/1.989/1.989 s`, Metal `1.615/1.629/1.629 s`; Fixture B wall `2.043/2.148/2.148 s`, Metal `1.618/1.624/1.624 s`; peak process `1.43 GB`, swap delta zero | Frozen/candidate/frozen BF, ABF, ADF, CoM row/column, DPC row/column, and iDPC exports byte-identical; selected diffraction hashes `255b94c5a4b37122` and `cc1b9e849138c351`. This is not comparable to no-bin load or saved-result reopen. |

The M2 Air row is newer than the July Apple measurements but does not replace
them: it measures a physical low-memory application path with explicit detector
bin 4, while the July rows measure no-bin library paths or calibration-cache
construction.

## WebGPU local-file and display paths

| ID | Date and exact revision | Hardware | Source and load plan | Cache state and benchmark definition | Result | Scientific and calibration provenance |
|---|---|---|---|---|---|---|
| WEBGPU-512-FULL | 2026-07-20, [`b61572e4`](https://github.com/bobleesj/quantem.gpu/commit/b61572e47058e4b3fa48835541f667d46b762cf0) | Chrome, Apple adapter string `apple metal-3`; Mac model not retained | `512x512x192x192`, source `uint16`, count-audited lossless-low8 `uint8` browse output, no crop/bin; file group 8, worker count 8, staging pipeline | Prepared local-file indexes/sidecars; 946-cycle repeated soak, not cold | p50 `0.772 s`, range `0.726-0.879 s`; decoded `9.7 GB` | First/middle/last corrected-frame checksums exact to CUDA. No detector calibration enters full-stack load. |
| WEBGPU-1024-FULL-REJECTED | 2026-07-21, [`6c8ca5d0`](https://github.com/bobleesj/quantem.gpu/commit/6c8ca5d0a66bf78d88d6310fba5a1b9a2ea50326) | Chrome, NVIDIA RTX PRO 6000 Blackwell | True `1024x1024x192x192`, no crop/bin, full-stack browse attempt | First observed attempt; cache state not retained | about `97.2 GB` device memory before failure | Failed before profile/checksum readback. Rejected; not a parity or performance signoff. |
| WEBGPU-DET-BIN | 2026-07-20, [`cee0ba5c`](https://github.com/bobleesj/quantem.gpu/commit/cee0ba5ca3725b03054ecf5e6a14e304bb93d4ed) | Headed Chrome, NVIDIA RTX PRO 6000 Blackwell | Full `512x512x192x192` and true `256x256` crop; explicit detector bin `2/4/8`; bad pixels zeroed before bin; block-index sidecars | Prepared local-file page profiles; crop arm has 20 repeats, not cold | full low8 `1.199/1.212/1.106 s`; crop p50 `0.774/0.755/0.733 s`, p95 `0.798/0.813/0.775 s`; native `uint16` bin 2 two-pass `2.651 s` | Corrected-frame checksums exact to the zero-bad-before-bin reference. No detector calibration mask is used. |
| WEBGPU-256-CROP | 2026-07-20, [`b61572e4`](https://github.com/bobleesj/quantem.gpu/commit/b61572e47058e4b3fa48835541f667d46b762cf0) | Chrome, Apple adapter string `apple metal-3`; Mac model not retained | True `256x256x192x192` scan crop from a full source, `uint8` browse output, no detector bin; worker 2, group 4, batch 8 | Prepared local-file indexes/sidecars; 946-cycle repeated soak | p50 `0.338 s`, range `0.316-0.464 s` | Crop checksums exact to the corresponding CUDA full-load slice. This is an explicit crop benchmark and is never substituted for full-scan evidence. |
| WEBGPU-BF-256 | 2026-07-20, [`b61572e4`](https://github.com/bobleesj/quantem.gpu/commit/b61572e47058e4b3fa48835541f667d46b762cf0) | Chrome, Apple adapter string `apple metal-3`; Mac model not retained | True `256x256` scan crop, detector `192x192`, BF radius 30 px; selected-block exact compressed sidecar | Prepared selected-block source; 946-cycle page-total soak | p50 `0.210 s`, range `0.185-0.246 s`; product stage about `0.100 s` | BF product max/mean absolute error `0` to CUDA; calibration is the fixed 30 px BF radius. |
| WEBGPU-BF-512 | 2026-07-20, [`b61572e4`](https://github.com/bobleesj/quantem.gpu/commit/b61572e47058e4b3fa48835541f667d46b762cf0) | Chrome, Apple adapter string `apple metal-3`; Mac model not retained | Full `512x512x192x192`, BF radius 30 px; selected-block exact compressed sidecar | Prepared selected-block source; 946-cycle page-total soak | p50 `0.378 s`, range `0.358-0.473 s`; visible product-first run `0.307-0.336 s` | BF product max/mean absolute error `0` to CUDA; calibration is the fixed 30 px BF radius. |
| WEBGPU-BF-1024 | 2026-07-20, [`cee0ba5c`](https://github.com/bobleesj/quantem.gpu/commit/cee0ba5ca3725b03054ecf5e6a14e304bb93d4ed) | Chrome, NVIDIA RTX PRO 6000 Blackwell | True `1024x1024x192x192`, BF radius 30 px; selected-block exact compressed sidecar | Prepared selected-block source; four-run wall median | wall p50 `4.92 s`; page/profile `4.85 s`; product stage `1.56 s`; selected payload `6.88 GB`; output `4.19 MB` | Product max/mean absolute error `0`, mismatches `0` to an independent Python reference; fixed 30 px BF radius. Not full-stack browse signoff. |
| WEBGPU-BF-1024-STRESS | 2026-07-20, [`b61572e4`](https://github.com/bobleesj/quantem.gpu/commit/b61572e47058e4b3fa48835541f667d46b762cf0) | Chrome, Apple adapter string `apple metal-3`; Mac model not retained | Synthetic `1024x1024` dispatch/output stress made from four repeats of real `512` evidence, BF radius 30 px | Prepared selected-block source; 946-cycle soak | p50 `1.170 s`, range `1.142-1.631 s`; product stage about `0.595 s` | Product max/mean absolute error `0`; scaling diagnostic only, not a true 1024 acquisition claim. |
| WEBGPU-VISIBLE-512 | 2026-07-20, [`b61572e4`](https://github.com/bobleesj/quantem.gpu/commit/b61572e47058e4b3fa48835541f667d46b762cf0) | Visible Chrome, Apple adapter string `apple metal-3`; Mac model not retained | Full `512x512x192x192` prepared local-file path | Prepared local-file indexes/sidecars; one latest visible load plus warm GPU-resident interactions | visible full load `0.933 s`; warm drag `0.5-0.9 ms` | Corrected-frame checksum gate passed; BF/ADF/DPC interactions reuse GPU-resident data. `0.933 s` is a single visible run, not a median. |
| WEBGPU-DPC-512 | 2026-07-20, [`cee0ba5c`](https://github.com/bobleesj/quantem.gpu/commit/cee0ba5ca3725b03054ecf5e6a14e304bb93d4ed) | Headed Chrome, NVIDIA RTX PRO 6000 Blackwell | Full `512x512x192x192`, no crop/bin, GPU-resident DPC/iDPC display | Warm GPU-resident display medians and full recompute medians; local-file harness rejects URL fallback | display row/column/iDPC p50 `14.9/13.2/13.2 ms`; recompute p50 `13.7/19.3/22.7 ms`; idle `60 FPS` | Same BF mask and fixed rotation as Python reference; fitted mask/rotation values were not retained publicly. DPC max error `7.63e-6`; iDPC mean/max error `4.70e-6/3.05e-5` from float32 FFT order. |

## Resident kernels and compressed save

These rows exclude source loading unless the benchmark definition says
otherwise. Missing host models are explicit; those rows remain historical
diagnostics even when numerical parity is strong.

| ID | Date and exact revision | Hardware | Source and load plan | Cache state and benchmark definition | Result | Scientific and calibration provenance |
|---|---|---|---|---|---|---|
| CUDA-BF-512 | 2026-07-19, [`0456e15e`](https://github.com/bobleesj/quantem.gpu/commit/0456e15ebfecd3627794118a930295e55a5a709b) | CUDA GPU; exact model not retained | GPU-resident full `512x512x192x192`, full scan, no crop/bin | Warm resident-kernel before/after microbenchmark; source load excluded | `4.96 -> 1.35 ms` | Exact integer BF sums, max error `0`; mask values were not retained publicly. Historical diagnostic. |
| CUDA-ADF-512 | 2026-07-19, [`0456e15e`](https://github.com/bobleesj/quantem.gpu/commit/0456e15ebfecd3627794118a930295e55a5a709b) | CUDA GPU; exact model not retained | GPU-resident full `512x512x192x192`, full scan, no crop/bin | Warm resident-kernel before/after microbenchmark; source load excluded | `16.16 -> 3.86 ms` | Exact integer ADF sums, max error `0`; annular mask values were not retained publicly. Historical diagnostic. |
| CUDA-DF-512 | 2026-07-19, [`0456e15e`](https://github.com/bobleesj/quantem.gpu/commit/0456e15ebfecd3627794118a930295e55a5a709b) | CUDA GPU; exact model not retained | GPU-resident full `512x512x192x192`, full scan, no crop/bin | Warm resident-kernel before/after microbenchmark; source load excluded | `62.64 -> 1.84 ms` | Exact integer dense-DF sums, max error `0`; mask values were not retained publicly. Historical diagnostic. |
| CUDA-COM-512 | 2026-07-19, [`0456e15e`](https://github.com/bobleesj/quantem.gpu/commit/0456e15ebfecd3627794118a930295e55a5a709b) | CUDA GPU; exact model not retained | GPU-resident full `512x512x192x192`, full scan, no crop/bin | Warm resident-kernel before/after microbenchmark; source load excluded | `200.42 -> 12.39 ms` | Row/column CoM max error `0`; coordinate order is `(row, col)`. Historical diagnostic. |
| MPS-SAVE-U16-512 | 2026-07-25, [`3061501`](https://github.com/bobleesj/quantem.gpu/commit/30615019cfe293ae9759006ae89c0e378b7065fd) and public API [`83bb608`](https://github.com/bobleesj/quantem.gpu/commit/83bb6089e11604b5828e6f94a70d49e487e75929) | Apple Metal GPU; exact Mac model not retained | Chunk-backed `512x512x192x192` `uint16`, no crop/bin, batch 2048, Bitshuffle/LZ4 HDF5 | Warm resident source; save wall includes compression and HDF5 writes, source load excluded | sweep `1.69-1.80 s`; public default confirmation `1.91 s`; output `1.205 GB` | Exact decoded samples, mismatches `0`. No detector calibration enters save. Historical diagnostic because host/storage models are missing. |
| SSB-CUDA-512-FULL | 2026-07-19, [`0456e15e`](https://github.com/bobleesj/quantem.gpu/commit/0456e15ebfecd3627794118a930295e55a5a709b) | NVIDIA RTX PRO 6000 Blackwell | Prepared real `512x512` SSB field, full-BF policy, float32/complex64 | Warm prepared phase+loss kernel; source load and preparation excluded | mean `32.5 ms`, p50 `32.2 ms`, p95 `33.3 ms` | Same complete BF disk, aberrations, objective, and loss reference. The same-revision SSB performance record identifies the Blackwell workstation; the exact BF count was not retained in the summary row. |
| SSB-MPS-512-R30 | 2026-07-28, [`e8d49866`](https://github.com/bobleesj/quantem.gpu/commit/e8d49866ea16cc57c0073d734c448cbbf601a5a5) | Apple M5 MacBook Pro (`Mac17,2`), 10-core GPU, 24 GB unified memory | Prepared real `512x512` Hermitian `G_qk`, radius-30 BF policy, float32/complex64 | Warm prepared kernels; source load and preparation excluded | phase p50/p95 `76.88/78.98 ms`; phase+loss p50/p95 `76.52/77.41 ms`; loss `0.2932657`; object mean `10.86 ms` | Same BF policy, aberrations, precision, objective, and frozen loss. Hardware identity is retained in the same-revision SSB performance record and frozen MPS reference fixture. |
| SSB-MPS-512-FULL | 2026-07-28, [`e8d49866`](https://github.com/bobleesj/quantem.gpu/commit/e8d49866ea16cc57c0073d734c448cbbf601a5a5) | Apple M5 MacBook Pro (`Mac17,2`), 10-core GPU, 24 GB unified memory | Prepared real `512x512` Hermitian `G_qk`, full active BF policy, float32/complex64 | Warm prepared kernels; source load and preparation excluded | phase p50/p95 `476.32/509.44 ms`; phase+loss p50/p95 `537.58/557.51 ms`; loss `0.0885396` | Same full-active BF policy, aberrations, precision, objective, and frozen loss. Hardware identity is retained in the same-revision SSB performance record and frozen MPS reference fixture; the summary row does not retain the exact BF count. |
| SSB-MPS-1024-SYNTH | 2026-07-28, [`e8d49866`](https://github.com/bobleesj/quantem.gpu/commit/e8d49866ea16cc57c0073d734c448cbbf601a5a5) | Apple M5 MacBook Pro (`Mac17,2`), 10-core GPU, 24 GB unified memory | Prepared synthetic `1024x1024` Hermitian `G_qk`, 8809 BF, float32/complex64 | Warm prepared kernels; source load and preparation excluded | object p50 `142.7 ms`; phase+loss p50 `669.1 ms`; resident `G_qk` `37.02 GB` | Same synthetic BF policy and exact objective. Hardware identity is retained in the same-revision SSB performance record and frozen MPS reference fixture. Scaling diagnostic, not a real-acquisition claim. |

## What changed in this audit

- A single exact revision and byte-identical real fixture now cover current
  full-scan CUDA GPU 0 and Python MPS load timings at detector bins 1/2/4/8 on
  two Apple devices. This replaces the stale representative CUDA/MPS numbers
  in the overview where the configuration is comparable; the older rows remain
  below as historical, differently configured evidence.
- Current product timing is separated from parity. Detector-bin-4 integer
  products and CoM pass, but cross-backend iDPC fails its frozen `1e-5` gate.
  Native-detector public MPS CoM/iDPC also fails because the interaction
  sidecar uses detector bin 2; those fast results are not presented as native
  resolution.
- Release-mode Native Swift/Metal catalog, first-index, prepared-index,
  display, statistics, and FFT diagnostics now identify both Apple devices.
  Prepared index reopen remains explicitly distinct from raw-source load.
- The stochastic IO `8.90 s` result is now “first process” rather than “cold”
  because the storage-cache eviction procedure was not retained.
- The visible WebGPU `0.933 s` result is now explicitly a single headed run,
  not a median. The comparable 946-cycle prepared-source median remains
  `0.772 s`.
- The physical M2 Air result is added as the latest Apple application evidence,
  but it does not replace the July no-bin library measurements because it uses
  explicit exact-sum detector bin 4.
- No frozen parity value, output hash, or tolerance was recaptured. Rows with a
  missing exact host model are labeled historical diagnostics rather than
  silently upgraded to release claims.

Detailed accepted and rejected experiments remain in the
[optimization ledger](../maintainer/backend-optimization-matrix.md). Benchmark
interpretation follows the [methodology](methodology.md).
