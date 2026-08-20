# Verified benchmark results

This page is the concise provenance index for numerical performance claims in
the public documentation. It does not turn historical diagnostics into current
release promises. Each row names the original measurement revision, cache
state, load plan, benchmark definition, and scientific agreement gate.

“First process” means a process-isolated first source encounter for which the
operating-system storage cache was not forcibly evicted. It is not called cold.
“Prepared” means an index, sidecar, or derived product cache already existed.

## Current platform profile

- **Date tested:** 2026-08-19 local time; retained UTC artifacts extend into
  2026-08-20.
- **Baseline revisions:** Phil `334b7b5135fe29787540370a00f280fa138430a2`;
  CUDA execution mirror `8c47a466d573f74e425faff611939a17fa6efbf2`.
  Their production compute trees are byte-equivalent for the profiled paths.
- **Clean follow-up stack:** local branch `mps-subsecond-pipeline` at
  `70bc3663c1c7cc495e77348c9fe7594545c66fa8`. It contains exact streamed
  screening (`5d56535`), strict MPS source validation (`1d2e3c9`),
  deterministic CUDA SSB fitting (`fa9ab6f`), native Metal SSB (`e1da9bc`),
  exact prepared QH5 binning (`e0e92b4`), optimized word-major binning
  (`ff3c7fd`), provenance-bound resident summaries (`d65911a`), and exact
  fused-accumulator widening (`70bc366`). These unpublished commits do not
  retroactively change baseline timings.
- **Fixture C:** independent real `512x512x192x192` native-`uint16` source,
  27 compressed shards, 3,169,920,193 bytes, 28-file manifest SHA-256
  `741e7bcf13ffd77bcacfeeabc0b7edb7b427448273ceba2a166426b8f73f509a`.
- **Fixture D:** independent real `512x512x192x192` native-`uint16` source,
  27 compressed shards, 3,165,551,746 bytes, master SHA-256
  `4802ec16ba241fef439e9dcb1c28e94f9cf9d95f773df9c5c8c3b5f7ed8192c4`;
  dataset/v0.1 identity
  `1be810b96fdff8e384ad4cb6ebd49adff9b4ab0a6503cd5fed9106e09f5aa286`.

Every current load uses the complete `512x512` scan, no scan or detector crop,
scan bin 1, and the explicit detector bin shown. The operating-system storage
cache was not forcibly evicted, so the source measurements are **warm**, not
cold. CUDA and WebGPU use D; MPS/Swift use C. They are not a fixture-controlled
backend ranking.

### Current warm load/decode/bin

The boundary is synchronized first-usable resident output from the public
loader. WebGPU uses the loader's internal library boundary rather than the
outer browser harness. Resident payload and process/card peaks remain distinct.

| Platform | Revision | Detector bin | Output detector | Resident dtype | Repetitions | p50 | p95 | Maximum | Memory observation | Parity | Device tested |
|---|---|---:|---:|---|---:|---:|---:|---:|---|---|---|
| **CUDA** | `8c47a466` | 1 | `192x192` | `uint16` | 7 | **0.386 s** | **0.396 s** | **0.397 s** | 18.00 GiB payload; 22.78 GB total-card peak | Pass | NVIDIA RTX PRO 6000 Blackwell Max-Q, GPU 1 |
| **CUDA** | `8c47a466` | 2 | `96x96` | `uint32` | 7 | **0.396 s** | **0.401 s** | **0.402 s** | 9.00 GiB payload; 12.41 GB total-card peak | Pass | NVIDIA RTX PRO 6000 Blackwell Max-Q, GPU 1 |
| **CUDA** | `8c47a466` | 4 | `48x48` | `uint32` | 7 | **0.390 s** | **0.413 s** | **0.419 s** | 2.25 GiB payload; 4.03 GB total-card peak | Pass | NVIDIA RTX PRO 6000 Blackwell Max-Q, GPU 1 |
| **CUDA** | `8c47a466` | 8 | `24x24` | `uint32` | 7 | **0.381 s** | **0.401 s** | **0.402 s** | 0.5625 GiB payload; 1.94 GB total-card peak | Pass | NVIDIA RTX PRO 6000 Blackwell Max-Q, GPU 1 |
| **Python MPS** | `334b7b5` | 1 | `192x192` | `uint16` | 7 | **2.273 s** | **2.445 s** | **2.449 s** | 19.327 GB logical payload; process RSS 0.93 GB | Pass | Apple M5 Max, 40-core GPU, 128 GB |
| **Python MPS** | `334b7b5` | 2 | `96x96` | `uint16` | 7 | **0.707 s** | **0.720 s** | **0.723 s** | 4.832 GB logical payload; process RSS 0.73 GB | Pass | Apple M5 Max, 40-core GPU, 128 GB |
| **Python MPS** | `334b7b5` | 4 | `48x48` | `uint16` | 7 | **0.605 s** | **0.613 s** | **0.615 s** | 1.208 GB logical payload; process RSS 0.73 GB | Pass | Apple M5 Max, 40-core GPU, 128 GB |
| **Python MPS** | `334b7b5` | 8 | `24x24` | `uint16` | 7 | **0.586 s** | **0.592 s** | **0.594 s** | 0.302 GB logical payload; process RSS 0.74 GB | Pass | Apple M5 Max, 40-core GPU, 128 GB |
| **WebGPU** | `334b7b5` | 1 | `192x192` | `uint8` | 5 | **0.824 s** | **0.892 s** | **0.892 s** | Chrome-tree RSS max 5.39 GB | Exact tested frames and products | Chrome 151, Apple M5 Max Metal-3 |
| **WebGPU** | `334b7b5` | 2 | `96x96` | `float32` sums | 5 | **1.281 s** | **1.300 s** | **1.300 s** | Chrome-tree RSS max 5.76 GB | Exact sampled count sums | Chrome 151, Apple M5 Max Metal-3 |
| **WebGPU** | `334b7b5` | 4 | `48x48` | `float32` sums | 5 | **1.044 s** | **1.050 s** | **1.050 s** | Chrome-tree RSS max 5.57 GB | Exact sampled count sums | Chrome 151, Apple M5 Max Metal-3 |
| **WebGPU** | `334b7b5` | 8 | `24x24` | `float32` sums | 5 | **0.979 s** | **0.986 s** | **0.986 s** | Chrome-tree RSS max 5.57 GB | Exact sampled count sums | Chrome 151, Apple M5 Max Metal-3 |
| **CPU reference** | `334b7b5` | 1 | `192x192` | `uint16` | 1 | **34.37 s** | — | — | Peak RSS 39.14 GB | Independent exact adjudicator | Apple M5 Max CPU |
| **CPU reference** | `334b7b5` | 2 | `96x96` | `uint16` | 1 | **54.22 s** | — | — | Peak RSS 10.34 GB | Independent exact adjudicator | Apple M5 Max CPU |
| **CPU reference** | `334b7b5` | 4 | `48x48` | `uint16` | 1 | **43.04 s** | — | — | Peak RSS 3.20 GB | Independent exact adjudicator | Apple M5 Max CPU |
| **CPU reference** | `334b7b5` | 8 | `24x24` | `uint16` | 1 | **38.13 s** | — | — | Peak RSS 2.18 GB | Independent exact adjudicator | Apple M5 Max CPU |

Fixture D bin 1 is value-audited lossless `uint8`; bins 2/4/8 are exact detector
sums stored as `float32`. WebGPU's Chrome RSS is not a complete device-memory
measurement and does not prove the physical 8 GB laptop gate. CPU timings are
diagnostic adjudication only, never a silent production fallback.

### Current native exact resident summary

Revision `d65911a` adds a package-owned exact summary beside a validated native
resident cache. Revision `70bc366` adds an overflow-checked path that widens
the exact `uint32` accumulators already produced by a fused source pass, so an
integrator need not reread the resident detector volume. The measured plan is
the complete `512x512` scan from fixture C,
no scan or detector crop, scan bin 1, exact detector sum bin 4 from `192x192` to
`48x48`, native `uint16` source, and `uint16` resident output. The source
identity is
`9f0ddb932c631b63cb573c38d747fa41941ee585c5389d33bdafb4add962b768`;
the resident payload SHA-256 is
`2a876d00ca1512955006a40433341b26aee766dec077ddced8368011f4ec52b3`.

The summary stores exact BF, ABF, and ADF `uint32` maps; total, detector-row,
and detector-column `uint64` moments; and one selected `uint32` diffraction
pattern. Read validates source identity, resident payload, geometry, dtype,
scan region/bin, detector bin, count audit, detector-band definition, selected
scan coordinate, artifact sizes, and every artifact SHA-256 before returning a
product. It never represents a prepared summary as source loading.

| Device | Revision | Repetitions | Cache state | First complete product p50 | p95 | Maximum | Process wall p50 | p95 | Maximum | Maximum RSS | Parity |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| Apple M5 Max, 40-core GPU, 128 GB | `d65911a` | 7 fresh processes | Prepared exact summary | **0.026 s** | **0.027 s** | **0.027 s** | **0.120 s** | **0.144 s** | **0.150 s** | **97.4 MB** | Nine same-device products byte exact |
| Apple M2 MacBook Air (`Mac14,2`, 8 GB) | `d65911a` | 7 fresh processes | Prepared exact summary | **0.029 s** | **0.030 s** | **0.030 s** | **0.110 s** | **0.124 s** | **0.130 s** | **92.0 MB** | Nine same-device products byte exact |

The original `d65911a` creation path remains a valid fallback when only a
prepared resident cache exists. It traverses the validated 1.208 GB resident
cache, but it does not open or decompress the 3.17 GB compressed-HDF5 source.
These historical fallback measurements remain here because their input state
differs from source-fused creation.

| Device | Revision | Repetitions | Cache state | Resident load wall | Metal product/moment kernel | Summary write | Process wall | Process swaps |
|---|---|---:|---|---:|---:|---:|---:|---:|
| Apple M5 Max, 40-core GPU, 128 GB | `d65911a` | 1 | Prepared resident cache, summary absent | **0.369 s** | **8.0 ms** | **8.0 ms** | **0.86 s** | **0** |
| Apple M2 MacBook Air (`Mac14,2`, 8 GB) | `d65911a` | 1 | Prepared resident cache, summary absent | **1.397 s** | **103.0 ms** | **18.0 ms** | **2.61 s** | **0** |

On the Air, the resident-cache pass took 1.313 s while the fused exact Metal
product/moment kernel took 103 ms. Mapped-page population, not reduction, is
therefore the dominant creation cost. Once exact moments exist, one
instrumented derived-product stage took **11.389 ms**: 0.565 ms center/mean on
CPU, 1.107 ms rotation/alignment on CPU, 2.095 ms Metal iDPC, and 7.622 ms
float-surface statistics. The legacy `gpu=0.000` aggregate excludes those small
command buffers and is rejected as GPU telemetry for this path.

Revision `70bc366` removes that additional resident traversal when the same
process has just completed the validated fused source pass. The consumer
rechecks a conservative accumulator bound, widens the three exact moment maps
in one small Metal dispatch, and writes the unchanged
`quantem.gpu.resident-summary/v1` schema. The table reports only the incremental
summary work; it is not a compressed-source load time.

| Device | QuantEM.GPU revision | Consumer overlay | Repetitions | Input state | Additional resident traversal | Widen wall | Widen GPU | Summary write | Process swaps | Parity | Date tested |
|---|---|---|---:|---|---|---:|---:|---:|---:|---|---|
| Apple M5 Max, 40-core GPU, 128 GB | `70bc366` | `105942d3` tracked-diff SHA-256 | 1 | Same validated fused source pass | None | **0.703 ms** | **0.328 ms** | **11 ms** | **0** | Seven summary artifacts byte exact; nine products same-device byte exact | 2026-08-20 |
| Apple M2 MacBook Air (`Mac14,2`, 8 GB) | `70bc366` | `105942d3` tracked-diff SHA-256 | 1 | Same validated fused source pass; low page residency and pre-existing swap | None | **0.569 ms** | **0.104 ms** | **18 ms** | **0** | Seven cross-device integer artifacts byte exact; all floating products below `1e-5` | 2026-08-20 |

The Air's surrounding source encounter took 6.444 s in a low-residency state;
three sequential rechecks improved from 5.534 s to 2.723 s to 2.101 s while
Metal stayed between 1.450 s and 1.542 s. Those values are a cache-state
sequence, not independent repetitions, so none is promoted into the current
warm-load table. The follow-up changed only post-pass summary materialization;
it did not make the compressed source sub-second.

All seven summary artifact hashes match across Phil and the Air. Every reopen
also reproduced same-device BF, ABF, ADF, CoM row/column, DPC row/column, iDPC,
and selected diffraction byte-for-byte against the full resident calculation.
This is prepared-product evidence, not the original compressed-source first
encounter and not a headed application-paint measurement.

### Current streamed screening

Screening is a separate build/reopen boundary, not source loading. The accepted
MPS follow-up accumulates the complete detector sum in `uint64`, validates the
provisional mask against the final full-scan mask, and reruns BF/DF only when
those masks differ.

| Platform | Revision | Operation | Source plan | Statistic | Time | Memory state | Numerical state | Device tested |
|---|---|---|---|---|---:|---|---|---|
| **Python MPS** | `5d56535` | Exact screening build | Full `512x512x192x192` `uint16`; no crop/bin; 170 rows, four chunks; exact fallback pass | Single run | **6.711 s** | Streamed source; derived products retained | Mean DP and CoM byte-exact; BF/DF value-exact | Apple M5 Max, 40-core GPU, 128 GB |
| **Python MPS** | `5d56535` | Validated screening-v3 reopen | Prepared derived products from the same source identity | p50 | **20.803 ms** | Saved-result reopen | Cache identity and products validated | Apple M5 Max, 40-core GPU, 128 GB |

The rejected one-pass candidate derived its mask from only the first chunk and
failed full-scan mean-DP/BF/DF parity. Its timing is intentionally absent from
the current ledger and remains discoverable only as a rejected experiment in
the [optimization ledger](../maintainer/backend-optimization-matrix.md).

### Current resident products

All rows exclude source loading. CUDA uses D at detector bin 1; MPS uses C at
detector bin 4; WebGPU uses D at detector bin 1. CUDA and MPS rotation/iDPC are
small-field CPU operations after GPU CoM, not GPU-kernel claims.

| Platform | Mean DP | BF | ADF | DF | CoM row/column | DPC row/column | iDPC | State and parity |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **CUDA** | **18.392 ms** | **3.768 ms** | **5.586 ms** | **3.747 ms** | **13.002 ms** | — | **21.272 ms** | p50 of 7; integer/mean exact; CoM and iDPC pass current CPU gates |
| **Python MPS** | **74.805 ms** | **2.502 ms** | **4.404 ms** | **2.642 ms** | **4.637 ms** | — | **12.678 ms** | p50 of 21; integer/mean/CoM exact; same-runtime iDPC exact |
| **WebGPU** | **50.9 ms** | **5.5 ms** | **15.0 ms** | **43.4 ms** | **82.9 ms** | **0.9/0.7 ms** | **1.4 ms** | p50 of 5; integer/mean exact; per-pixel float errors not retained |
| **CPU reference** | — | — | — | — | — | — | **177.6 ms** | Independent product traversal was 31.08 s; reference only |

### Current SSB reconstruction and calibration

CUDA calibration uses calibration SHA-256
`4a2d9cc36943973dbe0f1d5e40858160f0a6393cd56d9f54e073a358b3eff8e8`,
200 kV, 21.4 mrad semiangle, 0.49492961 Å scan sampling, full automatically
detected BF disk, float32/complex64, seeded Optuna TPE 200 trials, and
Nelder–Mead refinement. The MPS row uses C, explicit detector bin 2, 2,275
calibrated BF positions, and calibration SHA-256
`8815ddd710f33973ac11d504cd679f16d9f5d6bf3043d0480e682ecc0a053941`.
WebGPU uses a separately frozen native-detector exact `uint8` companion, 3,418
active aperture positions, and compute-matched revision `5cd285250911974c738e9c911bd00a170873bf45`.
Native Swift/Metal uses a separate frozen 512×512 exact-`uint8` full-BF
fixture at `e1da9bc86a0c1ae6edc60e1205a9966e6826f315`: 9,074 logical BF
planes, 2,459 executed aperture planes, no scan crop or bin, detector bin 1,
and float32/complex64 compute. Its source SHA-256 is
`6046f7855b6925aafc86a52cc9ef06156ebf617d63b25c5a2a10fd94762ae3ae`.
The operating-system page cache was warm.

| Platform | Operation | Statistic | Time | Numerical state | Device tested |
|---|---|---|---:|---|---|
| **CUDA** | Complex object | p50 of 7 | **13.883 ms** | Deterministic | NVIDIA RTX PRO 6000 Blackwell Max-Q, GPU 1 |
| **CUDA** | Exact phase | p50 of 7 | **32.035 ms** | Deterministic | NVIDIA RTX PRO 6000 Blackwell Max-Q, GPU 1 |
| **CUDA** | Exact phase and loss | p50 of 7 | **32.335 ms** | Deterministic | NVIDIA RTX PRO 6000 Blackwell Max-Q, GPU 1 |
| **CUDA** | 200-trial TPE plus Nelder–Mead | p50/p95/max of 3 | **11.168/11.235/11.242 s** | Byte-identical fitted parameters, phase, object, and loss | NVIDIA RTX PRO 6000 Blackwell Max-Q, GPU 1 |
| **Python MPS** | Exact phase and loss | One synchronized run | **497.187 ms** | CUDA phase max `1.2815e-6` rad; loss error `7.45e-9` | Apple M5 Max, 40-core GPU |
| **Native Swift/Metal** | Complex object, complete Hermitian cache | p50 of 7 | **8.911 ms** | CUDA phase relative L2 `5.8695e-5`; maximum `5.6288e-6` rad | Apple M5 Max, 40-core GPU |
| **Native Swift/Metal** | Exact phase-variance loss, complete Hermitian cache | p50 of 7 | **25.120 ms** | Cached-versus-streamed loss relative error below `5e-5` | Apple M5 Max, 40-core GPU |
| **Native Swift/Metal** | 200-trial TPE plus Nelder–Mead | p50 of 3 | **6.061 s** | Identical fitted parameters and loss, seed 42 | Apple M5 Max, 40-core GPU |
| **WebGPU** | Complex object | Readback wall p50 of 5 | **32.5 ms** | Deterministic hash 5/5 | Chrome 151, Apple M5 Max Metal-3 |
| **WebGPU** | Exact phase | Readback wall p50 of 5 | **102.1 ms** | Deterministic hash; one retained scheduling outlier | Chrome 151, Apple M5 Max Metal-3 |
| **WebGPU** | Exact phase and loss | Readback wall p50 of 5 | **189.4 ms** | Phase byte-identical to no-loss 5/5 | Chrome 151, Apple M5 Max Metal-3 |

An earlier CUDA atomic-objective fit split into two fitted minima under the
same seed and therefore failed the frozen repeatability gate. It is not a
current benchmark row. Native Swift calibration also repeats deterministically
but uses a different real fixture and optimizer implementation, so its timing
is not a direct CUDA ranking. The prepared MPS companion candidate is rejected
because its stored columns do not match its declared detector-bin coordinate
grid. WebGPU implements reconstruction but not TPE or Nelder–Mead calibration.
Levenberg–Marquardt is not implemented in any current SSB backend.

### Current native Swift/Metal boundary

At `d65911a`, the release suite executed 76 tests, skipped five opt-in
real-QH5/performance cases, and had no failures. A separate real-QH5 run
executed 71 tests with one performance skip and no failures. Four real QH5
frames matched the Swift CPU reference exactly for decode, detector bin 4,
BF/ABF/DF/total, and row/column moments. The physical full-scan summary evidence
above adds exact BF/ABF/ADF and overflow-safe total/row/column moments plus
same-device derived-product parity. Package-level numerical DPC/iDPC unit tests
remain pending; the derived-product timing came from an isolated headless
consumer harness, not a GUI or application-paint boundary.

At `70bc366`, the expanded release suite executed 81 tests, skipped five
opt-in/data cases, and had no failures. Its new focused checks cover the
audited low-count plan, scan-bin and incomplete-edge contribution bounds,
high-dynamic-range rejection, arithmetic overflow, and exact Metal widening.
The real-QH5 gate again passed four of four checks.

The earlier `334b7b5` profile remains the owner of prepared-index reopen
**5.339/5.760/5.842 ms** p50/p95/max and 512×512 FFT **15.622 ms** first,
**0.291/0.584 ms** warm p50/p95. Those measurements were not rerun or silently
reassigned to `d65911a`.

The later native SSB follow-up `e1da9bc` adds `MetalSSBKernels`, complete-cache
and bounded-memory exact policies, and a standalone release benchmark. Its
accepted reconstruction, loss, and fit values appear once in the SSB table
above. The [native Metal SSB migration record](../maintainer/native-metal-ssb-migration.md)
owns API lineage, cache-policy details, and artifact fingerprints. Neither page
represents warm library compute as application wall time or physical 8 GB
device signoff.

### Current evidence fingerprints

| Evidence bundle | SHA-256 |
|---|---|
| Apple atomic matrix | `358da791e27433a7f2cd5bab3e7880d1907a8f07976bd9fa529882685de0c84e` |
| CUDA atomic rows | `e68c6b56a95b6844a9eb354b1c68ca68e2f2f815395dc74ee6b4a3e6f7945272` |
| WebGPU/CPU report | `a2f75c7a595d5fc457b5c32afd0ea8aac5dd20f5db30c496b386e30acedce7d1` |
| Exact MPS screening adjudication | `1094cc68e2bf9952916fb12ac6489119a4f4ee4be2ece7f8b1c1a4f1ed411fa3` |
| Raw MPS/CUDA SSB parity | `17a7ef5750444377c7d16c18bfadef39607ea3d684c91dad93681ca887d7154e` |
| Deterministic CUDA full fit | `d262c1ed8fa55728811735bc974ef4fcc413e60aff83dfcd7396b4ad681f4527` |
| Exact Air resident summary | `4f8f366553cf8ae13b5b732a24a070f6cc404127ef6e421599d00b5c27a3688c` |
| Source-fused Air summary follow-up | `123b77e3424994980379a942da21dfbdf0d0921b2de0a862652831c4bfe814a9` |

## Historical and rejected results

Current tables stop above. Older campaigns are retained once in their
domain-specific maintainer records instead of being copied into this page.

| Record | Status | Canonical owner |
|---|---|---|
| Earlier three-host full-scan campaign | Superseded for current headline timing; still useful same-fixture history | [Load acceptance evidence](../maintainer/backend-4dstem-load-checklist.md) |
| July native CUDA/MPS IO and compressed-save work | Historical diagnostic; some host/storage fields were not retained | [Optimization ledger](../maintainer/backend-optimization-matrix.md) |
| WebGPU local-file, selected-block, and display campaigns | Historical implementation evidence, not current full-stack or physical-8-GB signoff | [Optimization ledger](../maintainer/backend-optimization-matrix.md) and [WebGPU history](../maintainer/history/index.md) |
| SSB size sweeps and rejected kernel layouts | Historical or rejected unless promoted into the current table above | [SSB performance history](../maintainer/ssb-performance.md) |
| Physical M2 Air application loading | Separate application-level evidence; never substituted for a library benchmark | [M2 Air Metal evidence](../maintainer/m2-air-lz4-match-unroll-2026-08-18.md) |
| Rejected first-chunk screening and nondeterministic CUDA calibration | Failed scientific gates; retained to prevent repetition, never shown as current speed | [Revision and change ledger](changes.md) and [optimization ledger](../maintainer/backend-optimization-matrix.md) |

A historical number may be quoted only with its original revision, fixture,
cache state, scientific plan, hardware, and acceptance status. The
[revision and change ledger](changes.md) records why a current row replaced or
remained separate from an older campaign.
