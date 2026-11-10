# Verified benchmark results

This page is the concise provenance index for numerical performance claims in
the public documentation. It does not turn historical diagnostics into current
release promises. Each row names the original measurement revision, cache
state, load plan, benchmark definition, and scientific agreement gate.

“First process” means a process-isolated first source encounter for which the
operating-system storage cache was not forcibly evicted. It is not called cold.
“Prepared” means an index, sidecar, or derived product cache already existed.

## CUDA streaming index, 2026-11-07

The [streaming index record](io/reports/2026-11-07-stream-index.md) measures
13.0% more completed-pattern throughput with identical counts and encoded bytes.
The smaller tile mapping reaches 91% of available DRAM throughput in its kernel
profile; the complete streaming pipeline remains below its unestablished limit.

## Full-scan native Metal SSB, 2026-11-05

The [Direct Ptycho preparation study](ssb/reports/2026-11-05-direct-ptycho-preparation.md)
records exact FFT/model preparation improvements and faster bounded QEM column
decoding. It separates these gains from the unfinished complete-workflow target.

The [512 × 512 × 192 × 192 M5 measurement](ssb/reports/2026-11-05-mps-full-scan.md)
records 200-trial fifth-order fitting, exact transfer parity, a rejected batching
candidate, and native-app limitations. The two-second full-size target is not met.

## Broad SSB candidate overhead, 2026-11-01

The [density batching comparison](ssb/reports/2026-11-01-broad-density-batching.md) preserves every candidate and matched CUDA fit output while reducing host proposal overhead. Default standard fitting is unchanged; broad fitting remains a larger workload.

## Joint ANS detector queries, 2026-10-25

The [seven-acquisition CUDA record](virtual-detectors/reports/2026-10-25-seven-ans-acquisitions.md)
measures 1.98× faster ADF and 2.65× faster displaced DF queries with exact
full-image agreement. Measurements interleave both implementations under GPU
contention and separate loading, query and host transfer from rendering.

## SSB reconstruction and interaction, 2026-10-24

The [direct phase viewer handoff check](ssb/reports/2026-10-24-direct-live-handoff.md)
records a corrected cache invalidation when opening a viewer at unchanged
optics. Cached numerical regularization updates and first preparation are
timed separately; these observations do not establish browser frame rate.

The [native Metal redraw record](ssb/reports/2026-10-24-native-metal-ssb-redraw.md)
times `MetalSSBJointModel.run` with every BF pixel, all 14 terms, tilt
and depth on Apple M5 Max: 512² fifth order at 3× falls from 215 ms to 29–34 ms
per frame (median), 256² to 10 ms. Parity against the previous kernel is
within 4e-6 rad. Displayed app frame rate is not part of this record.

The [anonymous C10 cache and FFT transfer benchmark](rendering/reports/2026-10-24-ssb-c10-cache-transfer.md)
retains full data and higher-order corrections. It separates CUDA cache
experiments, an isolated WebGPU transfer improvement and actual notebook
paints. The 2× notebook measurement does not improve on its baseline; 4×
measurements vary. The 60 fps target remains unmet. Raw source identities and
images remain in the private evidence archive.

## CUDA HDF5 saving, 2026-11-07

The [saving record](io/reports/2026-11-07-hdf5-save.md) measures seven complete
ANS-resident acquisitions: 48.26 s before and 25.91 s after bounded batching
and fewer host transfers, including flush and publication. Counts and
calibration match exactly; shared GPU/storage timings are not hardware limits.

## HDF5 and QEM Disk–GPU loading, 2026-10-23

Start with the [IO benchmark record](io/reports/2026-10-23-io-loading-limits.md). It is the
current entry point for this campaign; the older records below keep their
original source revisions and timing boundaries.

| Record | What it establishes |
|---|---|
| [Current three-backend loading check](io/reports/2026-10-23-io-loading-limits.md#routine-cross-backend-loading-check) | Two full 512² inputs on CUDA, MPS and Swift/Metal, five cold/cached trials, matched references and persistent private history |
| [Fresh 512² CUDA regression check](io/reports/2026-10-23-io-loading-limits.md#routine-512-cuda-loading-check) | Five cold and five cached loads per format/input, frozen encoded/product parity, matched replay within the 20% review threshold |
| [Five-acquisition backend matrix](io/reports/2026-10-23-io-loading-limits.md#complete-workflows) | CUDA, Python MPS and Swift/Metal HDF5 loads, QEM loads and verified conversions, with cache definitions |
| [512² CUDA loading status](io/reports/2026-10-23-io-loading-limits.md#validated-512-cuda-loading) | Two full 512 × 512 × 192 × 192 inputs, exact original-count comparisons and measured load times |
| [Routine conversion](io/reports/2026-10-23-io-loading-limits.md#routine-conversion-with-optional-checks-disabled) | New Python defaults with source hashing and exhaustive verification off; saving and fsync included |
| [CUDA saving profile](io/reports/2026-10-23-io-loading-limits.md#focused-cuda-save-profile-for-handoff) | Write/flush attribution, limitations and archived agent handoff; further optimization deferred |
| [Compression accounting](io/reports/2026-10-23-qem-compression.md) and [index layout](io/reports/2026-10-23-qem-layout-cuda-profile.md) | Encoded payload, indexes, metadata and size relative to bitshuffle/LZ4 HDF5 |
| [Verification](io/reports/2026-10-23-qem-verification.md), [CUDA](io/reports/2026-10-23-io-cuda.md), [MPS](io/reports/2026-10-23-io-mps.md), [Swift/Metal](io/reports/2026-10-23-io-swift-metal.md) | Earlier full scale, source identities, exact-count coverage and backend-specific limitations |

**Status:** the tested 512² × 192² ARINA HDF5/QEM CUDA loading paths are
validated. Large-QEM CUDA write/flush time remains variable and unresolved.
Routine Python conversion defaults to `verify=False, source_hashes=False`;
QEM payload checksums and file-data fsync remain enabled. This development
record is not a claim about a newly published package or app build.

Raw evidence, source snapshots and the handoff are retained under private
experiment `20261023-io-limits`; the earlier backend scale uses
`20261023-qem-backend-verification`. Reusable runners are described in the
[test guide](../developer/testing.md#performance-entry-points). The
[change ledger](changes.md#2026-10-23-optional-conversion-checks) records the
default and interpretation changes.

The [qualified loading-limits comparison](io/reports/2026-10-23-io-loading-limits.md)
measures five complete 192 × 192 detector acquisitions on CUDA, Python MPS and
Swift/Metal from one frozen source tree. It separates first/repeated HDF5
readiness, full verified conversion, and measured empty-cache/repeated QEM loads.
Every backend checked 215,016,800,256 original counts exactly. Largest cold QEM
medians are 7.790 s on CUDA, 2.929 s on MPS and 2.907 s on Swift/Metal; storage,
cache and verification boundaries prevent interpreting this as a GPU ranking.
Broader format-gate failures and blocked cells remain explicit. Local default
HDF5/QEM smoke tests passed. The first CUDA save profile identifies file writes
and final fsync as the dominant cost. A later private direct-write candidate
passed exact checks but did not establish a repeatable real QEM save gain; the
buffered default remains. Raw-write controls do not replace the canonical table.
Further storage investigation is deferred; no maintenance or writer fix is
claimed by the profiling handoff.

## CUDA QEM loader correction, 2026-10-23

The [cache-aware loader comparison](io/reports/2026-10-23-cuda-qem-loader-fix.md) measures
10.61 to 6.59 s with the file cache empty, and 10.44 to 4.78 s with half cached,
for the complete 38.333 GB QEM. Ordinary repeated loading is 5.10 s on the
largest of five inputs. All 215,016,800,256 original values compare exactly;
targeted CUDA tests pass. The broader FP8/data acceptance gaps remain explicit.
Historical HDF5, MPS and Swift results below retain their original revisions.

## Five default acquisitions: Disk–GPU loading, 2026-10-23

The [original five-acquisition comparison](io/reports/2026-10-23-default-five-io.md) covers
192 × 192 detectors on CUDA, MPS and Swift/Metal. It selects completed
measurements, with first/repeated HDF5 loads, full verified conversion and QEM
loads kept separate. HDF5 uses bitshuffle + LZ4; QEM uses ANS. MPS and Swift/Metal
ran on a MacBook Pro with Apple M5 Max and 128 GB unified memory. The selected
five inputs use 30.0% less storage as CUDA-produced QEM.

The [original six-acquisition table](io/reports/2026-10-23-default-six-io.md),
[compression analysis](io/reports/2026-10-23-qem-compression.md), and
[index-size/CUDA profile](io/reports/2026-10-23-qem-layout-cuda-profile.md) remain historical
evidence, including the excluded 96 × 96 input. The profile separates observed
storage-read delays from a 28.76 GB/s pinned-transfer control and documents
background device and disk activity. Later variable loads do not replace the
original medians.

The [2048² CUDA cache recheck](io/reports/2026-10-23-cuda-qem-cache-recheck.md) confirms
10.20–10.45 s authenticated loads with 37.3–38.3 GB fetched from storage, while
a separate full-byte-count transfer control takes 1.33 s. File cache residency
and brief background GPU activity are recorded explicitly.

## QEM backend verification, 2026-10-23

The [QEM conversion record](io/reports/2026-10-23-qem-verification.md) separates physical
CUDA, Python MPS, and native Swift/Metal measurements, with exhaustive count
verification and explicit unsupported cases. The eleven-acquisition scale has
separate [CUDA](io/reports/2026-10-23-io-cuda.md), [MPS](io/reports/2026-10-23-io-mps.md), and
[Swift/Metal](io/reports/2026-10-23-io-swift-metal.md) tables for loading, encoding, storage
size, saving and QEM reopening.

## Acquisition-worker startup, 2026-10-23

The [loader startup record](io/reports/2026-10-23-loader-startup.md) separates reusable
CUDA preparation, full-native H5 loading, staged fitting and real browser paint.
It records exact saved-product and full-volume checks; the end-to-end latency
targets remain unmet. This is not a release qualification.

## MPS HDF5 loading, 2026-10-23

The [bounded loading record](io/reports/2026-10-23-mps-hdf5-pipeline.md) measures overlapping
decode and reusable staging at full resolution. The 2048² case improves from a
34.80-second repeated control to 16.26–16.54 seconds, with identical complete
encoded residency. Broader format-gate failures remain explicitly recorded.

## SSB research optimization, 2026-10-22

The [staging-memory record](ssb/reports/2026-10-22-ssb-staging-memory.md) measures lazy
per-detector staging and smaller Fourier batches. The 256-square cases retain
about 4.3 GiB less memory with exact phase and no consistent timing regression.
Full 512-square peak memory is unchanged.

The [independent TPE reuse record](ssb/reports/2026-10-22-independent-tpe-reuse.md)
measures exact proposal reuse in the broad-search candidate on real 256/512
scans and separately profiles synthetic 1024 reconstruction. The solver is
hybrid; its CPU proposal stage is faster and CUDA phase kernels are unchanged.

The [four-case preparation reuse record](ssb/reports/2026-10-22-rotation-comparison-reuse.md)
qualifies the local comparison implementation. The separate
[broad-fifth CUDA study](ssb/reports/2026-10-22-broad-fifth-cuda-study.md) reports private-model
prototype gains and exact reference checks; it does not change the public
higher-order estimator or qualify a release.

## SSB joint fitting, 2026-10-21

The [dated installed-wheel SSB record](ssb/reports/2026-10-21-ssb-joint-benchmark.md)
contains full-field, regional, move/resize and fresh-kernel widget evidence.
Use [the reusable benchmark command](ssb/joint-fit.md) for new measurements.
MPS measurements passed; CUDA qualification and release publication remain
pending. This record does not replace the historical loader results below.

The platform-profile rows below were measured between 2026-08-18 and 2026-08-22, before
`io.load` on CUDA and MPS kept acquisitions ANS encoded (October 2026). The
rows time the loaders and resident products of their recorded revisions; they
are dated records, not timings of the current `io.load`.

## Platform profile, 2026-08

The August 2026 cross-platform load, screening, resident-product and SSB profile is archived in
[Platform profile, 2026-08](../maintainer/history/platform-profile-2026-08.md).

## Historical and rejected results

The 2026-08 tables are archived. Older campaigns are retained once in their
domain-specific maintainer records instead of being copied into this page.

| Record | Status | Canonical owner |
|---|---|---|
| Earlier three-host full-scan campaign | Superseded for current headline timing; still useful same-fixture history | [Load acceptance evidence](../maintainer/backend-4dstem-load-checklist.md) |
| July native CUDA/MPS IO and compressed-save work | Historical diagnostic; some host/storage fields were not retained | [Optimization ledger](../maintainer/backend-optimization-matrix.md) |
| WebGPU local-file, selected-block, and display campaigns | Historical implementation evidence, not current full-stack or physical-8-GB signoff | [Optimization ledger](../maintainer/backend-optimization-matrix.md) and [WebGPU history](../maintainer/history/index.md) |
| WebGPU full-native batch-2 load candidate | Refuted for performance: exact output but prepared-index/source-pages-unspecified p50/p95/max `1.128484/1.140200/1.141501 s` missed the strict subsecond gate and did not replace the accepted `1.080860 s` single observation | Sealed evidence `75eec9cccd5c1a9814068a83a23e67253227a363` |
| SSB size sweeps and rejected kernel layouts | Historical or rejected unless promoted into the current table above | [SSB performance history](../maintainer/history/ssb-performance-to-2026-10-18.md) |
| Physical M2 Air application loading | Separate application-level evidence; never substituted for a library benchmark | [M2 Air Metal evidence](../maintainer/m2-air-lz4-match-unroll-2026-08-18.md) |
| Rejected first-chunk screening and nondeterministic CUDA calibration | Failed scientific gates; retained to prevent repetition, never shown as current speed | [Revision and change ledger](changes.md) and [optimization ledger](../maintainer/backend-optimization-matrix.md) |

A historical number may be quoted only with its original revision, fixture,
cache state, scientific plan, hardware, and acceptance status. The
[revision and change ledger](changes.md) records why a current row replaced or
remained separate from an older campaign.

- [Initial fit and fifth-order refinement](ssb/reports/2026-10-22-ssb-initialization-and-refinement.md): matched initialization study and optimization lessons.
