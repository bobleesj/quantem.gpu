# Experimental native Metal entropy series

**Status: experimental, 2026-09-11 (America/Los_Angeles).**
This opt-in Swift SPI is included on `main`. It remains experimental:
merging the source does not make it a stable API, a release, or a completed
performance qualification.

Native implementation revision: the commit that introduced this document
revision, `the commit that introduced the 2026-09-11 section below on branch `metal-tans-exact-interactive` (`git log -- docs/api/experimental_metal_entropy_series.md`)` (exact tile index, annulus atlas, interactive
grouping, packet-owner kernel and pipelined queries). Earlier revisions of this
page described the 2026-09-08 promotion; pin the revision you measured.

Use this prototype to query a compatible, already encoded acquisition series
without keeping a decoded 4D tensor on the GPU. It is **ANS-based**: dense
streams use paired table-based ANS (tANS); sparse events and literal exceptions
retain exact counts. The native execution path uses Swift and Metal, not a
Python process. It consumes a prepared archive; it does **not** yet encode or
save one from an arbitrary scientist's original HDF5 file.

The compute implementation belongs to QuantEM.GPU. A native app links these
SwiftPM products and retains UI, admission, cancellation scheduling, calibration,
cache identity, and latest-wins publication ownership. No app code is included
in this integration. Related CUDA interfaces are documented separately in
[Exact compact resident series](../developer/compact-resident.md); their
producer, archive and performance capabilities must not be attributed to Metal.

## Try the opt-in API

Check out `main`, then pin its exact commit in your consumer:

```bash
git clone --branch main \
  https://github.com/bobleesj/quantem.gpu.git
cd quantem.gpu
git rev-parse HEAD
swift test -c release
```

Use that printed full revision with SwiftPM's `revision:` requirement and
depend on the `Metal4DSTEMStreamingIO` product. The `main` tip evolves;
preserve the exact revision with every result.

```swift
import Foundation
import Metal
@_spi(EntropySeriesPrototype) import Metal4DSTEMStreamingIO

// Run on one serialized worker, never the UI/main thread.
// The host must supply a compatible archive and independently admitted budgets.
let series = try ExperimentalMetalEntropySeries(
  directory: archiveURL,
  acquisitions: Array(0..<66),
  device: device,
  maximumAdditionalBytes: admittedSourceBytes)
defer { series.release() }

// Optional one-time preparation; additional index <= 2 GiB plus preparation
// headroom. This retains losslessly packed 2D sums, not a raw 4D source.
try series.prepareDetectorIndex(maximumIndexBytes: admittedIndexBytes)
// Interactive launch geometry; from here on queries use the packet-owner
// kernel (see "Packet-owner kernel and pipelined queries").
series.configureInteractiveGrouping(mixedModelTails: true, chooseCheaperBase: true)

// Both inputs are binary 192x192 arrays; the app resolves physical calibration.
let mask = zip(apertureMask, series.validDetectorMask).map {
  UInt8($0.0 != 0 && $0.1 != 0 ? 1 : 0)
}
let virtualImages = try series.detectorImages(
  mask: mask, maximumAdditionalBytes: admittedQueryBytes)
let diffractionImages = try series.diffractionImages(
  scanRow: 256, scanColumn: 256)

// Pipelined form: at most `maximumDetectorQueriesInFlight` (2) unfinished.
let submission = try series.submitDetectorImages(
  mask: mask, maximumAdditionalBytes: admittedQueryBytes,
  selectedAcquisitions: Array(0..<66))
submission.whenGPUCompleted { worker.async { publish(try? series.finishDetectorImages(submission)) } }
```

The example's URL, device, masks and budgets are supplied by the application,
not defaults or a calibration recipe. Importing the SPI is an explicit opt-in
to a breaking experimental interface. There is no new Python API for this SPI.

| Call or property | Contract |
|---|---|
| Initializer | Unique in-range acquisition indices; validates metadata and SHA-256 record integrity before publishing the encoded resident owner |
| `shape`, `acquisitionIndices` | Archive logical shape and the loaded acquisition selection; a subset does not change the archive's leading dimension |
| `diffractionImages` | One complete 192x192 `uint32` buffer per loaded acquisition; exact widening of original `uint16` values, including invalid-pixel counts |
| `detectorImages` | One complete 512x512 `uint32` virtual sum per requested acquisition; binary mask must explicitly include validity policy |
| `selectedAcquisitions:` overload | Unique loaded indices, returned in requested order; a subset query is not an all-series update |
| `diffractionImages(scanRow:scanColumn:selectedAcquisitions:)` | Exact 192x192 `uint32` patterns for a loaded subset, in requested order; the remaining acquisitions can follow in a later call |
| `rebase: true` | Recompute without the previous-image delta seed; never changes counts or coverage |
| `prepareDetectorIndex` | Exact bounded packed-tile index; failure preserves the previous source/index and completed outputs |
| `prepareDetectorIndex(maximumIndexBytes:cacheDirectory:layout:)` | `layout` chooses the exact tile layout: `.centerFine` (100 tiles, 2.1 GB) or `.uniform8` (576 exact 8x8 tile sums, about 8.3 GB). The index budget ceiling is 8 GiB; the physical admission check (allocation + index + 1 GiB within the recommended working set) is unchanged. The planner's tile price follows the layout (0.5 and 2.0); it only ranks exact decompositions |
| `prepareDetectorIndex(maximumIndexBytes:cacheDirectory:)` | Same index, restored from a digest-verified on-disk cache when it matches the archive manifest digest, acquisition selection and layout; otherwise built and then exported. Returns a `DetectorIndexPreparation` (restored, seconds, cache written, cache error) |
| `configureInteractiveGrouping(mixedModelTails:chooseCheaperBase:)` | Interactive launch geometry for a prepared series: mixed-model tail regrouping, decoder entry prefetch, cheaper-base planning, and from this call on the packet-owner kernel for the series' queries unless `QUANTEM_TANS_PACKET_OWNER=0`. Counts, coverage and precision unchanged |
| `beginDetectorAtlas(maximumBytes:)` | Start an empty exact atlas with an explicit budget; requires the prepared index and 1 GiB headroom; drops any prior atlas. See "Exact annulus atlas" |
| `appendDetectorAtlasField(mask:)` | Add the exact image of one binary mask for every loaded acquisition, computed into temporary outputs; no returned image or delta seed changes; audited bit for bit. Waits for and verifies any unfinished submissions first |
| `detectorAtlasFieldCount`, `detectorAtlasBytes`, `lastDetectorAtlasField` | Atlas size, and the stored mask the last query started from (nil when it did not) |
| `detectorDenseMask` | 192x192 bytes, 1 where a detector pixel is stored as a dense tANS column and 0 where it is a sparse-event column; planning metadata for predicting decode cost, not a validity mask |
| `submitDetectorImages(mask:maximumAdditionalBytes:rebase:selectedAcquisitions:)` | Pipelined exact subset query: plans, encodes and commits, then returns a `DetectorSubmission` without waiting. Same counts as `detectorImages`; the seed is committed at submission. Throws when `maximumDetectorQueriesInFlight` submissions are unfinished |
| `finishDetectorImages(_:)` | Waits for that submission and every earlier one (no wait after `whenGPUCompleted` fired), verifies, and returns a `DetectorQueryResult`: images in requested order, `gpuSeconds`, `decodedColumns`, `usedPrevious`, `atlasField`, `commandTiming`. On failure it throws and every affected seed rolls back; earlier finished images are untouched |
| `DetectorSubmission` | `acquisitions` (return order), `sequence` (submission order), `whenGPUCompleted(_:)` (one call on an arbitrary thread once the GPU finished it and all earlier queries; completion is not verification) |
| `detectorQueriesInFlight`, `maximumDetectorQueriesInFlight` | Unfinished submissions, and the limit (2: the three-slot output ring keeps every rollback exact). `detectorImages` and index preparation throw while a submission is unfinished |
| `archiveCheckpointSHA256` | SHA-256 of the archive manifest bytes; the identity a consumer keys its derived caches by |
| `residentBytes`, `detectorIndexBytes` | Retained package buffer accounting (encoded records, globals, output rings, index, atlas, and the packet-owner kernel's rank bytes and packed decode table once built); not peak driver allocation, compressed memory, or process RSS |
| `lastDetectorGPUSeconds` | Completed command GPU timing for the last query; not input-to-screen latency or displayed FPS |
| `release` | Releases owned source/index/state; later queries fail. Already returned buffers remain caller-retained |

Every call is made from one serialized worker, including release; the one
exception is that `whenGPUCompleted` handlers run on an arbitrary thread and
must hop back to that worker before calling the series. `detectorImages`,
`diffractionImages`, index preparation and atlas calls are synchronous.
`submitDetectorImages` / `finishDetectorImages` are the one pipelined path: up
to two queries in flight on one detector command queue, whose ordering places
each query's reads after the previous query's writes. Outputs are read-only
for the consumer: the package retains each completed image as that
acquisition's exact delta seed, and a returned image is rewritten by the third
later query of its acquisition, so read or copy it before submitting that
query. General concurrent calls and mutable output borrowing are not
qualified. Queries do not reread the source or allocate a decoded 4D resident.
The separate bounded full-count audit intentionally reads back decoded packets
for SHA-256 verification; it is not the interactive query path.

### Per-acquisition delta seeds and output rings (2026-09-10)

Delta seeds are kept **per acquisition**: the seed is the binary mask and the
completed image of that acquisition's last query, whatever subset it belonged
to. A request groups its acquisitions by identical seed mask, so one residual
decomposition serves each group and a request that mixes freshly updated and
older acquisitions still returns one complete, exact result. Dropping a seed
(index preparation, release) is always exact; it only removes an accelerator.

Each acquisition rotates through three complete output images. A returned
buffer is not written again until that acquisition has completed two further
queries, so a consumer can keep displaying the previous image while the next
one is computed, and an in-flight render never observes a partial write. This
adds 3 MiB per queried acquisition (198 MiB for all 66).

### All-acquisition interactive throughput (2026-09-10)

`configureInteractiveGrouping(mixedModelTails:chooseCheaperBase:)` configures the launch for
queries that update every retained acquisition from one interaction. Four changes, all of which
alter only launch geometry, which exact decomposition is dispatched, or which exact kernel runs it,
never counts, coverage or precision:

* the mixed-model-tail savings gate is removed (a 1/8 threshold left the regrouping inactive on
  all-acquisition frames, where roughly 0.3% of model groups were being regrouped),
* the decoder prefetches its next table entry,
* `chooseCheaperBase` compares continuing from the previous image against recomputing from the tile
  index and dispatches whichever leaves less entropy work,
* from this call on, the series' queries run on the packet-owner kernel (2026-09-11, below) unless
  `QUANTEM_TANS_PACKET_OWNER=0`. The index built before the call used the shared-model kernel;
  atlas fields appended after it use whichever kernel is selected.

The per-step timings this section used to carry (17-18 ms GPU for a 1 px all-66 step of an ADF
annulus walk on the shared-model kernel) are the baseline of the packet-owner section below. The
host cost of a query fell separately: the dispatch table for a fixed acquisition set and output
ring slot is cached instead of re-encoded, and the per-context 32-lane model grouping runs
concurrently across contexts. These are query timings on prepared GPU-resident data, not
application frame rates.

### Tile layout and content-keyed seeds (2026-09-10)

Measured over 66 acquisitions of the sealed archive on one Apple M5 Max with the GPU confirmed idle,
per all-acquisition frame of an ADF annulus walk, every configuration bit-identical to an unseeded
recompute:

| step | centerFine | uniform8 (tile price 2.0) |
|---|---|---|
| 1 px | 18.7 ms | 16.5 ms |
| 2 px | 29.2 ms | 30.3 ms |
| 4 px | 62.0 ms | 62.2 ms |
| 8 px | 124.3 ms | 77.8 ms |
| 16 px | 145.4 ms | 68.6 ms |
| full recompute | 113.0 ms | 64.9 ms |

At the former flat tile price of 0.5 the uniform8 planner bought tiles that cost more than the
columns they removed at mid-size steps (4 px: 77.6 ms); 1.3, 2.0 and 3.0 were measured and 2.0 is
used. Seed groups are keyed by mask content rather than by the call that wrote them, so acquisitions
refreshed by separate calls with the same geometry batch together again: the first all-acquisition
query after a per-acquisition catch-up is one group (21.8 ms) instead of up to one group per
acquisition.

### Exact annulus atlas (2026-09-10)

A fast all-acquisition drag has no stable frame rate when each frame continues from the previous
one: the decode cost grows with the distance moved since the last publication, and a slower frame
lets the pointer move further before the next one. The atlas removes that dependence. It holds the
exact complete image of each of a set of caller-chosen masks, packed losslessly in 256-scan blocks
(about 25.3 MB per mask over 66 acquisitions). A query may start from the stored image whose mask is
closest to the request and decode only the pixels where the two masks differ:

    image(M) = image(M0) + sum over pixels p of (M[p] - M0[p]) * column_p

Every coefficient is -1, 0 or +1, and the identity uses the stored mask bytes only, so it is exact
for any request, any sub-pixel centre and any rasterization. The planner compares this base against
continuing from the previous image and recomputing from the tile index, and dispatches the cheapest.

`beginDetectorAtlas(maximumBytes:)` reserves an explicit budget (the caller admits it plus 1 GiB
headroom) and requires the prepared detector index. `appendDetectorAtlasField(mask:)` adds one mask:
an unseeded tile-index query written to temporary outputs, so no returned image, ring slot or delta
seed changes, followed by blocked packing and a bit-for-bit audit against that query's output. About
0.12 s per field over 66 acquisitions, so a consumer adds fields between interactive queries.

Measured on one Apple M5 Max with the GPU confirmed idle, 66 acquisitions, ADF annulus (inner 40,
outer 80) on a 2 px lattice of centres, per all-acquisition frame of a 7-frame walk; every frame
verified bit-identical to an unseeded recompute with the atlas off:

| step | previous image or tile index: median / max | with the atlas: median / max |
|---|---|---|
| 2.5 px | 35.2 / 43.2 ms | 16.6 / 24.3 ms |
| 3.0 px | 45.5 / 54.1 ms | 14.5 / 18.3 ms |
| 3.3 px | 53.0 / 59.1 ms | 12.6 / 21.2 ms |

Thirty unseeded probes at random sub-pixel centres 0.4-1.1 px from a lattice centre, half near the
pattern centre and half 17-20 px out, took 8.2-23.6 ms GPU, all bit-identical. The cost follows the
distance to the nearest stored mask, not the drag step. A 2 px lattice within 22 px of the pattern
centre is 377 masks, about 9.5 GB. Geometries outside the stored masks' product and radii get no
benefit and use the other two bases. The 2026-09-10 tables in this and the two sections above were
measured on the shared-model kernel; the packet-owner kernel below runs the same decompositions
faster without changing which one the planner picks.

### Packet-owner kernel and pipelined queries (2026-09-11)

Interactive queries run on a second exact detector kernel,
`tans_detector_packet_owner_batch`. One SIMD group owns one (record, 512-scan
packet): it loops over every dense 32-lane model group of that record, adds the
packet's sparse events, and writes each of its 512 output scans once, so no
device output atomics are used. Fixed in the kernel after measurement on all 66
acquisitions: the transposed butterfly pair reduction, grouped bit-reservoir
refill (three pairs per check), fused output initialization (the exact seed or
zero base is stored with the sums instead of a separate fill pass), the
record's grouped work descriptors walked directly instead of a padded
per-record grid, device-resident tables, one packed 32-bit decode-table entry
per state (symbol pair, bits, escape, next base), and balanced sparse events
using per-record rank bytes and a ballot column search.

Exactness argument: the kernel consumes the same streams and the same bits in
the same per-lane order as the shared-model kernel, and every contribution is
the same integer; only the order in which those integers are added changes,
and uint32 modular addition is associative and commutative, so the images are
bit-identical. It is not a re-derivation of the codec. The shared-model kernel
(`tans_detector_shared_model_batch`) remains as an independent implementation:
it builds the tile index, it builds atlas fields unless the owner kernel is
selected, and it is the bit-for-bit cross-check oracle
(`QUANTEM_TANS_LOCKSTEP_CROSSCHECK=1` in `TANSLockstepThroughputExperimentTests`
recomputes each walk's final geometry and every atlas probe on the shared-model
kernel and compares hashes).

One runtime toggle, `QUANTEM_TANS_PACKET_OWNER`:

| Value | Kernel selection |
|---|---|
| unset | A series uses the shared-model kernel until `configureInteractiveGrouping` is called on it, then the packet-owner kernel for its queries. The tile-index build and any series that never makes the call (the seed and interaction fixture suites) stay on the shared-model kernel |
| `1` | The packet-owner kernel for every query of every series from construction, index and atlas builds included; the way to run the fixture exactness suites through the interactive kernel |
| `0` | The shared-model kernel everywhere, interactive queries included; the legacy fallback |

The owner kernel's tables are built on its first dispatch and cached for the
life of the series: rank bytes for every resident record (about 96 MB for 66
acquisitions, derived on the GPU from the immutable sparse flags) and the
packed decode table. `residentBytes` includes both from then on, so it grows
after the first interactive query (with `=1`, after the first query of any
kind). This is accounting of retained buffers, not a change to what the
archive or index occupy.

`submitDetectorImages` / `finishDetectorImages` let a host encode query n+1
while n decodes: one detector command queue, at most two unfinished
submissions (one fewer than the three output ring slots, so a failed query's
rollback never touches a returned image), seeds committed at submission and
rolled back on failure, and `whenGPUCompleted` for a non-blocking finish.
Counts are those of `detectorImages`. Whether a host should submit ahead is
its own latency policy: it raises throughput and can raise pointer-to-image
lag.

Evidence, with its limits. Measured on the 66-acquisition sealed archive on
one Apple M5 Max whose GPU was shared with other applications (30-65% busy
from screen sharing and streaming during every run, so absolute times drift
20-30% between runs; only paired candidate/reference runs in the same GPU
hold were compared):

* Lockstep all-66 ADF annulus walk at 0.83 px per frame (uniform8 index,
  `TANSLockstepThroughputExperimentTests`): about 17-18 ms GPU per frame on
  the shared-model kernel to about 9.6 ms on the packet-owner kernel, a paired
  ratio of about 0.56 (about 1.8x). Every frame bit-identical to an unseeded
  recompute, 30/30 atlas probes exact, every shared-model cross-check exact.
* Live application, pointer-exact all-66 ADF annulus drag, before and after
  the kernel change on the same machine and load: 100 px/s 63 to 107 fps,
  50 px/s 80 to 119 fps; input-to-published p95 at 100 px/s about 20 ms.
  These are one application's frame counters on one dataset and one gesture
  family under a shared GPU. They are not a universal 120 Hz claim, not
  idle-GPU numbers, and not evidence for BF/ABF/edge gestures, other
  devices, or other archives.

### Exact tile index cache (2026-09-10)

`exact-tile-index.json` + `exact-tile-index.bin` in the consumer-chosen directory
hold only the packed 2D tile sums (about 2 GiB for 66 acquisitions), never the
encoded source. The manifest records the archive manifest digest, the retained
acquisition indices, the layout name, the image count and, per field, the tile
geometry, packed width, word count, byte range and SHA-256. Restoring verifies
every field digest and the layout arithmetic before any field is retained; a
mismatch throws and leaves no index, so the consumer builds one instead. The
payload is written first and the manifest last through temporary files, so a
partial write is never a valid cache.

This is what allows an interactive tier (one acquisition per pointer frame)
and a background catch-up tier (the other acquisitions in small batches) to
share one series without invalidating each other's seeds. Measured on one Apple M5 Max
(128 GB) with the 66-acquisition archive and the exact index prepared:
one-pixel BF/ABF/ADF/half-plane drags on one acquisition take 0.9–1.9 ms wall
(p95 ≤ 2.1 ms), four-pixel drags ≤ 2.8 ms, a full single-acquisition
recompute 2.2–3.9 ms, while the first query after a multi-second pause costs
about 20–45 ms (see the idle experiments). Every 48-step delta chain equals an
unseeded recompute bit for bit (`TANSSingleAcquisitionLatencyExperimentTests`,
`TANSPerAcquisitionSeedTests`).

`MetalDisplayStatistics.analyzeUInt32Batch` in `MetalImageRuntime` reuses the
existing display kernels for equal-shaped images and requested scales, with
two GPU synchronization points and only range/256-bin summary readback.

## Fixed archive and precision contract

- Logical layout is `[acquisition, scan row, scan column, detector row,
  detector column]`: each acquisition is `512x512x192x192 uint16`.
- Full scan and detector coverage, scan bin 1, detector bin 1, crop none.
  No lossy encoding, downcast, preview, approximate reduction or raw-resident
  fallback is permitted.
- Each acquisition has sixteen 16,384-scan records, with 512-scan entropy
  streams. Dense codec: `source112-tans1024-pair-v1`; sparse codec:
  `position9-flag1-count8-rank256-v1`. Small-value coding is a representation,
  not a reduced scientific dtype; literal paths retain values through 65,535.
- Original counts remain unmasked in storage. Detector sums use the supplied
  binary validity/aperture mask. `192 * 192 * 65535 < 2^32`, so complete sums
  and signed-delta updates fit the exact unsigned modular arithmetic.
- The directory contains `checkpoint.json`, its declared global NPZ state,
  and all declared record shards. Paths, extents, shapes, offsets, model tables
  and completion/checksums are checked. Source-only metadata must declare the
  exact count-preserving index-rebuild contract.
- Canonical profile labels are `metal-entropy-source-v1` and
  `metal-entropy-prepared-v1`. Two immutable historical prototype labels are
  recognized by fingerprint, with identical validation. These labels do not
  make other ANS archives compatible. No files are rewritten during opening.
- SHA-256 verifies integrity against the manifest, not trusted authorship or
  independent scientific truth. Use audited archives; arbitrary hostile archives
  and a general interoperable format are not qualified by this prototype.
- Calibration, physical sampling, source identity, and mask policy must remain
  source-bound in the consumer. This SPI is not a complete metadata/calibration
  round-trip or a replacement for the general IO contract.

## Dated measurements and their limits

Retained prototype observations on Apple M5 Max, 128 GB, as of 2026-09-08:
66 complete acquisitions, exact geometry above. These are historical local
prototype measurements, **not native UI acceptance of the promoted main**.

| Endpoint | Retained observation | Meaning |
|---|---|---|
| Encoded archive to GPU resident | 14.56 s | One prepared-archive observation; excludes original HDF5 encoding |
| Optional exact index construction | 4.532 s | One observation, additional preparation |
| Initial native panels available | 19.51 s | One app observation; initial saved products, not a new arbitrary mask computation |
| One acquisition prepared load | 0.714 s | One observation; not cold original-source load |
| One acquisition full-count audit | 6.390 s | Decode + CPU readback + SHA-256, not pure decoder wall time |

Retained repeated all-66 resident query measurements (`n=3` per gesture,
complete 66-image result at return; small sample, p95 equals max):

| Detector gesture | Wall p50 (ms) | p95/max (ms) |
|---|---:|---:|
| BF | 41.155 | 41.437 |
| ABF | 53.449 | 53.636 |
| ADF | 134.442 | 134.620 |
| Large edge motion | 201.865 | 203.210 |
| Wide ADF radius change | 50.051 | 50.409 |
| Small nudge | 28.188 | 28.641 |

These timings depend on the mask difference and reusable exact index. They do
not establish 10-20 scientific FPS for general large gestures. Retained native
large-ADF input-to-publication p50 was 618.137 ms, p95/max 908.825 ms (`n=10`),
and complete all-image presentation timestamps were missing for most events.
A 120 Hz display or short GPU command duration does not prove 120 scientific
FPS. Warm resident queries must not be described as loading or encoding. The
2026-09-11 section above holds the current kernel's annulus-walk numbers; this
table predates the exact index, atlas and packet-owner kernel.

Encoded source accounting was 84,681,338,880 bytes; the optional exact index was
2,108,620,800 bytes. Private GPU allocations still consume unified memory even
when process RSS is small. There is no claim that all 66 acquisitions fit on
an 8 GB or 24 GB machine. Admission must reject insufficient budgets.

Prior parity evidence includes 504 exact complete-image query comparisons
against frozen products or a full-decoder reference, not 504 independent
original-source volume audits. One complete 18 GiB acquisition has an
independent original-count SHA-256 oracle. All-66 independent volume auditing
remains a separate gate.

## Reproduce verification

Publication check on the implementation revision above, 2026-09-11:
full release Swift suite **255 tests / 37 explicit opt-in skips / 0 failures**.
The fixture suites (`ExperimentalMetalEntropySeriesTests`,
`TANSPerAcquisitionSeedTests`, `MetalTANSInteractionTests`,
`TANSEncodeAheadTests` with their `QUANTEM_TANS_*_FIXTURE` variables set to
the sealed archive) ran **11 tests / 1 skipped / 0 failures** twice: once on
the default path and once with `QUANTEM_TANS_PACKET_OWNER=1`, so the
index/delta/edge/subset/reordered/rollback comparisons hold on both kernels.
The one skip is `MetalTANSInteractionTests`' sustained all-66 gesture test,
which needs `QUANTEM_TANS_GESTURE_FIXTURE`. Strict Swift formatting and the release build pass.
The live-application figures above come from the consuming app's own
counters, not from a package test. Private fixtures, raw logs and generated
HTML are not published.

```bash
swift test -c release
swift build -c release
xcrun swift-format lint --strict --recursive \
  src/quantem/gpu/swift/Sources src/quantem/gpu/swift/Tests

# On an admitted device with the complete sealed archive:
QUANTEM_TANS_FULL_SERIES_FIXTURE=/path/to/encoded-series \
  swift test -c release --filter MetalTANSFullSeriesTests
export QUANTEM_TANS_SPI_FIXTURE=/path/to/encoded-series
export QUANTEM_TANS_DETECTOR_FIXTURE=$QUANTEM_TANS_SPI_FIXTURE
export QUANTEM_TANS_SEED_FIXTURE=$QUANTEM_TANS_SPI_FIXTURE
export QUANTEM_TANS_INTERACTION_FIXTURE=$QUANTEM_TANS_SPI_FIXTURE
FILTER='ExperimentalMetalEntropySeriesTests|TANSPerAcquisitionSeedTests|MetalTANSInteractionTests|TANSEncodeAheadTests'
swift test -c release --filter "$FILTER"                              # shared-model kernel
QUANTEM_TANS_PACKET_OWNER=1 swift test -c release --filter "$FILTER"  # packet-owner kernel
# Paired lockstep walk with the shared-model cross-check oracle:
QUANTEM_TANS_LOCKSTEP_CROSSCHECK=1 \
  swift test -c release --filter TANSLockstepThroughputExperimentTests
```

Fixtures and frozen products are not distributed here. Without them, opt-in
tests skip; a skip is not parity or capacity evidence. The full-series test
checks all-66 loading/point queries and one independent full-count hash. The SPI
test compares index/delta/edge/subset/reordered results against full-source
queries and checks failure/release behavior. They are package tests, not a
headed app journey.

## What remains before stable integration

- [ ] Native arbitrary-source encoder, save/reopen API, and documented portable
  format/version migration with complete calibration/provenance round-trip.
- [ ] Standalone encode and full-decode wall distributions for one and all 66
  acquisitions, with disk IO, hashing, upload and index preparation separated.
- [ ] Repeated cold original-HDF5 to encoded-resident measurements with declared
  source-page/cache state, peak Metal/process/compressed/swap memory and parity.
- [ ] All-66 independent original-source count auditing, more acquisition
  shapes/dtypes, hostile-input validation and additional physical devices.
- [ ] Full product coverage through this SPI: mean DP, CoM/DPC/iDPC and FFT
  integration are not supplied by this entropy class.
- [ ] Mutable-output ownership isolation, thread-safe concurrent lifecycle
  beyond the two-deep submit/finish pipeline, bounded cancellation latency and
  asynchronous publication integration.
- [ ] General large BF/ABF/ADF gestures at 10-20 all-66 scientific FPS,
  with native input-to-publication and complete presentation evidence. The
  all-66 ADF annulus drag is measured (2026-09-11 section); BF/ABF/edge
  gestures, idle-GPU runs and complete presentation timestamps are not.
- [ ] Stable API promotion and app/package release qualification.

Optimization work is tracked in
[QuantEM.GPU issue #8](https://github.com/bobleesj/quantem.gpu/issues/8).
The integration makes the native experiment reusable and reviewable; it does not
close these gates.
