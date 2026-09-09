# Experimental native Metal entropy series

**Status: experimental, 2026-09-08 (America/Los_Angeles).**
This page describes the opt-in Swift SPI on branch
`sep-8-experimental-metal-entropy-series`. It is not a stable API, a release,
or a claim that the implementation is available on `main`.

Implementation revision: `6a0c09ad227adcf084c21bdf1fbcdb67b266c584`,
based on upstream `3c0e15ae2a61a3f4503e307575dad27e4ddeb934`.
Later documentation-only commits do not change that implementation.

Use this prototype to query a compatible, already encoded acquisition series
without keeping a decoded 4D tensor on the GPU. It is **ANS-based**: dense
streams use paired table-based ANS (tANS); sparse events and literal exceptions
retain exact counts. The native execution path uses Swift and Metal, not a
Python process. It consumes a prepared archive; it does **not** yet encode or
save one from an arbitrary scientist's original HDF5 file.

The compute implementation belongs to QuantEM.GPU. A native app links these
SwiftPM products and retains UI, admission, cancellation scheduling, calibration,
cache identity, and latest-wins publication ownership. No app code is included
in this branch. Related CUDA interfaces are documented separately in
[Exact compact resident series](../developer/compact-resident.md); their
producer, archive and performance capabilities must not be attributed to Metal.

## Try the opt-in API

Check out the dated branch, then pin its exact commit in your consumer:

```bash
git clone --branch sep-8-experimental-metal-entropy-series \
  https://github.com/bobleesj/quantem.gpu.git
cd quantem.gpu
git rev-parse HEAD
swift test -c release
```

Use that printed full revision with SwiftPM's `revision:` requirement and
depend on the `Metal4DSTEMStreamingIO` product. Experimental branch tips may
change; preserve the exact revision with every result.

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

// Both inputs are binary 192x192 arrays; the app resolves physical calibration.
let mask = zip(apertureMask, series.validDetectorMask).map {
  UInt8($0.0 != 0 && $0.1 != 0 ? 1 : 0)
}
let virtualImages = try series.detectorImages(
  mask: mask, maximumAdditionalBytes: admittedQueryBytes)
let diffractionImages = try series.diffractionImages(
  scanRow: 256, scanColumn: 256)
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
| `rebase: true` | Recompute without the previous-image delta seed; never changes counts or coverage |
| `prepareDetectorIndex` | Exact bounded packed-tile index; failure preserves the previous source/index and completed outputs |
| `residentBytes`, `detectorIndexBytes` | Retained package buffer accounting; not peak driver allocation, compressed memory, or process RSS |
| `lastDetectorGPUSeconds` | Completed command GPU timing for the last query; not input-to-screen latency or displayed FPS |
| `release` | Releases owned source/index/state; later queries fail. Already returned buffers remain caller-retained |

All calls are synchronous and must be serialized, including release. Outputs
are independent of later package publications, but **treat them as read-only**:
the package can retain a completed image as an exact delta seed. Copy before
consumer mutation. General concurrent calls and mutable output borrowing are
not qualified. Queries do not reread the source or allocate a decoded 4D
resident. The separate bounded full-count audit intentionally reads back
decoded packets for SHA-256 verification; it is not the interactive query path.

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
prototype measurements, **not new-branch native UI acceptance**.

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
FPS. Warm resident queries must not be described as loading or encoding.

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

Publication check on the implementation revision above, 2026-09-08:
full release Swift suite **213 executed / 20 explicit opt-in skips / 0 failures**;
real entropy-focused suite **21 executed / 2 explicit opt-in skips / 0 failures**.
The two remaining skips are long sustained-gesture and topology/statistics
timing experiments, not the all-66 load, frozen products or SPI index tests.
Strict Swift formatting, release build, 41 documentation/resource tests and
the documentation site build pass. No headed app run was performed on this
new branch. Private fixtures, raw logs and generated HTML are not published.

```bash
swift test -c release
swift build -c release
xcrun swift-format lint --strict --recursive \
  src/quantem/gpu/swift/Sources src/quantem/gpu/swift/Tests

# On an admitted device with the complete sealed archive:
QUANTEM_TANS_FULL_SERIES_FIXTURE=/path/to/encoded-series \
  swift test -c release --filter MetalTANSFullSeriesTests
QUANTEM_TANS_DETECTOR_FIXTURE=/path/to/encoded-series \
  swift test -c release --filter MetalTANSInteractionTests
QUANTEM_TANS_SPI_FIXTURE=/path/to/encoded-series \
  swift test -c release --filter ExperimentalMetalEntropySeriesTests
```

Fixtures and frozen products are not distributed here. Without them, opt-in
tests skip; a skip is not parity or capacity evidence. The full-series test
checks all-66 loading/point queries and one independent full-count hash. The SPI
test compares index/delta/edge/subset/reordered results against full-source
queries and checks failure/release behavior. It is a package test, not a
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
- [ ] Mutable-output ownership isolation, thread-safe concurrent lifecycle,
  bounded cancellation latency and asynchronous publication integration.
- [ ] General large BF/ABF/ADF gestures at 10-20 all-66 scientific FPS,
  with native input-to-publication and complete presentation evidence.
- [ ] Stable API promotion and app/package release qualification.

Optimization work is tracked in
[QuantEM.GPU issue #8](https://github.com/bobleesj/quantem.gpu/issues/8).
The branch makes the native experiment reusable and reviewable; it does not
close these gates.
