# Native 4D-STEM load and cache contract

The repository-root Swift package exposes reusable HDF5 discovery, exact load
geometry, Metal kernels, and resident-cache integrity for native macOS clients.
It does not own SwiftUI, folder-selection policy, memory-pressure UI, cache
admission or eviction, or application state.

## Products and dependencies

Native clients import two products for local 4D-STEM loading:

| Product | Owns | Dependencies |
|---|---|---|
| `Native4DSTEMIO` | HDF5 and EMD catalog discovery, QH5 indexing, validated bounded source windows, source identity, value audits, resident-cache and exact-summary IO | `CNativeHDF5`, vendored `CHDF5.xcframework`, zlib, Foundation, CryptoKit |
| `Metal4DSTEMKernels` | Exact load geometry, streaming geometry, typed exact binning, QH5 decode, BF/ABF/ADF, CoM, and DPC/iDPC primitives | Metal, Foundation, CryptoKit |

Neither product imports SwiftUI, AppKit, UIKit, or Python.

The native HDF5 bridge accepts unsigned 8-bit and unsigned 16-bit detector
sources. `Metal4DSTEMLoadPlan.sourceBytesPerValue` is therefore exactly 1 or 2.
The persistent resident cache currently stores `uint16` or `uint32`; audited
`uint8` is a compact decode/staging representation, not a silently relabeled
`uint8` resident cache.

## Catalog and load geometry

Prepare a Python-free catalog and construct an explicit load plan:

```swift
import Metal4DSTEMKernels
import Native4DSTEMIO

let catalog = try Native4DSTEMCatalogBuilder(cacheDirectory: indexDirectory)
  .prepare(input: selectedFolder, mode: .indexed)
let dataset = catalog.datasets[0]
let indexedSource = try Native4DSTEMIndexedSource.open(dataset: dataset)
let region = try Metal4DSTEMScanRegion.full(
  sourceRows: dataset.scanRows,
  sourceColumns: dataset.scanCols
)
let plan = try Metal4DSTEMLoadPlan(
  sourceScanRows: dataset.scanRows,
  sourceScanColumns: dataset.scanCols,
  detectorRows: dataset.detectorRows,
  detectorColumns: dataset.detectorCols,
  sourceBytesPerValue: indexedSource.sourceBytesPerValue,
  scanRegion: region,
  scanBin: 1,
  detectorBin: 4
)
```

The client must preserve all of these package-provided fields in provenance:

- source and output scan rows and columns;
- source and output detector rows and columns;
- source, staging, and output dtype;
- half-open scan region `[rowStart, rowStop) × [columnStart, columnStop)`;
- scan and detector bin factors;
- exact reduction semantics, source-audit identity, staging and output layout,
  maximum output count, and payload bytes.

The client additionally records its memory budget, chosen plan, and reason for
any automatic reduction. Those are application-policy fields, not defaults
selected by QuantEM.GPU.

Do not infer a crop or silently call detector-binned data native resolution.
QuantEM.GPU does not choose detector bin 1, 2, or 4 from a device name. For the
specific full-scan 512 by 512, detector 192 by 192 case, exact detector bin 2
produces a 512 by 512 by 96 by 96 packed-uint16 payload of 4,831,838,208 bytes
(4.5 GiB) when an identity-bound audit proves every four-pixel sum fits
`uint16`. That byte calculation is not a physical-device admission decision.

## Bounded native indexed windows

`Native4DSTEMIndexedSource` opens the prepared QH5 sidecars, validates them
against their exact canonical source paths, sizes, modification times, detector
geometry, dtype, block geometry, compressed ranges, and complete frame
coverage. It rejects stale, trailing, repeated, incomplete, or incompatible
indexes. Moving a source file makes its old path-bound index stale; regenerate
the index instead of weakening this check.

The public frame order is row-major:

\[
n = R_r N_{R_c} + R_c,
\]

where \((R_r, R_c)\) is `(scanRow, scanColumn)`. A caller can partition the
logical source without changing its scientific shape, dtype, binning, or crop:

```swift
let decodedBytesForFourScanRows =
  UInt64(4 * dataset.scanCols) * indexedSource.decodedBytesPerFrame
let windows = try indexedSource.windows(
  maximumDecodedBytes: decodedBytesForFourScanRows,
  alignToScanRows: true
)
```

Each `Native4DSTEMIndexedWindow` reports one half-open global frame range, its
decoded byte count, and the exact shard/chunk/index-word slices required to
decode it. For a `512 × 512 × 192 × 192 uint16` source, four-row windows are
288 MiB each, 128 windows cover all 262,144 scan positions, and
`logicalDecodedBytes` remains 19,327,352,832 bytes (18 GiB).

Opening and partitioning read only prepared index sidecars. They do not open or
map compressed HDF5 shards, decode frames, allocate a resident volume, execute
Metal, compute products, or choose a device budget. Consequently, index-open
latency is not a first-load or first-product benchmark. The consuming layer
supplies the transient byte ceiling and owns scheduling, cancellation, memory
admission, and cache lifecycle.

## Typed exact binning

Construct and validate the scientific contract before allocating a resident
payload or encoding a command:

```swift
let sourceAudit = try Metal4DSTEMExactSourceAudit(
  sourceIdentitySHA256: sourceIdentity,
  sourceDtype: .uint16,
  badPixelIndices: badPixels,
  maximumSourceCount: maximum,
  pixelsAbove255: pixelsAbove255
)
let exact = try Metal4DSTEMExactBinner.provenance(
  plan: plan,
  sourceAudit: sourceAudit,
  stagingDtype: .uint16,
  outputDtype: .uint16
)
```

The audit digest binds source identity, source dtype, sorted bad-pixel indices,
maximum source count, and the above-255 count. Detector bin 2 accepts a
`uint16` maximum of 16,383 and rejects 16,384 because four equal source counts
would sum to 65,536. Use `uint32` output when the proven bound does not fit
`uint16`; do not clip or downcast.

`Metal4DSTEMExactBinner.encodeBatch(...)` accepts a frame-major
`stagedSource`. It must contain only the selected scan columns, and the audited
bad pixels must already be zeroed in every frame. The method validates batch
coverage, offsets, buffer lengths, Metal's 32-bit geometry parameters, dtypes,
and output bounds before creating a command encoder. It writes either
detector-word-major `uint32` values or packed `uint16` low/high lanes, including
a zero high lane for an odd final detector pixel. It does not allocate buffers,
commit, synchronize, choose a memory budget, or select a bin factor.

Sampling propagation is deliberately narrower than a full calibration
transform:

```swift
let sampling = try exact.propagatingSampling(
  sourceScan: sourceScanSampling,
  sourceDetector: sourceDetectorSampling
)
```

Uniform complete bins scale row and column sampling by the corresponding bin
factor and report the first working-bin center in source-pixel coordinates.
Incomplete edge bins return no single uniform working sampling. Detector
center, affine calibration, masks, and radii require their own typed coordinate
transform and are not silently rewritten by this API.

## Streaming geometry

`Metal4DSTEMStreamingPlan` is deterministic when given a load plan, scratch
budget, depth, and staging dtype:

```swift
let depth = Metal4DSTEMStreamingPlan.recommendedDepth(
  physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory
)
let streaming = try Metal4DSTEMStreamingPlan(
  loadPlan: plan,
  scratchBudgetBytes: scratchBudget,
  preferredDepth: depth,
  stagingBytesPerValue: 1
)
```

The application supplies the memory budget and decides whether the plan is
admissible. `totalScratchBytes` is not a full-process peak estimate. The client
must reserve memory for the resident volume, maps, audits, products, FFT work,
cache IO, and native UI.

## Lossless compact staging

`Native4DSTEMValueRangeAudit` permits a uint16 source to use its low byte only
when all of the following match the current dataset:

```swift
let audit = try Native4DSTEMValueRangeAuditIO.read(from: auditURL)
let isLossless = audit.provesLosslessUInt8(
  sourceIdentitySHA256: exactSourceIdentitySHA256,
  sourceDtype: dataset.sourceDtype,
  badPixelIndices: dataset.badPixelIndices
)
```

The audit records the exact source identity, dtype, bad-pixel set, maximum, and
number of values above 255. A filename or shape match is insufficient. New
files use schema `quantem.gpu.value-range-audit/v1`; the reader accepts the
earlier client-specific schema only so existing audited fixtures remain usable.

The accepted Air fast path uses
`decodeU16AuditedLow8ScalarFunction` followed by
`binU16AuditedLow8ScalarU16WordMajorFunction`. The binning kernel writes exact
packed uint16 detector-word-major values and accumulates BF, ABF, ADF, and CoM
moments. The caller must retain the general uint16 path when the audit does not
prove compact staging is lossless.

The direct threadgroup decode/bin kernel remains diagnostic. Do not enable it
as the consumer default. The removed frame-owned binning experiment was never
dispatched and is not part of this contract.

## Resident cache

`Metal4DSTEMResidentCacheMetadata` format 2 records scientific meaning as well
as file integrity:

- dataset and ordered source identities;
- source identity SHA-256 and, whenever narrowing requires it, the complete
  sealed value-range audit plus its canonical SHA-256;
- source and output shapes and dtypes;
- half-open scan region, scan bin, and detector bin;
- bad-pixel indices, maximum count, and values above 255;
- payload bytes, payload identity, and payload SHA-256.

Write shared Metal storage without creating another multi-gigabyte copy:

```swift
let complete = try Metal4DSTEMResidentCacheIO.write(
  pointer: residentBuffer.contents(),
  length: residentBuffer.length,
  payloadURL: payloadURL,
  metadataURL: metadataURL,
  metadata: metadata
)
```

`write` validates shape, dtype, exact output bound, bin, crop, payload size,
bad-pixel provenance, and the sealed audit before publishing the payload. It
writes a temporary payload, renames it, seals the metadata with SHA-256, and
removes the payload if metadata publication fails. Format 1 metadata is
invalidated rather than interpreted under the stronger format 2 contract.

On reopen, call `readMetadata(from:)` and then
`validatePayload(at:metadata:verifySHA256:)`. The default SHA-256 verification
is the scientific integrity path. Passing `verifySHA256: false` verifies only
the sealed file identity and size and must be labeled as such by the client.
An incomplete or rejected cache falls back to the original indexed source; it
must never change scan coverage, binning, dtype, or metadata silently.

## Exact resident summary

After a resident payload has been sealed, a client may persist exact compact
products and sufficient statistics with
`Metal4DSTEMResidentSummaryIO.write(...)`:

```swift
let summaryMetadata = try Metal4DSTEMResidentSummaryIO.write(
  to: summaryDirectory,
  residentMetadata: residentMetadata,
  detectorBands: detectorBands,
  selectedScanRow: selectedRow,
  selectedScanColumn: selectedColumn,
  artifacts: exactArtifacts
)
```

`exactArtifacts` must contain every `Metal4DSTEMResidentSummaryRole`: BF, ABF,
ADF, total intensity, detector-row moment, detector-column moment, and selected
diffraction. Virtual images and selected diffraction are little-endian
`uint32`; total and coordinate moments are little-endian `uint64` so CoM
derivation cannot overflow at the retained full-scan scale.

Reopen against the same sealed resident metadata and detector-band definition:

```swift
let summary = try Metal4DSTEMResidentSummaryIO.read(
  from: summaryDirectory,
  residentMetadata: residentMetadata,
  detectorBands: detectorBands
)
```

The reader validates the `quantem.gpu.resident-summary/v1` schema, source and
resident identities, output shape/dtype, half-open scan region, scan and
detector bins, count audit, detector bands, selected scan coordinate, artifact
shape/dtype/size, and every artifact SHA-256. A mismatch fails closed; it never
returns a partly trusted product set.

This is a prepared-product cache. Reading it does not open, read, or decompress
the original HDF5 source and must not be reported as a source-load benchmark.
The application owns the decision to create, retain, evict, or present it.

## Package benchmark boundary

`metal-4dstem-binning-benchmark` measures only the synchronized exact-binning
kernel after a deterministic source buffer is already staged in unified
memory. Its JSON reports source and working shapes, all three dtypes, bin
factors, staged and output bytes, device limits, p50/p95/max wall and GPU time,
and output SHA-256. It explicitly excludes HDF5 discovery, storage reads,
decompression, cache creation or reopen, scientific products, and UI. Never
publish its kernel time as a first-load or application wall time.

## Client ownership

The consuming application owns:

- folder selection, latest-request-wins scheduling, cancellation, and UI;
- memory budget and reserve selection;
- the decision and visible reason for automatic detector binning;
- cache admission, eviction, disk reserve, and memory-pressure response;
- command-buffer orchestration, presentation, and first-draw measurement.

QuantEM.GPU owns the typed geometry, resource estimates, exact kernels, source
identity, cache format validation, and numerical reference tests. Clients must
consume these Swift products through one exact package revision and must not
copy the Metal or native HDF5 sources into the application.
