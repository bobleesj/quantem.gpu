# Native 4D-STEM load and cache contract

The repository-root Swift package exposes reusable HDF5 discovery, exact load
geometry, Metal kernels, and resident-cache integrity for native macOS clients.
It does not own SwiftUI, folder-selection policy, memory-pressure UI, cache
admission or eviction, or application state.

## Products and dependencies

Native clients import two products for local 4D-STEM loading:

| Product | Owns | Dependencies |
|---|---|---|
| `Native4DSTEMIO` | HDF5 and EMD catalog discovery, QH5 indexing, source identity, value audits, resident-cache IO | `CNativeHDF5`, vendored `CHDF5.xcframework`, zlib, Foundation, CryptoKit |
| `Metal4DSTEMKernels` | Exact load geometry, streaming geometry, QH5 decode, detector binning, BF/ABF/ADF, CoM, DPC/iDPC primitives | Metal, Foundation |

Neither product imports SwiftUI, AppKit, UIKit, or Python.

## Catalog and load geometry

Prepare a Python-free catalog and construct an explicit load plan:

```swift
import Metal4DSTEMKernels
import Native4DSTEMIO

let catalog = try Native4DSTEMCatalogBuilder(cacheDirectory: indexDirectory)
  .prepare(input: selectedFolder, mode: .indexed)
let dataset = catalog.datasets[0]
let region = try Metal4DSTEMScanRegion.full(
  sourceRows: dataset.scanRows,
  sourceColumns: dataset.scanCols
)
let plan = try Metal4DSTEMLoadPlan(
  sourceScanRows: dataset.scanRows,
  sourceScanColumns: dataset.scanCols,
  detectorRows: dataset.detectorRows,
  detectorColumns: dataset.detectorCols,
  sourceBytesPerValue: dataset.sourceBytes,
  scanRegion: region,
  scanBin: 1,
  detectorBin: 4
)
```

The client must record all of these fields in provenance:

- source and output scan rows and columns;
- source and output detector rows and columns;
- source and resident dtype;
- half-open scan region `[rowStart, rowStop) × [columnStart, columnStop)`;
- scan and detector bin factors;
- whether detector sums use a narrower exact integer representation;
- the memory budget, chosen plan, and reason for any automatic reduction.

Do not infer a crop or silently call detector-binned data native resolution.
The M2 Air policy validated for the retained BTO fixtures uses the full 512 by
512 scan, scan bin 1, and explicit exact detector sum bin 4 from 192 by 192 to
48 by 48.

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
earlier Live4DSTEM schema only so existing audited fixtures remain usable.

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

`Metal4DSTEMResidentCacheMetadata` records scientific meaning as well as file
integrity:

- dataset and ordered source identities;
- source identity SHA-256;
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

`write` validates shape, dtype, bin, crop, payload size, and bad-pixel
provenance before publishing the payload. It writes a temporary payload,
renames it, seals the metadata with SHA-256, and removes the payload if metadata
publication fails.

On reopen, call `readMetadata(from:)` and then
`validatePayload(at:metadata:verifySHA256:)`. The default SHA-256 verification
is the scientific integrity path. Passing `verifySHA256: false` verifies only
the sealed file identity and size and must be labeled as such by the client.
An incomplete or rejected cache falls back to the original indexed source; it
must never change scan coverage, binning, dtype, or metadata silently.

## Client ownership

Live4DSTEM owns:

- folder selection, latest-request-wins scheduling, cancellation, and UI;
- memory budget and reserve selection;
- the decision and visible reason for automatic detector binning;
- cache admission, eviction, disk reserve, and memory-pressure response;
- command-buffer orchestration, presentation, and first-draw measurement.

QuantEM.GPU owns the typed geometry, resource estimates, exact kernels, source
identity, cache format validation, and numerical reference tests. Clients must
consume these Swift products through one exact package revision and must not
copy the Metal or native HDF5 sources into the application.
