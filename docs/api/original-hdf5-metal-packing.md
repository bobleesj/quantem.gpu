# Original Arina HDF5 to exact Metal residency

Load original data through the existing compact loader:

```swift
let catalog = try Native4DSTEMCatalogBuilder(cacheDirectory: indexDirectory)
  .prepare(input: masterOrRelatedHDF5)
let indexed = try Native4DSTEMIndexedSource.open(dataset: catalog.datasets[0])
let resident = try MetalCompactH5Loader.load(
  source: indexed, device: device, maximumAdditionalBytes: availableBytes)
```

This reads and decompresses the original acquisition into private lossless
packed Metal buffers. It does not write a packed file, load a saved virtual
image, or allocate the entire dense 4D tensor. The returned resident supports
the same detector, diffraction-pattern, and DPC APIs as a prepared-file resident.
Release it with `releaseResidentStorage()` when the owning session ends.

`availableBytes` is the caller's remaining allocation budget after accounting
for other residents and application/system headroom. Loading checks bounded
staging first, then measured packed growth; it fails instead of binning data
when the budget is insufficient. This is not a universal compression ratio.

For an explicit persistent packed file, use
`prepare(source:destinationURL:device:)`, then `load(sourceURL:device:)`.
The `.qgix` output is a packed container, not an editable HDF5 copy. That separate
workflow computes logical hashes and performs file writes; its preparation
time must not be described as the direct loader's latency.

Fast decode/packing defaults are automatic; application callers do not need
experimental environment variables. The current resident receipt uses
`representation: packed` and records its dtype and storage schema separately.
Old receipt names are not alternate API spellings.

For repeat visits, pass `packingPlanURL` in an application-owned user cache.
This stores source-bound packing layout metadata, not a 4D count copy. Optional
`preparedDPC` reuses exact, source-bound 2D sums. Both are validated against the
current source. A missing, stale, corrupt, or over-budget layout falls back to
fresh packing; deleting it is safe. Neither option replaces rereading and
decompressing the original counts. The backend does not choose a global cache
directory, update the original HDF5, or upload data.

Report indexing, source reads, decode, packing/products, and first visible frame
separately. An index-cache hit is not a resident-cache hit: the original counts
are still reread and decoded on each direct load. No subsecond guarantee is made.

## Scientific and ownership guarantees

- Original Arina bitshuffle/LZ4 input is read in bounded native-precision
  windows. No full dense 4D allocation, crop, bin, or lossy conversion is used.
- Every source count is preserved, including source-marked bad/hot pixels.
  The source mask is used only to estimate an initial detector position; it
  does not remove counts. This estimate is not angular calibration.
- The entire acquisition is range-audited. Working uint8 is selected only
  when every count is <=255; otherwise the resident retains uint16 counts.
- Every count is checked against its packed representation on Metal before
  publication, either by exact scalar reconstruction or source bit-plane
  comparison. The latter does not need to materialize a dense count tensor.
  Independent NumPy/HDF5 tests check decoder and downstream scientific parity.
  Direct loads leave raw/working logical SHA-256 fields nil because they do not
  hash a dense tensor. File preparation records those hashes. GPU roundtrip
  checking is not the same as an independent source digest.
- The caller owns the output location and cache eviction. Existing outputs
  are immutable; cancellation or failure removes only this call's partial file.
- Preparation is synchronous. UI callers must use a background worker and
  forward the optional progress callback without blocking the main thread.
- Direct loading uses a bounded decode window and retains each completed packed
  shard. File preparation uses two slots to overlap GPU work with ordered
  logical hashing and file writes. A slot cannot be reused until the writer
  releases it. Cancellation drains writes before removing an incomplete output.
- Source reads cover the compressed blocks needed by each window, rather than
  rereading every containing source chunk. Reported CPU/GPU phase durations can
  overlap; their sum is not end-to-end latency.

Current restrictions: uint8/uint16 indexed Arina inputs, scan count divisible
by 32, and supported decoder block geometry. Unsupported inputs fail explicitly;
the implementation must not silently bin, crop, or zero-fill them. Only the
Swift/Metal producer is implemented here; this is not a CUDA/WebGPU
qualification claim.

## Regression gate

On macOS with Command Line Tools, Metal, and the project's Python test
dependencies (including h5py/hdf5plugin):

```sh
PYTHONPATH=src python -m pytest -q \
  tests/hardware/metal/test_original_packing.py tests/contracts/io/test_compact_h5.py
```

The original-file tests compile **release-optimized Swift**, generate synthetic
native 192×192 detector acquisitions with two external shards, and compare all
counts and detector/DPC products against an independent NumPy oracle. Cases
cover genuine uint8, narrowable uint16, high-count uint16 including 65,535,
and a late high count, plus all-zero, saturated-uint16 and every-bit-width cases.
The saturated case verifies DPC products above the uint32 range.
A smaller 64×64 detector fixture spans three decode
windows to test slot reuse and a high count discovered after the first window.
Both direct and file paths run these oracles. Tests also verify the reference
reader, cancellation, budget rejection, absence of a direct-path output cache,
immutable output, and unchanged original HDF5 files. Native Open Folder,
multi-acquisition switching, real display latency, and bounded memory remain
separate physical UI acceptance gates.

## Repeatable loading benchmark

```sh
swift run -c release metal-original-hdf5-benchmark \
  "$QGPU_INPUT" "$QGPU_INDEX_DIR" --repeats 7 \
  --plan-directory "$QGPU_PLAN_DIR" --reuse-products \
  --oracle "$QGPU_ORACLE_JSON"
```

Input can be an original master, related HDF5, or acquisition folder. The runner
cycles through the discovered acquisitions, keeps one complete 4D resident,
and releases it before each next selection. A return visit rereads and
reconstructs the source. The optional oracle maps each source identity to an
independently computed full-count SHA-256 in scan-major uint32 little-endian
order. With `--oracle`, every reconstructed resident is fully audited on every
visit, outside its loading timer; it must not inherit a previous visit's pass.
Without that oracle, selected diffraction samples are only repeatability checks,
not an independent full-volume validation. Omit the two cache options to measure
fresh packing and products on every visit.

`--detector-trials N` additionally measures fixed BF, ABF, and ADF transitions.
`--series` retains the discovered acquisitions together and exercises concurrent
detector submissions; it is a different memory and timing workflow from the
single-resident loop. Complete map hashes must be compared with the independent
reference, and these kernel timings must not be labeled native presented FPS.

JSON lines report catalog time separately from indexed-open-to-resident-ready
time, include the first load, and expose GPU decode/fused intervals without
counting a fused interval twice. Allocation-after-load and a conservative
planned bound are not sampled peak memory. OS source pages are uncontrolled:
these are not certified cold-I/O measurements. No FPS or UI-visible latency is
inferred. Retain executable/resource hashes, source revision, raw JSON lines,
and separate memory telemetry with each benchmark registration.
