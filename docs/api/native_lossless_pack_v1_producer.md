# Native Lossless Pack Format v1 producer

`NativeLosslessPackV1Producer` converts one supported original `uint16` HDF5
family into an exact, portable Lossless Pack Format v1 cache without allocating
the dense 4D tensor.
It is a reusable library boundary for native applications. It does not choose a
device memory policy, manage a user interface, evict caches, or decide when a
cache should be created.

The public lifecycle is deliberately three steps:

```text
original HDF5 family
  -> inspect and authenticate
  -> plan memory, disk, shards, and execution backend
  -> produce temporary container and receipt
  -> synchronize receipt, then publish the cache as the commit marker
```

This revision implements the bounded CPU reference producer. A request for
Metal, CUDA, Vulkan, or WebGPU fails before output allocation; it never falls
back silently. The execution backend and `gpuAccelerated` flag are recorded in
the receipt. The CPU implementation establishes the exact producer contract
that future accelerator implementations must match.

## Scientific contract

The logical source order is

```text
(scan_row, scan_column, detector_row, detector_column)
```

The current producer accepts an indexed HDF5 family with one full detector
frame per bitshuffle/LZ4 chunk and exact `uint16` source counts. It preserves:

- the full scan and detector shape;
- scan bin 1, detector bin 1, and crop `none`;
- source-family SHA-256 identity and ordered member hashes;
- row and column scan and detector calibration;
- the exact ordered detector-exclusion identity; and
- the complete source `uint32` detector-mask identity, distinct from the sparse
  ordered exclusion list; and
- independent SHA-256 identities for the raw logical `uint16` source and the
  mask-applied Lossless Pack Format v1 exact-`uint8` working array.

The current producer emits the exact-`uint8`/bitpacked profile. It admits a
nonexcluded count only when it is in the exact range 0 through 255. A larger
value fails with guidance to use the Lossless Pack Format v1 exact-`uint16`/LZ4
profile.
An excluded detector stream may be omitted only after the producer proves that
its raw `uint16` value is constant across every scan. The receipt retains that
constant so the raw source can be reconstructed exactly. Mask-applied
scientific products continue to see zero at excluded pixels.

## Step 1: inspect

Inspection discovers one unambiguous dataset, builds or validates its small QH5
indexes, authenticates every source member, and records scientific metadata. It
does not allocate the logical volume.

```swift
import Native4DSTEMIO

let producer = NativeLosslessPackV1Producer(cacheDirectory: indexDirectory)
let inspection = try producer.inspect(input: selectedHDF5OrFolder)

print(inspection.sourceShape)
print(inspection.sourceDtype)
print(inspection.sourceIdentitySHA256)
```

For a catalog that contains multiple datasets, the caller selects one entry and
uses `inspect(dataset:)`. This keeps selection policy outside QuantEM.GPU.

## Step 2: plan

Planning happens before decoded, packed, or output payload allocation. The
caller supplies the transient-memory ceiling. QuantEM.GPU computes the shard
layout, worst-case packed residency, peak producer transient bytes, container
reserve, receipt reserve, and combined output-disk requirement.

The transient bound includes the prepared QH5 index files, the largest
compressed block, decoded-block scratch, one decoded scan tile, one mask-applied
working tile, shard widths and offsets, and the active packed shard. It is a
buffer-accounting bound, not process RSS or operating-system compressed memory.

```swift
let plan = try producer.plan(
  inspection: inspection,
  destination: preparedCacheURL,
  maximumTransientBytes: transientBudgetBytes,
  executionBackend: .cpuReference
)

print(plan.predictedPackedResidentMaximumBytes)
print(plan.predictedPeakTransientBytes)
print(plan.predictedOutputDiskMaximumBytes)
print(plan.encodingProfile)
```

The disk check uses the nearest existing destination ancestor, so a caller may
plan a new cache directory without creating it first. An explicit
`availableOutputDiskBytes` is available for deterministic tests or for a
consumer that has already measured its own storage boundary.

The portable layout fixes `scanTile` to 32. The planner selects a complete
tile-aligned shard divisor no larger than 4096 scan positions. For the intended
192 × 192 detector geometry, planning gives:

| Scan shape | Scan positions per shard | Shards | Claim |
|---|---:|---:|---|
| 512 × 512 | 4096 | 64 | layout arithmetic covered by unit tests |
| 1024 × 1024 | 4096 | 256 | layout arithmetic covered by unit tests |

These rows prove shape-generic planning only. They are not physical full-source
production, load-time, resident-memory, or application acceptance results.

## Step 3: produce

Production rereads and verifies the source identity, decodes bounded 32-scan
tiles, proves admissible widths and excluded streams, packs one shard at a time,
and writes contiguous lossless-pack payload and header datasets through the native HDF5
bridge.

```swift
let receipt = try producer.produce(
  plan,
  shouldCancel: { taskIsCancelled }
)

print(receipt.destination)
print(receipt.outputSHA256)
print(receipt.observedPackedResidentBytes)
print(receipt.accountedPeakTransientBytes)
```

Cancellation is checked before production and between bounded units. Failure,
cancellation, source mutation, corruption, an unsupported value, or a resource
bound violation leaves no final cache or final receipt. The temporary files are
removed. On success, the synchronized JSON receipt is staged first and the
cache is linked last as the commit marker, without overwriting an existing
file. A consumer treats the cache path as published only when its matching
receipt is already present and valid.

The returned cache and receipt are complete caller-owned files. The producer
does not expose borrowed buffers, retain a GPU allocation, or keep a hidden
runtime resource alive after return.

## Receipt and provenance

`NativeLosslessPackV1ProductionReceipt.currentSchema` is
`quantem.gpu.lossless-pack-production/v1`. The public format schema is
`quantem.gpu.lossless-pack-format/v1`. The receipt records:

- source members, paths, byte counts, hashes, and aggregate source identity;
- source and working logical hashes, shapes, dtypes, and encoding profile;
- prepared-index bytes and the largest indexed compressed block;
- scan tile, scans per shard, shard count, scan bin, detector bin, and crop;
- excluded pixel indices, their ordered identity, and proven raw values;
- detector-mask SHA-256 and whether it came from source metadata or an explicit
  all-admitted mask;
- calibration fields and their canonical identity;
- requested execution backend and whether it was GPU accelerated;
- predicted packed residency, transient memory, output file, receipt file, and
  combined output-disk bounds;
- observed packed residency, buffer-accounted peak transient bytes, output
  bytes, and per-shard payload/header bytes and hashes; and
- source authentication, decode/packing, container writing, stability/hash, and
  pre-publication wall timing.

Timing fields describe cache production. They are not original-source app load,
first usable image, exact product completion, cache reopen, display publication,
or interaction latency.

## Native HDF5 bridge

`CNativeHDF5` owns the low-level, contiguous HDF5 writer used by the Swift
lifecycle:

```c
qh5_lossless_pack_v1_writer_open(...);
qh5_lossless_pack_v1_writer_append_shard(...);
qh5_lossless_pack_v1_writer_close(...);
qh5_lossless_pack_v1_writer_abort(...);
```

Shards are appended once in ordinal order. `close` transfers no resources back
to the caller; `abort` closes the open HDF5 objects and removes only the
producer's temporary path. Application code should use the checked Swift
inspect-plan-produce surface rather than calling this storage bridge directly.

## Qualification status

The repository test fixture is a real bitshuffle/LZ4 HDF5 file with calibrated
row and column sampling and a constant excluded detector stream. Tests cover
independent Swift decoding of every prepared value, exact raw and working
hashes, Python reference-parser interoperability, cancellation, source mutation,
memory and disk admission, explicit backend rejection, and 512/1024 planning.

Full 512 × 512 or 1024 × 1024 physical production and application acceptance
remain separate gates. Do not infer those results from the representative
fixture or the planning tests.

See [Lossless Pack Format v1](compact_4dstem_h5.md) for the binary format and
[Native 4D-STEM load and cache contract](native_4dstem_io.md) for
consumer-side loading and exact products.
