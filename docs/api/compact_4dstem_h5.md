# Lossless Pack Format v1

QuantEM.GPU Lossless Pack Format v1 stores an exact mask-applied integer 4D-STEM
working array without constructing its dense logical tensor. The stable logical
order is

```text
(scan_row, scan_column, detector_row, detector_column)
```

All coordinates are zero-based `(row, column)`, scan positions are C-order, and
detector pixels are C-order. Binning and crop are fixed to `1` and `none`. A
backend must reject conflicting metadata, unsupported widths, truncated
ranges, bad integrity hashes, or an output type that cannot represent the
requested exact reduction.

The public format has two exact encoding profiles. Their legacy binary revision
numbers select the decoder and remain in existing files for compatibility; they
are not separate public format generations.

| Public format | Encoding profile | Legacy binary revision | Scan tile | Header representation |
|---|---|---:|---:|---|
| Lossless Pack Format v1 | exact `uint16`/LZ4 | 1 | 128 | one u8 width per pixel/tile |
| Lossless Pack Format v1 | exact `uint8`/bitpacked | 3 | 32 | per-pixel base, 32-tile checkpoints, and packed width nibbles |

The legacy index magic selects the encoding profile. A reader must not use the
raw-LZ4 interpretation for a direct-bitpacked file or transfer performance
evidence between the two profiles.

Authenticated detector exclusions always read as zero through the scientific
working-array API. The exact-`uint16`/LZ4 profile can retain those raw streams in
its payload and declare `masked_detector_payload_policy` as
`retained_exactly_in_payload`. The exact-`uint8`/bitpacked profile can instead
reconstruct excluded raw streams when its manifest carries producer-proven
constant uint16 values in
`masked_detector_raw_values` and authenticates the exact excluded-index sequence
with `masked_detector_pixels_sha256`. A result must say whether it was compared
with the raw source or with the mask-applied working array; raw sentinels never
enter masked scientific products.

## Container user block

The HDF5 file begins with this little-endian prelude. Integer offsets are file
offsets, not HDF5 object offsets.

| Field | Type | Required value or meaning |
|---|---:|---|
| magic | 8 bytes | `QGPUH5\0\x01` |
| JSON bytes | u32 | UTF-8 manifest byte count |
| JSON CRC-32 | u32 | IEEE CRC-32 of the exact manifest bytes |
| binary offset | u32 | start of the binary index |
| binary bytes | u32 | complete binary-index byte count |

The JSON region ends no later than the binary offset. Both ranges must be
inside the file. A reader validates the CRC before interpreting the manifest.

The exact-`uint16`/LZ4 binary index begins with:

| Field | Type | Required value or meaning |
|---|---:|---|
| legacy magic | 8 bytes | `QGIX\0\0\0\x01` |
| shard count | u32 | positive |
| payload chunk bytes | u32 | exactly 128 for the raw-LZ4 profile |
| scan rows, scan columns | 2 x u32 | positive logical scan shape |
| detector rows, detector columns | 2 x u32 | positive logical detector shape |
| scans per shard | u32 | positive; the final 128-scan tile may be partial |
| excluded count | u32 | no greater than detector pixel count |
| excluded pixels | repeated u32 | unique row-major detector indices |
| source identity | 32 bytes | binary SHA-256 identity |
| shard records | repeated 96 bytes | exactly `shard count` records |

The full scan plane must satisfy

```text
scan_rows * scan_columns == shard_count * scans_per_shard
```

No trailing binary-index bytes are allowed. Payload, length, and width ranges
must be disjoint and inside the file.

Each 96-byte shard record contains seven u64 values followed by two u32 values
and a 32-byte digest:

```text
payload_offset, payload_bytes,
lengths_offset, lengths_bytes,
widths_offset, widths_bytes,
decoded_bytes,
descriptor_count, chunk_count,
decoded_sha256
```

`decoded_sha256` authenticates the exact bit-packed decoded payload before a
backend publishes the shard as resident.

### Exact `uint8`/bitpacked binary index

The direct-bitpacked profile uses legacy magic `QGIX\0\0\0\x03` and replaces the
raw-LZ4 header's payload
chunk size with three u32 values: reserved zero, `scan_tile=32`, and
`header_encoding=1`. Every shard must contain complete 32-scan tiles.

The 96-byte shard record is reused with strict direct-bitpacked meanings:

- `payload_offset`, `payload_bytes`: directly addressable little-endian u32
  payload; `decoded_bytes` must equal `payload_bytes`;
- `lengths_offset`, `lengths_bytes`, `chunk_count`: all zero because there is no
  raw-LZ4 envelope;
- `widths_offset`, `widths_bytes`: compact u32 headers rather than raw-LZ4 u8
  widths;
- `descriptor_count`: compact header word count; and
- `decoded_sha256`: SHA-256 of the exact direct payload bytes.

For `T = scans_per_shard / 32`, each detector pixel stores
`ceil(T / 32)` checkpoint words followed by `ceil(T / 8)` words holding eight
four-bit widths each. Header word zero is the detector pixel's payload base.
Subsequent checkpoint words equal the cumulative widths at tiles 32, 64, and
so on. Widths are 0 through 8; unused tail nibbles are zero. The final pixel's
base plus cumulative width must end exactly at `payload_bytes / 4`. Every
excluded detector pixel has all-zero widths and consumes no payload.

The binary index authenticates each direct-bitpacked payload but does not hash
the compact headers. Qualification therefore also requires an externally frozen
whole-file SHA-256, or a future schema revision that embeds header digests. A
filesystem path or adjacent sidecar by itself is not an integrity identity.

## Required exact-`uint16`/LZ4 JSON agreement

The exact-`uint16`/LZ4 profile retains the legacy manifest schema
`quantem.gpu.packed-detector-h5/v1`. These fields must
agree exactly with the binary index:

- `source_shape`
- `source_identity_sha256`
- `shard_count`
- `scans_per_shard`
- `payload_chunk_bytes`
- `masked_detector_pixels`

The source dtype is `uint16`, `scan_bin` and `detector_bin` are 1, `crop` is
null, and `status` is `complete`. An exact-`uint16`/LZ4 reader accepts
`working_dtype` equal to
`uint8` or `uint16`, but it derives values from descriptor widths rather than
casting to that label.

Early raw-LZ4 files can declare `working_dtype: uint8` while retaining 16-bit width
metadata for excluded raw-sentinel columns. Those columns have authenticated
zero payloads. Compatibility is limited to widths above eight whose detector
pixel is in the authenticated exclusion list. A width above eight for any
nonexcluded pixel conflicts with the legacy manifest and fails closed.

The general exact-uint16 producer declares `working_dtype: uint16` and retains
every source sample, including samples at excluded detector pixels, in its
packed payload. It declares
`masked_detector_payload_policy: retained_exactly_in_payload`, which permits
the raw reconstruction API to read those retained samples. The mask-applied API
still returns zero for excluded detector pixels, so detector products never
consume excluded values. An exact-`uint16`/LZ4 source with exclusions but
without this explicit
policy is not admissible as a portable raw reconstruction source.

### Optional source-bound detector calibration

A prepared source may carry one reusable detector calibration in the JSON
manifest. Readers validate it before exposing it to a product UI:

```json
{
  "detector_calibration": {
    "schema": "quantem.gpu.detector-calibration/v1",
    "source_identity_sha256": "<same identity as the compact source>",
    "detector_center_px": [97.26, 95.67],
    "bright_field_radius_px": 41.0,
    "dpc_rotation_degrees": 176.98,
    "dpc_component_order_exchanged": false,
    "method": "mean-diffraction-half-maximum"
  }
}
```

Detector coordinates are always `[row, column]`. Center coordinates and the
bright-field radius must be finite and in range. The DPC rotation and component
order are optional, but must either both be present or both be absent. Binding
the calibration to `source_identity_sha256` prevents a calibration copied from
another file from being silently accepted. Existing exact-`uint16`/LZ4 sources
without this
optional object remain valid and can be calibrated once after residency.

### Optional prepared reopen integrity

A prepared copy may add `encoded_envelope_sha256` to every JSON shard record.
The digest covers the one continuous byte range containing that shard's
compressed payload, compressed-length table, and descriptor-width table. The
binary index and scientific payload remain unchanged. Preparation must write a
new file; it must never replace the acquired or initially encoded source.

`prepare_compact_h5_metadata_copy` creates this source-preserving copy and can
also attach the source-bound detector calibration above. The function validates
the original index, hashes the exact stored envelopes, rewrites only the
reserved HDF5 user block, rereads the result, and publishes the destination
atomically. For example:

```python
from quantem.gpu.io._compact_h5 import prepare_compact_h5_metadata_copy

prepare_compact_h5_metadata_copy(
    "scan-gpu-native.h5",
    "scan-gpu-native-prepared.h5",
    detector_calibration={
        "detector_center_px": [97.26, 95.67],
        "bright_field_radius_px": 41.0,
        "dpc_rotation_degrees": 176.98,
        "dpc_component_order_exchanged": False,
        "method": "mean-diffraction-half-maximum",
    },
)
```

The first qualification load uses every binary `decoded_sha256`: decode each
shard on the accelerator, read the decoded bytes back, and verify the exact
lossless resident representation. A prepared reopen can instead authenticate
the already-read encoded envelope and omit that multi-gigabyte GPU readback.
This is a speed optimization, not a weaker source identity: any payload,
length-table, or width-table mutation fails its prepared hash before the source
is published. A loader must fall back to decoded SHA-256 when any shard lacks
the optional hash, and callers may force decoded verification for an audit.

Benchmarks must label these modes separately as `decoded-sha256` and
`authenticated-encoded-envelope`; their timings are not interchangeable.

## Required exact-`uint8`/bitpacked JSON agreement

The exact-`uint8`/bitpacked profile retains the legacy manifest schema
`quantem.gpu.packed-detector-h5/v3`. It must bind the
binary index to the complete source and working representation with:

- `source_identity_sha256`, `source_raw_logical_sha256`, `source_shape`, and
  `source_dtype: uint16`;
- `working_dtype: uint8`, `prepared_uint8_sha256`, and the exact
  `working_value_definition`;
- `detector_mask_sha256` and the ordered row-major
  `masked_detector_pixels` list;
- `masked_detector_pixels_sha256`, the SHA-256 of the exact ordered
  little-endian u32 `masked_detector_pixels` sequence, whenever any detector
  stream is omitted;
- `masked_detector_raw_values`, an exact uint16 list aligned one-to-one with
  that pixel list, whenever any detector stream is omitted;
- `scan_bin: 1`, `detector_bin: 1`, `crop: null`, `scan_tile: 32`, and exact
  `shard_count`; and
- `payload_codec: direct-bitpacked-u32` with `status: complete`.

Legacy files that omit these provenance fields are not portable
exact-`uint8`/bitpacked acceptance fixtures even when an adjacent audit can fill
the gaps. Produce a new immutable
enriched and sealed copy or rebuild the artifact; do not weaken readers based
on local path conventions.

`detector_mask_sha256` preserves a distinct provenance identity: it hashes the
complete little-endian uint32 calibration/source mask, including its original
mask values and zero entries. It must not be reinterpreted as the digest of the
sparse index list. `masked_detector_pixels_sha256` binds that sparse list to the
binary index and equals
`sha256(pack_little_endian_u32(masked_detector_pixels))`.

The exact-`uint8`/bitpacked working representation intentionally returns zero
at excluded pixels.
`masked_detector_raw_values` is ordered by the same strictly increasing
row-major pixel indices as `masked_detector_pixels`; each item is an integer in
the exact uint16 range 0 through 65535. Before publishing the file, the producer
must independently prove that every omitted sample across every scan and shard
equals its recorded value. The field records the result of that proof; it is not
a license for a reader to infer a sentinel from width or mask metadata.

The native [Lossless Pack Format v1 producer](native_lossless_pack_v1_producer.md) implements the
source-authenticated inspect-plan-produce lifecycle, bounded shard construction,
atomic publication, and versioned receipt for this contract. Its current CPU
reference backend is explicit in provenance and is not presented as GPU
compression.

The Python reference exposes `value()` and `selected_diffraction()` as
mask-applied working reads. `raw_value()` and `raw_diffraction()` restore the
producer-proven constants only after `raw_reconstruction_available` is true.
Scientific detector sums continue to exclude those pixels.

Pre-release direct-bitpacked files with nonempty `masked_detector_pixels` but
missing either
`masked_detector_pixels_sha256` or `masked_detector_raw_values` remain readable
only as explicitly nonportable mask-applied candidates.
`require_raw_reconstruction()` and every raw read fail closed for them. They
cannot qualify as portable raw-lossless sources, even if an adjacent audit
happens to contain the missing values. Produce a new immutable enriched and
whole-file-sealed copy; do not mutate the earlier file.

`prepare_compact_h5_metadata_copy()` creates that distinct portable copy only
when the caller supplies both the immutable input's expected whole-file SHA-256
and producer-proven `masked_detector_raw_values`. The writer verifies the input
seal, computes `masked_detector_pixels_sha256` from the binary-index sequence,
writes through a temporary path, closes it, reparses it, requires raw
reconstruction, and only then atomically publishes the destination. The new
destination needs its own whole-file SHA-256 before qualification.

## Exact-`uint16`/LZ4 descriptor and payload layout

One descriptor covers 128 consecutive scans for one detector pixel. Descriptor
order is detector-pixel major, then scan-tile major:

```text
descriptor_index = detector_pixel * tile_count + scan / 128
tile_count = ceil(scans_per_shard / 128)
```

The file stores one unsigned width byte per descriptor. The resident descriptor
is reconstructed as one u32:

```text
descriptor = (payload_word_offset << 5) | width
```

Widths are 0 through 16. Offset uses the high 27 bits. A tile consumes exactly
`4 * width` u32 words because it contains `128 * width` bits. Consecutive
offsets must meet exactly, and the last descriptor must end at
`decoded_bytes / 4`. A width of zero represents 128 zeros without payload.

For local scan `s`, detector pixel `p`, and descriptor width `w`:

```text
bit = (s % 128) * w
word = payload_word_offset + bit / 32
shift = bit % 32
value = low w bits spanning word and word + 1 when required
```

Bits are least-significant-bit first inside little-endian u32 words. There is no
implicit saturation, low-byte cast, transpose, crop, or bin.

The decoded bitstream is split into independent 128-byte raw-LZ4 blocks. One
stored u8 length per block encodes `compressed_bytes - 1`. Lengths must cover
the compressed payload exactly. Chunk count is `ceil(decoded_bytes / 128)`;
only the last decoded chunk can be shorter, and its size remains u32-aligned.

## Mask and exact interaction semantics

The exclusion list is separate from user detector geometry. Every selected
diffraction pattern returns zero at an excluded pixel. A virtual detector first
normalizes its row-major zero-or-one mask by clearing excluded pixels.

Virtual-detector sums use exact u32 output only when the maximum representable
sum implied by the selected per-pixel widths fits u32. Otherwise the current
adapter rejects the request and asks for a narrower mask or a future u64 path.
After a full rebase, detector movement can use signed membership deltas. The
next scan map is published only after the whole GPU command succeeds.

FFT work is independent of detector reduction. An FFT-off interaction must
report zero FFT dispatches. A backend may not recompute FFTs during detector
movement and still label that measurement FFT-off.

## Bounded resident lifecycle

A conforming exact-`uint16`/LZ4 loader processes one shard at a time:

1. validate both metadata copies and all file ranges;
2. read one compressed payload plus its length and width metadata;
3. reconstruct and validate exact descriptors;
4. decode one shard into bounded staging;
5. verify the shard's decoded SHA-256, or on a prepared reopen authenticate its
   encoded envelope while the source bytes are resident;
6. copy the decoded payload and descriptors to backend-private storage;
7. release transient shard storage; and
8. publish the complete source only after every shard succeeds.

For a failed or cancelled load, no partially resident source is returned. The
19.3 GB logical uint16 cube for a `512 x 512 x 192 x 192` source is never a
temporary or resident allocation in this path.

A direct-bitpacked loader instead validates the compact headers, reads/authenticates the
direct payload and its headers, and copies those compact buffers to
backend-private storage. It does not run LZ4 or expand descriptors for all
pixel/tile pairs. Publication still occurs only after every shard and the
whole-file header integrity identity pass.

Benchmarks report metadata, whole-file integrity, source read, host header
validation, direct-payload integrity, descriptor preparation, private upload,
GPU header validation or decode, resident-ready, first selected diffraction,
and first detector-ready phases separately. A run is not described as cold
unless storage and operating-system cache state were deliberately controlled
and recorded.

## Exact uint16 preparation pipeline

The general real-source path uses the Lossless Pack Format v1 exact-`uint16`/LZ4
profile. It does not widen or reinterpret the exact-`uint8`/bitpacked profile.
The source audit and frozen inventory first produce a
source-bound uint16 contract and independent product oracles. Metadata-only
QH5 indexes then let the preserved native packer read every source frame,
round-trip every uint16 value, and emit detector-major shards. Finally,
`build_compact_h5_uint16.py` validates complete descriptor coverage and writes
the raw-LZ4 lossless-pack container through a temporary file.

Acceptance requires the frozen source identity, complete raw logical SHA-256,
mask-applied working uint16 SHA-256, detector-mask SHA-256, source-bound
calibration, native packer binary SHA-256, every packed shard SHA-256, compact
whole-file SHA-256, and independent original-HDF5 sample parity. Producer
completion is CPU evidence only. CUDA residency, VRAM, reduction, switching,
and presentation claims require a separate uncontended physical CUDA run.

## Adapter status and gates

| Adapter | Contract code | Current qualification gate |
|---|---|---|
| Python reference | `quantem.gpu.io._compact_h5` | strict profile dispatch, fail-closed metadata, direct-bitpacked payload/header checks, and bounded independent raw-HDF5 comparisons |
| Swift and Metal | `MetalCompactH5Loader` and `packed_h5.metal` | physical Metal load, all decoded shard hashes, selected diffraction, complete detector-map oracles, bounded allocation, FFT-off interaction |
| WebGPU and WGSL | `compact-h5.ts` | same synthetic golden, physical browser adapter, complete real-file products, browser memory receipt |
| CUDA and NVRTC | `compact_h5.py` and runtime-compiled kernels | exact-uint16/LZ4 retained; direct-bitpacked payload/header implementation and device-hidden NVRTC compile pass, while physical CUDA adapter, complete real-file products, and device allocation receipt remain required |

WebGPU and CUDA must consume this binary contract. They must not reinterpret an
Android implementation detail, accept unauthenticated low-byte narrowing, or
create a backend-specific public format.
