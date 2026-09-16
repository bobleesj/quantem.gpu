# QEM 0.0.1 codec specification

This page specifies the bytes needed by independent implementations. It is a
project specification, not a claim of community ratification. The
[envelope and calibration contract](qem-format.md) applies to both codecs.
All multi-byte words are little-endian. Counts use C order:
`(scan_row, scan_column, detector_row, detector_column)`.

## Integer codec: runtime-column-rans-spatial-v2

Required geometry: four positive axes, dtype `uint8` or `uint16`, `version=1`,
`interval=512`. `valid` is a hexadecimal detector bit mask, with the first pixel
in the most significant bit of its byte. Unused final bits are padding. The
mask affects spatial sums, not original count decoding.

Chunks cover flattened scan positions consecutively, without gaps or overlaps.
Each chunk declares `first`, `scans`, and six `arrays`. Each array declares a
body-relative byte `offset` and an element `count`. Before each array, align the
running body cursor upward to a multiple of eight. Count payload offsets must
fit uint32. Let `P = detector_rows * detector_columns`, `B = ceil(scans/512)`.

| Array | Element type | Meaning / element count |
| --- | --- | --- |
| 0 | uint8 | Count-stream payload |
| 1 | uint32 | Payload byte offsets, `B*P+1`, beginning at zero |
| 2 | uint8 | Model selector, `B*P` |
| 3 | uint32 | Packed spatial sums |
| 4 | uint64 | Spatial payload word offsets, `B*F+1`, beginning at zero |
| 5 | uint8 | Spatial sum widths, `B*F` |

Count stream `block*P + pixel` stores one detector pixel across up to 512
consecutive scans. Its sample count is `min(512, scans-block*512)`. Consecutive
offsets delimit each stream; offsets are nondecreasing and end at payload size.

### Count models

| Selector | Bytes and decoded values |
| --- | --- |
| 253 | No bytes; all zeros |
| 255 | One uint16 constant, repeated for the stream length |
| 254 | One uint16 per scan, including when logical dtype is uint8 |
| 252 | Sorted uint16 events: position is `event >> 7`, value is `(event & 127)+1`; omitted positions are zero |
| 0 through 63 | Byte-renormalized rANS, specified below |

Other selectors are invalid. Sparse positions must be strictly increasing and
within the stream. Every decoded value must fit the declared logical dtype.
Do not truncate out-of-range uint16 values into uint8.

The normative 64-by-33 frequency table is
{download}`qem-rans-tables-v1.json <../../src/quantem/gpu/io/qem-rans-tables-v1.json>`.
Every row sums to 1024. **Use these integer frequencies, not a regenerated
floating-point probability distribution.** For each symbol, `start` is the sum
of preceding frequencies. The reference source is
{download}`_qem_reference.py <../../src/quantem/gpu/io/_qem_reference.py>`.

A rANS stream begins with its uint32 state, in `[2**23, 2**31)`. For each scan:

```python
slot = state & 1023
symbol = the_symbol_whose_interval_contains(slot)  # [start, start+frequency)
state = frequency * (state >> 10) + slot - start
while state < 2**23:
    state = (state << 8) | next_byte()
if symbol == 32:
    value = next_uint16_little_endian()
else:
    value = symbol
```

After all samples, state must equal `2**23` and every byte must be consumed.
An encoder processes samples in reverse order, emits escape high/low bytes
before renormalization, and reverses the emitted bytes after the final state.
Different model choices are permitted; measurement equality, not identical
compressed bytes, defines conformance.

### Spatial sums

Fields consist first of detector 8-by-8 tiles, then 32-by-32 tiles, each ordered
by tile row then tile column. Partial edge tiles stop at detector boundaries.
`F = ceil(rows/8)*ceil(columns/8) + ceil(rows/32)*ceil(columns/32)`.
Each field is a uint32 sum of original **valid** pixels for one scan. The largest
tile sum is bounded by `1024*65535`, so this does not overflow uint32.

Spatial stream `block*F + field` contains up to 512 sums. Its width is 0 through
32 bits. Values are concatenated least-significant-bit first into uint32 words;
the final word is zero-padded. A width-zero stream has no words. Offsets count
uint32 words, not bytes. Readers must bound every span before using an index.
Index correctness is checked against sums of decoded counts, not just checksums.

## Float codec: empad-xor-row-packed-v1

Required geometry: four positive axes with detector `(128,128)`, dtype
`float32`, `version=1`. This describes bit-exact storage, not a detector
calibration algorithm. The codec can also store compatible NumPy float arrays.

Each chunk covers 1 through 512 scans. Its payload consists of packed uint32
words, followed immediately by `scans*128` descriptors. A descriptor contains
four uint32 values: `base`, `width`, `shift`, `word_offset`.

For each 128-pixel detector row, pixel bit patterns are XORed with `base`.
The resulting integers, shifted right by `shift`, are packed using `width`
bits per pixel, least-significant-bit first. Decode pixel `column` as:

```python
delta = extract_bits(payload, word_offset*32 + column*width, width)
pixel_uint32 = base ^ (delta << shift)
```

Reinterpret those uint32 bits as float32; do not numerically cast integers to
floats. `0 <= width <= 32`, `0 <= shift <= 32-width`. A row consumes `width*4`
words. Descriptors must cover the payload consecutively. An all-constant chunk
still has a four-byte payload. `logical_sha256` hashes original float32 bytes
in C order; signed zero and NaN payloads are significant.

The `empad` object records source identity and any saved mean-dark recipe.
`background.values_float32_le`, if present, is base64 encoding of 128x128
little-endian float32 values. CPU reference decoding returns original raw
measurements and reports `background_applied=False`; it retains the recipe in
`metadata["qem_empad"]`. Native products may apply this recipe once. A corrected
display and original decoded measurements are different scientific outputs.

## Conformance and extension policy

Validate the envelope, supported versions, metadata, chunk bounds and body
checksums before exposing measurements. The GPU-free validator also checks
float descriptors; decoding separately verifies entropy completion and float
logical checksums. Checksum success does not establish correct derived indexes.

Unknown codecs and major metadata schemas must fail with a clear update request.
Readers preserve unknown optional metadata, but must not treat unknown units
as known calibration. Use namespaced keys inside `extensions` for additions
which are not yet part of the shared vocabulary. Codec changes require a new
codec identifier; do not change an existing bitstream definition in place.

Use the [conformance bundle](qem-interoperability.md) to check independent
implementations. Agreement between implementations maintained in this one
repository is useful evidence, not independent community adoption.
