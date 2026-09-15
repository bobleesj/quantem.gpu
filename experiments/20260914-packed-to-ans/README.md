# Native packed to ANS return path

The reverse conversion is now implemented as an experimental SPI:

```swift
let ans = try packed.makeANSResident(maximumAdditionalBytes: availableBytes,
  shouldCancel: { cancellationRequested })
// Publish/accept the replacement, then explicitly retire the old source.
packed.releaseResidentStorage()
```

It accepts a live original-count 512×512×192×192 uint8/uint16 packed resident
with source identity and exact DPC moments. It does not open source files or
support arbitrary persisted/calibrated MAPED layouts. Serialized worker calls
are required; it must not run on the UI thread or race resident release.

## Implementation

One reusable private 16K-scan window feeds the existing native ANS encoder.
For uint16 this window is 1.208 GB. The existing encoder may allocate another
bounded scratch window; this is not a scratch-free algorithm. The complete
dense 4D dataset is never allocated. DPC summaries, source identity and detector
validity are reused. Display exclusions do not alter the raw conversion values.

The first window decoder decoded each count independently. The replacement
assigns a SIMD group to one block-32 bit-plane cell: one header decode, shared
plane loads, and five shuffle stages transpose planes into exact counts. Other
packed layouts retain the scalar fallback.

## Measurements and correctness

Apple M5, 24 GB, same seven full uint16 acquisitions as the forward experiment.

- Initial scalar reverse: **10.368 s for the first acquisition**.
- SIMD reverse: **13.034 s summed for seven**, approximately 1.81–1.99 s each.
- Repeat: **13.074 s summed for seven**, approximately 1.82–1.99 s each.
- Final ordinary-default forward check: 6.030 s forward wall with two workers;
  reverse 13.356 s summed. Sampled reverse device high-water **15.494 GB**.
  This is not continuous process/driver peak memory or a 16 GB qualification.
- These sums exclude parity between sources; they are not uninterrupted series
  wall times. Both directions still fail the 1–2 s seven-source target.
- Final ANS resident total returns to **7,084,989,324 bytes**, the same total
  as the initial sources. Packed total was 12,045,870,148 bytes.
- Each direction in each seven-source round trip passes 21 full detector maps
  and 77 full diffraction patterns against pre-conversion ANS references.
- Synthetic original-count oracle passes 1,048,576 count comparisons across
  offset/staging combinations, including 32767, 32768, and 65535. The reverse
  window is checked count by count as well.
- Reverse early cancellation preserves the packed original. Full real-data
  original-HDF5 count authentication, reverse mid-cancel qualification, general
  shape/type coverage, UI switching and interaction/FPS remain pending.

In the repeat, combined unpack/encode takes 1.68–1.81 s per acquisition;
compact copy takes 0.061–0.080 s, CPU prefix about 0.006 s, and consolidation
0.040–0.064 s. The combined stage is still the bottleneck. These are existing
builder stage timers, not independent hardware-counter/occupancy measurements.

Build failures retained in this record: the first build missed two imports;
the next was invalidated by editing a converter source while it compiled. Both
were corrected before successful runtime tests. A wrong helper arity in the
SIMD reader was caught by inspection and fixed before its first runtime test.

## Decision and follow-up

Keep both APIs experimental and keep viewer defaults unchanged. Concurrency
alone does not meet the target. Further work needs faster paired-ANS encoding
and bounded conversion scheduling, followed by native UI acceptance and
continuous peak-memory qualification. Do not describe retaining an old ANS
copy as a free reverse conversion: it consumes additional resident memory.

Reproduce with `bash experiments/20260914-packed-to-ans/build.sh`, then run
`ans-roundtrip-synthetic` and `packed-to-ans-probe FOLDER INDEX_FOLDER 7 2`
from the Swift release build directory. Set `QGPU_PAIRED_CONVERSION_STAGING=1`
only for the additional forward scratch experiment. No push/release occurred.
