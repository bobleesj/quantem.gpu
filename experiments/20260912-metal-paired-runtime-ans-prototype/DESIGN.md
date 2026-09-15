# Exact paired runtime tANS prototype

Status: native integration candidate. The CPU oracle, Metal codec, original-HDF5
builder, resident queries, and Live4DSTEM adapter are wired locally; publication
is pending review and explicit push authorization.

## Scientific contract

- Input is one complete native `uint8` or `uint16` detector column over a
  512-scan coding interval.
- Every stored count is reproduced exactly, including `255`, `256`, `32768`,
  and `65535`.
- Two consecutive counts form one entropy symbol, reducing the dependent tANS
  state transitions from 512 to 256 per ordinary stream.
- No crop, bin, clipping, invalid-pixel replacement, or scientific dtype
  conversion is permitted.
- Alternative zero, constant, sparse, and literal modes are exact storage
  choices. They do not change the logical dtype.

The proposed ABI name is `metal-runtime-paired-tans-v1`. It must not reuse
`streamed-counts-v1`, `paired-polar-counts-v1`, or
`source112-tans1024-pair-v1`: those retained representations have different
payload and offset layouts.

## Codec ABI

One chunk contains any positive multiple of 512 scans. Streams are ordered
`packet * detectorPixels + detectorPixel`, where each packet is 512 scans.

| Array | Type | Shape | Meaning |
| --- | --- | --- | --- |
| `payload` | `uint8` | encoded byte count plus eight guard bytes | Concatenated stream bytes |
| `offsets` | `uint32` | `streamCount + 1` | Half-open byte bounds in `payload` |
| `models` | `uint8` | `streamCount` | Entropy model or exact alternative mode |
| `decoding` | `uint32` | `32 x 1024` | Pair, bit count, and next-state base |

The first implementation should retain ordinary `uint32` offsets. Grouped
16-bit relative offsets are a separate optimization and are not required to
prove the paired transition design.

### Stream modes

- `64...95`: paired tANS using model `mode - 64`.
- `252`: sparse events. Each event is one little-endian `uint16` containing
  `(scan << 7) | (count - 1)`; this mode is used only when every count is at
  most 128.
- `253`: all-zero, with an empty payload.
- `254`: literal native counts, each as little-endian `uint16`.
- `255`: constant nonzero, stored as one little-endian `uint16`.

The paired tANS payload begins with a little-endian `uint16` header:

```text
bits 0...2   = meaningful-bit count modulo eight
bits 3...5   = zero (reserved; reject if nonzero)
bits 6...15  = initial 1024-state tANS state
```

The remaining bytes are a reverse-read bit stack. Values are pushed least
significant bit first by the encoder. The decoder pops complete fields from
the high end, preserving each field's integer value.

### Pair alphabet and escape

There are 1,089 frequency symbols: `a * 33 + b` for directly modeled
`a,b < 32`, plus symbol 1,088 for escape. A decoding-table entry packs:

```text
bits 0...11  = a | (b << 6), or 4095 for escape
bits 12...15 = transition bit count
bits 16...31 = next-state base
```

An escape first stores a 13-bit word:

- `0...4095`: exact pair `a = word & 63`, `b = word >> 6`.
- `4096`: followed by exact `uint16 a`, then exact `uint16 b`.
- `4097...8191`: reserved and invalid.

The wide encoder pushes `b`, then `a`, then marker `4096`; reverse decoding
therefore observes marker, `a`, `b`. This preserves the complete uint16 range
without making common six-bit pairs wider.

## Metal/HDF5 integration boundary

The eventual Metal implementation can consume the private dense window and
the same command buffer produced by
`OriginalHDF5Packing.forEachExactDecodedWindow`. Use a 512-aligned bounded
window; never publish or retain that dense window.

For each decoded window:

1. Dispatch one thread per stream to select the exact mode/model and encode a
   bounded transposed scratch bit stack.
2. Prefix the stream byte sizes into shared `uint32` offsets. The existing CPU
   prefix is acceptable for the first measured prototype because it is not a
   dependent entropy transition.
3. Allocate the exact private payload and compact entropy, sparse, constant,
   and literal streams into it.
4. Retain only payload, offsets, models, the shared decoding table, source
   identity, shape, and logical dtype. Reuse scratch and metadata allocations
   across windows and acquisitions.

The CUDA control flow and integer recurrence in `_compact/kernels/paired.cu`
are the algorithmic reference. CUDA syntax, CuPy ownership, grouped offsets,
and the polar index cannot be copied into Metal. The current runtime rANS
source and retained source112 Metal consumer are independent ABIs.

## Promotion gates

1. Freeze the generated frequency and transition-table SHA-256 values. Compare
   every supported `(model, symbol, oldState)` encode step with the independent
   decode-table transition.
2. Round-trip exact deterministic and randomized 512-count streams for both
   logical dtypes, including zero, constant, sparse, ordinary entropy,
   incompressible literal, odd escape, and `65535` sentinel cases.
3. Reject truncated payloads, reserved header bits, invalid escape markers,
   duplicate/out-of-order sparse positions, and nonterminal tANS states.
4. Metal and CPU must produce identical modes, offsets, payload bytes, and
   decoded counts on the small fixtures.
5. Compare complete diffraction patterns and BF/ABF/ADF integer images against
   the existing original-HDF5 packed oracle. Promotion requires a full-volume
   count audit for every acceptance acquisition; sampled frames are only a
   development check.
6. Record decode, paired encode, prefix, compact, resident-ready, resident
   bytes, peak allocation, and teardown separately. A speed claim requires
   matched A/B/A runs and unchanged counts, shape, dtype, and memory ceiling.
7. Verify cancellation, source-file mutation, malformed HDF5, and memory-budget
   failure publish no partial resident.

## Native integration evidence

On `apple-m5-24gb`, seven exact `512x512x192x192 uint16` HDF5 acquisitions
were opened directly into paired tANS without a saved ANS file, cropping,
binning, or clipping. The native app retained 9.27 GB of paired residents
(9.35 GB including presentation surfaces), versus approximately 14.75 GB for
the prior packed-resident representation.

The hook-driven native journey completed with zero problems and zero console
errors: all seven opens, return visits, rapid switching, BF, ABF, ADF, iDPC,
FFT, scan drag, large custom-detector drag, seven-way comparison, comparison
detector drag, and comparison scan drag. Exact DPC moments are accumulated from
each transient original-decode window and retained only as derived scan maps;
the dense windows are still discarded. Sampled diffraction, detector-sum, and
DPC-moment parity all pass against independent exact calculations.

Observed full-app first visibility was 3.79 s and subsequent first loads were
2.88–3.09 s; already-resident return visits were 0.07–0.11 s. These are warm,
ordinary-run observations, not cold-I/O or subsecond claims. Further codec and
multi-resident interaction optimization is intentionally deferred. Selected-DP
dragging presented 118.6–119.7 updates/s, while all-seven large ABF and ADF
gestures presented only 6.3–17.3 updates/s; the 120 Hz multi-resident detector
target is therefore explicitly not met.
