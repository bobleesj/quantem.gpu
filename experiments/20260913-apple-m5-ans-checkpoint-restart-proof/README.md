# CPU proof: paired runtime tANS checkpoint restart

This experiment checks exact restart at every interior pair boundary for the
production paired runtime tANS representation. It runs entirely on the CPU and
uses only Python's standard library.

Run it from this directory:

```sh
PYTHONDONTWRITEBYTECODE=1 python -m unittest -v
```

The oracle in `paired_runtime_tans_checkpoint.py` mirrors the table builder in
`src/quantem/gpu/swift/Sources/Metal4DSTEMStreamingIO/PairedRuntimeTANSTables.swift`
and the encoder, fallback mode selection, and reverse reader in
`src/quantem/gpu/swift/Sources/Metal4DSTEMKernels/Resources/paired_runtime_tans.metal`.
It covers the deployable entropy modes 64 through 95. The experimental
interleaved-state modes 96 through 127 are deliberately outside this proof.

The table test matches all three frozen Swift ABI hashes. The encoder tests
also match the checked-in synthetic Metal fixture payload hashes for ordinary
entropy data, both escape encodings, sparse fallback, literal fallback, zero,
and constant streams. These byte-for-byte fixture comparisons validate that
the CPU port is using the paired runtime production transitions and packing
rules, rather than a newly invented tANS variant.

For each of four entropy fixtures, the restart test visits every interior
boundary between decoded pairs: ordinary symbols, six-bit pair escapes,
wide uint16 escapes including 65,535, and a 509-value stream with an odd final
pair. At each boundary it:

1. Decodes the original payload up to the boundary and snapshots the state,
   reverse-reader cursor, reservoir, available bits, and remaining meaningful
   bits.
2. Resumes the same payload from that snapshot and checks that both decoded
   parts reproduce every source value, ending in state zero with no bits left.
3. Re-frames each side as its own byte-aligned segment. The first segment is
   encoded to terminate at the saved tANS state; the second segment starts
   from that saved state and terminates at zero.
4. Checks that the two segment bit strings, in reverse-encoder write order,
   concatenate to the original meaningful bit string exactly. It then decodes
   both standalone segment payloads and compares all values again.

Every split is checked, not just a hand-picked boundary. Segment headers carry
their own tail-bit counts and unused high padding bits are zero-filled; the
test observes all eight tail alignments (0 through 7) across the segments.
This is a bit-level partition: a restart record must preserve the reverse
reader reservoir and cursor when continuing the original byte stream. When
materializing two separately byte-aligned payloads, the bit fragments are
repacked and each segment gets a fresh header.

The fallback tests cover production modes 252 (sparse), 253 (all zero), 254
(uint16 literal), and 255 (constant uint16). They also verify that malformed
entropy headers, nonzero byte-tail padding, truncated entropy data, malformed
sparse events, malformed literals, and unsupported modes are rejected.
Fallback modes do not have tANS pair checkpoints because they contain no tANS
state chain.

## Scope

This is a CPU oracle and frozen-fixture parity check, not a Metal-kernel run.
It proves the algebra and byte-boundary behavior against the checked-in
production table hashes and synthetic-codec payload digests. It does not
measure GPU behavior or establish a resident checkpoint file format.
