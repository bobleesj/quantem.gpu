# Reverse-reader word reuse

FC10 removes an overlapping word reload in the ANS reverse reader. After
advancing backward four bytes, the previous low word becomes the new high
word. Load only the new earlier word. Retain the original prime at boundaries
where the new cursor is below four. The default is disabled.

The compressed representation, precision, offsets, state transitions, escape
decoding and output reductions do not change. No additional resident storage
is needed. Reduced source-level loads are a hypothesis, not evidence of less
device-memory traffic or a performance gain: caches/compiler may already help.

The mixed-data correctness probe specializes FC10 and covers full signed
output equality, word alignment, truncation, tail padding and trailing bytes.
Full seven-source A/B/A must follow before any speed or correctness claim.

Start the resident benchmark with `QGPU_ANS_RESIDENT_LOOP=1`, prepare the
existing radial1 leaf16 index, then switch `reuse_word` false/true/false with
`kernel=packet-owner2`, `streams_per_lane=2`, `batch=false`, `packet_splits=1`.
Raw and indexed results are separate. Keep all seven residents alive between
arms. No load time, hash work or UI presentation is included in the API timer.

## Result: hypothesis refuted

Six five-cycle arms completed with 4,200 exact full-map hashes against the
frozen independent reference and complete in-process array equality. Warm
medians exclude each first cycle:

| Path / transition | Control before (ms) | Word reuse (ms) | Control after (ms) |
|---|---:|---:|---:|
| Raw ADF center-1 | 25.440 | 25.261 | 22.967 |
| Raw ADF center-8 | 191.186 | 191.054 | 186.482 |
| Raw ADF base | 468.305 | 501.018 | 477.166 |
| Indexed ADF center-1 | 22.185 | 22.976 | 20.179 |
| Indexed ADF center-8 | 44.524 | 46.421 | 42.088 |

Fewer source-level loads did not improve these timings. Do not enable FC10 by
default. No cache-counter evidence establishes the reason for the regression.
The next useful measurement is instruction/stall attribution of the dependent
decode and reduction loop, not another assumption that all loads hit memory.

Seven-source resident storage stayed at 11,877,814,048 bytes; Metal current
allocation stayed at 11,883,921,408. This includes the existing 2,603,403,760-byte
exact regional index, unused by raw arms. No extra scratch or decoded volume
was added by word reuse. UI testing and production promotion remain undone.

## Profiling availability

The local Metal device reports only the `timestamp` counter set and
`GPUTimestamp` counter through `MTLDevice.counterSets`. `xcrun --find xctrace`
fails because Instruments is not installed with this Command Line Tools setup.
Therefore these runs cannot substantiate occupancy, cache misses, instruction
stalls or a theoretical latency floor. A future diagnostic-only split of decode
versus reduction must not be reported as a full scientific-image speedup.

The resident benchmark remains loaded but idle after the six arms; no further
GPU work is queued. The tested opt-in variants are local only, not installed in
the app or pushed. Send further JSON commands to the existing owner rather than
loading seven copies again when the next experiment uses these same pipelines.
