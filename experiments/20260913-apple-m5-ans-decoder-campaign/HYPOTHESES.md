# Residual decoder hypothesis ledger

Source and experiment review, 2026-09-13. This ledger contains 20
approaches, **not 20 tested optimizations**. The status column distinguishes
earlier measurements from proposed work. New proposals have no timing or
correctness qualification unless a subsequent campaign result says otherwise.

Scope: exact paired tANS decoding in `prt_fast_pair`, `prt_fast_pop`,
`prt_fast_refill`, and `paired_runtime_tans_detector_packet_owner2`, in
`src/quantem/gpu/swift/Sources/Metal4DSTEMKernels/Resources/paired_runtime_tans.metal`.
The default owner decodes two independent selected streams per lane, 256 pairs
per stream, then performs two SIMD sums and packet-local updates per pair.
No proposed shader probe needs a format change or additional resident buffer.
Some can increase registers or instruction footprint; that is not evidence
that hardware occupancy improves or remains unchanged.

## Twenty approaches and their evidence

| # | Approach | Status and concrete interpretation |
|---|---|---|
| 1 | Reuse the overlapping reverse-reader word | **Measured rejection, FC10.** Indexed center-1 22.185 / 22.976 / 20.179 ms. Already implemented; do not repeat under a new name. Fewer source loads did not establish less device traffic. |
| 2 | Retain 16 lane-owned output accumulators | **Measured rejection, FC11.** Indexed center-1 23.920 / 28.048 / 21.701 ms. No evidence justifies assuming another large register array will win. |
| 3 | Replace unique-writer dense atomics with plain shared writes | **Measured rejection, FC12.** Indexed center-1 20.725 / 22.147 / 21.016 ms. Existing fences remain necessary for sparse contributions. |
| 4 | Validate immutable table transitions once | **Measured narrow gain, FC13.** Repeated raw center-1 25.297 / 22.453 / 24.482 ms; indexed gain did not beat both controls. Keep payload, escape and terminal validation. Not a demonstrated indexed-residual gain. |
| 5 | Remove per-pair reduction to assess decoder cost | **Measured diagnostic, FC14.** ADF small region 25.01 / 21.88 / 22.99 ms for image/checksum/image. Checksums cannot replace detector images or certify elementwise parity. This is evidence for targeting decode, not an optimization candidate. |
| 6 | Refill only when the next table transition needs bits | **Measured rejection, FC15.** Indexed center-20 73.921 / 77.823 / 76.522 ms. Demand-driven refill changes the dependency chain and did not establish a general gain. |
| 7 | Remove zero-bit pop branch in eager 64-bit reader | **Measured inconclusive, FC16.** Residual sequence 46.205 / 43.951 / 41.722 / 38.683 / 43.196 ms. Earlier report explicitly lacked FC16 malformed-stream qualification. Further testing is confirmation, not a new hypothesis. |
| 8 | Decode multiple pairs through a larger lookahead table | **Measured rejection.** Four-bit macro table adds 4.125 MiB/resident and roughly doubles center-1 latency. Outside zero-additional-resident-memory scope. Do not casually retry with a larger table. |
| 9 | Moderate fixed refill threshold of 16 or 24 bits | **Measured, FC17; no promotion.** Both variants passed exact parity and malformed fixtures but neither qualified for combined speed improvement. See this directory's README. |
| 10 | Issue both reader table loads before either transition completes | **Measured, FC18; no combined speedup.** Residual median improved 45.86→44.03 ms; paired combined ratio was 1.003. See `../20260913-apple-m5-ans-phased-readers/`. |
| 11 | Fuse full-width escaped pair extraction | **New planned.** For escape word 4096 in eager64 only, refill if available<32, check validity/availability, decrement available by32 once, extract a UInt32, and split high16/low16. Potential only where full uint16 escapes occur; count their frequency before interpreting negligible aggregate effects. |
| 12 | Unroll exactly two pair iterations | **Measured, FC19; no promotion.** Factors 2/4/8 passed parity fixtures but did not qualify for a combined gain. See `../20260913-apple-m5-ans-pair-unroll/`. |
| 13 | Separate common symbol decoding from escape handling | **New planned.** In the ordinary path compute `a=code&63`, `b=(code>>6)&63` directly, then overwrite only for escape. Keep state transition before escape consumption. This tests control/data scheduling, not fewer decoded symbols; compiler may already produce identical instructions. |
| 14 | Delay next prime until next actual refill | **New planned, low priority.** Remove unconditional next-window prime at refill exit; prime at next refill entry. Different from FC10 word reuse and FC15 demand selection: changes prefetch timing. May save terminal prefetch but lose latency hiding, so expected benefit is uncertain. Preserve padded readable-window contract. |
| 15 | Bypass redundant zero-cursor prime loads | **New planned.** At refill exit when cursor==0, initialize pending words/shift to zero instead of reading payload[0..7]. Subsequent zero-byte refills contribute no bits. Validate tiny/truncated streams and unchanged error reporting; compiler may already suppress some loads. |
| 16 | Specialize all-entropy SIMD chunks | **Measured rejection, FC21.** Exact seven-source indexed ADF center-20 p50 was 70.16 / 70.94 / 70.19 ms (control/candidate/control); the candidate was about 1.1% slower, with exact maps and unchanged residents. The 93.62% all-entropy census justified the test but did not imply a speedup. Keep the general path. |
| 17 | Hoist constant-stream sums outside pair loop | **New planned.** Accumulate each lane's constant contribution once, combine across SIMD once, then seed/update all 512 partials with that value; remove constant lanes from dense loop. Preserve modulo-UInt32 signed arithmetic and original failure checks. Nearly all ADF streams are entropy, so this is probably a BF/special-data improvement only. |
| 18 | Use aligned vector loads for raw literal pairs | **New planned, low priority.** Where the validated address satisfies alignment, replace four byte loads by one 32-bit load and split into uint16 values, with original byte path otherwise. No format changes. Raw stream incidence limits utility; unaligned casts are not a safe universal replacement. |
| 19 | Shorten reader state live ranges through a dedicated ordinary struct | **New planned.** Ordinary eager64 specialization needs no macro queue or reader32 fields. Explicit dedicated helper/struct can test whether compiler already strips these fields. No resident-memory saving is claimed; register allocation might be identical. Keep generic reader as validation reference. |
| 20 | Reorder existing selected residual pixels for locality | **New planned, outside first shader probes.** Stable-sort existing selected/coefficient pairs by a locality key without adding resident data. Modular integer addition is order independent; preserve paired coefficients and unique ownership. Entropy model varies by packet, so a single detector-pixel ordering cannot guarantee same-table lanes. Include sorting/upload time in end-to-end timing. |

## Completed probes and current next measurement

### Campaign checkpoint

- #7 and #9 (FC16 zero-bit-pop and FC17 refill16/24): measured with exact
  parity; neither variant qualified for combined speed. The earlier malformed
  fixture gap for FC16 is closed for the small fixture.
- #10 (FC18 phased loads): residual median improved 45.86 to 44.03 ms, but the
  paired combined ratio was 1.003. Not promoted.
- #12 (FC19 unroll2/4/8): fixtures pass; no repeatable combined gain.
- Additional infrastructure: one cached exact query plan and scratch set per
  resident. Repeat calls allocate zero scratch buffers; no reliable combined
  timing win established. This is not reduced resident dataset storage.

The original proposal/status rows above distinguish pre-experiment hypotheses
from measured outcomes. Unlisted proposals remain unimplemented.

Probe **#16, all-entropy SIMD chunks** was tested against the same exact ADF
transition after its census of 1,067 residual detector pixels. Of 57,344
complete 64-stream tiles, 51,437 (89.70%) were entropy-only; at the kernel's
SIMD32 decision scope, 110,722/118,272 (93.62%) groups were entropy-only.
Despite those high fractions, the candidate p50 was 70.94 ms versus 70.16 and
70.19 ms controls (about 1.1% slower). Full-map parity, unchanged resident
bytes, and release passed. The branch remains off; prevalence alone was not a
performance predictor.

The indexed all-seven batch A/B/A is complete in
`../20260913-apple-m5-ans-indexed-batch-profile/`: normal indexed return was
70.94/76.80/70.48 ms unprofiled, so coordinated submission regressed about 8.6%
and is rejected. Its first profile did not initialize
`PairedRuntimeDetectorProfiler`, because the environment flag was set after
source construction. The corrected startup-profile run is
`../20260913-apple-m5-ans-stage-profile-normal/`. All 420 profiled source
updates had valid exact-map parity and valid timestamp intervals. Across seven
sources, median polar/index encoder interval unions were 40.58/32.02/35.37 ms
for concurrent/batched/concurrent; residual unions were 50.74/48.18/49.79 ms;
their overlapping combined unions were 68.40/68.79/67.01 ms. The unprofiled
all-seven controls were 71.01/76.84/70.76 ms. Thus residual is the largest
individual stage, but there is no speedup and no proof of occupancy or
bandwidth saturation. Do not add per-source durations as if they were
all-seven wall.

The SIMD32 diagnostic measured complete groups, counting the partial 11-stream
tail separately; the fast route applied only when every lane had a valid
entropy reader. FC21 retained per-stream model/table pointers and existing
begin/pair/finish checks. Exactness gates passed, but all-seven latency did not
improve, so the candidate remains rejected and default stays unchanged.

For #11, do not call the generic pop with count32: `(1u << count)-1u` is not
valid for count32. After refill and checks, use:

```metal
reader.available -= 32u;
uint both = uint(reader.reservoir >> reader.available);
a = both >> 16u;
b = both & 65535u;
```

The first 13-bit escape tag is still decoded exactly as before. Keep the
original implementation as the reader32/macro/lazy fallback. On insufficient
bits, set validity false and fail through the existing path; never publish a
partly decoded result as successful. Explicitly cover 0..31 initially available
bits, cursor<4, the final pair, tail padding, truncation and high signed sums.

## Threshold16/24 exactness review

The decoded bit count is `(code >> 12) & 15`, hence at most15 even before the
trusted-table proof (the actual immutable tables prove at most10). Therefore
any eager threshold at least16 guarantees a transition can pop whenever no
refill is requested. A refill when available<threshold either supplies the
needed bits or leaves the existing pop check to report insufficient data.

The maximum reservoir availability after a full refill is threshold-1+32:
47 for16 and55 for24. This is below64 and keeps the FC16 zero-bit shift safe.
The initial partial byte supplies at most7 bits. Escapes must retain their
independent 13- and16-bit refill checks, whose maxima also remain below64.

Changing refill timing changes cursor and available together while preserving
their bit balance. `prt_fast_finished` already accepts the permitted bytes
prefetched before begin through
`available == 8 * (begin-cursor)` and `cursor <= begin`; it does not require
the old exact refill schedule. Do not simplify that terminal check to cursor
equality. Do not move or remove padding/trailing-data checks.

For the initial attributable experiment disable lazy, macro and reader32
specializations, keep payload layout and table validation unchanged, and ensure
the threshold switch changes the actual prepared pipeline. The source proof
supports correctness but does not replace full-source elementwise comparisons
and threshold-specialized malformed-stream tests.

## Evidence references and acceptance

Measured entries reference sibling experiment README files:
`20260912-apple-m5-ans-word-reuse`, `ans-reduction-bounds`, `ans-plain-history`,
`ans-trusted-table`, `ans-decode-checksum`, `ans-lazy-refill`,
`20260913-apple-m5-ans-pop`, and the macro section of
`20260912-apple-m5-ans-adf-optimization`.

Other measured dead ends outside these20 entries include cooperative packet
ownership, packet splitting, smaller coordinated batches, sparse splitting,
dense compaction, and larger partial scratch. They should not be presented as
untested ways to solve this same residual case.

Use the same seven full sources and exact signed transitions, full output
equality, warmup policy, resident allocation and bracketed controls. The stage
isolation reference is index+residual=combined=ordinary signed indexed delta;
also retain the existing independently frozen full-map oracle for ordinary
trajectories. A stage-only comparison does not itself create a new independent
raw-count oracle. Preserve adversarial uint16 and malformed-stream gates for
each compiled candidate. No checksum substitution, additional dense volume,
format change, or UI FPS claim follows from these proposals.

Only timestamp counters were available in preceding work. Instruction stalls,
cache misses, bandwidth saturation, spilling and occupancy remain hypotheses.
Zero additional resident allocation does not imply unchanged registers,
transient allocation or process peak memory.
