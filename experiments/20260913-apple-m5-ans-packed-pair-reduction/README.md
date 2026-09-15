# Packed `ulong` pair reduction

Status: failed before the A/B/A measurement. The CPU arithmetic oracle passed,
but Apple MSL rejected `simd_sum(ulong)` and the Metal integer-literal spelling
before the benchmark emitted its ready event. The compiler diagnostic and
empty timing record are retained under `results/`; no timing or map-parity
result exists for this candidate. Follow up using the separately registered
`20260913-apple-m5-ans-paired-vector-reduction` experiment.

## Hypothesis

Does replacing the two `simd_sum(uint)` operations for the two values in each
dense tANS pair with one bias-packed `simd_sum(ulong)` in `packet-owner2`
reduce exact seven-source large-ADF update time when scan512 and trusted-table
decoding are held on in every arm, without changing any full detector map or
resident-memory ceiling?

The candidate packs two independent signed channel sums into the low and high
32-bit halves of one `ulong`. For `N` streams per lane, use
`bias = N * 65535 * 2`. For every lane, add that bias to each signed sum before
packing, reduce the `ulong` across the 32 SIMD lanes, unpack, then subtract
`32 * bias` from each channel. Inactive lanes still contribute the fixed bias.
The prototype must accumulate each lane's stream products in signed integers
before biasing.

The worst tested lane has four streams with coefficient magnitude 2, so each
signed channel partial is bounded by `4 * 65535 * 2 = 524,280`. Each biased
lane field is in `[0, 1,048,560]`; after 32 lanes either field is at most
`33,553,920`, well below `2^32`. Thus the low field cannot carry into the high
field. The 64-bit packed sum also remains below `2^64`. The oracle checks this
bound and exact reconstruction for every coefficient tuple in `[-2, 2]^N`,
`N ∈ {1, 2, 4}`, with uint16 endpoints/midpoints and inactive SIMD lanes.

Run the CPU-only preflight with:

```sh
python3 experiments/20260913-apple-m5-ans-packed-pair-reduction/oracle_one_ulong.py
swift build -c release --disable-sandbox --product metal-paired-runtime-tans-series-benchmark
python3 experiments/20260913-apple-m5-ans-packed-pair-reduction/run.py \
  --exe .build/release/metal-paired-runtime-tans-series-benchmark \
  --folder ~/data/maped-seven-tilts \
  --cache /tmp/ans-packed-pair-reduction-cache-20260913-1 \
  --out experiments/20260913-apple-m5-ans-packed-pair-reduction/results
```

One initial harness invocation stopped before GPU work because its `--exe`
path incorrectly assumed a nested SwiftPM build directory. The executable is
at the repository-level `.build/release/`; this is recorded in the manifest's
`launch_preflight_attempts` and the corrected command above.

This oracle validates only integer packing, signedness, and carry behavior. It
does not establish Metal `simd_sum(ulong)` support, shader correctness, speed,
or production readiness.

## Frozen A/B/A protocol

- Host: Apple M5 24 GB. Use seven distinct original full
  `(512, 512, 192, 192)` `uint16` acquisitions from the same
  `tilt-series-seven-native-v1` fixture. No crop, binning, clipping, or count
  conversion.
- Fixed configuration in every arm: indexed `packet-owner2`, 2 streams per
  lane, packet split 1, scan512 polar query, trusted-table decode enabled,
  compact offsets off, and every other benchmark option copied from the
  accepted scan512 + trusted-table configuration. Only the reduction
  representation changes.
- Arms: A1 existing pair of `simd_sum(uint)` operations; B opt-in packed
  `simd_sum(ulong)`; A2 the unchanged baseline. Prepare both pipelines before
  measurements, then use one unmeasured warmup and 20 measured cycles per arm.
  Each cycle applies `adf-center-8` followed by `adf-center-20` to all seven
  residents.
- Parity: before timing promotion, require exact full-map hashes for every
  source, mask, and cycle. Freeze/compare against the 14 source-mask hashes in
  `20260913-apple-m5-ans-scan512-trusted-table-compose/manifest.json` and
  require A1, B, and A2 to agree. Also require stable source identities and
  clean release of all seven residents.
- Memory: add no resident or scratch buffers. Require every arm at or below
  11,877,814,048 resident bytes and 11,883,921,408 Metal allocated bytes, the
  accepted best-path ceilings. Record both arm allocations; any candidate
  growth is a failed gate even if it remains below the ceiling.
- Timing: compare unprofiled all-seven detector-update wall p50/p95 for
  `adf-center-20` using the A1/A2 bracket control. Loading and UI work remain
  outside the timing boundary. Do not claim a UI frame rate from this backend
  measurement.

## Outcome

Metal library compilation rejected `simd_sum(ulong)` on this toolchain, and
also flagged `524_280u` as an invalid Metal integer literal. No A1/B/A2 arm
began and no GPU update samples were produced. The stderr diagnostic is in
`results/stderr.log`; empty stdout is retained as `results/raw.jsonl`. The
experiment status is `failed`, not a performance refutation. The initial
executable-path mistake is also retained in the manifest.
