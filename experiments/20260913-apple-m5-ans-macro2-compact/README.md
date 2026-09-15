# Two-bit paired-tANS macro with compact offsets

Status: complete; speed hypothesis refuted. This was a single-variable decoder
experiment. Exact output parity passed, but the candidate was materially slower.

## Question

Does a two-bit lookahead macro table reduce the exact seven-source, indexed
large-ADF update time versus the ordinary decoder, when compact offsets are
enabled in both arms and all seven full datasets remain resident?

## Fixed protocol

- Apple M5, seven distinct original `(512, 512, 192, 192)` `uint16`
  acquisitions; no crop, binning, clipping, or count conversion.
- Same-process A1 / B / A2, one unmeasured warmup and 20 measured cycles per
  arm, applying `adf-center-8` then `adf-center-20` each cycle.
- Indexed `packet-owner2`, two streams per lane, one packet split, unbatched
  submissions, concurrency seven, `packet-groups` polar query in every arm.
- Compact offsets remain on in every arm. A1/A2 use ordinary per-pair tANS;
  B uses the prepared two-bit macro table. The table width is fixed before
  loading residents; only the macro pipeline selection changes between arms.
- Require exact per-source full-map hashes for every measured cycle, identical
  seven-source identities and allocations across arms, allocation no greater
  than the ordinary-offset reference ceiling, and successful release of all
  seven residents. The reference ceiling is 11,877,814,048 resident bytes and
  11,883,921,408 sampled Metal bytes.
- Timing includes planning, preparation, GPU submission, wait, and readback;
  excludes load and UI. No claim about displayed UI FPS.

## Memory hypothesis

The proposed 2-bit table has 1,179,648 bytes per source (ordinary 1024-word
table plus 1024 states × 4 lookaheads × two UInt32 words × 32 models). Seven
tables add about 8.26 MB to the compact-offset arm. Compact offsets previously
saved about 247.7 MB across seven, but actual allocations—not this estimate—are
the acceptance gate.

## Run

The release benchmark builds the two-bit table and exhaustively validates its
packed entries before uploading it. Runtime A/B/A then checks every full output
map on all seven sources. Use the project's release benchmark build:

```sh
swift build -c release --disable-sandbox --product metal-paired-runtime-tans-series-benchmark
```

Run the retained seven-source test with fresh cache and output directories:

```sh
python3 experiments/20260913-apple-m5-ans-macro2-compact/run.py \
  --exe .build/release/metal-paired-runtime-tans-series-benchmark \
  --folder ~/data/maped-seven-tilts \
  --cache /tmp/ans-macro2-compact-cache-20260913-1 \
  --out experiments/20260913-apple-m5-ans-macro2-compact/results
```

Retain failures and regressions in this experiment directory and update the
manifest and `experiments/RUNS.md` with measured artifacts.

## Result

The test completed with one warmup and 20 measured cycles per A/B/A arm.
Center-20 all-seven p50/p95 were A1 `70.24/74.93 ms`, macro2 `126.67/134.72
ms`, and A2 `70.06/74.14 ms`. Thus macro2 was `80.6%` slower than the bracketed
control p50. Median per-source GPU interval rose from about `10 ms` in the
controls to `17.9 ms` for macro2. All full-map hashes matched on every cycle,
including both masks and all seven sources.

The measured series resident was `11,638,345,518 B`; sampled Metal allocation
was `11,644,452,864 B`, both equal across arms and below the established
ordinary-offset ceiling. Each of the seven residents held a `1,179,648 B`
macro table, and all seven were released. This rejects the two-bit macro table
for this workload; it remains off by default. The experiment did not collect
hardware counters, so cache or occupancy effects are not asserted as the cause.

The filtered Swift XCTest command could not run on this Command Line Tools
installation (`no such module 'XCTest'`). However, resident creation built and
validated all `131,072` packed two-bit table entries before uploading them; the
seven-source detector run then passed full-map parity against ordinary decode.
