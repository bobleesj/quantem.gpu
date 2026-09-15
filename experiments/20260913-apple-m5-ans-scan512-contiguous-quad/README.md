# Scan512 contiguous-quad packed-window reuse

Status: measured; not promoted. This is an opt-in index-query specialization;
the default packet-groups path and the qualified scan512 path remain unchanged.

## Hypothesis

The scan512 query currently assigns each thread four scans spaced 128 apart.
For every selected field, it independently calculates and fetches each packed
value's bit window. A contiguous quad maps each thread to four neighboring
scans, loads their shared packed-word window once, and extracts four values
from those words. The field order, coefficient, and UInt32 accumulation order
stay the same. The candidate adds no resident data or persistent allocation.

The planner is held fixed. The current exact ADF center-8→20 plan uses 371
selected fields and 1,067 residual pixels; the previously tested joint planner
only removed one field with the same residual count. The existing field-parallel
query added 16 groups per packet and regressed, while scan512 stripe-4 showed no
reliable p50 win. This candidate changes scan assignment and packed-word reuse
within each thread.

This can only address the polar interval. The retained stage profile measured
25.813 ms polar/index union and 47.736 ms residual ANS union at p50, with a
55.098 ms combined union. Even eliminating the entire polar stage would not
meet the 20–30 ms all-seven target, so this experiment is one bounded
contribution.

## Exact performance and parity run

Build the benchmark with the experiment source, then run the existing
seven-source resident-loop A/B/A harness through this focused wrapper:

```sh
swift build -c release --disable-sandbox --product metal-paired-runtime-tans-series-benchmark
python3 experiments/20260913-apple-m5-ans-scan512-contiguous-quad/run.py \
  --exe .build/release/metal-paired-runtime-tans-series-benchmark \
  --folder ~/data/maped-seven-tilts \
  --cache /tmp/ans-scan512-contiguous-quad-cache \
  --out experiments/20260913-apple-m5-ans-scan512-contiguous-quad/results
```

The run uses scan512 / contiguous-quad / scan512 A/B/A, with one warmup and 20
measured cycles per arm for ADF center-8 followed by center-20. It requires
seven distinct complete `(512, 512, 192, 192)` uint16 acquisitions, exact
full-map hashes for all seven sources, unchanged resident and Metal allocation
bytes, and release of all residents. Every other resident-loop setting is held
to the existing scan512 experiment. The timer covers indexed detector update,
not source loading or UI presentation. No timing has been run for this planned
record.

## CPU packed-window reference

The focused Swift reference test packs 512 values at each width from 0 through
32, then compares the contiguous four-value window against a bit-by-bit scalar
decoder for every thread lane. It includes width zero, scan/word boundaries,
cross-word values, 32-bit values, and UInt32 base wraparound:

```sh
swift test --filter PairedRuntimeTANSPolarQuadReferenceTests
```

## apple-m5-24gb result

Two independent release A/B/A runs each passed exact full-map parity for all
seven sources and kept resident/current Metal allocations unchanged at
11,877,814,048 / 11,883,921,408 bytes. The first attempt stopped before timing
because the harness omitted the default `macro=false` field from its expected
configuration; that harness gate was corrected before the two valid runs.

| Run | A1 p50/p95 ms | Candidate p50/p95 ms | A2 p50/p95 ms | Paired median gain | Candidate wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 61.29 / 65.21 | 60.21 / 65.58 | 61.57 / 65.11 | 0.14 ms | 11/20 |
| 2 | 61.33 / 63.47 | 60.30 / 62.72 | 60.68 / 67.87 | 0.97 ms | 12/20 |

Positive paired gain means the candidate was faster than the midpoint of its
bracketing controls. The small, inconsistent gains and p95 behavior do not
establish a robust improvement; this candidate is not promoted. The best prior
large-move result remains about 59.44 ms p50, so this does not close the
20–30 ms target. The kernel is only the polar/index contribution; ANS residual
decoding remains the dominant measured stage.
