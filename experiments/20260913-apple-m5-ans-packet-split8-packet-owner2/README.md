# Packet-owner2 split-8 A/B/A on seven full sources

Status: completed. This is an isolated follow-up to the corrected packet-owner2
split-only test. It does not alter either earlier packet-split record.

## Question

Does packet-owner2 with packet split 8 improve the exact seven-source indexed
ADF 8→20 update versus split 1, when every other query, residency, and
submission setting is held fixed?

## Protocol

- Use the same seven distinct original full `(512, 512, 192, 192)` `uint16`
  acquisitions in `~/data/maped-seven-tilts`. No crop, binning, clipping,
  count narrowing, or source substitution.
- A/B/A is packet split `1/8/1`. All three arms explicitly use
  `packet-owner2`; the B request is routed around the previous 1/4 wrapper so
  that no adaptive-partials or split-4 option leaks into the candidate.
- Hold the indexed `packet-groups` polar query, compact offsets enabled,
  trusted table disabled, two streams per lane, unbatched submission,
  concurrency 7, no profiling, and all other detector settings fixed.
- Prepare split pipelines before timing. Use one unmeasured warmup and 20
  measured cycles per arm, each applying `adf-center-8` then
  `adf-center-20` to all seven residents.
- Require stable source identities, exact full-map hashes on all sources and
  cycles, stable allocation after candidate warmup, Metal allocation no more
  than `11,883,921,408` bytes and no more than `122 MiB` above compact-offset
  ready allocation, and release of all seven residents.
- The timing boundary is the unprofiled indexed detector update; initial
  loading and UI presentation are excluded.

## Result

All exact-map, memory, and release gates passed. Metal allocation stayed at
11,636,195,328 bytes in all arms (247,726,080 bytes below the 11,883,921,408
byte ceiling); all seven residents were released.

| Arm | Packet split | ADF 8→20 p50 | ADF 8→20 p95 |
| --- | ---: | ---: | ---: |
| A1 | 1 | 71.16 ms | 74.38 ms |
| B | 8 | 68.91 ms | 70.77 ms |
| A2 | 1 | 70.33 ms | 73.89 ms |

The candidate p50 is 2.6% below the mean of the bracketing controls; the paired
median difference is −1.22 ms and B is faster in 15/20 paired cycles. This is a
small, configuration-specific gain, not a breakthrough. This experiment uses
the slower `packet-groups` query with trusted-table decoding disabled; it does
not compose with the current best `scan512 + trusted-table` configuration, so
it does not improve the current best large-ADF result (59.44 ms p50). Split-8
is not promoted. See the retained raw JSONL and summary for per-cycle hashes and
timings.

## Run

```sh
swift build -c release --disable-sandbox --product metal-paired-runtime-tans-series-benchmark
python3 experiments/20260913-apple-m5-ans-packet-split8-packet-owner2/run.py \
  --exe .build/release/metal-paired-runtime-tans-series-benchmark \
  --folder ~/data/maped-seven-tilts \
  --cache /tmp/ans-packet-split8-packet-owner2-cache-20260913 \
  --out experiments/20260913-apple-m5-ans-packet-split8-packet-owner2/results
```
