# Scan512 + trusted-table + split-8 composition

Status: refuted. This is a bounded follow-up to the completed split-4 test.
It changes only packet split count; the current best scan512 query and
factory-validated trusted-table path remain enabled in every arm.

## Question

Does packet split-8 reduce exact all-seven ADF center-8→center-20 update time
compared with split-1 on the current best path, without changing any count,
full detector map, source identity, resident bytes, or allocation cap?

## Frozen protocol

- Apple M5, 24 GB; seven distinct complete `uint16 (512, 512, 192, 192)` source
  acquisitions from `~/data/maped-seven-tilts`; no crop, binning, clipping, or
  count narrowing.
- A/B/A packet split `1/8/1`. `scan512`, trusted-table decoding, compact
  offsets, two streams per lane, unbatched submission, and concurrency 7 stay
  fixed in every arm. Only `packet_splits` changes.
- One warmup and 20 measured cycles per arm. Each cycle applies
  `adf-center-8` then `adf-center-20` to all seven residents.
- Require distinct stable source hashes, full-map hash equality across all
  arms and cycles, resident bytes ≤ 11,877,814,048 B, Metal allocation ≤
  11,883,921,408 B, and explicit release of all seven residents.
- Report per-cycle paired timings and p50/p95. This backend measurement is not
  UI frame rate. A/noisy/B improvements do not count; promote only if B beats
  both controls beyond their observed spread.

The split-4 composition on this best path was already exact but showed no
speedup. Split-8's earlier modest gain was measured on the slower packet-groups
path only, so this is a single composition test rather than a presumption that
gains transfer.

## Result

The measured ADF 8→20 p50/p95 values were:

| Arm | Split | p50 (ms) | p95 (ms) |
| --- | ---: | ---: | ---: |
| A1 | 1 | 59.59 | 62.35 |
| B | 8 | 59.84 | 63.02 |
| A2 | 1 | 60.10 | 62.95 |

B is only 0.010 ms below the mean of its bracketing-control p50 values, well
inside run noise; its p95 is worse. This candidate is refuted as a speed
optimization. Full-map parity passed, resident bytes remained 11,877,814,048 B,
Metal allocation remained 11,883,921,408 B, and all seven residents were
released. The result does not approach the 20–30 ms large-jump target.

## Run

```sh
swift build -c release --disable-sandbox \
  --product metal-paired-runtime-tans-series-benchmark
python3 experiments/20260913-apple-m5-ans-scan512-trusted-split8/run.py \
  --exe .build/release/metal-paired-runtime-tans-series-benchmark \
  --folder ~/data/maped-seven-tilts \
  --cache /tmp/ans-scan512-trusted-split8-cache-20260913-1 \
  --out experiments/20260913-apple-m5-ans-scan512-trusted-split8/results
```
