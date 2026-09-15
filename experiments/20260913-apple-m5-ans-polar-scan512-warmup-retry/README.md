# Polar query scan512 warmup-controlled replication retry

Status: completed. This is a new experiment record because the prior warmup
run failed in harness validation before producing a measured A/B/A result.
The validator now uses the requested cycle count when checking warmup ordering.

## Question

Does scan512 reproduce the initial ADF latency reduction after one unmeasured
warmup per A/B/A arm, while preserving exact maps and the seven-source memory
contract?

## Fixed protocol

- Apple M5, seven distinct full `(512, 512, 192, 192)` uint16 acquisitions.
- No crop, binning, clipping, count conversion, or reduced source count.
- Same resident loop, indexed `packet-owner2`, two streams per lane, one packet
  split, unbatched submission, concurrency seven.
- A1 baseline, B scan512, A2 baseline; each arm gets one unmeasured warmup and
  then 20 measured cycles of `adf-center-8` followed by `adf-center-20`.
- Require exact full-map hashes, unchanged source identities/resident bytes/
  Metal allocation, and successful release of all seven sources.
- This is detector-update latency only; file loading and UI presentation are
  outside the timing boundary.

## Run

```sh
python3 experiments/20260913-apple-m5-ans-polar-scan512/run.py \
  --exe .build/release/metal-paired-runtime-tans-series-benchmark \
  --folder ~/data/maped-seven-tilts \
  --cache /tmp/ans-polar-scan512-warmup-retry-cache-20260913-1 \
  --out experiments/20260913-apple-m5-ans-polar-scan512-warmup-retry/results
```

For `adf-center-20`, all-seven p50/p95 ms were A1 70.22/73.01, B scan512
60.58/66.62, and A2 71.71/75.52. B is 14.6% faster in p50 than the mean of
the bracketed controls. Exact full-map hashes matched across all seven sources
and both masks; source identities were unchanged, resident bytes remained
11,877,814,048, Metal allocation remained 11,883,921,408 bytes, and all seven
residents were released. See `manifest.json` and `results/` for raw evidence.

The earlier harness-only failure remains retained under
`../20260913-apple-m5-ans-polar-scan512-warmup/`. This candidate is a bounded
win, not a 20–30 ms solution; it does not measure UI frame rate.
