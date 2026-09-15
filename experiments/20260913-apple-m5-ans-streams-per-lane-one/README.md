# Indexed tANS one reader per lane

Status: refuted. One reader per lane preserved exact output and resident
memory, but did not improve the large-ADF update. No hardware counters were
collected, so the slowdown is not attributed to a specific occupancy or
register-pressure cause.

## Question

Does one `packet-owner2` reader per lane improve the exact indexed seven-source
large-ADF update without increasing resident memory or changing detector maps?

## Protocol

- Seven distinct full `(512, 512, 192, 192)` uint16 acquisitions on Apple M5.
- No crop, binning, clipping, count conversion, or additional resident buffers.
- One resident process and A1 2 → B 1 → A2 2 readers/lane; each arm has one
  unmeasured warmup then 20 measured cycles of ADF center-8 followed by
  center-20.
- Hold packet-owner2, indexed packet-groups query, one packet split, unbatched
  submissions, and concurrency seven fixed.
- Require exact per-source/per-cycle full-map hashes, stable identities and
  allocations, and successful release of all seven residents.
- Measured boundary includes planning, preparation, submission, waiting and
  readback; excludes loading and UI presentation.

## Run

```sh
python3 experiments/20260913-apple-m5-ans-streams-per-lane/run.py \
  --exe .build/release/metal-paired-runtime-tans-series-benchmark \
  --folder ~/data/maped-seven-tilts \
  --cache /tmp/ans-streams-per-lane-one-cache-20260913-1 \
  --out experiments/20260913-apple-m5-ans-streams-per-lane-one/results \
  --candidate-streams 1
```

Center-20 p50/p95 ms were A1 68.90/77.03, B 72.66/76.95, and A2 70.50/74.44.
The candidate p50 is about 4.2% slower than the mean of the controls (69.70
ms); its p95 is also slightly slower than the control mean. All per-cycle
full-map hashes matched; resident bytes stayed at 11,877,814,048, Metal
allocation stayed at 11,883,921,408 bytes, and all seven residents were
released. This rejects one reader/lane for this workload, not every geometry.
