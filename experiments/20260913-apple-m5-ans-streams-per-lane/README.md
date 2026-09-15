# Indexed tANS streams per lane

Status: refuted. The test preserved exact output and resident memory, but four
readers/lane was substantially slower. The cause is not pinned to register
pressure or occupancy because no such hardware counters were collected.

## Question

Does changing `streams_per_lane` from 2 to 4 reduce the exact indexed seven-
source `adf-center-20` update time without changing output or resident memory?

## Fixed protocol

- Apple M5; seven distinct complete `(512, 512, 192, 192)` uint16 acquisitions.
- No crop, binning, clipping, count changes, or additional resident buffers.
- One process keeps the same seven original sources resident throughout.
- A1 2 readers/lane, B 4 readers/lane, A2 2 readers/lane. Every arm gets one
  unmeasured warmup then 20 measured cycles of `adf-center-8` followed by
  `adf-center-20`.
- Query/index variant stays `packet-groups`; the candidate is isolated to
  streams-per-lane. Submission is unbatched with concurrency seven.
- Require exact full-map hashes for every source/cycle/mask, unchanged source
  identities/resident bytes/Metal allocation, and release of all seven sources.
- Timing includes planning, preparation, submission, wait, and readback, but
  excludes loading and UI presentation.

## Run

```sh
python3 experiments/20260913-apple-m5-ans-streams-per-lane/run.py \
  --exe .build/release/metal-paired-runtime-tans-series-benchmark \
  --folder ~/data/maped-seven-tilts \
  --cache /tmp/ans-streams-per-lane-cache-20260913-1 \
  --out experiments/20260913-apple-m5-ans-streams-per-lane/results
```

For center-20, p50/p95 ms were A1 69.79/74.37, B 170.43/176.05, and A2
68.95/74.13. The candidate is 2.46× slower than the mean of the controls.
All per-cycle full-map hashes matched; resident bytes stayed at 11,877,814,048,
Metal allocation stayed at 11,883,921,408 bytes, and all seven residents were
released. This rules out four readers/lane for this workload; it is not a
reason to claim anything about all detector geometries.
