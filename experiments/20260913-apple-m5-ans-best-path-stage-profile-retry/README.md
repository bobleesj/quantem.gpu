# Best-path seven-source ANS stage profile retry

Status: complete diagnostic. The first attempt is retained separately and
marked failed because its validator incorrectly required profiling on control
arms. This retry leaves A1/A2 controls unprofiled and enables timestamps on the
scan512 + trusted-table candidate. It is a diagnostic, not a speed benchmark.

## Question and protocol

On the exact seven-source ADF 8→20 update using the best validated composition,
which GPU interval is larger: polar/index work or residual decoding?

- Apple M5, seven unique full uint16 `(512, 512, 192, 192)` acquisitions.
- A1/A2 are unchanged packet-groups controls; B is scan512 with trusted
  table. The only instrumentation change is candidate-arm timestamps.
- One warmup plus 20 measured cycles per arm, full detector-map parity, stable
  source identities, unchanged resident/allocation ceilings, explicit release.
- The source profiler is enabled before loading. A1/A2 controls remain
  unprofiled. The candidate warmup also receives profiling, but it is excluded
  from the measured-cycle interval analysis below.

## Run

```sh
swift build -c release --disable-sandbox --product metal-paired-runtime-tans-series-benchmark
python3 experiments/20260913-apple-m5-ans-best-path-stage-profile-retry/run.py \
  --exe .build/release/metal-paired-runtime-tans-series-benchmark \
  --folder ~/data/maped-seven-tilts \
  --cache /tmp/ans-best-stage-profile-cache-20260913-retry1 \
  --out experiments/20260913-apple-m5-ans-best-path-stage-profile-retry/results
```

## Results

All 140 measured candidate records for `adf-center-20` (20 cycles × 7 sources)
contain valid calibrated GPU timestamps and pass full-map parity. The unchanged
resident footprint was 11,877,814,048 B and peak measured Metal allocation was
11,883,921,408 B. All seven residents were released.

The analyzer unions absolute GPU intervals across the seven source commands
within each cycle. This accounts for commands running concurrently; overlapping
polar and residual intervals must not be added.

| Cross-source GPU interval union | p50 (ms) | p95 (ms) |
|---|---:|---:|
| Polar/index | 25.813 | 35.918 |
| Residual ANS decode | 47.736 | 52.144 |
| Combined union | 55.098 | 59.304 |
| Full command span | 55.147 | 59.375 |

Residual decoding is the larger stage interval, and together both stages nearly
cover the full measured GPU command span. The small difference between their
individual p50 values and the combined union is expected because work overlaps
across sources. This makes residual decode the next target, but a faster
residual kernel alone cannot guarantee the 20–30 ms goal because polar/index
work still contributes roughly 26 ms of union at p50.

The candidate's profiled center-20 end-to-end update p50/p95 was 58.90/62.77 ms.
Do not use that as an unprofiled speed claim or UI frame-rate measurement. The
best unprofiled combined-path A/B/A remains 59.441 ms p50 in
[the composition run](../20260913-apple-m5-ans-scan512-trusted-table-compose/README.md).
The latest direct scan512 A/B/A result was 14.6% faster than its packet-groups
controls; the trusted-table composition added 2.5% versus its own controls.
Those are separate comparisons, not cumulative claims.

Recompute the interval unions from the retained summary with:

```sh
python3 experiments/20260913-apple-m5-ans-best-path-stage-profile-retry/analyze_stage_union.py
```

Do not add overlapping source intervals, and do not treat profiled wall latency
as normal interactive performance.
