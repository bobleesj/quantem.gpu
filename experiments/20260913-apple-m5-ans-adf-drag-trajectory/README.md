# Seven-source ADF drag trajectory

## Question

What is the exact all-seven backend latency at each successive detector mask
while dragging a circular ADF aperture one detector column at a time across
20 pixels?

Each cycle clears to an empty detector mask, sets the centered ADF, and then
applies 20 consecutive one-column moves. This records a sequence of adjacent
deltas, not only a single base→one-pixel sample or a large endpoint jump. A/B/A
uses the same kernel configuration in all three arms to expose drift/noise; it
is not a kernel comparison.

## Frozen gates

- apple-m5-24gb Apple M5, 24 GB; seven distinct full uint16
  `(512, 512, 192, 192)` acquisitions.
- `scan512`, trusted-table decode, unbatched ordinary path, and concurrency 7
  fixed. 20 cycles after one warmup per arm.
- Exact full-map hashes for all 21 masks, unchanged source identities, resident
  bytes ≤ 11,877,814,048 B, Metal allocation ≤ 11,883,921,408 B, and complete
  release.
- Backend timing only; not UI frame cadence or 120-FPS presentation.

The retained exact CPU parity maps are 147 MiB for the 21 masks, below the
benchmark's 150-MiB bound. No additional GPU residency is requested.

## Run

```sh
swift build -c release --disable-sandbox \
  --product metal-paired-runtime-tans-series-benchmark
python3 experiments/20260913-apple-m5-ans-adf-drag-trajectory/run.py \
  --exe .build/arm64-apple-macosx/release/metal-paired-runtime-tans-series-benchmark \
  --folder ~/data/maped-seven-tilts \
  --cache /tmp/ans-adf-drag-trajectory-cache-20260913-1 \
  --out experiments/20260913-apple-m5-ans-adf-drag-trajectory/results
```

## Interpretation

Report p50/p95 for every adjacent drag step, plus the slowest step and the
first/last step. Do not compare this directly to the 8→20 large-jump endpoint
without stating that the transition differs.
