# Seven-source ADF drag trajectory (21-mask bounded retry)

The first attempt loaded and released all seven residents correctly but stopped
before timing because the command rejected 21 masks at its existing 20-mask
limit. This retry raises the mask count to 21 only when retained exact CPU
reference maps remain within the existing 150 MiB bound. The measured GPU
allocation ceiling is unchanged.

## Question and workload

Measure all-seven backend latency from a centered ADF through 20 successive
one-column detector moves, separately for each step. Use seven distinct full
uint16 `(512, 512, 192, 192)` sources, `scan512`, trusted-table decode, ordinary
unbatched execution, concurrency 7, one warmup, and 20 cycles per same-config
A/B/A arm.

## Exactness and memory gates

Require exact full-map hashes for all 21 masks, unchanged source identities,
resident bytes ≤ 11,877,814,048 B, Metal allocation ≤ 11,883,921,408 B, and
complete release. Retained A1 CPU maps are 147 MiB (below 150 MiB). No extra GPU
residency is requested. This is backend timing, not UI FPS.

## Run

```sh
swift build -c release --disable-sandbox \
  --product metal-paired-runtime-tans-series-benchmark
python3 experiments/20260913-apple-m5-ans-adf-drag-trajectory-retry/run.py \
  --exe .build/arm64-apple-macosx/release/metal-paired-runtime-tans-series-benchmark \
  --folder ~/data/maped-seven-tilts \
  --cache /tmp/ans-adf-drag-trajectory-retry-cache-20260913-1 \
  --out experiments/20260913-apple-m5-ans-adf-drag-trajectory-retry/results
```

## Result

All 20 one-column moves passed exact full-map parity across the same-config
A/B/A arms. Per-step median latency (median across the three arm medians) was
18.56–20.80 ms, with a 20.01 ms mean. The worst nearest-rank p95 across steps
and arms was 22.78 ms. Seven-source resident bytes stayed at 11,877,814,048 B
and sampled Metal allocation at 11,883,921,408 B; all residents were released.
This meets the 20–30 ms backend target for adjacent one-column drag steps. It
does not imply 120-FPS UI presentation, nor does it reduce the separate 8→20
large-jump latency of roughly 59 ms.
