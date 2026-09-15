# Bounded ordinary-path concurrency (fixed control and schema)

This is a fresh, isolated A/B/A run after two harness-only stops. The first
attempt did not preserve the trusted-table flag in A1; the first retry fixed
that environment control, then exposed a stale strict-validator schema that
did not know the emitted `macro: false` field. Both failures are retained in
their own experiment folders and neither is treated as timing evidence.

## Question

Does limiting the ordinary, unbatched seven-source ANS update path to four
in-flight source submissions reduce the exact all-seven large-ADF wall time
when `scan512` and trusted-table decoding are held constant?

## Frozen workload and gates

- apple-m5-24gb Apple M5, 24 GB; seven distinct full uint16
  `(512, 512, 192, 192)` acquisitions.
- Exact `adf-center-8` then `adf-center-20`, one warmup and 20 measured cycles
  per arm. A/B/A concurrency is 7 / 4 / 7; only concurrency changes.
- Require all per-cycle full-map hashes to match, unchanged seven-source
  identities, resident bytes at or below 11,877,814,048 B, Metal allocation
  at or below 11,883,921,408 B, and explicit release of every resident.
- This is backend update latency, not load-to-visible time or UI FPS.

The ordinary per-source path is retained in all arms. Submission is bounded by
launching the next source update only when one of the current in-flight updates
completes. GPU measurements are serialized; no competing GPU benchmark is
allowed.

## Run

```sh
python3 experiments/20260913-apple-m5-ans-update-concurrency-retry2/run.py \
  --exe .build/arm64-apple-macosx/release/metal-paired-runtime-tans-series-benchmark \
  --folder ~/data/maped-seven-tilts \
  --cache /tmp/ans-update-concurrency-retry2-cache-20260913-1 \
  --out experiments/20260913-apple-m5-ans-update-concurrency-retry2/results
```

## Decision rule

Promote four in-flight updates only if B beats both bracket controls beyond
their spread, does not regress p95 or the per-source critical path, preserves
exact maps, and stays inside both allocation ceilings. Otherwise retain seven.
