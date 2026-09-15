# Four-way striped scan512 accumulation (harness retry)

Status: planned. The first registered attempt is retained at
`../20260913-apple-m5-ans-scan512-striped-accumulation/`; it stopped before
timing because the generic A1 control policy disabled trusted-table mode. This
retry adds an opt-in benchmark flag that preserves trusted-table in A1. Default
benchmark behavior is unchanged.

## Hypothesis

Does factor-4 scan512 polar accumulation improve the exact seven-source
`adf-center-20` update at unchanged memory while trusted-table decoding remains
fixed and enabled in all arms?

## Fixed protocol

- apple-m5-24gb Apple M5; seven distinct original full `(512,512,192,192)` uint16
  acquisitions; no crop, binning, clipping, or count conversion.
- A1/A2 use `scan512`; B uses `scan512-stripe4`. Trusted-table stays enabled
  for all arms; the stripe factor is the only changed kernel parameter.
- One warmup plus 20 measured cycles per arm. Check both `adf-center-8` and
  `adf-center-20`, the complete detector maps for all sources, and frozen A1
  hashes.
- Enforce unchanged resident ceiling 11,877,814,048 B and device allocation
  ceiling 11,883,921,408 B; verify seven distinct source identities and release.
- Resident update timing only; source loading and UI rendering are excluded.
  This is not a UI frame-rate claim.

## Run

```sh
swift build -c release --disable-sandbox --product metal-paired-runtime-tans-series-benchmark
python3 experiments/20260913-apple-m5-ans-scan512-striped-accumulation-retry/run.py \
  --exe .build/release/metal-paired-runtime-tans-series-benchmark \
  --folder ~/data/maped-seven-tilts \
  --cache /tmp/ans-scan512-stripe4-retry-cache-20260913-1 \
  --out experiments/20260913-apple-m5-ans-scan512-striped-accumulation-retry/results
```

## Result

Pending.
