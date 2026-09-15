# Reproduce the best exact seven-source ADF path on the current checkout

## Question

Does the current dirty checkout still reproduce the established scan512 plus
trusted-table large-ADF result on seven full sources, with exact maps and no
increase over the resident/allocation ceilings?

## Protocol

- Apple M5, seven distinct native uint16 512×512×192×192 acquisitions from
  `~/data/maped-seven-tilts`.
- Preserve the current best composition: scan512 query, radial1 leaf16 index,
  trusted table off/on/off, packet-owner2, two streams per lane, one packet
  split, bounded source concurrency 7, no compact offsets.
- One warmup and 20 measured cycles per arm. Each cycle visits ADF center-8
  then center-20; all seven resident outputs are compared by full-map hashes.
- Require per-source identity hashes to be distinct and stable; resident bytes
  ≤11,877,814,048 B; sampled Metal allocation ≤11,883,921,408 B; all seven
  residents released after the run.
- Timings include planning, detector computation, submission, wait, and
  readback. They exclude loading and UI rendering.
- The tested executable is freshly built from the current dirty checkout into
  `/tmp/qgpu-ans-large-adf-build-20260913`; sources, executable, and runner
  hashes are retained in the manifest.

## Run

```sh
swift build --scratch-path /tmp/qgpu-ans-large-adf-build-20260913 \
  -c release --disable-sandbox \
  --product metal-paired-runtime-tans-series-benchmark
python3 experiments/20260913-apple-m5-ans-scan512-trusted-table-compose/run.py \
  --exe /tmp/qgpu-ans-large-adf-build-20260913/arm64-apple-macosx/release/metal-paired-runtime-tans-series-benchmark \
  --folder ~/data/maped-seven-tilts \
  --cache /tmp/ans-current-source-reproduction-cache-20260913-1 \
  --out experiments/20260913-apple-m5-ans-current-source-reproduction/results
```

## Interpretation

This is a current-source baseline reproduction, not an optimization by itself.
It determines whether subsequent candidates should compare against the earlier
59.44 ms measurement or a changed current baseline. No UI frame-rate claim is
made from this backend timing.
