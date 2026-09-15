# Seven-source one-pixel ADF drag step

## Question

How long does an exact seven-source ADF update take when the aperture moves one
detector pixel, rather than the previously reported large `adf-center-8` to
`adf-center-20` jump?

The resident loop starts each cycle from an empty detector mask, builds the
centered ADF base, then moves the center by one column. The second mask is the
timed target. This isolates the normal adjacent drag delta; it is not a UI FPS
measurement and does not model skipped pointer positions.

## Frozen gates

- apple-m5-24gb Apple M5, 24 GB; seven distinct full uint16
  `(512, 512, 192, 192)` acquisitions.
- `scan512`, trusted-table decode, ordinary unbatched path, and concurrency 7
  remain fixed in every arm. Three same-configuration arms bracket drift.
- One warmup and 20 cycles per arm; exact full-map hashes and source identities
  must match. Resident and Metal allocation ceilings are 11,877,814,048 B and
  11,883,921,408 B. All seven sources must be released.
- Backend update wall time only; no visible-frame or 120-FPS claim.

## Run

```sh
python3 experiments/20260913-apple-m5-ans-adf-one-pixel/run.py \
  --exe .build/arm64-apple-macosx/release/metal-paired-runtime-tans-series-benchmark \
  --folder ~/data/maped-seven-tilts \
  --cache /tmp/ans-adf-one-pixel-cache-20260913-1 \
  --out experiments/20260913-apple-m5-ans-adf-one-pixel/results
```

## Interpretation

Compare the `adf-center-1` samples against the large-jump baseline separately.
Only the measured one-pixel backend response can support an adjacent-drag claim.
