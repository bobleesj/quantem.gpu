# Polar query scan512 warmup-controlled replication

Status: failed before measured A/B/A. The seven sources loaded and were
released. The harness incorrectly validated the one-cycle warmup's mask order
as though it had 20 cycles; no performance result was produced. The preserved
failure is followed by a separately registered retry.

## Question

Does scan512 reproduce the initial 13.6% all-seven center-20 ADF p50 reduction
after matching per-arm warmup, with exact maps and unchanged seven-resident
memory?

## Fixed protocol

- Apple M5, seven distinct full `(512, 512, 192, 192)` uint16 acquisitions;
  no crop, binning, clipping, or count changes.
- Same one-process resident loop, indexed packet-owner2, two streams per lane,
  one packet split, unbatched seven-source updates, and masks in order
  `adf-center-8` then `adf-center-20`.
- A1 control, scan512 candidate, A2 control. Each arm first receives one
  unmeasured warmup cycle, then 20 measured cycles.
- Exact per-source full-map hash equality, source identity, unchanged resident
  and Metal allocation, and final release of all residents are required.
- A fresh metadata cache and fresh output directory are used. The previously
  completed screening artifacts remain untouched under the original experiment.

## Run

```sh
python3 experiments/20260913-apple-m5-ans-polar-scan512/run.py \
  --exe .build/release/metal-paired-runtime-tans-series-benchmark \
  --folder ~/data/maped-seven-tilts \
  --cache /tmp/ans-polar-scan512-warmup-cache-20260913-1 \
  --out experiments/20260913-apple-m5-ans-polar-scan512-warmup/results
```

## Result

Failure details are retained in `manifest.json` and `results/failure.json`.
This was a harness-only failure, not a kernel parity or runtime failure; the
20–30 ms target has not been reached. The benchmark excludes loading and UI
presentation and measures all-seven detector-update wall time.
