# Scan512 plus trusted-table detector update (retry)

Status: failed before the B arm; no A/B/A result.

This reruns the same exact test as
`../20260913-apple-m5-ans-scan512-trusted-table/`, fixing only a validator
compatibility issue: the current benchmark reports `macro=false` in addition
to the older configuration fields. The retry explicitly requires macro mode
off and records the complete configuration, including the trusted-table toggle.
The first attempt is retained separately and was released cleanly after its A1
warmup; it produced no measured cycles. This retry completed A1 at 61.66 ms
large-ADF p50 (66.06 ms p95), then the explicit scan512 compatibility guard
rejected trusted-table mode before B. All seven residents were released. The
next experiment removes only that conservative guard after checking that the
query and decode pipelines are independent, and records a new A/B/A run.

## Fixed protocol

- Apple M5, seven distinct full `(512, 512, 192, 192)` uint16 sources.
- Scan512 fixed in all arms; trusted-table off/on/off A/B/A.
- One warmup plus 20 measured cycles per arm; `adf-center-8` then
  `adf-center-20` each cycle.
- Exact full-map hash parity for every source and cycle, stable source identity,
  and clean release of all seven sources.
- Hard caps: 11,877,814,048 resident bytes and 11,883,921,408 Metal bytes.
- Detector-update wall-time only; no claim about load time or UI FPS.

```sh
swift build -c release --disable-sandbox --product metal-paired-runtime-tans-series-benchmark
python3 experiments/20260913-apple-m5-ans-scan512-trusted-table-retry/run.py \
  --exe .build/release/metal-paired-runtime-tans-series-benchmark \
  --folder ~/data/maped-seven-tilts \
  --cache /tmp/ans-scan512-trusted-table-retry-cache-20260913-1 \
  --out experiments/20260913-apple-m5-ans-scan512-trusted-table-retry/results
```

## Results

No result yet. The candidate remains unproven until the matched A/B/A completes.
