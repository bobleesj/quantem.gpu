# Compose scan512 with trusted-table decoding

Status: complete; modest incremental gain, not promoted as a standalone 120 Hz solution.

The scan512 query and trusted-table tANS decoder are separate pipeline stages.
The previous runtime guard rejected their composition even though the trusted
table affects only the validated decoder-state bound check. This experiment
removes that one conservative guard and measures the composed path.

## Fixed protocol

- Apple M5; seven distinct full `(512, 512, 192, 192)` uint16 acquisitions.
- No cropping, binning, clipping, or count conversion.
- Scan512 fixed in all arms; only trusted-table toggles off/on/off.
- One warmup plus 20 measured cycles per arm, `adf-center-8` then
  `adf-center-20`.
- Exact full-map hashes for every source and cycle, stable source identity, all
  residents released, resident memory <= 11,877,814,048 B and Metal allocation
  <= 11,883,921,408 B.
- Timings include indexed planning, detector update, submission, wait, and
  readback; loading and UI rendering are excluded.

## Run

```sh
swift build -c release --disable-sandbox --product metal-paired-runtime-tans-series-benchmark
python3 experiments/20260913-apple-m5-ans-scan512-trusted-table-compose/run.py \
  --exe .build/release/metal-paired-runtime-tans-series-benchmark \
  --folder ~/data/maped-seven-tilts \
  --cache /tmp/ans-scan512-trusted-table-compose-cache-20260913-1 \
  --out experiments/20260913-apple-m5-ans-scan512-trusted-table-compose/results
```

## Results

Completed exact A/B/A on seven full sources. Center-20 p50/p95 ms:

| Arm | Trusted table | p50 | p95 |
|---|---:|---:|---:|
| A1 | off | 60.693 | 64.938 |
| B | on | 59.441 | 62.109 |
| A2 | off | 61.257 | 65.479 |

The candidate is 1.534 ms (2.52%) below the mean control p50 (60.975 ms).
Full-map parity, resident/allocation ceilings, and release gates passed. This
is an incremental result; the separate scan512 experiment showed the larger
~14.6% gain. Neither gets the update near the 20–30 ms goal, and these timings
are not UI frame-rate measurements. See [RUNS.md](../RUNS.md) and the retained
manifest/raw records for full provenance.
