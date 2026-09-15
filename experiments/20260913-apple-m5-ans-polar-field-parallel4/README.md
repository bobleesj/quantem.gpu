# Four-way field-parallel polar query

Status: refuted. This was a focused kernel-topology experiment on the polar
index; it does not alter the resident data format or add persistent resident
buffers.

Preflight attempt 1 failed before dataset loading because the wrapper passed a
keyword unsupported by the trusted-table retry helper. No GPU work or timing
was produced. The runner now uses the underlying fingerprint helper directly;
the failed preflight is retained here rather than counted as a benchmark arm.

Run attempt 1 reached and completed A1 (scan512): p50 57.75 ms / p95 61.60 ms
for ADF center-8→20, with exact full-map parity and the established resident
allocation. It then stopped before B because the benchmark configuration
validator omitted `scan512-field4`; the candidate has no timing or parity result.
This harness failure and clean release are retained under `results-retry1/`.
The validator now accepts the already-implemented variant. Retry 2 is recorded
separately under `results-retry2/` so neither attempt overwrites the other.

## Hypothesis and gates

Does assigning the four SIMD groups to four disjoint selected-field partitions
shorten the field-query critical path enough to improve the exact seven-source
large-ADF update versus the current scan512 query?

- Apple M5, seven unique full `(512, 512, 192, 192)` `uint16` acquisitions.
- Exact `adf-center-8` to `adf-center-20`, scan512 and trusted-table fixed.
- A1/B/A2: scan512 / field4 / scan512; one warmup and 20 measured cycles per
arm, with the original 7-source ADF path and full detector-map hashes.
- No crop, binning, clipping, narrowing, or extra persistent resident memory.
- Resident bytes must remain at or below 11,877,814,048 B; Metal allocation
  must remain at or below 11,883,921,408 B; all seven sources must be released.
- This measures resident backend updates only, not file loading, rendering, or
  UI presentation.

Each 128-thread group covers 32 scans. Four SIMD groups each sum one quarter of
the selected fields, then a threadgroup-memory reduction combines the four
exact UInt32 partials per scan. The dispatch uses 16 groups per 512-scan
packet. It trades more groups and metadata staging for a four-times shorter
serial field loop; whether that trade helps is an empirical question.

## Run

```sh
swift build -c release --disable-sandbox --product metal-paired-runtime-tans-series-benchmark
python3 experiments/20260913-apple-m5-ans-polar-field-parallel4/run.py \
  --exe .build/release/metal-paired-runtime-tans-series-benchmark \
  --folder ~/data/maped-seven-tilts \
  --cache /tmp/ans-polar-field-parallel4-cache-20260913-1 \
  --out experiments/20260913-apple-m5-ans-polar-field-parallel4/results
```

## Result

Retry 2 completed the full A/B/A grid with exact detector-map parity and
released all seven residents. For ADF center-8→20, A1/B/A2 p50 was
59.02/67.64/59.27 ms; p95 was 60.97/73.81/65.11 ms. The candidate was about
14% slower than the median of its controls. ADF center-8 p50 was
30.98/40.61/30.39 ms, showing the same regression. Resident bytes remained
11,877,814,048 B and ready Metal allocation 11,883,921,408 B in all three arms.
The candidate's correctness and storage-cost gates passed; its speed hypothesis
did not. The likely cost is the 16-way group expansion and duplicated metadata
staging, which outweighed the shorter per-group serial field loop.

The complete data and first failed harness attempt remain registered in
`manifest.json`; measured retry-2 data are in `results-retry2/`.
