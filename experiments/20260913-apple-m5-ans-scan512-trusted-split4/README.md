# Scan512 + trusted-table + split-4 composition

Status: completed and rejected for promotion. The exact seven-source A/B/A run
showed no speedup from composing split-4 with the current best path. Two earlier
attempts failed in the harness before measuring B; their raw outputs and
release records remain in `results/` and `results-retry/`.

The question is whether packet split-4 improves the established scan512 plus
trusted-table path for the exact seven-source large-ADF update. All arms keep
scan512 and the factory-validated trusted table enabled. Only packet split
changes (1 / 4 / 1). The candidate pipeline is explicitly compiled with both
Metal function constants FC9=4 and FC13=true; the table is validated before
that pipeline can be selected. Unsupported combinations remain fail-closed.

## Frozen workload and gates

- Apple M5, 24 GB; seven distinct full uint16 `512×512×192×192` acquisitions.
- Exact ADF center-8 followed by center-20, one warmup plus 20 measured cycles
  per arm; A/B/A order with packet splits 1/4/1.
- Scan512 and trusted-table stay enabled for A1, B, and A2. No cropping,
  binning, clipping, count conversion, or UI timing claim.
- Require full-map parity against A1 and cycle-by-cycle hashes across arms,
  unchanged source identities, unchanged resident bytes, the established
  11,883,921,408-byte Metal allocation cap, and release of all residents.

## Run

```sh
swift build -c release --disable-sandbox --product metal-paired-runtime-tans-series-benchmark
python3 experiments/20260913-apple-m5-ans-scan512-trusted-split4/run.py \
  --exe .build/release/metal-paired-runtime-tans-series-benchmark \
  --folder ~/data/maped-seven-tilts \
  --cache /tmp/ans-scan512-trusted-split4-cache-20260913-1 \
  --out experiments/20260913-apple-m5-ans-scan512-trusted-split4/results
```

Use a fresh cache/output path. The runner refuses an existing output directory.
The effective configuration is checked explicitly: scan512 and trusted-table
remain enabled in A1/B/A2, with packet splits 1/4/1. The inherited validator
normalizes only the unrelated default `macro=false` field before applying its
frozen configuration gate.

## Result

On Apple M5 with seven distinct full `uint16 512×512×192×192` sources, all
three arms passed exact full-map parity, kept the same resident and Metal
allocation, and released all seven residents. For the large ADF 8→20 move:

| Arm | Packet splits | p50 (ms) | p95 (ms) |
|---|---:|---:|---:|
| A1 control | 1 | 59.31 | 64.57 |
| B candidate | 4 | 59.78 | 63.41 |
| A2 control | 1 | 59.35 | 65.03 |

The candidate was 0.47 ms slower than the median of its bracketing controls.
This composition is rejected; it does not improve on scan512 plus the
validated trusted table. The measured output is retained under `results-retry2/`.
The two prior failures were harness-only: the first did not preserve trusted
table for A1; the second did, but the frozen validator rejected a newly added
`macro=false` config field. No kernel-speed conclusion was drawn from either.
