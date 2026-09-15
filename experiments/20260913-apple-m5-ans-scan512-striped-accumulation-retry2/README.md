# Four-way striped scan512 accumulation (schema retry)

Status: planned. The first attempt exposed A1's default control-reset behavior;
the next corrected that behavior, then exposed an older Python validator that
did not know the explicit `macro: false` field added to the benchmark response.
Both harness failures are retained in their own experiment folders; neither
produced a timing result. This run checks that field explicitly, removes it only
for the legacy validator comparison, then restores it in the retained record.

## Hypothesis and gates

- Does factor-4 scan512 polar accumulation improve the exact seven-source
  `adf-center-20` update while trusted-table stays fixed on?
- Seven distinct original full `(512,512,192,192)` uint16 acquisitions; no
  crop, binning, clipping, or count conversion.
- A1/A2 use `scan512`; B uses `scan512-stripe4`; trusted-table is on in all arms.
- One warmup plus 20 measured cycles per arm, checking both ADF masks and exact
  full detector maps against frozen A1 hashes and across all arms.
- Unchanged resident ceiling 11,877,814,048 B; Metal allocation ceiling
  11,883,921,408 B; seven distinct source identities; explicit release.
- Resident update only. Source loading and UI rendering are excluded; this is
  not a UI frame-rate claim.

## Run

```sh
swift build -c release --disable-sandbox --product metal-paired-runtime-tans-series-benchmark
python3 experiments/20260913-apple-m5-ans-scan512-striped-accumulation-retry2/run.py \
  --exe .build/release/metal-paired-runtime-tans-series-benchmark \
  --folder ~/data/maped-seven-tilts \
  --cache /tmp/ans-scan512-stripe4-retry2-cache-20260913-1 \
  --out experiments/20260913-apple-m5-ans-scan512-striped-accumulation-retry2/results
```

## Result

Refuted for promotion. All exact full-map parity, seven-source identity,
unchanged resident/allocation ceilings, and release gates passed. For the
large ADF `adf-center-20`, p50 was A1 59.60 ms, stripe4 59.17 ms, A2 58.85 ms;
the candidate landed between the controls. Its p95 was 61.42 ms versus 62.79
and 63.16 ms, respectively, but one 20-sample A/B/A session is insufficient
to promote on that noisy tail result. For `adf-center-8`, stripe4 p50 was
31.22 ms versus 30.20 and 30.67 ms, a regression. No resident buffer or
measured Metal allocation changed. This experiment shows no reliable
performance win, so factor-4 striped accumulation remains experimental and
is not selected by default.
