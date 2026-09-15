# Four-way striped scan512 polar accumulation

Status: planned. This is a single-variable, exact-map A/B/A experiment. It
tests the new factor-4 scan512 accumulator while keeping the trusted-table
decoder active in all arms, matching the current best composition.

## Hypothesis

Does dividing the scan512 polar reduction into four independent unsigned
accumulators reduce seven-source `adf-center-20` latency without changing any
detector-map value or increasing the established resident/allocation ceiling?

## Fixed protocol

- apple-m5-24gb Apple M5, seven distinct original full `(512,512,192,192)` uint16
  acquisitions; no crop, binning, clipping, or count conversion.
- A1 and A2 use `scan512`; B uses `scan512-stripe4`. Trusted-table decoding is
  enabled for all three arms, so only the polar accumulator factor changes.
- One warmup plus 20 measured cycles per arm, checking both exact ADF masks
  (`adf-center-8` then `adf-center-20`) and every source's full detector map.
- Same-process A/B/A, frozen A1 hashes, seven unique source identities, exact
  resident ceiling 11,877,814,048 B, Metal allocation ceiling 11,883,921,408 B,
  and explicit release of all seven sources.
- Measurement is the resident update only; source load and UI rendering are
  excluded. This is not a UI frame-rate claim.

## Run

```sh
swift build -c release --disable-sandbox --product metal-paired-runtime-tans-series-benchmark
python3 experiments/20260913-apple-m5-ans-scan512-striped-accumulation/run.py \
  --exe .build/release/metal-paired-runtime-tans-series-benchmark \
  --folder ~/data/maped-seven-tilts \
  --cache /tmp/ans-scan512-stripe4-cache-20260913-1 \
  --out experiments/20260913-apple-m5-ans-scan512-striped-accumulation/results
```

The stripe kernel changes summation order only within unsigned UInt32 modular
arithmetic; exact full-map parity remains mandatory. A build alone does not
compile the runtime Metal shader, so runtime pipeline creation and this parity
gate are required.

## Result

Pending.
