# Reader32 under packet-groups

Status: refuted. The 32-bit reader is intentionally incompatible with the
scan512 query path today, so this test holds packet-groups fixed in all arms.
This isolates the reader specialization while keeping the new scan512 path out
of the measurement.

## Question and protocol

Does the 32-bit reverse bit-reader improve exact seven-source large-ADF
updates versus the default 64-bit reader when the query path is packet-groups?

- Apple M5, seven distinct full `(512, 512, 192, 192)` `uint16` sources.
- A1 reader64, B reader32, A2 reader64; packet-groups throughout; one warm-up
  plus 20 measured cycles per arm, `adf-center-8` then `adf-center-20`.
- Indexed `packet-owner2`, two streams/lane, one packet split, concurrency
  seven; other detector specializations disabled.
- Exact full-map hashes, stable source identities, resident bytes no higher
  than 11,877,814,048, Metal allocation no higher than 11,883,921,408, and
  explicit release of all seven sources.
- Reader32 is prepared before the run; scan512 is neither prepared nor
  selected in this isolated run. The inherited readiness
  validator required a scan512 pipeline, so the runner first asserts actual
  scan512 preparation is false and then reuses its independent shape, dtype,
  identity, and allocation checks. Timings include planning, submission, GPU
  wait, and readback; loading and UI presentation are excluded.

## Run

```sh
swift build -c release --disable-sandbox --product metal-paired-runtime-tans-series-benchmark
python3 experiments/20260913-apple-m5-ans-reader32-packet-groups/run.py \
  --exe .build/release/metal-paired-runtime-tans-series-benchmark \
  --folder ~/data/maped-seven-tilts \
  --cache /tmp/ans-reader32-packet-groups-cache-20260913 \
  --out experiments/20260913-apple-m5-ans-reader32-packet-groups/results
```

## Result

Center-20 p50/p95 was A1 71.28/75.87 ms, B 73.60/77.78 ms, A2 69.24/72.92
ms. Reader32 was slower than both controls; its speed hypothesis is refuted on
this workload. All full-map checks and the resident/allocation gates passed;
all seven sources were released. Allocation was unchanged at 11,883,921,408
Metal bytes. This does not measure or refute a future checkpointed decoder.
