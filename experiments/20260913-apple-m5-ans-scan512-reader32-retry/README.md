# Scan512 + reader32 retry

Status: failed (the requested composition is explicitly unsupported). The first
reader32 attempt stopped on a validator schema mismatch. This retry fixed that
schema compatibility issue, passed the A1 parity/timing arm, then the benchmark
rejected B because `scan512` deliberately requires `QGPU_PAIRED_RUNTIME_READER32=0`.
No reader32 candidate timing exists for this combination.

## Question

Can the 32-bit reverse reader compose with the scan512 query path?

## Protocol

- Apple M5, seven distinct full `(512, 512, 192, 192)` `uint16` acquisitions.
- Fixed scan512 query path, indexed `packet-owner2`, two streams/lane, one
  packet split, concurrency seven; all other optional specializations disabled.
- A1 reader32 off, B reader32 on, A2 reader32 off; one warm-up plus 20 measured
  cycles per arm, each cycle `adf-center-8` then `adf-center-20`.
- Exact full-map hashes and source identities; resident ceiling
  11,877,814,048 bytes; Metal allocation ceiling 11,883,921,408 bytes; release
  all seven sources.
- Reader32 pipeline prepared before loading. Timing includes planning,
  submission, GPU wait, and readback; excludes source loading and UI.

The retry validator ignores only the newly reported `macro: false` field when
calling the older shared validator, after asserting it is false. The A1 warm-up
and measured arm were exact; A1 center-20 p50/p95 was 60.64/63.07 ms. The B arm
was rejected by the intended incompatibility guard before candidate timings,
and all seven sources were released. The next valid test uses packet-groups in
all arms, isolating reader32 without scan512.

## Run

```sh
swift build -c release --disable-sandbox --product metal-paired-runtime-tans-series-benchmark
python3 experiments/20260913-apple-m5-ans-scan512-reader32-retry/run.py \
  --exe .build/release/metal-paired-runtime-tans-series-benchmark \
  --folder ~/data/maped-seven-tilts \
  --cache /tmp/ans-scan512-reader32-cache-20260913-retry \
  --out experiments/20260913-apple-m5-ans-scan512-reader32-retry/results
```

## Result

Scan512 and reader32 are currently mutually exclusive by the backend's
fail-closed compatibility guard. No speedup claim is possible from this run.
