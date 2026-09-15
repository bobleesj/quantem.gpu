# Paired `uint2` SIMD reduction

Status: refuted as a speed optimization. The exact vector path compiled and
passed the full seven-source A/B/A parity and memory gates, but it did not
produce a robust latency improvement.

## Hypothesis

Does replacing the two scalar `simd_sum(uint)` calls with a single
component-wise `simd_sum(uint2)` reduce the exact indexed large-ADF update time
when scan512 and trusted-table decoding are fixed in all arms, without changing
any output map or allocation?

The vector reduction is algebraically identical to two independent UInt32
reductions: it adds corresponding components modulo 2^32. For this workload,
two streams per lane and coefficients in `[-2, 2]` bound each component well
within Int32, so the modulo result is also the exact signed sum. This avoids
the unsupported `simd_sum(ulong)` experiment without adding scratch or resident
storage. The compiler may still scalarize the vector operation, so only the
controlled timing can establish whether there is a speed benefit.

CPU-only exact preflight:

```sh
python3 experiments/20260913-apple-m5-ans-paired-vector-reduction/oracle_uint2.py
```

## Frozen A/B/A protocol

- apple-m5-24gb Apple M5 24 GB; seven distinct original `(512,512,192,192)` uint16
  acquisitions from `tilt-series-seven-native-v1`; no crop, bin, clipping, or
  count conversion.
- Indexed `packet-owner2`, 2 streams/lane, packet split 1, scan512, trusted-table
  enabled in all arms, compact offsets off. Only the component-reduction form
  changes: scalar A1, vector B, scalar A2.
- One warmup and 20 measured cycles per arm, each applying ADF center 8 then
  center 20. Compare the seven full maps in every cycle to one another and to
  the frozen source/map hashes in the scan512 + trusted-table anchor manifest.
- Resident ceiling 11,877,814,048 B; Metal allocation ceiling
  11,883,921,408 B; no extra buffers. Include full cleanup/release evidence.
- Timing covers indexed planning, detector update, submission, wait, and
  readback; source loading and UI presentation are outside this measurement.

Run:

```sh
swift build -c release --disable-sandbox --product metal-paired-runtime-tans-series-benchmark
python3 -B experiments/20260913-apple-m5-ans-paired-vector-reduction/run.py \
  --exe .build/release/metal-paired-runtime-tans-series-benchmark \
  --folder ~/data/maped-seven-tilts \
  --cache /tmp/ans-paired-vector-reduction-cache-20260913-1 \
  --out experiments/20260913-apple-m5-ans-paired-vector-reduction/results
```

This is a backend update benchmark, not a UI FPS result. Any speed change must
pass exact map parity, identity, allocation, and A/B/A gates to count.

## Outcome

The 20-sample `adf-center-20` p50/p95 results were A1 59.09/63.27 ms, B
59.29/62.20 ms, and A2 60.34/62.74 ms. B is nominally 0.7% below the mean of
the two control medians, but is 0.3% slower than A1; this is within observed
run-to-run spread and is not a demonstrated speedup. Exact maps matched the
frozen seven-source hashes on every arm. Resident bytes remained 11,877,814,048
and Metal allocation remained 11,883,921,408 in all arms. All seven residents
were released; post-release Metal allocation was 5,701,632 bytes. The
experiment does not support enabling this specialization by default.
