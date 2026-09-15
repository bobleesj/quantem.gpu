# Adaptive partials: packet-groups with bounded scratch accounting

Status: refuted. This is a third, separate experiment record; the two previous
experiment folders and registry rows remain unchanged. The corrected A/B/A
completed all three arms and found adaptive partials slower than both controls.

## Corrected validation

- A1 (`packet-owner2`) must retain the compact-offset ready-time
  `series_resident_bytes`.
- B warmup (`adaptive-partials`) may increase that aggregate by 0–122 MiB.
  The observed prior warmup delta was 124,780,544 bytes (about 119 MiB).
- B's measured response and both A2 responses must retain the exact B-warmup
  `series_resident_bytes`; Metal allocation must also remain fixed after that
  warmup.
- Existing runner checks continue to require full exact maps/hashes, stable
  source identities, unchanged compact resident inputs, Metal growth no more
  than 122 MiB, absolute Metal allocation no more than 11,883,921,408 bytes,
  and release of all seven residents.
- On successful completion, the manifest records source-identity stability,
  per-arm series resident bytes, and candidate aggregate growth explicitly. It
  does not claim `series_resident_bytes` is unchanged.

The workload and settings otherwise match the corrected packet-groups protocol:
seven distinct full uint16 sources, packet-groups query, compact offsets on,
trusted-table off, exact ADF 8→20, and one warmup plus 20 measured cycles per
A/B/A arm.

## Measured result

| Arm | ADF 8→20 p50 | ADF 8→20 p95 | Metal allocation |
|---|---:|---:|---:|
| A1 packet-owner2 | 70.75 ms | 74.47 ms | 11,636,195,328 B |
| B adaptive-partials | 73.67 ms | 76.94 ms | 11,760,975,872 B |
| A2 packet-owner2 | 71.63 ms | 73.41 ms | 11,760,975,872 B |

All measured full maps matched exactly across all seven sources and both masks;
all residents were released. Candidate partial scratch added 124,780,544 B
(about 119 MiB), remaining under both the 122 MiB growth allowance and the
11,883,921,408 B absolute Metal cap. The candidate p50 was 3.5% slower than the
mean of the two controls, and its p95 was about 4.1% slower. The extra partial
clear/write/read traffic and finish reduction outweighed the additional
threadgroup parallelism here; each stream still has the same dependent ANS
decode chain. Do not promote this kernel on the basis of this workload.

This is a result for the compatible packet-groups query with trusted-table off.
It does not measure adaptive-partials composed with the current best
scan512-plus-trusted-table path, which the present compatibility guards reject.

## Run

```sh
swift build -c release --disable-sandbox --product metal-paired-runtime-tans-series-benchmark
python3 experiments/20260913-apple-m5-ans-adaptive-partials-packet-groups-bounded-scratch/run.py \
  --exe .build/release/metal-paired-runtime-tans-series-benchmark \
  --folder ~/data/maped-seven-tilts \
  --cache /tmp/ans-adaptive-partials-packet-groups-bounded-cache-20260913 \
  --out experiments/20260913-apple-m5-ans-adaptive-partials-packet-groups-bounded-scratch/results
```

This isolated wrapper imports the packet-groups experiment harness, relaxes
only its incorrect aggregate-resident equality check, and retains the imported
packet-groups wrapper and its underlying adaptive-partials runner as
fingerprinted dependencies.
