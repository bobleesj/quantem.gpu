# Adaptive residual partials with packet-groups query: seven-source A/B/A

Status: planned. This is a corrected, separate experiment record. No benchmark
was run while creating it. The earlier scan512 candidate experiment is
preserved unchanged; scan512 requires `packet-owner2` and cannot time the
`adaptive-partials` candidate.

## Question

Does the existing `adaptive-partials` detector kernel improve the exact,
all-seven `adf-center-8` → `adf-center-20` update relative to `packet-owner2`,
within the established Metal allocation ceiling, when both kernels use the
compatible `packet-groups` query variant?

## Frozen protocol

- Seven distinct original full `(512, 512, 192, 192)` `uint16` sources from
  `~/data/maped-seven-tilts`; no crop, binning, clipping, count narrowing, or
  synthesized/renamed acquisitions.
- Same process and resident objects for A1/B/A2. Indexed polar query uses
  `packet-groups` throughout. Compact offsets are enabled on every arm.
- Other detector settings are fixed: 2 streams/lane, one packet split,
  unbatched, seven-way concurrency, no profiling, and trusted-table disabled
  uniformly. Trusted-table is incompatible with `adaptive-partials` in the
  current implementation; this experiment does not combine them.
- Arms: A1 `packet-owner2`, B `adaptive-partials` with 32 maximum partial
  groups, A2 `packet-owner2`. Each arm gets one unmeasured warmup followed by
  20 measured cycles of `adf-center-8` then `adf-center-20`.
- Require exact full-map hashes for all seven sources and both masks, unchanged
  source identities/resident bytes, and release of all seven residents.
- Ready allocation is the compact-offset baseline. Candidate allocation may
  grow by at most `122 MiB` above that baseline and may never exceed
  `11,883,921,408` bytes. After B's warmup allocates lazy scratch, B's measured
  response and A2 must report the same allocation.
- This is unprofiled detector-update latency, including planning, preparation,
  submission, wait, and readback. It does not measure loading, UI presentation,
  or display refresh rate.

## Run

```sh
swift build -c release --disable-sandbox --product metal-paired-runtime-tans-series-benchmark
python3 experiments/20260913-apple-m5-ans-adaptive-partials-packet-groups/run.py \
  --exe .build/release/metal-paired-runtime-tans-series-benchmark \
  --folder ~/data/maped-seven-tilts \
  --cache /tmp/ans-adaptive-partials-packet-groups-cache-20260913 \
  --out experiments/20260913-apple-m5-ans-adaptive-partials-packet-groups/results
```

The isolated wrapper reuses the A/B/A resident-loop implementation and
validators from `../20260913-apple-m5-ans-adaptive-partials/run.py`, overriding
the experiment identity, output manifest, query configuration, and request
variant. The imported runner is included as a fingerprinted source dependency.
