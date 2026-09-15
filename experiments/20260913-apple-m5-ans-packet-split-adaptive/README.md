# Exact packet-split A/B/A on seven sources

Status: planned. This experiment is prepared only; no Metal benchmark was run.

## Hypothesis

Does packet split 4 improve the exact seven-source indexed `adf-center-8` →
`adf-center-20` update versus packet split 1, with packet-owner2 and the
packet-groups query held fixed, while preserving maps and the existing memory
ceiling?

## Fixed protocol

- Seven distinct original full `(512, 512, 192, 192)` `uint16` sources from
  `~/data/maped-seven-tilts`; no crop, binning, clipping, count narrowing, or
  source substitution.
- Same process/residents throughout. Packet-owner2 kernel, indexed
  packet-groups query, compact offsets on, trusted-table off, two streams per
  lane, unbatched submission, concurrency 7, and no profiling.
- Arms: A1 packet split 1, B packet split 4, A2 packet split 1. The runner
  starts with `QGPU_PAIRED_RUNTIME_PREPARE_PACKET_SPLITS=1`, sets the per-arm
  packet split in each request, and prepares both variants before timing.
- Each arm gets one unmeasured warmup and 20 measured cycles of
  `adf-center-8` followed by `adf-center-20`.
- Require exact full-map hashes for every source/mask/cycle, unchanged source
  identities and resident bytes, Metal allocation below
  `11,883,921,408` bytes and candidate growth no more than 122 MiB, stable
  allocation after the B warmup, and explicit release of all seven residents.
- Timing includes unprofiled indexed query planning, preparation, submission,
  GPU completion, and readback; excludes initial loading and UI presentation.

## Run

```sh
swift build -c release --disable-sandbox --product metal-paired-runtime-tans-series-benchmark
python3 experiments/20260913-apple-m5-ans-packet-split-adaptive/run.py \
  --exe .build/release/metal-paired-runtime-tans-series-benchmark \
  --folder ~/data/maped-seven-tilts \
  --cache /tmp/ans-packet-split-adaptive-cache-20260913 \
  --out experiments/20260913-apple-m5-ans-packet-split-adaptive/results
```

The wrapper imports the established packet-groups A/B/A runner, fingerprints
that imported runner as a dependency, and changes only the experiment identity,
manifest path, and packet-split setting per arm.
