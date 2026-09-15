# Packet-owner2 split-only A/B/A on seven full sources

Status: planned. This is a corrected, separate experiment. The preceding
packet-split harness remains preserved as a failed harness record: its B arm
inadvertently used `adaptive-partials`, so that result is not valid evidence
for packet split alone.

## Hypothesis

Does packet-owner2 with packet split 4 improve the exact seven-source indexed
`adf-center-8` → `adf-center-20` update compared with split 1, with the
packet-groups query and all other settings held fixed?

## Fixed protocol

- Seven distinct original full `(512, 512, 192, 192)` `uint16` sources from
  `~/data/maped-seven-tilts`; no crop, binning, clipping, count narrowing, or
  source substitution.
- Same process/residents throughout. Every A/B/A request explicitly forces
  `packet-owner2`; only `packet_splits` changes 1/4/1. The expected response
  configuration is independently validated as packet-owner2 with that split.
- Indexed packet-groups query, compact offsets on, trusted-table off, two
  streams per lane, unbatched submission, concurrency 7, no profiling, and
  other detector options fixed.
- `QGPU_PAIRED_RUNTIME_PREPARE_PACKET_SPLITS=1` is set before launch; each
  query request selects its own packet split. Split pipelines are prepared
  before measurements.
- One unmeasured warmup and 20 measured cycles per arm, each cycling
  `adf-center-8` then `adf-center-20`.
- Require exact full-map hashes for every source/mask/cycle, stable source
  identities and series resident bytes, Metal allocation below
  `11,883,921,408` bytes and candidate growth no more than 122 MiB, stable
  allocation after B warmup, and release of all seven residents.
- Timing excludes initial source loading and UI presentation.

## Run

```sh
swift build -c release --disable-sandbox --product metal-paired-runtime-tans-series-benchmark
python3 experiments/20260913-apple-m5-ans-packet-split-packet-owner2/run.py \
  --exe .build/release/metal-paired-runtime-tans-series-benchmark \
  --folder ~/data/maped-seven-tilts \
  --cache /tmp/ans-packet-split-packet-owner2-cache-20260913 \
  --out experiments/20260913-apple-m5-ans-packet-split-packet-owner2/results
```

The runner imports the prior packet-split wrapper to reuse its experiment
orchestration, then overrides both request and expected configuration so every
arm uses packet-owner2. The imported wrapper and its packet-groups/adaptive
runner dependencies are fingerprinted as source lineage.
