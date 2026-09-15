# Adaptive residual partials: seven-source A/B/A

Status: planned. This harness is prepared for a controlled exact comparison;
no benchmark was run while creating it.

## Question

Does the existing `adaptive-partials` detector kernel improve the exact,
all-seven `adf-center-8` → `adf-center-20` update relative to `packet-owner2`,
without exceeding the established Metal allocation ceiling?

## Frozen protocol

- Seven distinct original full `(512, 512, 192, 192)` `uint16` sources from
  `~/data/maped-seven-tilts`. No crop, binning, clipping, count narrowing, or
  synthesized/renamed acquisitions.
- Same process and resident objects for A1/B/A2. Indexed polar query uses
  `scan512` throughout. Compact offsets are enabled on every arm to reclaim
  the established approximately 247.7 MB before considering candidate scratch.
- Other detector settings are fixed: 2 streams/lane, one packet split,
  unbatched, seven-way concurrency, no profiling, and trusted-table disabled
  on all arms. The current implementation explicitly rejects combining
  trusted-table with `adaptive-partials`; therefore this run does not claim or
  test that combination. No production compatibility guard is bypassed.
- Arms: A1 `packet-owner2`, B `adaptive-partials` with
  `PARTIAL_MAX_GROUPS=32`, A2 `packet-owner2`. Each arm gets one unmeasured
  warmup followed by 20 measured cycles of `adf-center-8` then
  `adf-center-20`.
- Require exact full-map hashes for all seven sources and both masks, unchanged
  source identities/resident bytes, and release of all seven residents.
- Ready allocation is the compact-offset baseline. Candidate allocation may
  grow by at most `122 MiB` above that baseline and may never exceed
  `11,883,921,408` bytes. After B's warmup allocates the lazy scratch, B's
  measured response and A2 must report the same allocation; further growth
  fails the run.
- Candidate scratch is lazy and reused by the implementation. The planning
  estimate for center-20 is about 17.8 MB/source (about 124.8 MB total); the
  harness checks observed allocation rather than assuming that estimate.
- This is unprofiled detector-update latency, including planning, preparation,
  submission, wait, and readback. It does not measure file loading, UI
  presentation, or display refresh rate.

## Run

Build the release benchmark, then run the harness with distinct cache/output
paths:

```sh
swift build -c release --disable-sandbox --product metal-paired-runtime-tans-series-benchmark
python3 experiments/20260913-apple-m5-ans-adaptive-partials/run.py \
  --exe .build/release/metal-paired-runtime-tans-series-benchmark \
  --folder ~/data/maped-seven-tilts \
  --cache /tmp/ans-adaptive-partials-cache-20260913 \
  --out experiments/20260913-apple-m5-ans-adaptive-partials/results
```

The runner imports the established scan512 resident-loop helpers, pins the
experiment-specific configuration, validates allocation growth and exact
parity, retains raw JSONL/stderr/summary artifacts, and records release
evidence. A failed attempt retains its partial records and failure details.
