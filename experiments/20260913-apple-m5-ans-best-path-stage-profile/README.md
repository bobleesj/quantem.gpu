# Best-path seven-source ANS stage profile

Status: preregistered. Diagnostic only; timestamp profiling is not a release
latency measurement and does not itself optimize the kernel.

## Question

On the current best measured composition (scan512 polar query + trusted-table
decoder), which stage remains the larger GPU interval for the exact seven-source
ADF 8→20 update: polar/index computation or residual decoding?

The earlier valid profile used the packet-groups query path and cannot be used
to infer the best composition's stage balance. This run profiles the A/B/A
arms with identical timestamp instrumentation. A1/A2 are the packet-groups
controls; B is scan512 with trusted-table decoding enabled.

The first attempt loaded all seven residents, produced an exact A1 warmup map,
and released every source, but the inherited validator expected profiling on
the control arm. The runtime correctly forces profiling off for that arm, so
the harness rejected the configuration. No stage timing is accepted from this
failed run; the separately registered retry profiles only the candidate arm.

## Fixed workload and gates

- Apple M5, seven unique complete uint16 `(512, 512, 192, 192)` acquisitions.
- No crop, bin, clip, count conversion, or per-source reduction.
- Exact full-map hashes for both ADF masks on all seven acquisitions.
- Identical resident and Metal-allocation ceilings to the existing composition
  test; all seven sources must be explicitly released.
- `QGPU_PAIRED_RUNTIME_PROFILE=1` before source construction and per-update
  profiling enabled for the full A/B/A diagnostic.
- One warmup plus 20 cycles per arm. Report profiled intervals only; compare
  unprofiled latency only to the separate validated A/B/A record.

## Run

```sh
swift build -c release --disable-sandbox --product metal-paired-runtime-tans-series-benchmark
python3 experiments/20260913-apple-m5-ans-best-path-stage-profile/run.py \
  --exe .build/release/metal-paired-runtime-tans-series-benchmark \
  --folder ~/data/maped-seven-tilts \
  --cache /tmp/ans-best-stage-profile-cache-20260913-1 \
  --out experiments/20260913-apple-m5-ans-best-path-stage-profile/results
```

The cache stores preparation metadata, not converted detector data. Profiled
wall times are diagnostic and must not replace unprofiled A/B/A measurements.

## Interpretation gate

Require valid timestamp records for every measured target source-update. Compute
the union of encoder intervals across the seven source commands; do not add
overlapping per-source stage medians. This profile determines whether the next
kernel experiment should attack scan512 polar/index work or the dependent ANS
residual chain. It does not prove occupancy or memory-bandwidth saturation.
