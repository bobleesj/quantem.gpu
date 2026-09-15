# Normal indexed seven-source Metal stage profile

Status: complete; per-encoder timestamp profiling is valid. This is a diagnostic rerun correcting the profiler lifecycle
in `20260913-apple-m5-ans-indexed-batch-profile`: `QGPU_PAIRED_RUNTIME_PROFILE`
must be present when resident sources are constructed. Stage timing is
diagnostic and may perturb scheduling; only the unprofiled phase is suitable
for user-facing backend timing.

## Question

Can the ordinary indexed seven-source `adf-center-8` → `adf-center-20`
transition produce valid timestamp-counter intervals for its existing Metal
encoders, with exact full-map parity and no resident-byte change?

## Fixed workload and gates

- Seven distinct original full `uint16` acquisitions, each
  `512×512×192×192`; no cropping, binning, clipping, or count changes.
- Load all seven once and retain them for all A/B/A measurements.
- Use the existing normal indexed benchmark harness, not a synthetic or
  stage-isolation route. The resident profiler is enabled at child-process
  launch, before source creation.
- Require exact complete-map parity for both masks on all seven sources,
  unique source identities, unchanged resident/allocation bytes, and explicit
  `stage_profile_valid == 1` on sampled target updates.
- Keep the profiler-enabled measurements separate from the unprofiled A/B/A
  timing. Do not interpret timestamp-sampling timings as release performance.

## Run

```sh
QGPU_PAIRED_RUNTIME_PROFILE=1 python3 \
  experiments/20260913-apple-m5-ans-indexed-batch-profile/run.py \
  --exe .build/release/metal-paired-runtime-tans-series-benchmark \
  --folder ~/data/maped-seven-tilts \
  --cache /tmp/ans-stage-profile-normal-cache \
  --out experiments/20260913-apple-m5-ans-stage-profile-normal/results
python3 experiments/20260913-apple-m5-ans-indexed-batch-profile/summarize.py \
  experiments/20260913-apple-m5-ans-stage-profile-normal/results/summaries.json
```

The fresh cache stores metadata/index preparation, not converted detector
data. This is a warm-resident detector-update measurement, not cold I/O or UI
FPS. The unmodified harness is reused and its SHA is recorded in the manifest.

## Interpretation gate

Report encoder-stage medians only when valid timestamp intervals exist. The
source emits per-encoder fields such as `stage_<encoder>_milliseconds` and
`gpu_<encoder>_<occurrence>_{start,end}_seconds`; their names come directly
from the existing encoder labels. If the profiler is unsupported or invalid,
mark this diagnostic failed and do not infer index-versus-residual time from
command-GPU durations.

## Result

All six A/B/A arms passed full-map parity; each retained seven unique source
identities and the same resident bytes (**11,877,814,048**) and sampled Metal
allocation (**11,883,921,408**). Every one of the 420 target source-updates in
the profiled A/B/A phase reported `stage_profile_valid == 1`; `polar` and
`residual` timestamps were present. All sources were released cleanly.

Unprofiled all-seven wall medians / p95s were **71.01 / 73.81 ms** (A1
concurrent), **76.84 / 79.85 ms** (B batched), and **70.76 / 75.18 ms** (A2
concurrent). A1/A2 differ by only 0.26 ms; this confirms no new speedup and
repeats the batch regression. Profiler-enabled wall times are diagnostic, not
release timings.

| Profiled A/B/A arm | Seven-source polar encoder interval union, median | Seven-source residual interval union, median | Combined encoder interval union, median | Seven-command GPU interval hull, median |
|---|---:|---:|---:|---:|
| A1 concurrent | 40.58 ms | 50.74 ms | 68.40 ms | 68.46 ms |
| B batched | 32.02 ms | 48.18 ms | 68.79 ms | 68.91 ms |
| A2 concurrent | 35.37 ms | 49.79 ms | 67.01 ms | 67.08 ms |

Each union combines timestamp intervals across the seven independent source
commands for one update cycle. Polar and residual intervals overlap across
sources, so their medians **must not be added**. The combined encoder union
closely covers the whole GPU command hull; the residual stage is the largest
individual union, but neither this nor the timestamps establish occupancy,
bandwidth saturation, or why the hardware cannot complete faster. Per-source
stage medians are around 6.4–7.1 ms for polar and 9.2–10.6 ms for residual;
these overlap across sources and are not seven-source wall-time contributions.

The result narrows the next work: keep the concurrent update path; do not
repeat the failed batch wrapper. A true single-dispatch multi-source kernel
would be a new scheduling architecture, not the measured batch API, and still
needs an exact A/B/A parity/performance test. The current 20–30 ms target
remains unmet; this experiment itself did not optimize production code.
