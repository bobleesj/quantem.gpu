# Indexed seven-source batch and profile test

Status: complete; batching hypothesis refuted. This is a diagnostic before
kernel changes: it compares the
normal indexed detector update API with all seven residents updated concurrently
through the batched submission API, then profiles host planning/preparation,
GPU command intervals, waits, and readback. No performance result is claimed
until the run and exact-parity checks finish.

## Question

Does batched submission improve the actual full seven-source `adf-center-8` →
`adf-center-20` update, and which measured stages account for the all-seven
return time? Previous stage-isolation experiments used the non-batched
diagnostic API; these measurements use `run` on the indexed resident path.

## Fixed workload and gates

- Seven distinct original full `uint16` acquisitions, each `512×512×192×192`.
- No clipping, cropping, binning, packing conversion, or count modification.
- All seven residents load once and stay allocated through both measurement
  phases. The same exact transition and source identities are used in A/B/A.
- Twenty measured repeats per arm after one excluded warm-up cycle; a cycle
  includes the full detector maps for both masks.
- First A/B/A phase enables timestamp/profile diagnostics; the second repeats
  unprofiled to confirm user-facing return times without profiler overhead.
- Every full detector map must match the A1 baseline bit-for-bit in all seven
  sources; resident bytes and distinct source IDs must remain unchanged.
- This measures a backend detector update, not native UI presentation or FPS.

## Run

```sh
python3 experiments/20260913-apple-m5-ans-indexed-batch-profile/run.py \
  --exe .build/release/metal-paired-runtime-tans-series-benchmark \
  --folder ~/data/maped-seven-tilts \
  --cache /tmp/ans-indexed-batch-profile-cache \
  --out experiments/20260913-apple-m5-ans-indexed-batch-profile/results
```

The cache directory is metadata/index preparation only; it is not a converted
dataset or retained resident checkpoint. Do not interpret this run as cold I/O.
The script stores raw JSONL, summaries, standard error, and an updated manifest.

## Result

Exact parity passed for every full map in six 20-cycle runs, with seven unique
source identities and unchanged resident bytes (**11,877,814,048**). Unprofiled
all-seven update timings for A/B/A were:

| Submission path | A1 median / p95 | B median / p95 | A2 median / p95 |
|---|---:|---:|---:|
| concurrent per-source commands | 70.94 / 75.84 ms | — | 70.48 / 74.92 ms |
| batched prepare/submit/wait | — | 76.80 / 81.74 ms | — |

The controls agree within 0.46 ms median. Batched submission was 5.86–6.32 ms
slower (about 8.3–9.0%); reject it for this workload. Device allocation stayed
at **11,883,921,408 bytes** in all arms.

The profiled phase gives command-level timing, not per-kernel counters. Across
the seven per-source command timestamps, the median union/span of command-GPU intervals was
68.46/68.61 ms in A1, 68.64/68.68 ms in B, and 68.59/68.67 ms in A2. That
interval covers almost the entire 70–77 ms all-seven wall return. Per-source
GPU command medians (~18.1–18.7 ms) overlap and must **not** be summed as wall
time. Median per-source CPU planning was 0.46–0.68 ms, preparation 1.05–1.78 ms,
readback 0.03–0.04 ms. The seven-source CPU preparation envelope was 2.10 ms
in each task-group control and 7.61 ms for batched submission. Batched API
preparation happens serially before submission, while task-group preparation
overlaps across sources; the wrapper also builds each effective mask serially
in the batched case. This explains its extra ~6 ms without a shorter GPU-command
envelope. Commit-to-completion intervals overlap GPU execution and are not
additive. The ordinary task-group path's wall time is largely covered by
command-GPU intervals, so the next optimization target is GPU work/scheduling,
not host readback. The command timestamps do not reveal shader occupancy,
hardware stalls, bandwidth use, or physical GPU utilization.

The intended Metal stage profiler was **not active**: this process set
`QGPU_PAIRED_RUNTIME_PROFILE` only after the resident sources were constructed,
but the profiler object is created at initialization. There are no
`stage_profile_valid` or `stage_*` fields in these samples. Do not use this run
as index-versus-residual kernel attribution; the follow-up needs the environment
flag set before launching the resident process.

The old isolated-stage benchmark is a different measurement boundary; this
normal `run` path includes indexed planning, preparation, submission, wait,
readback and exact full-map verification. Neither timing is UI FPS. The
20–30 ms goal remains unmet; a useful next experiment must remove a measured
fraction of the dependent residual decoder's GPU work, not just reorganize
host submission. No defaults or production code changed.
