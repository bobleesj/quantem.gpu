# Full-scan native Metal fitting on Apple M5

2026-11-05. This measurement does **not** meet a two-second end-to-end target.

## Workload

Public ARINA gold, 512 × 512 scan positions and 192 × 192 detector pixels,
on Apple M5 (10 GPU cores, 24 GiB unified memory). All scan positions and
7,644 selected bright-field pixels were retained. The existing median-fill
policy repaired four detector marker pixels. No binning or precision reduction
was introduced. Calibration: 300 kV, 30 mrad, 0.5 Å scan sampling, and
0.46 mrad detector sampling for the native benchmark.

Native sources match `ceec1d9715e10fcbec276971b27163b6ce0c4055` with the
typed central-difference compiler fix in `ddbe7a1`. Python transfer parity used
a preserved working snapshot; only the transfer implementation changed between
its matched before/after runs.

The maintained `metal-ssb-fit-benchmark` ran three fits with 200 trials,
Newton refinement, fifth-order refinement, and the default half-band check.
All three returned identical fitted optics, tilt, depth, and higher-order
coefficients. This is repeatability evidence, not an independent calibration
or cross-backend correctness claim.

| Stage (seconds) | First observation | Repeat 1 | Repeat 2 |
|---|---:|---:|---:|
| Fourier preparation | 3.749 | 22.461 | 10.120 |
| Joint-model preparation | 3.944 | 10.040 | 3.294 |
| Initial fit | 9.832 | 8.887 | 8.699 |
| Fit through fifth order, including initial fit | 16.766 | 15.578 | 15.390 |

Original HDF5 loading was 6.9 s. Creating the reusable benchmark fixture took
28.3 s including that load and is separate from interactive fitting. Source
pages were not cold-cache controlled. System memory pressure affected the
preparation measurements; these are active-desktop observations, not an idle
hardware limit. The runner's current top-level `load_s` includes subsequent
work; the separated numbers above use its stage records and load log instead.

## Candidate evaluation experiment

The maintained objective benchmark evaluated identical fixed candidate sets,
three repetitions each. Reducing candidates per Metal thread from four to two
preserved every recorded agreement exactly, but made low-order batches slower:

| Batch | Original GPU ms/candidate | Two candidates/thread |
|---|---:|---:|
| 8 candidates | 26.201 | 33.024 |
| 64 candidates | 26.195 | 34.354 |

The candidate was rejected. Trial counts, half-band checks, fit bounds and
precision remain unchanged. No fitting speedup is claimed.

## Validated fixes and remaining gates

- Torch-to-MLX transfer now blits from the Metal buffer handle without requiring
  host mapping of private storage. Contiguous, offset and transposed MPS tensors
  passed exact transfer tests. Six matched real-data Phase reconstructions
  (256 × 256 × 128 × 128, tilt and no tilt) matched the original phase hashes and
  fitted parameters exactly.
- Splitting a central-difference expression into typed Swift intermediates
  resolves a compiler type-check failure without changing its arithmetic.
  Three physical-Metal fifth-order gradient/search tests passed.
- The Python `benchmark-ssb` command still rejects the current staged optimizer
  because its guard expects the earlier GPU-only Newton metadata. Its failed
  invocation is not a qualifying timing or a CPU-fallback measurement.
- Native application acceptance and source-to-visible timings must be reported
  separately. Kernel fitting does not certify a complete UI workflow or a
  release. The two-second full-size target remains open.

## Native application observations

The native app loaded the full 512 × 512 × 192 × 192 acquisition with scan and
detector binning both 1. Two successful load observations reached confirmed
resident presentation in 9.764 s and 4.525 s; source-cache state was uncontrolled.
Other attempts were cancelled by system memory-pressure warnings. A local UI
fix replaces the stale loading percentage with an explicit stopped/retry state.

The app's BF-edge calibration measured 0.6035446 mrad/pixel and selected 7,647
BF pixels. These differ from the benchmark settings above, so the following
times are **not** a matched overhead comparison. Source calibration was not
rewritten. The app reported an initial fit in 33.231 s and a separately requested
fifth-order refinement in 41.488 s, publishing the refined image state at
34.993 s before its checks completed. The final state contained all 200 primary
trials, order 5, a 2048 × 2048 Auto output, matching requested/shown revisions,
and a trusted half-band result (0.23 rad). Endpoint Metal allocations were
14.122 GB; this is not a peak-memory measurement.

Native actions used opt-in test hooks. Screenshot permission was unavailable,
so visible-window acceptance remains pending. The app currently requires the
separate fifth-order refinement action. No end-to-end two-second claim, native
release qualification, or automatic-fifth-order UI claim follows from this test.

A replay with the corrected native UI harness completed both stages. Resident
presentation took 5.134 s. Initial Run-to-result was 19.502 s including automation
waits; app timers reported 3.883 s preparation and 14.883 s fitting. The separate
fifth-order image state was published at 2.122 s and its checks completed at
6.047 s (7.029 s including automation waits). Run through both stages was
26.650 s, excluding acquisition loading and calibration setup. Final phase hash
and fitted coefficients matched the earlier app run exactly. These warmer
observations are not an optimization speedup.

The corrected harness reports `workflow_pass=true`, `visual_evidence_pass=false`,
and overall `pass=false`: all three screenshots were denied by macOS. Metal
allocations sampled after the initial fit were 17.999 GB, falling to 14.122 GB
after refinement. Neither sample is a measured peak; this does not qualify a
16 GiB memory limit.

Private evidence ID: `mps-full-scan-20261105`.
