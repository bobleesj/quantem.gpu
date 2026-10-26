# MPS SSB execution and coherent reconstruction, 2026-10-17

> **Historical:** describes the code as of 2026-10-17; the phase-variance search (`objective="phase_variance"`) and its Nelder–Mead refinement are retired. See the [current fit methods](../../../api/direct_ptycho.md).

The Python MPS path uses four-candidate GPU execution tiles for the joint
objective and a guarded half-spectrum correction for fine-grid coherent
reconstruction. The search still proposes eight candidates at a time. Trial
order, bright-field reduction order, fit budget, calibration, precision and
the `phase_of_mean` estimator are unchanged.

## What changed

- Four-candidate execution tiles were faster than eight on the measured
  device. This changes dispatch geometry, not the optimizer's proposals.
  Tests compare every objective value, including a partial tile, exactly.
- The fine-grid correction reuses the symmetry established in native Metal
  commit `82a10eca`: for real source counts and the even C10/C12 probe,
  the non-DC corrected spectrum is anti-Hermitian. The MPS path computes its
  unique columns, retains the real DC, and restores paired columns on GPU
  before the existing complex inverse FFT.
- The half-spectrum path is enabled only when both Nyquist boundaries lie
  strictly outside the complete soft-aperture support. Higher-order optics,
  native sampling and unsupported geometry retain the full-grid path.
  Bright-field chunk sizes and summation order remain unchanged.

The native renderer averages individual phases; the current Python default
averages corrected waves before taking phase. Transferring the native
per-detector FFT sequence would change the estimator. This change transfers
its symmetry argument while retaining the Python estimator. Existing fused
objective kernels, Fourier storage and probe-geometry reuse are retained.
An additional queued-dispatch experiment gave no consistent benefit and was
not included.

## Fresh-process measurements

Baseline: `aed2219ccf8e0befc3aeba343862391fe4fed8c4`.
Device: Apple M5 Max, 40 GPU cores, 128 GiB unified memory. Each acquisition
retains the complete 512 × 512 × 192 × 192 uint16 source. A retains 8,807
bright-field pixels and returns 1536² Auto phase; B retains 8,930 and returns
2048². There is no crop, detector binning, BF subsampling or dtype change.

The joint workflow runs 200 trials plus Newton, including its existing
200-trial zero-depth comparison. Calibration and rotation are frozen for
the numerical comparison. This is not an independent ordinary fit followed
by a second joint workflow.

| Load through saved Auto phase | Baseline p50 / p95 / max | Optimized p50 / p95 / max |
| --- | ---: | ---: |
| A | 15.01 / 18.24 / 18.50 s | 12.32 / 12.82 / 12.90 s |
| B | 23.53 / 27.24 / 28.26 s | 18.96 / 19.70 / 19.70 s |

There are six observations per acquisition and version: five fresh processes
followed by one counterbalanced control per version. The controls alone
reduced wall time by 14% and 13%, respectively. Small-sample p95 values are
descriptive, not latency guarantees. Operating-system and shader caches were
not cleared; source, session, Fourier preparation and output were fresh.
Interpreter startup, later parity checks and session teardown are outside
the load-to-save timer. GPU stage boundaries synchronize completion.

| Stage, median seconds | A baseline | A optimized | B baseline | B optimized |
| --- | ---: | ---: | ---: | ---: |
| Source loading | 2.32 | 2.21 | 2.39 | 2.26 |
| BF backend selection | 0.109 | 0.107 | 0.105 | 0.105 |
| Fourier preparation | 1.03 | 0.84 | 0.98 | 0.88 |
| Fit and native phase/diagnostic | 9.09 | 7.28 | 16.18 | 13.20 |
| Auto coherent reconstruction | 1.23 | 0.64 | 2.52 | 1.27 |
| Array/metadata save | 0.007 | 0.007 | 0.012 | 0.012 |

Session construction and orchestration contribute another approximately
1.2 s. Instrumented controls measured construction at 1.17–1.23 s; this is
included in total wall time. Stage medians are independent and do not sum to
the median total. Loading and preparation code did not change; their timing
differences should not be credited to these kernels.

Peak MLX allocation was 18.47 GiB for A and 18.95 GiB for B. The sampled
Metal-driver high water was 57.67/59.09 GiB, with a 50 ms sampling interval.
These overlapping accounting measures must not be added. Process peak RSS
was approximately 3.00/1.50 GiB. Overall peaks are unchanged: the reduced
correction workspace is smaller than the loading/preparation high water.

## Numerical checks

All complete arrays were compared against frozen MPS GPU outputs. Native
phases, fitted coefficients, objective values and trial records are bit-exact.
Fine-grid reconstruction retains the existing absolute tolerance of 2e-5 rad
with zero relative tolerance; it is not advertised as bit-exact.

| Complete Auto phase | A maximum difference | B maximum difference |
| --- | ---: | ---: |
| Ordinary, identical fixed optics | 1.49e-7 rad | 1.19e-7 rad |
| Joint tilt/depth | 1.79e-7 rad | 9.24e-7 rad |

The ordinary controls disable tilt/depth on the same fixed optics; they are
separate reconstruction checks. Eight higher-order full-array checks cover
both acquisitions, both correction models and native/Auto sampling. All are
bit-exact against the previously frozen GPU results because the higher-order
fallback is unchanged.

The focused hardware/contract run passed 69 tests; two optional simulation
fixture tests were skipped. The legacy native trajectory gate still lacks
its external fixture and is not claimed as passed. This Python MPS change
does not qualify CUDA, native Swift, another device or a release.

## Consuming application

QuantEM.Live `16cc9a1e` ran the changed backend through its folder watcher,
automatic calibration, 200-trial joint search with Newton, native saves,
Auto reconstruction and first HTTP image delivery. Each version processed
three fresh arrivals per acquisition through a persistent server/worker.
These separate measurements include detection and application overhead;
server startup is excluded. No browser interaction ran during this timing
comparison.

| Arrival through first served Auto image | Baseline p50 / p95 / max | Optimized p50 / p95 / max |
| --- | ---: | ---: |
| A | 15.63 / 17.24 / 17.42 s | 14.19 / 16.20 / 16.43 s |
| B | 24.76 / 25.11 / 25.15 s | 22.37 / 23.32 / 23.42 s |

All six paired application results retained identical fitted parameters,
loss and complete native phases. Auto differences were at most 1.79e-7 rad
for A and 9.02e-7 rad for B. The original acquisition files remained unchanged.
A separate interactive run confirmed that the dashboard displayed A's saved
Auto phase and FFT with the expected joint-model metadata. Application
latency and the direct fixed-calibration benchmark have different boundaries
and must not be compared as the same operation.

Separate ordinary-mode application checks passed for both acquisitions. They
use the existing phase-variance search with 200 trials plus Nelder-Mead and
took 39.83/40.37 s from arrival through the first Auto image (one observation
each). The joint-search timing improvement is not a claim about ordinary
phase-variance fitting.

Evidence identifier: `20261017-mps-ssb-fit-dispatch`. Inputs, full arrays,
timing records, harnesses and application screenshots remain private.
