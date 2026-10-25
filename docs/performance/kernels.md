# GPU kernel registry

One row per kernel version. A result records the kernel that produced it
(`MetalKernelIdentity`: name, version, source digest, pipeline variant, and a
fingerprint over both), so any saved phase, export, or benchmark line can be
traced to a row here.

## Rules

- **One production kernel per job.** When a faster version is validated, it
  replaces the older one; the older one is retired, not kept as a silent
  fallback. Any case the new kernel cannot serve is named below and in code.
- **Version on change.** Editing a kernel's `.metal` source or its numerical
  contract requires a new version here and in `SSBKernelRegistry`.
  `SSBKernelRegistryTests` fails when a shipped source digest does not match
  its registered version, or when this page lacks the version or digest.
- **Active means measured.** A version becomes active only with parity tests
  passing and both benchmark and in-app headed timings recorded with date,
  device, and dataset. Retired versions stay listed with their last numbers.

Timings are median / p90 ms per reconstruction. Device for all rows: MacBook
Pro, Apple M5 Max, 40-core GPU, 128 GB, macOS 26.4. Data: the first acquisition,
512 × 512 × 192 × 192, 8,806 bright-field pixels, 300 keV, 30 mrad; 256²
rows use the same acquisition's full detector at a 256² scan
(`docs/performance/ssb/reports/2026-10-24-native-metal-ssb-redraw.md`).

## Native SSB reconstruction (`MetalSSBKernels`)

| Kernel | Status | Source digest | Serves | Validated | Fifth order, 512², 3× | Low order, 512², 3× | Fifth order, 256², 3× | Parity | Replaced by | Retired |
|---|---|---|---|---|---|---|---|---|---|---|
| `ssb-redraw` v0.2 | **active** | `b20d7914811d` | `phase_of_mean`, `complex_wave`, 1–4× | 2026-10-24 | benchmark 33.7 / 33.9; in app 30.3–33.0 (median ≈ 31.3, ≈ 32 FPS) | benchmark 31.2 / 31.5; in app 32.5–34.9 | benchmark 9.9 / 10.0 | `SSBRedrawReconstructionTests` (per-term and double-precision fifth-order references), `SSBComplexWaveTests` | — | — |
| `ssb-per-term` v0.1 | **retired** (test reference; explicit 8× only) | `1c390cc7b833` | 1–4× before 2026-10-24; 8× now | before 2026-10-24 | benchmark 214.9 / 223.8; in app ≈ 320 | benchmark 60.4 / 64.3 | benchmark 67.9 / 69.3 | reference for v0.2 | `ssb-redraw` v0.2 | 2026-10-24 |
| `ssb-mean-phase` v0.1 | **active** (`mean_phase`, C10/C12 only) | `4aa7c1da6391` | `mean_phase` | before 2026-10-24 | n/a (no higher orders) | not remeasured | not remeasured | existing joint tests | — | — |

The first render after a fit costs about 100–130 ms in the app (pipeline
variant compilation and warm-up); the steady drag rate above excludes it.
In-app timings are `MetalSSBJointPhase.wallSeconds` on an otherwise idle GPU
with one app instance on the built-in display.

## Native SSB fit (`MetalSSBKernels`)

The primary (low-order) fit's trust-region Newton climbs on the exact gradient and Hessian of the agreement
in C10, C12 (two Cartesian components), the sample tilt and the depth spread. One evaluation is two passes
over every (band frequency, bright-field pixel): the objective's sums, then, with each band frequency's
model sum known, the 51 k-sums its first and second derivatives need (`ssb_low_order.metal`). It replaces
the finite-difference stencil of 13 or 28 objective evaluations per Newton iteration.

| Kernel | Status | Source digest | Serves | Validated | One evaluation, film 512² (GPU) | Parity |
|---|---|---|---|---|---|---|
| `ssb-low-order-derivatives` v0.1 | **active** | `40da0bd64def` | primary-fit Newton (value, gradient, Hessian) | 2026-10-25 | pass 1 ≈ 16 ms + pass 2 ≈ 50 ms (≈ 7 objective evaluations; the stencil it replaces is 13–28) | `SSBModelReconstructionTests.testLowOrderDerivativesMatchFiniteDifferences` (gradient and Hessian against Richardson differences, ≤ 4e-4 relative; value against the agreement objective ≤ 1e-6) |

## Native SSB fifth-order gradient (`MetalSSBKernels`)

Timings: Apple M5 Max, the first acquisition kept branch (8,805 BF pixels, 44,247 band frequencies), wall time per
call of one candidate, host to GPU to host. The fifth-order objective `fifth_agreement` (in `ssb_fifth_order`,
registered with `ssb-per-term`) is 25.0 ms single / 22.0 ms each in a batch of 28.

| Kernel | Status | Source digest | Serves | Validated | Value + gradient, one candidate | Parity | Replaced by | Retired |
|---|---|---|---|---|---|---|---|---|
| `ssb-fifth-gradient` v0.1 | **active** (`SSBModelFit.HigherOrderSearch.exactGradient*`) | `dc8d80c38d16` | exact gradient of the fifth-order agreement (28 z: coefficients, depth, tilt); BF- or band-subsampled agreement for steering | 2026-10-25 | 66.2 ms (objective pass + gradient pass ≈ 2.65 objectives); gradient pass alone ≈ 41 ms | `SSBFifthOrderGradientTests` (double-precision central differences, worst 3.6e-6 of max\|g\|; value pass bit-identical to `fifth_agreement`) | — | — |

## Fit objectives (`MetalSSBJointModel` agreement)

GPU ms per candidate objective (median of three calls; command-buffer GPU time, which the wall time of the
call exceeds by at most 0.3 ms) at a fit's kept rotation. Fifth order: one candidate, four seeded random
candidates, the 57-point Newton diagonal stencil, 64 random candidates. Low order: one candidate, eight (one
search-trial batch) and 64 random candidates. Data: the first acquisition (8,805 BF pixels, 44,247 band
frequencies), a second 512² acquisition (9,296 BF, 44,255) and a third (8,928 BF, 77,842), all 300 keV,
30 mrad. Benchmark: `metal-ssb-fit-benchmark --objective-bench RUN.json`, 2026-10-25.

| Kernel | Status | Source digest | Serves | Validated | film | Second | Third | Parity | Replaced by | Retired |
|---|---|---|---|---|---|---|---|---|---|---|
| `ssb-fifth-agreement` v0.2 | **active** | `f2f700147e47` | `fifthOrderAgreements` (fifth-order fit, stencil Newton, gradient check) | 2026-10-25 | 11.0 / 6.6 / 5.6 / 6.6 | 11.1 / 7.2 / 5.8 / 6.9 | 18.4 / 11.5 / 9.8 / 12.0 | `SSBFifthOrderNumericsTests` (v0.1 and the float64 objective); measured: within 3.1e-7 of v0.1 on all three acquisitions | — | — |
| `ssb-fifth-agreement` v0.1 (`fifth_agreement` in `ssb-per-term`) | **retired** (test reference) | `1c390cc7b833` | before 2026-10-25 | before 2026-10-25 | 24.4 / 19.6 / 20.8 / 21.2 | 25.4 / 21.4 / 23.3 / 22.9 | 41.3 / 36.6 / 37.9 / 37.5 | reference for v0.2 | `ssb-fifth-agreement` v0.2 | 2026-10-25 |
| `ssb-joint-agreement` v0.2 | **active** | `d698aa9cb17a` | `agreements` with `.angleSums` (low-order search trials, branch scores) | 2026-10-25 | 8.7 / 5.1 / 5.1 | 10.2 / 5.9 / 5.8 | 16.3 / 9.5 / 9.8 | `SSBFifthOrderNumericsTests.testLowOrderAngleSumsMatchReference`; measured: within 1.3e-8 of v0.1 | — | — |
| `ssb-joint-agreement` v0.1 (`joint_objective_angle_sums` in `ssb_joint`) | **retired** (test reference) | `4aa7c1da6391` (with `ssb-mean-phase`) | before 2026-10-25 | before 2026-10-25 | 35.3 / 8.9 / 9.4 | 41.1 / 10.3 / 10.9 | 65.3 / 18.2 / 17.8 | reference for v0.2 | `ssb-joint-agreement` v0.2 | 2026-10-25 |

Both v0.2 kernels evaluate the per-pair float32 terms of v0.1:

- The soft aperture of `joint_geometry_xy` is exactly 1 inside and 0 outside conservatively rounded radii; the
  full expression runs only for SIMD groups with a lane in the one-pixel edge ring (a one-trip `simd_ballot`
  loop, since the compiler if-converts plain uniform branches). Divisions use the correctly rounded reciprocal
  with a Markstein correction, which reproduces IEEE division. The edge, alpha^2 and the depth-sinc arguments
  are bitwise those of v0.1 on every (k, q) pair of film and of the third acquisition.
- `fast::sin` / `fast::sincos` equal `precise::` with fast math off except for 1-ulp differences in about one
  argument in 1e7, at less than half the cost.
- A threadgroup computes one to four candidates per thread with no padding candidates.

The fifth-order kernel also reassociates chi(k -/+ q), its only change of rounding: a threadgroup owns band
frequencies of at most two band rows and collapses the bivariate chi polynomial (built in double precision from
the same packed float32 terms) once per BF pixel and row, so each sideband is one degree-6 Horner evaluation
instead of the 14-term harmonic sum. Candidates with equal depth and tilt share depth-sinc weights. On the
synthetic test scene both fifth-order kernels are about 1e-6 from the float64 objective and v0.2 is no less
accurate.
