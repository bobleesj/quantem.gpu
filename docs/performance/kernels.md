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
