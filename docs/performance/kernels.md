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
| `ssb-redraw` v0.3 | **active** (in-app headed timing pending) | `73dc413569c7` | `phase_of_mean` (method `ssb`), 1–4× | 2026-10-25 | benchmark 30.8 / 31.4 (v0.2 30.5 / 31.3, runs alternating, paired median +0.1 ms); in app not yet measured | benchmark 39.9 (v0.2 40.3, alternating under background load; minimum 29.8 vs 29.4) | benchmark 10.3 / 10.7 (v0.2 10.1 / 10.3) | `SSBRedrawReconstructionTests` (per-term and double-precision references; `testRedrawResolvesNearlyCancellingSidebands`: 3.7e-8 rad rms where v0.2 is 4.5e-3); five measured acquisitions × eight optics sets × 1×, 2×, Auto against CUDA and a float64 definition: native−definition ≤ 1.08 and native−CUDA ≤ 1.13 times CUDA's own error (v0.2: up to 2.45 / 2.53), [report](ssb/reports/2026-10-25-ssb-redraw-precision.md) | — | — |
| `ssb-redraw` v0.2 | **retired** (low-frequency precision; superseded by v0.3) | `b20d7914811d` | `phase_of_mean` (method `ssb`), 1–4× (also `complex_wave` until its retirement, 2026-10-25) | 2026-10-24 | benchmark 33.7 / 33.9; in app 30.3–33.0 (median ≈ 31.3, ≈ 32 FPS) | benchmark 31.2 / 31.5; in app 32.5–34.9 | benchmark 9.9 / 10.0 | `SSBRedrawReconstructionTests` (per-term and double-precision fifth-order references); `SSBComplexWaveTests` until 2026-10-25 | `ssb-redraw` v0.3 | 2026-10-25 |
| `ssb-per-term` v0.2 | **retired** (test reference; explicit 8× only) | `05177e0c2cde` | 8×; v0.1 without the retired `complex_wave` amplitude kernel, per-term kernels unchanged | 2026-10-25 | as v0.1 | as v0.1 | as v0.1 | reference for `ssb-redraw` v0.2 and v0.3 | `ssb-redraw` v0.2 | 2026-10-24 |
| `ssb-per-term` v0.1 | **retired** (test reference; explicit 8× only) | `1c390cc7b833` | 1–4× before 2026-10-24; 8× now | before 2026-10-24 | benchmark 214.9 / 223.8; in app ≈ 320 | benchmark 60.4 / 64.3 | benchmark 67.9 / 69.3 | reference for v0.2 | `ssb-redraw` v0.2 | 2026-10-24 |
| `ssb-mean-phase` v0.1 | **retired** (estimator removed) | `4aa7c1da6391` | `mean_phase` (C10/C12 only) | before 2026-10-24 | n/a (no higher orders) | not remeasured | not remeasured | joint tests until 2026-10-25 | `phase` / `complex` (`ssb-direct`); methods are ssb, phase, complex | 2026-10-25 |

The first render after a fit costs about 100–130 ms in the app (pipeline
variant compilation and warm-up); the steady drag rate above excludes it.
In-app timings are `MetalSSBJointPhase.wallSeconds` on an otherwise idle GPU
with one app instance on the built-in display.

## Native regularized direct reconstruction (`MetalSSBKernels`)

Methods `phase` and `complex` of `MetalSSBJointModel.reconstruct(_:upsamplingFactor:method:regularization:)`
(`SSBDirectSolver`, `ssb_direct.metal`), the port of the CUDA `PreparedDirect`: per scan frequency the normal
equations of every supported output alias (`direct_normal_equations`: the weak-object transfer per BF pixel in
float32 with CUDA's expressions, products of open alias pairs, float32 batch sums met in float-float across BF
batches), then float64 Cholesky solves of the damped systems on the CPU, then one inverse FFT of the packed
eta + i phi spectrum (`direct_wave`). The layout (frequencies, aliases, priors, detector weights, dose) is kept
per data, factor, prior and weighting; the prepared normal equations per optics, one per method (a complex
system also serves `phase`), so a damping change only solves again.

Wall ms on an Apple M5 Max (`metal-ssb-fit-benchmark --direct-parity` / `--fixture-parity`, 2026-10-25): a
measured 512 × 512 acquisition (9,298 BF pixels, 0.373 Å step, Auto 3×, 3–6 aliases per frequency) and a
0.525 Å-step one (9,289 BF pixels, Auto 4×, 7–12 aliases). "Optics change" is a call at new optics (layout and
kernels kept: prepare + solve + wave; GPU prepare in brackets); "damping change" reuses the prepared system.
The first call of a process adds the layout (3–7 ms) and kernel specialization (about 50–80 ms). In the app
(headed, 2026-10-25): a C10 edit with Complex selected 484 ms, Phase's first render 353 ms.

| Kernel | Status | Source digest | Serves | Validated | Optics change 1× / 2× / 3× (0.373 Å) | Optics change 4× (0.525 Å) | Damping change | Parity | Replaced by | Retired |
|---|---|---|---|---|---|---|---|---|---|---|
| `ssb-direct` v0.2 | **active** | `4e0355f889a5` | methods `phase` and `complex`, 1–4×, and their drag windows | 2026-10-25 | phase 45–50 / 118–123 / 165–175; complex 47–64 / 153–161 / 167–190 (one thread per frequency and alias for complex systems of two or more aliases, `direct_normal_tiled`) | not remeasured (the tiled kernel serves complex) | as v0.1 | as v0.1 on every reconstruction of the five fixtures (see the parity record); drag previews within 1e-6 of the exact image's std inside their window (`SSBDirectDragWindowTests`, `--direct-drag`) | — | — |
| `ssb-direct` v0.1 | **retired** (`direct_normal_equations` alone; it still serves phase and 1× complex in v0.2) | `8d1237658f22` | methods `phase` and `complex`, 1–4× | 2026-10-25 | phase 53 / 128–148 / 175–183 (GPU 170); complex 54–57 / 204–210 / 280–287 (GPU 272–278) | phase 468 (GPU 450); complex 1228 (GPU 1201) | phase 0.7–4.7, complex 0.9–7.0 at 1–4×; 15 at 4× complex | five measured acquisitions × their optics × 1× and Auto × 5–7 regularization presets against CUDA (182 reconstructions): phase rms 4e-8–1.8e-7 rad, correlation ≥ 0.9999999999, complex amplitude relative rms ≤ 1.6e-7; against a float64 oracle the same size as CUDA's own error; `SSBDirectReconstructionTests` (synthetic, in-test float64 reference) | `ssb-direct` v0.2 | 2026-10-25 |

Drag previews (v0.2, `buildDragWindow` / `dragPreview`, design note
`maintainer/2026-10-25-ssb-direct-drag-windows.md`): a Chebyshev series of the prepared entries along the dragged
control from 29 exact preparations (50 for an azimuth). At 3× a window builds in about 5 s; a preview frame takes
5–9 ms in the benchmark and, in the app while dragging C10 or C12, 5.4–5.9 ms (phase) and 9.8–10.4 ms (complex) per
render with a render every 16–21 ms; it agrees with the exact image to about 1e-6 of its standard deviation.

## Native SSB fit (`MetalSSBKernels`)

The primary (low-order) fit's trust-region Newton climbs on the exact gradient and Hessian of the agreement
in C10, C12 (two Cartesian components), the sample tilt and the depth spread (`ssb_low_order_derivatives.metal`,
`SSBLowOrderDerivatives`). It replaces the finite-difference stencil of 13 or 28 objective evaluations per
Newton iteration. v0.2 evaluates in stages: a value pass (the objective's sums; the same 256-pixel partials and
expressions as v0.1's pass 1, with the soft-aperture shortcut, fast `sincos`, Markstein quotients and real
anchor branches of `ssb-joint-agreement` v0.2), then, where the Newton uses them, the derivatives. Their pass
forms 29 k-sums instead of v0.1's 51 by two exact reductions: the depth-sinc weights depend on the tilt only
through q·θ (so the tilt row and column derivatives are qx and qy times one derivative), and the C10/C12 phase
slopes are linear in k (each sum over them is two k-moments, each product of two is three). The combine expands
them to v0.1's 51 per band frequency. For Newton centers whose derivatives it will use (every iteration but the
pruned second one), a fused pass forms the value sums (bitwise those of the value pass) and the 43 derivative
sums that need no model sum N, projected on conj N in the combine. No derivatives are formed at a rejected step
or for the starts that the second iteration's pruning stops.

GPU ms per center (median of seven calls, command-buffer GPU time; `metal-ssb-fit-benchmark --objective-bench`
on the kept low-order fit of each app-matched case and three seeded random centers near it), Apple M5 Max,
2026-10-25. The v0.2 numbers were taken with heavier background GPU load than v0.1's: the unchanged
`ssb-joint-agreement` v0.2 single candidate measured 8.5 / 9.1 / 17.4 ms with v0.1 and 9.8 / 11.7 / 19.7 ms with
v0.2. Fits (median of three, interleaved with the v0.1 build): Newton 0.91 → 0.34 s (film), 1.47 → 0.57 s
(second acquisition), 1.83 → 0.74 s (third); same centers per start, same kept branch, low-order parameters
within 2e-6 nm / 3.2e-5° / 1e-6 mrad.

| Kernel | Status | Source digest | Serves | Validated | film (8,807 BF, 44,247 q) | Second (9,298 BF, 44,255 q) | Third (8,930 BF, 77,842 q) | Parity | Replaced by | Retired |
|---|---|---|---|---|---|---|---|---|---|---|
| `ssb-low-order-derivatives` v0.3 | **active** | `0158114dc413` | as v0.2: v0.2 with the retired `mean_phase` kernels removed from the shared `ssb_joint` source; derivative kernels unchanged | 2026-10-25 | as v0.2 | as v0.2 | as v0.2 | as v0.2 | — | — |
| `ssb-low-order-derivatives` v0.2 | **retired** (superseded by v0.3, same kernels) | `6178f0ed1613` | primary-fit Newton (value, gradient, Hessian) | 2026-10-25 | fused 29.4 (one center) / 27.6 (three); value pass 11.2 + derivative pass 24.4 | fused 27.5 / 26.2; value 12.0 + derivatives 25.9 | fused 49.8 / 45.1; value 19.5 + derivatives 44.0 | `SSBModelReconstructionTests.testLowOrderDerivativesMatchFiniteDifferences` (Richardson differences: gradient ≤ 3.8e-5, Hessian ≤ 3.5e-4 relative; value against the agreement objective ≤ 9e-7; fused and two-stage value bitwise equal, derivatives ≤ 7e-6; v0.1 value ≤ 2.2e-7, derivatives ≤ 1.3e-5), `SSBAgreementSearchTests.testStagedExactNewtonMatchesOneStage`; measured on the three acquisitions: value within 5.3e-10 of v0.1 | `ssb-low-order-derivatives` v0.3 | 2026-10-25 |
| `ssb-low-order-derivatives` v0.1 (`ssb_low_order`) | **retired** (test reference) | `40da0bd64def` | primary-fit Newton before 2026-10-25 | 2026-10-25 | 53.7 / 53.8 (pass 1 ≈ 16 + pass 2 ≈ 38) | 57.2 / 58.5 | 101.6 / 104.3 | reference for v0.2 | `ssb-low-order-derivatives` v0.2 | 2026-10-25 |

## Native SSB fifth-order gradient (`MetalSSBKernels`)

The exact-gradient search's calls, GPU ms per call (command buffer; the wall time of a call exceeds it by about
0.3 ms), one candidate unless noted, at the app-matched kept rotation: the value pass, the gradient pass after it, and
both together on all BF pixels; the same on every 8th BF pixel (coarse stage); the start Hessian batch (29
candidates on every 64th BF pixel). Min / median over 27 calls from three short runs alternating with v0.1 (the GPU
throttles under sustained load and shows about 38% background use from the desktop, so long single runs read up to
1.8x slower). Data as below: the first acquisition (8,807 BF pixels, 44,247 band frequencies), the second (9,298,
44,255) and the third acquisition (8,930, 77,842). Benchmark: `metal-ssb-fit-benchmark --objective-bench RUN.json
--search-kernels-only`, 2026-10-25.

| Kernel | Status | Source digest | Serves | Validated | film | Second | Third | Parity | Replaced by | Retired |
|---|---|---|---|---|---|---|---|---|---|---|
| `ssb-fifth-gradient` v0.2 | **active** (`SSBModelFit.HigherOrderSearch.exactGradient*`) | `9a751b5a12ba` | exact gradient of the fifth-order agreement (28 z: coefficients, depth, tilt) on all BF pixels or every n-th, after the `ssb-fifth-agreement` v0.3 value pass (`fifth_gradient_scale`, `fifth_gradient_rows`) | 2026-10-25 | all BF: value 9.8 / 11.0, gradient 16.5 / 18.4, both 27.8 / 29.2; every 8th: 1.35 / 1.40, 2.11 / 2.25, 3.45 / 3.63; Hessian batch 10.7 / 11.9 | all BF: 10.2 / 11.5, 17.9 / 19.7, 28.5 / 31.6; every 8th: 1.38 / 1.49, 2.21 / 2.33, 3.60 / 3.97; Hessian 11.3 / 12.6 | all BF: 18.4 / 18.8, 30.6 / 32.8, 49.1 / 52.9; every 8th: 2.21 / 2.50, 3.61 / 4.32, 5.97 / 7.06; Hessian 19.9 / 21.4 | `SSBFifthOrderGradientTests` (double-precision central differences of all 28 z: worst 2.9e-6 of max\|g\| on all BF pixels, 2.6e-6 on every 3rd; value pass bit-identical to `fifthOrderAgreements`); measured against v0.1 at the same points: gradients within 3.5e-6 of max\|g\| at the search starts (within 4.6e-5 at the optima, where the gradient is float32 noise), agreements within 1.9e-7 | — | — |
| `ssb-fifth-gradient` v0.1 | **retired** | `dc8d80c38d16` | before 2026-10-25 (value pass: `fifth_agreement` and its strided copy `fifth_strided_agreement`; BF- or band-subsampled agreement) | 2026-10-25 | all BF: 22.4 / 23.4, 33.0 / 35.4, 57.1 / 59.6; every 8th: 2.76 / 3.05, 4.15 / 4.25, 7.10 / 8.03; Hessian 22.4 / 23.6 | all BF: 23.1 / 24.5, 35.3 / 38.0, 60.5 / 62.9; every 8th: 2.93 / 3.08, 4.34 / 4.68, 7.44 / 7.97; Hessian 23.6 / 24.8 | all BF: 40.0 / 42.0, 60.1 / 64.3, 101.2 / 107.1; every 8th: 4.72 / 5.90, 7.25 / 9.66, 12.5 / 14.4; Hessian 39.7 / 43.1 | `SSBFifthOrderGradientTests` (worst 3.6e-6 of max\|g\|) | `ssb-fifth-gradient` v0.2 | 2026-10-25 |

v0.2 evaluates the per-pair terms of the `ssb-fifth-agreement` value pass (band tiles, row-collapsed Horner phases,
the exact soft-aperture shortcut, `fast::sincos` and Markstein division) and needs no per-pair basis: chi is a
polynomial of degree at most 6 in k, so the coefficient gradient is linear in the monomial sums
T_a = sum 2w [alpha1 ((k - q)^a - k^a) + alpha2 ((k + q)^a - k^a)], and by the binomial expansion (the k^a terms
cancel exactly) T_a = sum over a' < a of C(a, a') q^(a - a') [alpha2 + (-1)^|a - a'| alpha1] k^a'. Each thread (one
band frequency) accumulates the 15 even and 21 odd moments of k (36 vector FMAs per pair against monomials shared in
threadgroup memory; v0.1 evaluated three 28-term bases per pair) and shifts them by its q once; the host maps the 25
monomial sums to the 28 packed coefficients in double. A BF subsample keeps at least 32 pixels.

Fifth-order stage of the full native fit (`metal-ssb-fit-benchmark --cache CASE --repeats 3`, median, runs alternating
with v0.1 on the same busy machine): film 0.91 → 0.46 s, second 1.31 → 0.60 s, third 1.36 → 0.68 s; same kept branch
and rotation; Double-summed final agreement relative to v0.1 +1.5e-7, −1.5e-8, −2.6e-8 (float32 rounding of the
reassociated value pass; C10 within 3.5e-4 nm).

## Fit objectives (`MetalSSBJointModel` agreement)

GPU ms per candidate objective (median of three calls; command-buffer GPU time, which the wall time of the
call exceeds by at most 0.3 ms) at a fit's kept rotation. Fifth order: one candidate, four seeded random
candidates, the 57-point Newton diagonal stencil, 64 random candidates. Low order: one candidate, eight (one
search-trial batch) and 64 random candidates. Data: the first acquisition (8,805 BF pixels, 44,247 band
frequencies), a second 512² acquisition (9,296 BF, 44,255) and a third (8,928 BF, 77,842), all 300 keV,
30 mrad. Benchmark: `metal-ssb-fit-benchmark --objective-bench RUN.json`, 2026-10-25.

| Kernel | Status | Source digest | Serves | Validated | film | Second | Third | Parity | Replaced by | Retired |
|---|---|---|---|---|---|---|---|---|---|---|
| `ssb-fifth-agreement` v0.3 | **active** | `2d385e96d9c4` | `fifthOrderAgreements` (fifth-order fit, stencil Newton, gradient check) and the value pass of `ssb-fifth-gradient` v0.2 on all BF pixels or every n-th | 2026-10-25 | as v0.2 (10.7 / 6.3 / 5.9 / 7.5 back to back with v0.2 at 10.9 / 6.3 / 7.0 / 7.6) | as v0.2 | as v0.2 | v0.2's per-pair code plus a BF stride: bit-identical to v0.2 on every candidate set of the three acquisitions; `SSBFifthOrderGradientTests` (every 3rd BF pixel against the float64 objective) | — | — |
| `ssb-fifth-agreement` v0.2 | **retired** (superseded by v0.3, same per-pair terms) | `f2f700147e47` | `fifthOrderAgreements` before the BF stride | 2026-10-25 | 11.0 / 6.6 / 5.6 / 6.6 | 11.1 / 7.2 / 5.8 / 6.9 | 18.4 / 11.5 / 9.8 / 12.0 | `SSBFifthOrderNumericsTests` (v0.1 and the float64 objective); measured: within 3.1e-7 of v0.1 on all three acquisitions | `ssb-fifth-agreement` v0.3 | 2026-10-25 |
| `ssb-fifth-agreement` v0.1 (`fifth_agreement` in `ssb-per-term`) | **retired** (test reference) | `1c390cc7b833` | before 2026-10-25 | before 2026-10-25 | 24.4 / 19.6 / 20.8 / 21.2 | 25.4 / 21.4 / 23.3 / 22.9 | 41.3 / 36.6 / 37.9 / 37.5 | reference for v0.2 | `ssb-fifth-agreement` v0.2 | 2026-10-25 |
| `ssb-joint-agreement` v0.3 | **active** | `6283577c146a` | as v0.2: v0.2 with the retired `mean_phase` kernels removed from the shared `ssb_joint` source; agreement kernels unchanged | 2026-10-25 | as v0.2 | as v0.2 | as v0.2 | as v0.2 | — | — |
| `ssb-joint-agreement` v0.2 | **retired** (superseded by v0.3, same kernels) | `d698aa9cb17a` | `agreements` with `.angleSums` (low-order search trials, branch scores) | 2026-10-25 | 8.7 / 5.1 / 5.1 | 10.2 / 5.9 / 5.8 | 16.3 / 9.5 / 9.8 | `SSBFifthOrderNumericsTests.testLowOrderAngleSumsMatchReference`; measured: within 1.3e-8 of v0.1 | `ssb-joint-agreement` v0.3 | 2026-10-25 |
| `ssb-joint-agreement` v0.1 (`joint_objective_angle_sums` in `ssb_joint`) | **retired** (test reference) | `4aa7c1da6391` (with `ssb-mean-phase`) | before 2026-10-25 | before 2026-10-25 | 35.3 / 8.9 / 9.4 | 41.1 / 10.3 / 10.9 | 65.3 / 18.2 / 17.8 | reference for v0.2 | `ssb-joint-agreement` v0.2 | 2026-10-25 |

Both v0.2 kernels (and `ssb-fifth-agreement` v0.3, which adds the BF stride) evaluate the per-pair float32 terms of v0.1:

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

## Retired native engine (`MetalSSBEngine`, `ssb.metal`)

The engine's own non-model SSB and its phase-variance fit, retired 2026-10-25: native SSB fits and reconstructs
with the joint model only. These kernels were never registered in `SSBKernelRegistry` (the rows below have no
version), so trimming `ssb.metal` changed no registered digest. `ssb.metal` keeps `counts_to_complex`, the row
FFTs (`fft_rows_radix8`, `fft_rows_small_mixed_radix`), `transpose_complex32` and `extract_hermitian_half`, which
prepare the Fourier evidence of every joint model, and `ssb_object_phase`, which shows a reopened `.ssb` run.
Last measurements (2026-08-19, private 512² `uint8` fixture `native-ssb-fullbf-512-u8-v2`, Apple M5 Max,
[migration record](../maintainer/native-metal-ssb-migration.md)).

| Kernels | Status | Served | Last measured | Replaced by | Retired |
|---|---|---|---|---|---|
| `ssb_gamma_accumulate`, `ssb_gamma_accumulate_half`, `ssb_gamma_accumulate_half_aberrations`, `ssb_gamma_accumulate_full_aberrations`, `ssb_finalize_fourier_sum`, `ssb_precompute_cross_trig` | **retired** (non-model engine removed) | `MetalSSBEngine.reconstruct(aberrations:scanDetectorRotationDegrees:)`: SSB without tilt or depth, C10/C12 plus manual higher orders | warm reconstruction 8.9 ms p50, 512² | `ssb-redraw` (joint model, method `ssb`) | 2026-10-25 |
| `ssb_precompute_chi_trig`, `ssb_correct_half_ifft_small`, `ssb_ifft_small_phase_moments`, `ssb_correct_half_column_ifft512_hermitian`, `ssb_ifft512_nyquist_correction`, `ssb_ifft512_rows_hermitian_phase_moments`, `ssb_correct_full_for_phase_loss`, `ssb_accumulate_phase_moments`, `ssb_transpose_half_to_column_major`, `ssb_transpose_half_to_row_major` | **retired** (phase-variance fit removed) | `MetalSSBEngine.phaseVariance` and `optimize` (`SSBOptimizer`: 200 seeded TPE trials, Nelder–Mead); `SSBFitOptions(objective: .phaseVariance)` | exact loss 25.1 ms p50; 200-trial fit 6.06 s p50, 512² | `ssb-joint-agreement`, `ssb-low-order-derivatives`, `ssb-fifth-agreement`, `ssb-fifth-gradient` (model agreement) | 2026-10-25 |
