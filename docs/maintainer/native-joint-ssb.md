# Native SSB: the model-based, tilt-corrected fit

`MetalSSBKernels` runs Python's `SSB.find_aberrations` on a prepared native session: the same transfer model,
objective, search and branch test, so a native app gives the Python MPS result. The app owns selection and
presentation; the shared Swift package owns count extraction, Fourier preparation, the model, the fit and saved
numerical state. Python is not needed at runtime.

| `SSBFitOptions` | Python | Method |
| --- | --- | --- |
| `SSBFitOptions()` | `find_aberrations()` | model agreement, tilt-corrected (sample tilt and depth spread) |
| `SSBFitOptions(tilt: false)` | `find_aberrations(search="primary", tilt=False)` | model agreement, tilt and depth held at 0 |
| `SSBFitOptions(tilt: false, objective: .phaseVariance)` | `objective="phase_variance"` (retired in Python too) | retired (2026-10-25): saved options decode, `validate()` refuses a fit |

## Minimal workflow

```swift
let model = try engine.jointModel(scanDetectorRotationDegrees: rotation)  // the curl estimate
let fit = try SSBModelFit.findAberrations(
  model: model, rotationDegrees: rotation, darkField: darkField,
  scanSampling: (stepRow, stepColumn))                    // both 180-degree branches
let phase = try model.reconstruct(
  fit.fit.point, upsamplingFactor: calibration.ssbUpsamplingFactor())  // phase_of_mean, auto factor
```

`SSBRotationBranches.curlRotationDegrees` gives the rotation from exact centre-of-mass moments
(`dpc.find_optimal_rotation`); `findAberrations` fits it and its opposite branch with the same options, keeps
the branch whose dark-field atomic columns have positive phase (else the larger phase skewness), and measures
the standard comparison (`tiltFitGain`) once, for the kept branch. `checkRotation: false` fits the given
rotation alone. Prepare `engine` with the calibrated geometry and the original unsigned counts, after the
loader's flagged-pixel median fill when the Python session applied one. Preparing the engine again invalidates
joint models; `retarget` moves a model to another rotation without gathering its band evidence again.

## Scientific contract

| Quantity | Contract |
| --- | --- |
| C10, C12 | Angstrom in the engine (`SSBModelPoint` has nm accessors); `f = -C10` at the same reference plane |
| C10 reference | Model midplane; not a microscope entrance-plane focus measurement |
| Astigmatism angle | Radians, wrapped to [-pi/2, pi/2) |
| Tilt | `sampleTiltMrad` (row, column) in mrad, scan frame |
| Depth spread | Nonnegative Angstrom; a model depth spread, not a measured thickness |
| Objective | `agreements`: per band frequency `|sum_k G conj(gamma)|^2 / sum_k |gamma|^2`, 0.2 < q < 0.9 1/A, half-plane weights; float32 per term as the MPS kernel, double band sum |
| Agreement | `SSBFit.sampleAgreement` (Python `sample_agreement`): the fraction of G's power in the band that the model explains, S / P in [0, 1], with P = `MetalSSBJointModel.bandPower` = sum_q w_q sum_k \|G(q, k)\|^2 (Python `band_power`, the agreement's Cauchy-Schwarz bound), summed in Double in BF then band order on the host while the band is gathered, once per joint model. The same quantity for the low-order, standard-comparison (`standardAgreement`) and fifth-order fits; the raw S is `rawSampleAgreement`. `tiltFitGain` is S over the best standard-SSB S; a fifth-order fit's is the low-order gain times its agreement over `higherOrderNormalization` (nil without `compareStandard`). Saved fits keep the raw S under the key `sampleAgreement` and the fraction under `sampleAgreementFraction`; a fit saved before P was measured decodes with `sampleAgreement` and `standardAgreement` nil |
| Search | `SSBAgreementSearch`: the resident multivariate TPE (MT19937, 200 trials, seed 42) of `direct_ptycho._resident_search`, then a trust-region Newton on the exact gradient and Hessian of the agreement (`ssb-low-order-derivatives`), in double precision; a joint fit is polished once more from the phi12-mirrored best trial and the best trial in another (phi12, tilt, depth) cluster, kept only when its agreement is higher by more than 1e-6 relative. Only the given rotation is searched: the opposite 180-degree branch gets the exact image of the fit (even-degree chi terms and tilt negated, odd-degree terms and depth kept), which has the same agreement |
| Fifth order | `SSBFifthOrderGradientSearch` (`higherOrderSearch: .exactGradientCoarseToFine`, the default): exact gradients of the 28-coordinate agreement (`ssb-fifth-gradient`), a trust-region dense BFGS scaled by and started from a Hessian of every 64th BF pixel, the active set held at the box bounds (`SSBModelFit.fifthOrderBounds(higherOrderLimit:)`: the low-order box, then each of the 22 higher-order coordinates within ±`SSBFitOptions.higherOrderLimit` rad of probe phase at the aperture edge, 20 by default, which no longer binds, or the standard 2), first on every 8th BF pixel and then converged on all of them (a subsample keeps at least 32 pixels, so a small bright field uses all of them); accept/reject and the stop compare Double-summed agreements. The finite-difference Newton (`.finiteDifferenceNewton`) is retired natively, as in Python: saved options naming it decode, and a fifth-order fit with it is refused. `refineToFifthOrder` refines a shown low-order fit without its search, again at another limit when asked; `onRefinedFit` receives the fit before its checks |
| Half-band check | Default for a ±20 fit (`SSBFitOptions.checkHigherOrderConsistency`, `SSBFit.higherOrderConsistency`, `SSBHigherOrderConsistency`), the same definition on Metal, CUDA and MPS: the fit's band frequencies are split by ring parity floor(\|q\| / 0.05 1/A) mod 2 (q and -q in the same half); for each parity the fifth order is refitted at ±20 with the other half's band weights 0, both from the same full-data low-order start as the ±20 fit; the statistic is the rms over the aperture (\|k\| <= alpha, 96 x 96 samples of the unit disk), piston removed, of chi_A - chi_B through fifth order (C10 and C12 included, tilt and depth not). Trusted when <= 0.5 rad: the ±20 fit is kept. Otherwise the ±2 fit is returned (`options.higherOrderLimit` 2) with its higher orders marked not trusted (thick crystals give spurious terms at ±2 too), and the record keeps the checked ±20 fit. Record names as Python: `half_band_rms_rad`, `trusted`, `shown_limit`. Measured data: 0.16-0.35 rad on four frames of a thin two-dimensional crystal, about 1.33 rad on a thick crystal. Cost: two ±20 fits, a third (±2) on fallback; timings `higher_order_consistency_s`, `higher_order_fallback_s` |
| Final phase | `phase_of_mean` (corrected waves summed, one phase; scan-mean replicas zero). At 1x-4x the interactive redraw kernel (`ssb_redraw.metal`) serves every term through C56, tilt and depth for cached and streamed spectra: the anti-Hermitian half is evaluated and mirrored, each source frequency visits its output aliases per batch of 32 BF pixels, chi is a per-row Horner polynomial, and SIMD groups skip BF pixels with closed sidebands. It agrees with the earlier per-term kernel within 1e-5 rad (observed about 5e-8 on the synthetic test scene, 4e-6 on the full-size synthetic benchmark). 8x keeps the per-term kernel |
| Reconstruction methods | `SSBReconstructionMethod`, ShowPtycho's three: `.ssb` (the final phase above), `.phase` and `.complex` (regularized direct solves, below). The `mean_phase` and `complex_wave` estimators are retired (2026-10-25): their kernels and API are removed; the app opens results saved with them as saved and reconstructs edits with one of the three |
| Phase | `.phase` (Python `method="phase"`): the linear weak-phase model T = exp(i phi) ~ 1 + i phi solved per scan frequency for every supported output alias q (|q| < (2 semiangle + max angular sampling) / wavelength), with the same tilted, finite-depth transfer and fitted terms as SSB: (T^H W T + phase_damping S diag((1 + q^2/q0^2)^2)) x = T^H W G / dose, S = sum_k w_k A(k)^4; the DC family and, on even grids, the Nyquist row and column are excluded; aliases fill the output spectrum (times factor^2 with Hermitian mirrors) before one inverse FFT; result `angle(exp(i phi))`. Normal equations float32 per BF pixel (CUDA's expressions) and float-float across BF batches on Metal, systems solved in float64 on the CPU (`ssb-direct`). CUDA parity: below 2e-7 rad rms on every measured fixture and preset |
| Complex | `.complex` (Python `method="complex"`): the relative log-amplitude eta and the phase phi of T = exp(eta + i phi) solved together (columns t1 + t2 and i (t1 - t2)), amplitude damping on eta; `amplitude` = exp(eta) (relative; zero-mean eta). CUDA parity: phase as `.phase`, amplitude below 2e-7 of its mean |
| Regularization | `SSBDirectRegularization`: `phaseDamping` and `amplitudeDamping` (> 0, default 0.03), `smoothnessInverseAngstroms` q0 (>= 0, default 1; 0 = frequency-independent prior), `detectorWeighting` `.uniform` or `.inverseMean` (1 / max(mean count, 5 % of the median), normalized to mean 1 over every selected BF pixel, those outside the aperture's support included, as Python). `validate(for:)` gives Python's messages; `.ssb` ignores it. Results record `method` and `regularization` |
| Caching | The layout (frequencies, aliases, priors, weights, dose) is kept per data, factor, q0 and weighting; the normal equations per optics, rotation and layout, one per method (a complex system also serves `.phase`). A damping change solves again only (about 1–7 ms at 512 x 512, 1–3x); an optics change prepares again (about 0.17 s phase, 0.17-0.19 s complex at 3x on an Apple M5 Max; complex systems of two or more aliases use one thread per frequency and alias, `direct_normal_tiled`) |
| Drag previews | `buildDragWindow(along:at:...)` builds, one exact preparation per call, a Chebyshev series of the prepared normal equations along one `SSBDragControl` (C10, C12, phi12, a higher-order magnitude or azimuth, tilt, depth) with every other input held: 29 nodes (50 for an azimuth) over +-h, h from the transfer phase's spread (C10 +- 6.8 nm, C12 +- 3.4 nm at 300 kV, 30 mrad); up to three tiles per control, a quarter of the device's working set. `dragPreview` evaluates it and solves as `reconstruct` (5-10 ms at 3x); the result's `preview` marks it, it agrees with the exact image to about 1e-6 of its std, and it is never saved. Port of Python's CUDA `DragWindow`; design note `2026-10-25-ssb-direct-drag-windows.md` |
| Output sampling | Fourier-alias reconstruction at 1, 2, 3, 4 or 8x; not interpolation. `SSBOutputSampling.auto` (`ssbUpsamplingFactor`) picks the smallest of 1-4x covering nominal support and caps at 4x, as Python; `ssbUpsamplingRequirement` gives the uncapped requirement and `ssbUpsamplingChoice` the factor with the uncapped factor and a `capped` flag. `SSBOutputSampling.choices` lists Auto and explicit 1-4x for a viewer; 8x stays an explicit factor |
| Rotation | Double precision; the angle is converted in double, the BF coordinates rotated in float32 as MPS |
| Wavelength | Python's table at 80-400 kV, else the same formula |

The TPE port is checked trial by trial: on a synthetic float32 objective that both sides evaluate from the same
float32 trial rows, the 200 search trials equal Python's (`SSBAgreementSearchTests`, which
`scripts/ssb_agreement_gate.py` also feeds with the current Python records). The Newton after the search has
exact derivatives only: on the same objective, given its analytic derivatives, the tilt and `tilt=False` fits reach
its known maximum, and the staged Newton reaches the one-stage Newton's point with fewer derivative passes.
The exact-gradient fifth-order search is checked for a maximum of the Double-summed agreement on the synthetic
test scene: a small projected exact gradient, and no rise along any of the 28 coordinates
(`SSBFifthOrderGradientTests`). On three measured 512 x 512 acquisitions it reached a higher Double-summed
agreement than the retired finite-difference Newton (+1.2e-7 to +3.8e-7 relative) in about 1 s instead of
65-111 s; the parameters differed along the C10-C30-C50 direction, which the agreement resolves only to about
1e-3 to 1e-2 nm at float32. Measured parity (`MetalSSBMeasuredParityTests`, given an exported acquisition in
`SSB_PARITY_FIXTURE`) covers the BF disk and fitted detector sampling, the agreement at fixed parameters (1e-7
relative), the default and `tilt=False` fits on the model's exact derivatives (tolerances 0.01 nm, 0.2 degrees,
0.02 mrad; the retired stencil Newton measured 1e-5 nm, 1e-4 degrees and 1e-4 mrad, not re-measured since), the
curl rotation, the branch decision and the fixed-parameter phases (correlation 0.999999998 or better at 1x and
3x).

## Limits

- The joint model fits and reconstructs C10 through C56 with tilt and depth spread; an unsupported order or a
  repeated term is rejected.
- Supported scans are square 128, 256 and 512. There is no silent binning or padding for this model.
- Every fit is model-based (the agreement objective), natively and in Python.
- The native engine has no detector transposition (`scan_detector_rotation_flipped` is false).
- No absolute potential, acquisition focus sign or specimen thickness claim follows from contrast or the
  agreement alone.

## Saved results

`MetalSSBJointSavedRun` writes a phase-only `.jointssb` record with source identity, build identity,
calibration, ROI, parameters, rotation, output sampling, the fit (`SSBBranchedFit`, both branches when compared)
and a phase SHA-256. Its fit options record `higherOrderLimit` and `checkHigherOrderConsistency`, and a
fifth-order fit its `higherOrderConsistency` when computed; records saved before them decode
with the standard limit 2, unchecked, and none. Records carrying the retired coherence guard
(`computeHigherOrderGuard`, `higherOrderGuard`) decode and ignore it. Reopening checks source identity, shape
and checksum, then displays the exact stored phase without refitting. Records from the earlier native optimizer still decode, phase-variance fits included
(`SSBObjective.phaseVariance` is kept for decoding). `MetalSSBSavedRun` (`.ssb`, the retired non-model engine's
complex object) still opens; `MetalSSBEngine.phase(of:)` converts its object to the phase.

## Reproducible checks

```sh
swift test -c release -Xswiftc -enable-testing --filter MetalSSBKernelsTests
PYTHONPATH=src python scripts/ssb_agreement_gate.py        # the same search trials and exact-Newton fits
SSB_PARITY_FIXTURE=... SSB_PARITY_REFERENCE=... swift test -c release -Xswiftc -enable-testing \
  --filter MetalSSBMeasuredParityTests                     # a measured acquisition, kept outside this repository
SSB_DIRECT_FIXTURE=... [SSB_DIRECT_ALL=1] swift test -c release -Xswiftc -enable-testing \
  --filter SSBDirectMeasuredParityTests                    # ssb, phase and complex against CUDA arrays of the same optics
```

The joint fixtures (`SSB_JOINT_REFERENCE`, `SSB_JOINT_CALIBRATED_REFERENCE`) gate the objective (loss relative
error < 5e-5). Their phase images were of the retired `mean_phase` estimator; they are not compared, and
`scripts/build_metal_ssb_joint_reference.py` no longer writes them (nor needs an upsampling source). Missing
external fixtures are reported as skips, never passes. Keep acquisitions, screenshots and raw logs outside this
repository.

## Retired native engine paths

The engine's own non-model SSB and the phase-variance fit were removed on 2026-10-25; native SSB is model-based
only. `MetalSSBEngine` keeps Fourier preparation, the joint-model bridge and the saved-run phase conversion.
Removed: `MetalSSBEngine.reconstruct(aberrations:scanDetectorRotationDegrees:)`, `phaseVariance`, `optimize`,
`MetalSSBPhaseVarianceResult`, `SSBOptimizer` with `SSBOptimizationProgress`, the phase-variance branch of
`SSBModelFit` (and its `engine:` argument), `SSBFit.phaseVarianceLoss`, the `metal-ssb-benchmark` product, the
standalone `tests/metal` SSB harnesses and the frozen phase-variance fit trajectory. Their `ssb.metal` kernels
(`ssb_gamma_accumulate*`, `ssb_finalize_fourier_sum`, the chi and cross trigonometric tables, the fused 512 and
small-scan phase-loss kernels, the phase-moment and Nyquist-correction kernels and the half-plane transposes) are
deleted; `ssb.metal` keeps the count conversion, row FFTs, transpose, Hermitian extraction and
`ssb_object_phase`. These kernels were never registered (`docs/performance/kernels.md`), so no registered digest
changed.

## Retired estimators

`mean_phase` (the per-BF-pixel phase average, with its general and fused 1536-point kernels, `ssb-mean-phase`)
and `complex_wave` (the coherent sum's amplitude at 1x, C10/C12 only) were removed on 2026-10-25 in favour of the
regularized `phase` and `complex` reconstructions. `docs/performance/kernels.md` keeps their last registry rows.
Python retired the same two estimators.
