# Native SSB: the model-based, tilt-corrected fit

`MetalSSBKernels` runs Python's `SSB.find_aberrations` on a prepared native session: the same transfer model,
objective, search and branch test, so a native app gives the Python MPS result. The app owns selection and
presentation; the shared Swift package owns count extraction, Fourier preparation, the model, the fit and saved
numerical state. Python is not needed at runtime.

| `SSBFitOptions` | Python | Method |
| --- | --- | --- |
| `SSBFitOptions()` | `find_aberrations()` | model agreement, tilt-corrected (sample tilt and depth spread) |
| `SSBFitOptions(tilt: false)` | `find_aberrations(tilt=False)` | model agreement, tilt and depth held at 0 |
| `SSBFitOptions(tilt: false, objective: .phaseVariance)` | `find_aberrations(tilt=False, objective="phase_variance")` | retired natively (2026-10-25): saved options decode, `validate()` refuses a fit; Python keeps it |

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
| Search | `SSBAgreementSearch`: the resident multivariate TPE (MT19937, 200 trials, seed 42) of `ssb._resident_search`, then a trust-region Newton on the exact gradient and Hessian of the agreement (`ssb-low-order-derivatives`), in double precision; a joint fit is polished once more from the phi12-mirrored best trial and the best trial in another (phi12, tilt, depth) cluster, kept only when its agreement is higher by more than 1e-6 relative. Only the given rotation is searched: the opposite 180-degree branch gets the exact image of the fit (even-degree chi terms and tilt negated, odd-degree terms and depth kept), which has the same agreement |
| Fifth order | `SSBFifthOrderGradientSearch` (`higherOrderSearch: .exactGradientCoarseToFine`, the default): exact gradients of the 28-coordinate agreement (`ssb-fifth-gradient`), a trust-region dense BFGS scaled by and started from a Hessian of every 64th BF pixel, the active set held at the ±2 box bounds, first on every 8th BF pixel and then converged on all of them (a subsample keeps at least 32 pixels, so a small bright field uses all of them); accept/reject and the stop compare Double-summed agreements. `.finiteDifferenceNewton` (`SSBFifthOrderSearch`, Python `fifth_order_fit`) stays as the explicit trajectory reference. `refineToFifthOrder` refines a shown low-order fit without its search |
| Final phase | `phase_of_mean` (corrected waves summed, one phase; scan-mean replicas zero). At 1x-4x the interactive redraw kernel (`ssb_redraw.metal`) serves every term through C56, tilt and depth for cached and streamed spectra: the anti-Hermitian half is evaluated and mirrored, each source frequency visits its output aliases per batch of 32 BF pixels, chi is a per-row Horner polynomial, and SIMD groups skip BF pixels with closed sidebands. It agrees with the earlier per-term kernel within 1e-5 rad (observed about 5e-8 on the synthetic test scene, 4e-6 on the full-size synthetic benchmark). 8x keeps the per-term kernel |
| Reconstruction methods | `SSBReconstructionMethod`, ShowPtycho's three: `.ssb` (the final phase above), `.phase` and `.complex` (regularized direct solves, below). The `mean_phase` and `complex_wave` estimators are retired (2026-10-25): their kernels and API are removed; the app opens results saved with them as saved and reconstructs edits with one of the three |
| Phase | `.phase` (Python `method="phase"`): the linear weak-phase model T = exp(i phi) ~ 1 + i phi solved per scan frequency for every supported output alias q (|q| < (2 semiangle + max angular sampling) / wavelength), with the same tilted, finite-depth transfer and fitted terms as SSB: (T^H W T + phase_damping S diag((1 + q^2/q0^2)^2)) x = T^H W G / dose, S = sum_k w_k A(k)^4; the DC family and, on even grids, the Nyquist row and column are excluded; aliases fill the output spectrum (times factor^2 with Hermitian mirrors) before one inverse FFT; result `angle(exp(i phi))`. Normal equations float32 per BF pixel (CUDA's expressions) and float-float across BF batches on Metal, systems solved in float64 on the CPU (`ssb-direct`). CUDA parity: below 2e-7 rad rms on every measured fixture and preset |
| Complex | `.complex` (Python `method="complex"`): the relative log-amplitude eta and the phase phi of T = exp(eta + i phi) solved together (columns t1 + t2 and i (t1 - t2)), amplitude damping on eta; `amplitude` = exp(eta) (relative; zero-mean eta). CUDA parity: phase as `.phase`, amplitude below 2e-7 of its mean |
| Regularization | `SSBDirectRegularization`: `phaseDamping` and `amplitudeDamping` (> 0, default 0.03), `smoothnessInverseAngstroms` q0 (>= 0, default 1; 0 = frequency-independent prior), `detectorWeighting` `.uniform` or `.inverseMean` (1 / max(mean count, 5 % of the median), normalized to mean 1; needs every BF pixel inside the calibrated aperture). `validate(for:)` gives Python's messages; `.ssb` ignores it. Results record `method` and `regularization` |
| Caching | The layout (frequencies, aliases, priors, weights, dose) is kept per data, factor, q0 and weighting; the normal equations per optics, rotation and layout, one per method (a complex system also serves `.phase`). A damping change solves again only (about 1–7 ms at 512 x 512, 1–3x); an optics change prepares again (about 0.17 s phase, 0.17-0.19 s complex at 3x on an Apple M5 Max; complex systems of two or more aliases use one thread per frequency and alias, `direct_normal_tiled`) |
| Drag previews | `buildDragWindow(along:at:...)` builds, one exact preparation per call, a Chebyshev series of the prepared normal equations along one `SSBDragControl` (C10, C12, phi12, a higher-order magnitude or azimuth, tilt, depth) with every other input held: 29 nodes (50 for an azimuth) over +-h, h from the transfer phase's spread (C10 +- 6.8 nm, C12 +- 3.4 nm at 300 kV, 30 mrad); up to three tiles per control, a quarter of the device's working set. `dragPreview` evaluates it and solves as `reconstruct` (5-10 ms at 3x); the result's `preview` marks it, it agrees with the exact image to about 1e-6 of its std, and it is never saved. Port of Python's CUDA `DragWindow`; design note `2026-10-25-ssb-direct-drag-windows.md` |
| Output sampling | Fourier-alias reconstruction at 1, 2, 3, 4 or 8x; not interpolation. `SSBOutputSampling.auto` (`ssbUpsamplingFactor`) picks the smallest of 1-4x covering nominal support and caps at 4x, as Python; `ssbUpsamplingRequirement` gives the uncapped requirement. `SSBOutputSampling.choices` lists Auto and explicit 1-4x for a viewer; 8x stays an explicit factor |
| Rotation | Double precision; the angle is converted in double, the BF coordinates rotated in float32 as MPS |
| Wavelength | Python's table at 80-400 kV, else the same formula |

The TPE port is checked trial by trial: on a synthetic float32 objective that both sides evaluate from the same
float32 trial rows, the 200 search trials equal Python's (`SSBAgreementSearchTests`, which
`scripts/ssb_agreement_gate.py` also feeds with the current Python records). The finite-difference Newton
reference path (no derivatives passed) still reproduces all 573 tilt-fit and 244 standard-fit rows; the
production exact-derivative Newton ends about 1e-3 nm from Python's stencil Newton at an equal or higher
agreement. The exact-gradient fifth-order search reaches a higher Double-summed agreement than the
finite-difference Newton on three measured 512 x 512 acquisitions (+1.2e-7 to +3.8e-7 relative) in about 1 s
instead of 65-111 s; its parameters differ from the reference along the C10-C30-C50 direction, which the
agreement resolves only to about 1e-3 to 1e-2 nm at float32. Measured parity
(`MetalSSBMeasuredParityTests`, given an exported acquisition in `SSB_PARITY_FIXTURE`) covers the BF disk and
fitted detector sampling, the agreement at fixed parameters (1e-7 relative), the default and `tilt=False` fits
(1e-5 nm, 1e-4 degrees, 1e-4 mrad), the curl rotation, the branch decision and the fixed-parameter phases
(correlation 0.999999998 or better at 1x and 3x).

## Limits

- The joint model supports C10/C12 and tilt/depth. Nonzero higher-order aberrations are rejected explicitly.
- Supported scans are square 128, 256 and 512. There is no silent binning or padding for this model.
- Every native fit is model-based (the agreement objective). The phase-variance objective is Python's only.
- The native engine has no detector transposition (`scan_detector_rotation_flipped` is false).
- No absolute potential, acquisition focus sign or specimen thickness claim follows from contrast or the
  agreement alone.

## Saved results

`MetalSSBJointSavedRun` writes a phase-only `.jointssb` record with source identity, build identity,
calibration, ROI, parameters, rotation, output sampling, the fit (`SSBBranchedFit`, both branches when compared)
and a phase SHA-256. Reopening checks source identity, shape and checksum, then displays the exact stored phase
without refitting. Records from the earlier native optimizer still decode, phase-variance fits included
(`SSBObjective.phaseVariance` is kept for decoding). `MetalSSBSavedRun` (`.ssb`, the retired non-model engine's
complex object) still opens; `MetalSSBEngine.phase(of:)` converts its object to the phase.

## Reproducible checks

```sh
swift test -c release -Xswiftc -enable-testing --filter MetalSSBKernelsTests
PYTHONPATH=src python scripts/ssb_agreement_gate.py        # Python and native draw the same trials
SSB_PARITY_FIXTURE=... SSB_PARITY_REFERENCE=... swift test -c release -Xswiftc -enable-testing \
  --filter MetalSSBMeasuredParityTests                     # a measured acquisition, kept outside this repository
SSB_DIRECT_FIXTURE=... [SSB_DIRECT_ALL=1] swift test -c release -Xswiftc -enable-testing \
  --filter SSBDirectMeasuredParityTests                    # ssb, phase and complex against CUDA arrays of the same optics
```

The joint fixtures (`SSB_JOINT_REFERENCE`, `SSB_JOINT_CALIBRATED_REFERENCE`) gate the objective (loss relative
error < 5e-5); their phase images are of the retired `mean_phase` estimator and are no longer compared. Missing
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
changed. Python's phase-variance search is unchanged.

## Retired estimators

`mean_phase` (the per-BF-pixel phase average, with its general and fused 1536-point kernels, `ssb-mean-phase`)
and `complex_wave` (the coherent sum's amplitude at 1x, C10/C12 only) were removed on 2026-10-25 in favour of the
regularized `phase` and `complex` reconstructions. `docs/performance/kernels.md` keeps their last registry rows.
Python's estimators are unchanged.
