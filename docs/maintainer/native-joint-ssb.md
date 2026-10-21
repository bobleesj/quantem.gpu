# Native SSB: the model-based, tilt-corrected fit

`MetalSSBKernels` runs Python's `SSB.find_aberrations` on a prepared native session: the same transfer model,
objective, search and branch test, so a native app gives the Python MPS result. The app owns selection and
presentation; the shared Swift package owns count extraction, Fourier preparation, the model, the fit and saved
numerical state. Python is not needed at runtime.

| `SSBFitOptions` | Python | Method |
| --- | --- | --- |
| `SSBFitOptions()` | `find_aberrations()` | model agreement, tilt-corrected (sample tilt and depth spread) |
| `SSBFitOptions(tilt: false)` | `find_aberrations(tilt=False)` | model agreement, tilt and depth held at 0 |
| `SSBFitOptions(tilt: false, objective: .phaseVariance)` | `find_aberrations(tilt=False, objective="phase_variance")` | the older phase-variance SSB |

## Minimal workflow

```swift
let model = try engine.jointModel(scanDetectorRotationDegrees: rotation)  // the curl estimate
let fit = try SSBModelFit.findAberrations(
  model: model, engine: engine, rotationDegrees: rotation, darkField: darkField,
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
| Search | `SSBAgreementSearch`: the resident multivariate TPE (MT19937, 200 trials, seed 42) and trust-region Newton of `ssb._resident_search`, in double precision; a joint fit is polished once more from the phi12-mirrored best trial and the best trial in another (phi12, tilt, depth) cluster, kept only when its agreement is higher by more than 1e-6 relative |
| Final phase | `phase_of_mean` (corrected waves summed, one phase; scan-mean replicas zero) with the separable C10/C12 kernels of the MPS preview (row and column phasors, depth phasors, the Nyquist row in the main launch, two fine rows per thread at 2x, fine rows of one source row per thread with depth); the per-term kernel serves uncached spectra; `meanPhase` on request |
| Output sampling | Fourier-alias reconstruction at 1, 2, 3, 4 or 8x; not interpolation |
| Rotation | Double precision; the angle is converted in double, the BF coordinates rotated in float32 as MPS |
| Wavelength | Python's table at 80-400 kV, else the same formula |

The search port is checked trial by trial: on a synthetic float32 objective that both sides evaluate from the same
float32 trial rows, all 573 tilt-fit rows (the second polish from other basins included) and 244 standard-fit rows equal Python's (`SSBAgreementSearchTests`, which
`scripts/ssb_agreement_gate.py` also feeds with the current Python records). Python keeps its Newton state in
a three-float expansion, so fitted points agree to about 1e-8 there. Measured parity
(`MetalSSBMeasuredParityTests`, given an exported acquisition in `SSB_PARITY_FIXTURE`) covers the BF disk and
fitted detector sampling, the agreement at fixed parameters (1e-7 relative), the default and `tilt=False` fits
(1e-5 nm, 1e-4 degrees, 1e-4 mrad), the curl rotation, the branch decision and the fixed-parameter phases
(correlation 0.999999998 or better at 1x and 3x).

## Limits

- The joint model supports C10/C12 and tilt/depth. Nonzero higher-order aberrations are rejected explicitly.
- Supported scans are square 128, 256 and 512. There is no silent binning or padding for this model.
- The phase-variance objective uses the native phase-variance search (seeded TPE and Nelder-Mead); its trials
  are its own, not Python's.
- The native engine has no detector transposition (`scan_detector_rotation_flipped` is false).
- No absolute potential, acquisition focus sign or specimen thickness claim follows from contrast or the
  agreement alone.

## Saved results

`MetalSSBJointSavedRun` writes a phase-only `.jointssb` record with source identity, build identity,
calibration, ROI, parameters, rotation, output sampling, the fit (`SSBBranchedFit`, both branches when compared)
and a phase SHA-256. Reopening checks source identity, shape and checksum, then displays the exact stored phase
without refitting. Records from the earlier native optimizer still decode.

## Reproducible checks

```sh
swift test -c release -Xswiftc -enable-testing --filter MetalSSBKernelsTests
PYTHONPATH=src python scripts/ssb_agreement_gate.py        # Python and native draw the same trials
SSB_PARITY_FIXTURE=... SSB_PARITY_REFERENCE=... swift test -c release -Xswiftc -enable-testing \
  --filter MetalSSBMeasuredParityTests                     # a measured acquisition, kept outside this repository
```

The mean-phase fixtures (`SSB_JOINT_REFERENCE`, `SSB_JOINT_CALIBRATED_REFERENCE`) keep their gates: loss
relative error < 5e-5, phase relative L2 < 2e-4 and maximum phase difference < 2e-4 rad. Missing external
fixtures are reported as skips, never passes. Keep acquisitions, screenshots and raw logs outside this
repository.

## Full-resolution phase reconstruction

For a 512×512 scan with calibrated 3× output, the native renderer uses a fused
1536-point Metal FFT when the complete aperture support ends before both
Nyquist boundaries. The transfer function satisfies
`Gamma(-q, k) = -conj(Gamma(q, k))`. After removing DC and rotating by `-i`,
the corrected spectrum is Hermitian. The implementation transforms its unique
half, restores DC, evaluates each BF phase, and then averages the phases.
It does not replace `mean_phase` with `phase_of_mean`.

Each 1536-point transform combines three of the shared 512-point radix-8
transforms with a radix-3 stage. Correction is fused with the first transform;
the second transform accumulates phases directly. This path serves the
explicit `meanPhase` estimator. Calibrated BF geometry is
reused from the joint model. Eight-BF reduction groups,
logical BF normalization, native source counts, and float32/complex64
arithmetic are retained. FFT roundoff is checked against the general GPU
implementation; the specialized FFT is not advertised as bit-identical.

The intermediate workspace is bounded to 16 BF planes in Hermitian layout,
including partial cache and streamed tails. Cancellation occurs between GPU
batches, and result buffers remain independently owned. Other scan sizes,
sampling factors, and geometries with nonzero Nyquist support use the general
complex FFT path. Its correction, FFT and phase reduction share a command
buffer to avoid intermediate host waits. No source crop, BF sampling, or
resolution reduction is selected for speed.
