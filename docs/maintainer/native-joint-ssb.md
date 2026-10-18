# Native joint tilt/depth SSB

`MetalSSBKernels` now provides the depth-averaged transfer model used by the
Python MPS joint SSB implementation. The app owns selection and presentation;
the shared Swift package owns count extraction, Fourier preparation, the model,
optimization and saved numerical state. Python is not needed at runtime.

## Minimal workflow

```swift
let model = try engine.jointModel(scanDetectorRotationDegrees: rotation)
let start = SSBJointParameters(
  aberrations: .init(c10Angstroms: 0, c12Angstroms: 0, phi12Radians: 0),
  depthAngstroms: 200)
let fit = try SSBJointOptimizer().run(
  start: start, evaluate: model.loss, evaluateBatch: { try model.losses($0) })
let phase = try model.reconstruct(
  fit.best, upsamplingFactor: calibration.ssbUpsamplingFactor())
```

Prepare `engine` with the calibrated geometry and original unsigned counts.
Source calls and the engine are serial. Preparing the engine again invalidates
existing joint models; create a new model before evaluating that source. Pass `isCancelled` closures to long
operations. `SSBScanRegion` copies an explicitly selected 128, 256 or 512 square
in source `(row, column)` coordinates, retaining scan spacing and detector
pixels. Compute geometry and DC from that same region's detector totals.

## Scientific contract

| Quantity | Contract |
| --- | --- |
| C10, C12 | Angstrom; `f = -C10` at the same reference plane |
| C10 reference | Model midplane; not a microscope entrance-plane focus measurement |
| Astigmatism angle | Radians |
| Tilt | `(row, column)` in mrad, scan frame |
| Depth | Nonnegative Angstrom; model depth spread, not an independently measured thickness |
| Search | Original scan grid, every selected BF pixel, 0.2 < q < 0.9 inverse Angstrom |
| Objective | Negative explained Fourier power; lower is better |
| Final phase | Mean of individual corrected BF phases, radians |
| Output sampling | Fourier-alias reconstruction; not interpolation |
| Precision | Exact unsigned counts, Float32/complex64 science, Float64 final loss sum |

Scalar angle conversions and pi/wavelength use Double before conversion to
Float32. This matters near transfer zeros; changing the order of rounding can
fail the phase-parity gate even when scalar losses agree.

Automatic output sampling chooses the smallest supported factor in
`[1, 2, 3, 4, 8]` covering nominal circular-aperture support from energy,
convergence angle and scan spacing. It does not claim achieved resolution.
Calibrations requiring more than 8x fail with a corrective error.

The native seeded Parzen search runs 200 global trials by default, followed by
bounded Newton refinement from up to three distinct starts. Refinement uses
Cartesian astigmatism coordinates and at most 30 iterations. Its finite
differences are evaluated in GPU batches of up to eight candidates, sharing
probe geometry and measured Fourier reads. Each candidate retains the scalar
kernel's accumulation order. Only accepted iterates can become the final
solution; derivative probes outside the search bounds cannot.

Pass `evaluateBatch` to use GPU batching; the scalar closure remains supported.
The explicit `.nelderMead` refinement option preserves the previous method
and its default 200-iteration budget. Saved results record the chosen method;
older records without this field remain readable.

An iteration can evaluate multiple points. The global optimizer is not
proposal-identical to Optuna. Fixed-parameter model parity and optimizer
behavior are separate tests.
The search bounds are C10 ±300 A, C12 0..200 A, angle ±pi/2, tilt ±25 mrad per
component, and depth 20..600 A. A boundary result deserves review.

## Limits

- Scan-detector rotation is supplied and held fixed during the joint search.
  Trying both candidate angles is possible; this API does not certify the
  physical branch or resolve all ambiguities automatically.
- The joint model supports C10/C12 and tilt/depth. Nonzero higher-order
  aberrations are rejected explicitly.
- Supported scans are square 128, 256 and 512. There is no silent binning or
  padding for this model.
- Ordinary SSB retains its existing phase-of-mean estimator. Comparing it
  directly with the joint mean-phase output changes both model and estimator.
- No absolute potential, acquisition focus sign or specimen thickness claim
  follows from contrast or optimizer loss alone.

## Saved results

`MetalSSBJointSavedRun` writes a phase-only `.jointssb` record with source
identity, build identity, calibration, ROI, parameters, rotation, output
sampling, trial history and a phase SHA-256. Reopening checks source identity,
shape and checksum, then displays the exact stored phase without refitting.
The format is distinct from the legacy complex-wave `.ssb` record.

## Reproducible checks

Generate disposable synthetic references using the independent Python MPS
implementation and its alias-upsampling source:

```sh
python scripts/build_metal_ssb_joint_reference.py "$TEST_ARTIFACTS_DIR/reference" \
  --upsampling-source /path/to/quantem/gpu/ssb/mps/upsampling.py
SSB_JOINT_REFERENCE="$TEST_ARTIFACTS_DIR/reference" \
  swift test --filter MetalSSBJointTests
```

The reference covers zero depth, nonzero tilt/depth, signed C10 and 1x/2x/3x/4x
output. The frozen gates are loss relative error < 5e-5, phase relative L2
< 2e-4 and maximum phase difference < 2e-4 rad. Tests exercise both cached and
streamed Fourier input, exact UInt32 ROI extraction, optimizer repeatability,
cancellation and save/reopen. An optional calibrated fixture test additionally
covers native calibration and nonzero rotation. Missing external fixtures are
reported as skips, never passes.

Private measured-data evidence is archived under experiment
`20261015-native-joint`. Keep acquisitions, screenshots and raw logs outside
this repository. Native app interaction, numerical parity and release
qualification are separate gates.

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
the second transform accumulates phases directly. Calibrated BF geometry is
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
