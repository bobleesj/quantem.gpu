# SSB API

`quantem.gpu.SSB` is the public Python single-sideband ptychography contract for
CUDA and MPS. The browser WebGPU runtime mirrors reconstruction, phase, and the
exact-loss contract asynchronously. Native clients use the separate SwiftPM
product `MetalSSBKernels`. WebGPU does not currently implement aberration
fitting. Backend launch geometry, FFT layouts, and optimizer batching remain
implementation details.

## Inputs and outputs

`SSB.open()` accepts a supported detector source. `SSB(patterns, ...)` accepts an
existing backend-resident detector array. Both require electron voltage,
convergence semiangle, and scan sampling unless those values are available
from trusted source metadata.

`SSB.open` uses the canonical `io.load` path, including its ANS-only GPU
acquisition policy. Older prepared-packed files must be re-exported from their
original acquisitions; they cannot be reopened through a packed override.

`SSB.open` owns its loaded source until `close()` or context-manager exit,
including when reconstruction has not yet started. Pass an ordinary dense
array to `SSB` when borrowing caller-owned array storage.
Treat detector values as fixed for the lifetime of a session: SSB retains their
prepared Fourier data. After editing the array, construct a new `SSB` session.
SSB Fourier-stack preparation and reconstruction are additional work; they
are not included in a detector-viewer loading-time claim.

`find_aberrations()` returns one `SSBResult`. Its primary field is the complex64
`object_wave` with shape `(scan_row, scan_column)`. `phase` and `amplitude` are
derived as `angle(object_wave)` and `abs(object_wave)`. The result also records
the backend, fitted aberrations, rotation, loss, trial/refinement counts,
bright-field geometry, timings, reuse state, and provenance metadata.

## Shapes, coordinates, dtypes, and units

- detector input: `I[scan_row, scan_column, detector_row, detector_column]`;
- complex result: `object_wave[scan_row, scan_column]`, complex64;
- `bf_center`: `(detector_row, detector_column)`;
- `scan_sampling_A`: `(row, column)` when anisotropic, in Å;
- `C10` and `C12`: nm; and
- `phi12`: radians.

The default fit evaluates 200 seeded TPE candidates with the exact full active
bright-field phase-variance objective, chooses the minimum loss, and performs
Nelder–Mead refinement. It does not average optimizer candidates.

This is the implemented calibration workflow; Levenberg–Marquardt is not an
available refinement mode. In WebGPU, requesting `find_aberrations()` fails explicitly and
directs the caller to run the exact 200-trial plus Nelder–Mead workflow on CUDA
or MPS. The browser never substitutes fewer trials or a reduced objective.

## Find aberrations, then reconstruct

```python
aberrations = ssb.find_aberrations(tilt=True, save_to="results/aberrations")
aberrations.report()
ssb.show_trials(best=5)
result = ssb.reconstruct(aberrations, upsample=4, save_to="results/4x")
```

`find_aberrations()` searches on the native scan grid. The public `fit()` name
has been removed. `reconstruct()` applies the supplied parameters without
searching. CUDA uses `phase_of_mean` at every output factor; MPS uses
`mean_phase`. Both return phase-only complex waves with unit amplitude,
recorded as `amplitude_estimated=False`. That amplitude is not a measured
specimen transmission. Use explicit `phase_estimator="complex_wave"` for a
native thin-sample complex-object reconstruction when amplitude is needed;
the temporal averaging workflow selects that estimator explicitly.

`result.upsample` and `result.scan_sampling_A` record the output factor and
pixel spacing. Input sampling stays in the saved signature. Changing the
estimator, sampling, or scientific parameters invalidates saved-result reuse.

### Inspect and replay search trials

```python
ssb.show_trials(first=5)
ssb.show_trials(last=5)
ssb.show_trials(best=5)
aberrations.trials                 # DataFrame, indexed by stable trial ID
aberrations.aberrations["C10"]      # nm
aberrations.aberrations["C12"]      # nm
aberrations.aberrations["phi12"]    # radians
aberrations.tilt_mrad              # (row, column) mrad
aberrations.depth_spread_nm        # model depth spread
```

Supply exactly one positive selector. Images share phase contrast and include
scale bars and a parameter table. Trials are replayed from their recorded
settings rather than stored as a large image stack. Browsing preserves the
active reconstruction. `best` ranks the latest search only; losses from the
standard phase-variance objective and the joint tilt objective are not mixed.
The trial table explicitly names the objective. Local refinement is separate
from the search trials and its final result appears in `report()`.

```python
attempts = ssb.reconstruct(aberrations, trials=aberrations.trials.tail(5).index)
attempts.phase                     # (trial, row, column)
adjusted = ssb.reconstruct(aberrations, aberrations={"C10": 12.5})
```

Overrides preserve unspecified coefficients, tilt, depth spread and rotation.
They leave the supplied result unchanged. Replay uses each trial's original
rotation branch, even if the subsequent polarity check changed the session's
rotation. Saved search results retain this history.

## Finer preview sampling

Fit at native sampling, then reuse the fitted parameters for a finer preview:

```python
fitted = workflow.find_aberrations(tilt=True)
phase, loss = workflow.preview(
    fitted.aberrations,
    tilt_mrad=fitted.tilt_mrad,
    depth_spread_nm=fitted.depth_spread_nm,
    upsampling_factor=4,
)
```

CUDA supports factors 1, 2, 3, 4 and 8 with C10/C12 and optional tilt/depth
correction. The field of view stays fixed. The search and diagnostic loss stay
on the native scan grid; changing the factor does not refit parameters or
interpolate detector measurements. More output pixels do not guarantee more
resolved specimen detail. Factors above one do not yet support MPS, WebGPU or
higher-order aberrations.

When migrating hard-coded aberrations from releases before the nm correction,
divide old C10/C12 numbers and search bounds by 10. For example, an old value of
100 represented 100 Å and should now be supplied as 10 nm. Do not rescale angles
or values already recorded in nm. Saved records must declare nm explicitly and use the current schema.
Rerun the probe fit for older records; the loader does not guess their units.

## Errors and unsupported requests

- Missing voltage, semiangle, or scan sampling raises rather than inventing
  calibration.
- An unsupported backend or scientific request fails explicitly; SSB never
  falls back silently to CPU.
- `trials` must be non-negative and `refinement` is `"nelder-mead"` or `None`.
  With `tilt=True`, at least one trial is required. Use `reconstruct()` to
  apply known parameters without a search.
- To save results from an in-memory array, provide `source_path` for provenance.
  Direct arrays never reuse results from disk automatically: a path cannot
  identify which crop or edited array the scientist supplied.
- Native Swift requires a complete, finite `MetalSSBGeometry`, a 512×512 scan,
  and a sufficiently large plane-major `uint8` Metal buffer. It raises rather
  than cropping, binning, changing precision, or falling back to CPU.

## Provenance and exact reuse

`save_to` writes the complex object and a readable signature containing source
identity, source and output shapes/dtypes, calibration, physical parameters,
optimizer settings/history, bright-field geometry, loss, timings, backend,
package source identity, and Git revision. An exact signature match reopens the
saved result for a file-backed `SSB.open` session. Direct array sessions always
recompute when saving; no full-array hash or GPU-to-host copy is required to
check reuse. Any scientific mismatch recomputes instead of reusing stale
output. Inspect `result.reused`, `result.saved_path`, and `result.metadata`.

Searches start from the current session coefficients, including changes made
with `reconstruct(aberrations={...})`. Those starting coefficients participate
in the saved search identity, so a different refinement start cannot inherit
an older result.

## Minimal fit

```python
from quantem.gpu import SSB

with SSB.open(
    "scan_master.h5",
    backend="mps",
    voltage_kV=300,
    semiangle_mrad=30,
    scan_sampling_A=(0.264, 0.264),
) as workflow:
    result = workflow.find_aberrations(save_to="results/ssb")
```

Use `reconstruct()` when aberrations are known and no optimizer should run:

```python
result = workflow.reconstruct(
    aberrations={"C10": 12.5, "C12": 3.0, "phi12": 0.25},
    save_to="results/fixed-ssb",
)
```

`preview()` accepts the same complete aberration mapping and returns a
transient phase array plus an optional exact loss. The array has the full
selected output resolution: "preview" means it is not saved and does not
replace the fitted calibration or stored result, not that it is lower quality.
It does not create a second public result type.

### Phase averaging

The historical `preview(..., phase_estimator="mean_phase")` averages the phase of
each corrected bright-field detector contribution. At higher output sampling,
detector-dependent structure in the added frequency bands can compress contrast
through this nonlinear phase extraction. This is not an FFT brightness factor.

CUDA C10/C12 now defaults to phase after complex-wave averaging:

```python
phase, native_loss = workflow.preview(
    tilted.aberrations,
    tilt_mrad=tilted.tilt_mrad,
    depth_spread_nm=tilted.depth_spread_nm,
    upsampling_factor=4,
    phase_estimator="phase_of_mean",
)
```

Both estimators reuse the existing C10/C12 depth-aware correction kernel, the
same measured diffraction patterns and the supplied calibration. Fitting never
runs inside `preview`. Diagnostic loss retains the original native-grid
per-detector phase variance, even for wave averaging. Use
`compute_loss=False` when only the image is needed.

The estimator choice applies at **every** output factor, including 1x. CUDA
C10/C12 previews now use wave averaging at 1x as well; request `mean_phase`
explicitly to reproduce historical preview output. To compare 1x/2x/3x/4x
scientifically, use the same estimator at each factor. Switching from legacy
1x to wave-average 2x introduces an estimator change as well as a sampling change.

Reduced contrast compression from wave averaging does not establish
quantitative phase accuracy. Known-phase controls still show amplitude
attenuation, and improvement is not uniform across tested noise/sampling cases.
There is no fitted gain, display normalization or detector interpolation in
this option. Active higher-order magnitudes and non-CUDA backends are not
supported; zero higher-order magnitudes with retained angles are allowed.

This estimator does not change the native aberration-search objective.
It is also accepted by `reconstruct`, where it participates in saved-result reuse.
Live uses the public preview default for supported CUDA C10/C12 outputs.
If persisting the returned array, record `phase_estimator`,
`upsampling_factor`, native scan sampling, aberrations, tilt and depth alongside
it. Any application adopting the option must include the estimator in its
scientific product/cache identity, rather than reuse an older phase product.

Regression checks on the chosen physical CUDA device:

```bash
CUDA_VISIBLE_DEVICES=1 python -m pytest tests/hardware/cuda/test_ssb_wave_average.py -q
```

These compare public output against independent Torch per-detector inverse
transforms at 1x/2x/3x/4x, with and without depth/tilt, unchanged native loss,
return to the legacy estimator, and memory-batch invariance. They validate the
implementation, not absolute phase accuracy.

## Thick, tilted crystals

Standard SSB treats the sample as one thin plane. In a crystal a few nanometres
thick that leans by a few milliradians, each atomic column walks sideways with
depth (5 mrad over 10 nm is 0.5 A), which blurs the lattice along the tilt.
`find_aberrations(tilt=True)` fits the aberrations together with the sample tilt and a depth
spread, in one search with the same trial budget as the standard fit:

```python
standard = workflow.find_aberrations()              # C10, C12, phi12
tilted = workflow.find_aberrations(tilt=True)       # + sample tilt and depth spread, jointly

tilted.tilt_mrad          # (row, col) mrad, scan frame
tilted.depth_spread_nm    # model depth spread, not a measured thickness
tilted.tilt_fit_gain      # least-squares agreement relative to standard SSB (> 1: tilt explains more)
pd.concat([standard.report(), tilted.report()])   # one row per fit
```

After a tilt fit, `C10` is the defocus at mid-depth, so it can differ from the
standard fit's. The search is joint on purpose: fitting the tilt after a
standard fit leaves C10 behind and stalls (0.91-0.99 of the best agreement on
two acquisitions, every seed). With 200 trials every seed reached the same tilt
within 0.02 mrad; 300 and 400 trials gave the same answer.

`preview(aberrations, tilt_mrad=(row, col), depth_spread_nm=d)` renders the
thick-sample model for interactive viewers; with no depth spread the tilt has
no effect. `supports_tilt` says whether the session's backend implements it
(CUDA and MPS).

Limits:

- The phase-variance loss that `report()` shows does not reward tilt; judge a
  tilt by `tilt_fit_gain` and the lattice, not by the loss.
- A tilt at the search limit (`tilt_limit_mrad`, default 25) or a gain close to
  1 is not a measurement.
- One tilt direction can be loosely determined (on one film acquisition the
  row tilt varied by 0.6 mrad at 0.05 % of the agreement).
- The tilt is in the scan frame. The ptychography object frame is the scan
  rotated by the same scan-detector rotation.

Evidence: [SSB units and the thick-sample model](../maintainer/2026-09-24-ssb-units-and-thick-sample.md).

## Native Swift and Metal

`MetalSSBEngine` consumes exact plane-major BF columns with layout
`[logical_brightfield, scan_row, scan_column]`, source dtype `uint8`, and fixed
scan shape 512×512. The engine computes in float32/complex64. It retains every
logical BF term in normalization and skips only the proven-zero aperture union.

```swift
import Metal
import MetalSSBKernels

let device = MTLCreateSystemDefaultDevice()!
let engine = try MetalSSBEngine(
  device: device,
  geometry: calibratedGeometry,
  cacheBudgetBytes: availableSSBCacheBytes
)

try engine.prepare(brightfield: planeMajorUInt8Buffer)
let result = try engine.reconstruct(
  aberrations: MetalSSBAberrations(
    c10Nanometers: 72.98,
    c12Nanometers: 14.4,
    phi12Radians: 0.4686
  )
)
```

`result.object` and `result.fourierSum` are row-major complex64 Metal
buffers at the native 128×128, 256×256 or 512×512 scan size. Their
`result.provenance` records scan shape, source/compute dtype, scan bin
1, no scan crop, logical/executed/zero-aperture BF counts, cached/streamed BF
counts, and cache bytes. `phaseVariance(...)` evaluates the same complete
objective. `optimize(...)` defaults to 200 seeded TPE trials followed by
Nelder–Mead and returns the full trial record.

### Optional BF sampling for native optimization

`phaseVariance` and `optimize` accept `brightfieldFraction`, default **1.0**.
This fraction selects detector BF pixels for the objective, not scan positions,
aperture radius, detector binning or the final image resolution. Full aperture
remains the default throughout search and Nelder-Mead refinement.

```swift
let fit = try engine.optimize(
  start: initialAberrations, globalTrials: 200, brightfieldFraction: 0.25)
let final = try engine.reconstruct(aberrations: MetalSSBAberrations(
  c10Nanometers: Float(fit.best.c10Nanometers),
  c12Nanometers: Float(fit.best.c12Nanometers),
  phi12Radians: Float(fit.best.phi12Radians)))
```

Fractions below one intentionally approximate the full objective and may change
the fitted aberrations. They do not produce an equivalent full-BF loss. The
selection is uniform without replacement, fixed at seed 42, sorted into source
order and reused across all objective evaluations. Its size is the rounded
fraction of the logical BF selection, with at least two pixels (or all pixels
if fewer exist). Zero-aperture entries remain part of that logical normalization;
a subset with no active contribution fails with an instruction to increase it.
No intensity-based or central-disk-only selection is made.

The engine skips unselected cached or streamed columns, batches adjacent selected
columns, and reuses the full prepared cache. It does not allocate a second
Fourier-volume cache. Initial full evidence preparation and memory use are not
reduced. `reconstruct` is unchanged and always uses the engine's full selected
aperture, including after a sampled fit. Using fewer objective pixels is not a
kernel speedup or a guarantee of proportional total-workflow acceleration.

`SSBOptimizationResult.brightfieldSampling` stores the policy version, requested
fraction, total BF count and exact selected logical indices. Phase-variance
results carry the same selection and selected-work provenance. Saved native
runs retain this record; applications should label sampled losses explicitly
and retain selection provenance when exporting a phase-only result.

`scripts/check_metal_ssb_bf_sampling.sh` compares sampled loss against explicitly
constructed subset inputs at all three native sizes in cached, streamed and
hybrid modes, checks saved-fit selection, and checks bit-identical full
reconstruction before/after sampling. The unchanged 100% path is additionally
covered by `scripts/check_metal_ssb_scan_sizes.sh` against the frozen CUDA and
independent-equation fixtures. These fixtures validate implementation, not the
scientific quality of fitting a particular experiment with fewer BF pixels.

`cacheBudgetBytes: nil` requests a complete Hermitian `G(k)` cache. A finite
budget caches whole 32-BF batches and streams the remaining terms exactly.
Cache policy is an application decision: the application must choose and show
the memory policy, while QuantEM.GPU owns the estimator inputs, exact kernels,
and provenance. The Swift package does not own windows, controls, sessions, or
plots.

## Series reconstruction

`SSB.reconstruct_series()` discovers numbered acquisitions in one directory
and returns one `SSBSeriesResult`. Inclusive `first_frame` and `last_frame`
values select acquisition identifiers, not positional array indices. Set
`probe_reference_frame` to reuse one fitted probe; omit it to fit each
acquisition independently. The result retains the phase, bright-field and
dark-field stacks, frame/dataset identifiers, source/results directories,
backend request, optimizer settings, and per-acquisition records.

## Integration boundary

QuantEM.GPU owns SSB preparation, the exact objective, optimization,
reconstruction, typed results, persistence signatures, and backend parity. A
consuming application owns controls, progress presentation, cache scheduling,
and visualization. It must present approximate previews as previews and must
not promote them to exact calibration evidence.

See [Single-sideband ptychography](../kernels/ssb.md) for the complete
mathematics and [SSB performance evidence](../maintainer/ssb-performance.md)
for dated, revision- and device-qualified measurements. Performance numbers do
not live in this API contract because a new benchmark must not silently change
API semantics.

## Phase reconstruction order

CUDA C10/C12 output now combines corrected complex waves before taking their
phase, including tilt/depth correction and 1x, 2x, 3x, and 4x output.
`preview(..., phase_estimator="mean_phase")` remains an explicit historical
comparison. Native MPS and higher-order paths retain their existing estimator;
explicit wave averaging is currently restricted to CUDA C10/C12.

Taking a phase is nonlinear: `angle(mean(waves))` is not
`mean(angle(waves))`. Taking each angle first discards amplitude information
and can compress reconstructed contrast. Combining the corrected spectra
before one inverse FFT gives the same wave sum as summing separate inverse
FFTs, because both operations are linear. The calibration objective remains
the native-grid per-detector phase variance. Stronger contrast alone does not
establish quantitative phase accuracy.

QuantEM's `DirectPtychography` SSB implementation uses a related but distinct
weak-phase convention: multiply by `-1j * conj(gamma) / abs(gamma)`, inverse
transform, take the real part, normalize by BF weight, and sum contributions.
It does not average per-contribution phase angles. Do not describe our
DC-referenced `angle(mean(waves))` as numerically identical to that output.
