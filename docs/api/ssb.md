# SSB API

Use `SSB` from Python on MPS or CUDA. Find aberrations once, inspect the search,
and reuse those parameters for reconstruction. Start with the
{ref}`Python workflow <reconstruct-phase>` for a short example.
For actual images and saved outputs, run the {ref}`Gold notebook <gold-ssb-reconstruction>`.
This page covers parameters, calibration, ownership, and backend limits.

## Inputs and outputs

`SSB.open(path, ...)` loads a supported detector source. `SSB(data, ...)`
accepts an existing `Dataset4dstemGPU` or supported array. Both require explicit
`voltage_kV`, `semiangle_mrad`, and `scan_sampling_A`; read those values from
verified acquisition metadata or supply your microscope calibration.
`rotation_angle_deg` defaults to zero. SSB does not call DPC automatically;
provide a verified rotation or use a DPC estimate after checking
`dpc_result.use_transpose`. `det_sampling` is detector angular sampling in
mrad per pixel. When it is omitted, CUDA and MPS both use `semiangle_mrad`
divided by the radius in detector pixels at which the bright-field disk of the
full-detector mean pattern falls to half its plateau
(`quantem.gpu.ssb.brightfield.disk_edge_radius`): the disk edge is the image of
the aperture at the convergence semiangle. In `SSB(data, bf_center=..., bf_radius=...)`, disk geometry uses
the input detector grid. The session adjusts the center if it takes a
bright-field crop.

`SSB.open` uses the canonical `io.load` path, including its ANS-only GPU
acquisition policy.

`SSB.open` owns the acquisition it loads: it decodes the bright-field crop,
releases the acquisition, and releases the crop at `close()` or context-manager
exit. `SSB(data, ...)` borrows the caller-owned dataset or array and never
releases it; `close()` frees only the session's own crop and Fourier data.
Encoded inputs decode only the bright-field evidence needed by the session.
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
- complex result: `object_wave[scan_row, scan_column]`, complex64, on the session's output grid;
- `bf_center`: `(detector_row, detector_column)`;
- `scan_sampling_A`: `(row, column)` when anisotropic, in Å;
- `C10` and `C12`: nm; and
- `phi12`: radians.

For a dense CUDA input, native supported scan shapes are 128×128, 256×256,
512×512 and 1024×1024. Other smaller scan shapes are padded with the mean
diffraction pattern to a supported square. For example, a 32×32 input currently
produces a 128×128 working grid before any requested upsampling. Check
`ssb.scan_shape` and `result.phase.shape`; padded pixels are not additional
measured scan positions. Dense inputs larger than 1024 along either scan axis
are center-cropped by this CUDA path; choose the intended scan region explicitly
before construction if that default crop is unsuitable. Computational padding is separate from
`reconstruct(..., upsample=2)`.

The default fit evaluates 200 seeded TPE candidates with the exact full active
bright-field phase-variance objective, chooses the minimum loss, and performs
Nelder–Mead refinement. It does not average optimizer candidates.

This is the implemented calibration workflow; Levenberg–Marquardt is not an
available refinement mode. In WebGPU, requesting `find_aberrations()` fails explicitly and
directs the caller to run the exact 200-trial plus Nelder–Mead workflow on CUDA
or MPS. The browser never substitutes fewer trials or a reduced objective.

(ssb-find-and-reconstruct)=
## Find aberrations, then reconstruct

```python
from quantem.gpu import SSB

ssb = SSB.open(
    "acquisition.qem",
    voltage_kV=300,  # kV
    semiangle_mrad=30,  # mrad
    scan_sampling_A=0.264,  # Å per scan pixel
)
aberrations = ssb.find_aberrations(tilt=True, save_to="results/aberrations")
aberrations.report()
ssb.show_trials(best=5)
result = ssb.reconstruct(aberrations, save_to="results/native")
```

The numbers above illustrate the units; use your acquisition's calibration.
`tilt=True` jointly estimates specimen tilt and model depth spread as well as
aberrations. Omit it for the standard aberration search.
Keep `ssb` open for the examples below and call `ssb.close()` when finished.

`find_aberrations()` searches on the native scan grid.
`reconstruct()` applies the supplied parameters without
searching. CUDA uses `phase_of_mean` at every output factor; MPS uses
`mean_phase`. Both return phase-only complex waves with unit amplitude,
recorded as `amplitude_estimated=False`. That amplitude is not a measured
specimen transmission. Use explicit `phase_estimator="complex_wave"` for a
native thin-sample complex-object reconstruction when amplitude is needed.

`result.upsample` and `result.scan_sampling_A` record the output factor and
pixel spacing. Input sampling stays in the saved signature. Changing the
estimator, sampling, or scientific parameters invalidates saved-result reuse.

### Inspect and replay search trials

```python
ssb.show_trials(first=5)
ssb.show_trials(last=5)
ssb.show_trials(best=5)
aberrations.trials  # DataFrame, indexed by stable trial ID
aberrations["C10"]  # defocus, nm
aberrations["C12"]  # twofold astigmatism, nm
aberrations["phi12"]  # astigmatism angle, radians
aberrations.tilt_mrad  # sample tilt (row, column), mrad; None if not fitted
aberrations.depth_spread_nm  # model depth spread, nm; None if not fitted
result.phase  # radians
result.scan_sampling_A  # Å per output pixel
```

Coefficients use dictionary indexing; units remain nm for C10/C12 and radians
for phi12. `dict(aberrations)` makes a separate coefficient dictionary. Phase,
tilt, rotation and trial history remain properties of the fitted result.
The report table labels its angle column `phi12 (deg)`; it converts radians
to degrees for display.

Supply exactly one positive selector. Images share phase contrast and include
scale bars and a parameter table. Trials are replayed from their recorded
settings rather than stored as a large image stack. Browsing preserves the
active reconstruction. `best` ranks the latest search only; losses from the
standard phase-variance objective and the joint tilt objective are not mixed.
The trial table explicitly names the objective. Local refinement is separate
from the search trials and its final result appears in `report()`.

```python
attempts = ssb.reconstruct(aberrations, trials=aberrations.trials.tail(5).index)
attempts.phase  # radians; axes (trial, row, column)
adjusted = ssb.reconstruct(aberrations, aberrations={"C10": 12.5})  # nm
```

Overrides preserve unspecified coefficients, tilt, depth spread and rotation.
They leave the supplied result unchanged. Replay uses each trial's original
rotation branch, even if the subsequent polarity check changed the session's
rotation. Saved search results retain this history.

## Finer preview sampling

Fit at native sampling, then reuse the fitted parameters for a finer preview:

```python
result = ssb.reconstruct(aberrations, upsample="auto")
```

`"auto"` selects the smallest supported factor (1, 2, 3, 4 or 8) whose output
Nyquist frequency covers the ideal circular-aperture SSB bandwidth,
`2 * semiangle / wavelength`, on both scan axes. Equivalently, output spacing
must be at most `wavelength / (4 * semiangle)`, with semiangle in radians.
This uses the calibrated voltage, convergence semiangle and scan spacing.
It is a numerical grid criterion, not a measured resolution: detector coverage,
noise, coherence, aberrations and alias separation can reduce useful bandwidth.
See the [direct-ptychography upsampling paper](https://arxiv.org/abs/2507.18610).
The current CUDA implementation can produce stripes; automatic selection
does not certify their absence. More than 8× raises an explicit limit error.
MPS currently supports native output only.

For an explicit sampling comparison:

```python
result_2x = ssb.reconstruct(aberrations, upsample=2)
result_4x = ssb.reconstruct(aberrations, upsample=4)
```

Show both full fields with a marked, matching crop underneath:

```python
result.show(compare=result_4x, axsize=(7, 7))  # per-panel width and height, inches
```

The library selects the central quarter of each dimension and outlines it in
both images. All four panels share the first result's symmetric `phase_limits`.
Reconstruction carries those limits from the supplied aberration result, so its
phase, rotation diagnostics and finer outputs use the same linear mapping. The
results must cover the same physical field; this view does not align images.

`find_aberrations()` has no `upsample` argument: the search stays at 1×.
The {ref}`Gold comparison <gold-upsampling>` shows 1× and 4× outputs with shared
contrast and a matched crop. That Gold result has visible bands at 4×;
the example does not establish improved resolution.

## Fit a region and reconstruct the full scan

```python
aberrations = ssb.find_aberrations(
    scan_region=(128, 384, 128, 384),  # row start/stop, column start/stop; scan pixels
    tilt=True,
)
result = ssb.reconstruct(aberrations, upsample="auto")
result.show()
```

The stop indices are exclusive. Only the selected region contributes to the
aberration search; reconstruction still uses the full scan. `scan_region=None`
fits the full scan. The detector calibration remains tied to the full input.
Regional fitting currently requires a dense 4D scan source and a square region
of 128, 256, 512 or 1024 pixels per side, without implicit padding or cropping.
The result records `fit_scan_region`, and saved-result matching includes it.
Its saved search phase and rotation histograms describe that region. Applying
those parameters elsewhere assumes the optics and specimen tilt are shared;
a locally good fit does not establish that the whole specimen is uniform.

## Check the suggested and selected rotations

```python
dpc_result.report()  # the DPC-suggested scan-detector rotation
aberrations.report()  # input and SSB-selected angles, plus fitted coefficients
```

In a separate cell, show the phase images and histograms when SSB selects a
different rotation:

```python
aberrations.show("rotation", histogram=True, axsize=(7, 7))
```

DPC's curl minimum leaves a 180° ambiguity. SSB tests phase polarity under a
bright-atom-column assumption; this is a model-based choice, not independent
proof of the absolute orientation. The report uses explicit angles in degrees.
These saved diagnostic images are captured before conversion to the final
reconstruction display estimator, so their contrast may differ from `result.phase`.

The figure explains which angle was suggested, which angle SSB selected, and
how the phase asymmetry changed after selecting the branch. Each histogram measures phase
relative to its own median; both share the same bin edges and axis limits.
Negative asymmetry motivates testing the opposite branch. Positive asymmetry
on the selected branch supports the bright-column assumption.

When the input and selected rotations agree, this call returns `None` and the
cell produces no figure. Agreement alone does not establish orientation: an
inconclusive phase distribution also leaves the input unchanged. Inspect
`aberrations.column_sign` when that distinction matters. Omit `histogram=True`
to show only the phase images for a changed rotation.

A finite polarity score below −0.2 triggers the other branch. For CUDA joint
calibration (`tilt=True`), this reuses the fitted solution: C10, C12 and both
tilt components change sign; the astigmatism angle and depth spread stay fixed.
The joint objective is invariant under this transformation. Native phase and
diagnostic loss are recomputed on the selected branch because discrete Nyquist
terms prevent simply negating every stored pixel. No second search is run.
Other calibration paths retain their existing refinement behavior.

The inexpensive polarity measurement uses the already reconstructed phase.
It adds no extra reconstruction when the branch is acceptable or inconclusive.
A score between −0.2 and +0.2, or a non-finite score, is inconclusive and keeps
the input branch; non-finite phase data still require investigation. The score is
the median-centered third moment divided by the second moment to the power
3/2. The images, input angle and scores persist with a saved search result.
Earlier saved results without these arrays must be recomputed with `force=True`
before this diagnostic can be shown.

## View the phase and model probe

```python
result.show()  # one large phase image
```

In a separate notebook cell:

```python
result.show("probe")  # real-space and Fourier-space probe intensities
```

The returned Matplotlib figure renders once in a notebook. Phase has a
12 × 12 inch panel; each probe or trial panel is 6 × 6 inches. Use `axsize` to
change panel size. Probe and trial comparisons use at most two columns.

For example, replace the phase call with `result.show(axsize=(14, 14))`, or
use `result.show("probe", axsize=(7, 7))` for larger probe panels. These are
panel widths and heights in inches; image sampling stays unchanged.

Phase and both model-probe intensities use linear contrast over their full
ranges. Each probe has its own display range and scale bar: Å in real space,
mrad in Fourier space. Fourier intensity shows the aperture; defocus and
astigmatism are encoded in the complex Fourier probe's phase. Plotting does
not change the calculated waves.

A defocused model can have rings in real-space intensity. Twofold astigmatism
can make that pattern elliptical. A uniform Fourier-intensity disk does not
mean the probe is aberration-free: inspect the complex Fourier phase to see
those aberrations. Neither view independently validates the fitted coefficients.

To work with the complex arrays directly:

```python
probe = result.probe()  # centered complex real-space wave
probe_fourier = result.probe(space="fourier")  # centered complex reciprocal-space wave
result.probe_sampling_A  # Å per probe pixel, (row, column)
result.probe_sampling_mrad  # mrad per model-probe Fourier pixel, (row, column)
```

Voltage, semiangle and effective aberrations come from
`result`, including any coefficient overrides. Both waves have unit summed
intensity and scan-frame axes. The probe is the fitted circular-aperture optical
model, not an independently retrieved wave or a measured vacuum pattern.
Specimen tilt does not become beam tilt; a thick-model probe is at mid-depth.
The model grid resolves the aperture and expands to cover the defocus and
astigmatism footprint. It is independent of detector binning and object
upsampling. Finer model sampling does not add measured information.
The {ref}`phase–probe cell <gold-model-probe>` shows the calibrated scale bars.
CUDA is physically tested for this view; Python MPS uses MLX but has not yet
been checked on hardware for this addition.

## Upsampling support

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
Rerun the aberration search for older records; the loader does not guess their units.

## Errors and unsupported requests

- Missing voltage, semiangle, or scan sampling raises rather than inventing
  calibration.
- An unsupported backend or scientific request fails explicitly; SSB never
  falls back silently to CPU.
- `SSB(data.data, ...)` raises `TypeError`: the encoded storage inside a loaded
  dataset has no bounded read of its own. Pass the dataset, `SSB(data, ...)`, or
  open the file with `SSB.open(path, ...)`.
- `trials` must be non-negative and `refinement` is `"nelder-mead"` or `None`.
  With `tilt=True`, at least one trial is required. Use `reconstruct()` to
  apply known parameters without a search.
- To save results from an in-memory array, provide `source_path` for provenance.
  Direct arrays never reuse results from disk automatically: a path cannot
  identify which crop or edited array the scientist supplied.

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

## Scripts and fixed aberrations

```python
from quantem.gpu import SSB

with SSB.open(
    "acquisition.qem",
    voltage_kV=300,
    semiangle_mrad=30,
    scan_sampling_A=(0.264, 0.264),
) as batch_ssb:
    batch_aberrations = batch_ssb.find_aberrations(save_to="results/aberrations")
    result = batch_ssb.reconstruct(batch_aberrations, save_to="results/native")
```

The context manager closes the session after the block. In a still-open
notebook session, use `reconstruct()` when aberrations are known and no
optimizer should run:

```python
result = ssb.reconstruct(
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
phase, native_loss = ssb.preview(
    dict(aberrations),
    tilt_mrad=aberrations.tilt_mrad,
    depth_spread_nm=aberrations.depth_spread_nm,
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
import pandas as pd

standard = ssb.find_aberrations()         # C10, C12, phi12
tilted = ssb.find_aberrations(tilt=True)   # + specimen tilt and depth spread, jointly

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

- The phase-variance loss in `aberrations.loss` does not reward tilt; judge a
  tilt by `tilt_fit_gain` and the lattice, not by the loss.
- A tilt at the search limit (`tilt_limit_mrad`, default 25) or a gain close to
  1 is not a measurement.
- One tilt direction can be loosely determined (on one film acquisition the
  row tilt varied by 0.6 mrad at 0.05 % of the agreement).
- The tilt is in the scan frame. The ptychography object frame is the scan
  rotated by the same scan-detector rotation.

Evidence: [SSB units and the thick-sample model](../maintainer/2026-09-24-ssb-units-and-thick-sample.md).

## Native Swift and Metal

Native application authors should use the {ref}`Swift/Metal integration guide <native-swift-and-metal>`.

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
