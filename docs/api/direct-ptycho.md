# Choose a direct reconstruction

`DirectPtycho` owns acquisition preparation, SSB-based optical fitting and
three direct reconstruction methods. The default remains conventional SSB.
Regularized phase and complex reconstruction are experimental methods on CUDA
and MPS (Apple GPUs); both backends solve the same model, and their images
agree within twice CUDA's own float32 error. This API does not imply that a
linear weak-object model is accurate for every specimen.

This API is available in the development source candidate. See the
[acquisition tutorial](../python-workflow.md) for the matching installation.

```python
from quantem.gpu import DirectPtycho

ptycho = DirectPtycho.open("scan.qem")
optics = ptycho.fit_aberrations(tilt=True)

ssb = ptycho.reconstruct(optics, upsampling="auto")
phase = ptycho.reconstruct(
    optics, method="phase", phase_damping=0.03,
    upsampling="auto", save_dir="results",
)
complex_result = ptycho.reconstruct(
    optics, method="complex", phase_damping=0.03,
    amplitude_damping=0.1, upsampling="auto", save_dir="results",
)
```

For a quick CUDA comparison, reconstruct all three methods with one call:

```python
images = ptycho.reconstruct(optics, method="all", upsampling="auto", compute_loss=False)
images["ssb"].show(compare=images, region="center", show_fft=True, columns=3, cmap="gray")
```

The dictionary keys are `"ssb"`, `"phase"`, and `"complex"`. All three reuse the
same fitted optics and sampling; no new aberration search is performed when
`optics` is supplied. Their numerical solves still differ. Each regularized method prepares its own direct equations once; repeated
compatible reconstructions reuse those preparations. Rendering and FFT display take additional time. No method is automatically
claimed to be best. Individual calls and their defaults remain unchanged.

End a notebook cell on `optics` to inspect every fitted aberration and its units.
Use `optics.report()` for a full-precision pandas table with one row per
parameter and units in the row labels, or
`optics.report(to_dict=True)` for a dictionary, with coefficients under
`"aberrations"` and rotation, tilt and effective depth stored separately.
`optics.aberrations` remains the coefficient dictionary. The report also includes
recorded timing and optimizer diagnostics; `optics.report(show_convergence=True)`
adds the saved convergence plot without fitting again.
Use `optics.report(columns=2)` for two parameter/value groups side by side,
or `columns=3` for a wider, shorter table. The default is one group. This
changes only the table layout; dictionary exports retain the same structure.

To see convergence while fitting in Jupyter, request it on the fitting call:

```python
fit = ptycho.fit_aberrations(
    search="broad", compute_loss=False, size="large", show_convergence=True,
)
```

`size` controls the phase preview and chart size, not the acquisition or fit.
`columns` controls the final comparison layout: `columns=1` stacks the three
methods vertically, `columns=2` wraps them, and `columns=3` (the default) keeps
them side by side. `size` accepts a positive integer width in pixels per
panel, such as `size=750`, or `"small"`, `"medium"`, `"large"`, `"xlarge"`
(300, 450, 600, 900 pixels). `size=900` matches `"xlarge"`; static plots use
a 100-DPI reference. Browser fitting and HiDPI display can scale the rendered
figure. These settings do not alter reconstruction sampling. For example:

```python
fit = ptycho.fit_aberrations(
    search="broad", scan_region=scan_region, compute_loss=False,
    size=900, show_convergence=True, columns=1,
)
```

For a regional fit, each full field stays above its matching zoom within the
chosen column layout. The convergence chart remains a separate plot.
`show_convergence` defaults to `False`. For standard and broad searches, the
optional chart updates at most twice per second using objective values the
optimizer already computed. It shows fifth-order search and refinement after
the initial low-order fit. Larger normalized agreement means a better fit to
that model; it is not a resolution score. `compute_loss=False` still skips the
separate phase-variance diagnostic, and no result files are saved unless asked.
Use `show=False, show_convergence=True` for the chart without the phase preview.
For a primary-only fit, inspect `fit.show("convergence")` after fitting.

In a notebook, standard/broad fitting first shows a lightweight conventional
SSB preview. When fitting finishes on resident CUDA data, that same output
becomes three columns: **conventional SSB**, **regularized phase-only**, and
**amplitude + phase**. Every column shows phase in radians using the same fitted
optics, Auto sampling and shared contrast. The regularized previews use the
`reconstruct` defaults: phase strength 0.03, amplitude strength 0.03 when applicable,
smoothness crossover 1 Å⁻¹, and uniform weighting. They are comparison choices,
not a selected winner. Use explicit `reconstruct` calls to tune or retain alternatives.
Preparation happens once per model after fitting and is included in display
time, not the optical-fit time. `show=False` skips it. MPS and streamed sessions
continue to show conventional SSB here (on MPS, call `reconstruct` with `method="phase"`
or `"complex"`); streamed sessions have no regularized methods. Subsequent
phase-strength changes reuse the prepared phase equations.

To fit only a selected part of a loaded acquisition:

```python
fit = ptycho.fit_aberrations(
    search="broad", scan_region=(128, 384, 128, 384),
    compute_loss=False, size="large", show_convergence=True,
)
```

`scan_region` uses scan positions `(row_start, row_stop, col_start, col_stop)`
with exclusive stops. The square region must fit inside the acquisition and
currently have 128, 256, 512 or 1024 positions per side. Only that region
contributes to fitting. The initial preview shows full-field SSB with a box and
a matching zoom. The final CUDA comparison puts all three full fields above
their matching fit-region zooms. All previews apply the regional optics to the
full acquisition; no second full-field fit is run.
This display adds full-field reconstruction work; `show=False` skips it.
The returned `fit.phase` and diagnostics remain regional, and
`ptycho.reconstruct(fit, ...)` reconstructs the full field using those optics.
`scan_region` changes fitting data; `result.show(region="top-left", ...)`
only changes the displayed comparison zoom.

The first `reconstruct(method="phase")` prepares the equations. Repeating it with only
`phase_damping` changed reuses them and is much faster. Defocus, tilt,
method or sampling changes require new preparation. Plotting with `show()` is
separate from the solve, so numerical update speed is not browser frame rate.

The acquisition must supply its calibration, or it must be passed explicitly
as in the [acquisition tutorial](../python-workflow.md). All three calls above
apply the **same fitted aberrations, scan rotation, specimen tilt and effective
depth**. Effective depth is a parameter of the transfer model, not an independent
measurement of physical thickness. Auto sampling covers modeled support; it
is not evidence of measured spatial resolution.

| Method | Unknowns | Reconstruction |
|---|---|---|
| `"ssb"` | Conventional corrected SSB phase | Existing sideband reconstruction |
| `"phase"` | Phase, with amplitude fixed to one | Regularized weak-phase linear solve |
| `"complex"` | Relative log-amplitude and phase | Coupled regularized weak-object linear solve |

For the regularized methods, at each scan frequency we model the measured
Fourier data as `g = H_phase phi + H_amplitude eta`, with
`T ≈ 1 + eta + i phi`. Aliases supported by the aperture are solved jointly.
The optical transfer uses the same fitted higher-order aberrations, tilt and
uniform effective-depth averaging. The returned wave is `exp(eta + i phi)`.
This final representation does not make the underlying linear fit an exact
nonlinear scattering model or a multislice reconstruction.

The objective is measurement disagreement plus separate quadratic terms:

```text
||sqrt(W) (g - H_phase phi - H_amplitude eta)||²
  + s * phase_damping * phi* D phi
  + s * amplitude_damping * eta* D eta
```

`eta` and its term are absent in the phase method. `s` normalizes the penalty
by weighted aperture power. Both strengths are absolute dimensionless controls;
the amplitude value is **not a multiplier** of the phase value. Larger values
suppress poorly measured components more strongly, and can also suppress real
features. The default strength is 0.03 for each applicable channel.
These are Tikhonov (damped least-squares) strengths, hence the names. The
earlier spellings `phase_regularization`, `amplitude_regularization` and
`upsampling_factor` still work during a transition and raise a
`DeprecationWarning` naming the new keyword; passing both spellings raises
`TypeError`.

The default smoothness prior is `D(q) = (1 + (q/q0)²)²` with
`q0 = smoothness_inv_A = 1` in inverse Angstrom. Use `smoothness_inv_A=0`
for a flat prior. The same numerical strength with a flat prior and with the
smoothness prior does not describe the same reconstruction.
`detector_weighting="inverse_mean"` offers a diagonal variance approximation
with a 5%-of-median count floor. The default is `"uniform"`.

Both methods set the mean phase and mean log-amplitude to zero and exclude
the DC alias family and the even-grid Nyquist axes. Amplitude is relative;
these data alone do not establish absolute absorption. Regularization does
not supply missing independent measurements or validate apparent atom peaks.

## Review and reuse

```python
phase.show(compare=ssb, region="top-left", cmap="gray", size="large")
phase.reconstruction_options
phase.reconstruction_diagnostics
ptycho.show(view="regularization", upsampling="auto")
```

Pass several results to compare damping strengths in the same figure:

```python
phase_001.show(
    compare=[phase_003, phase_005], region="bottom-left", cmap="gray",
    interactive=True, show_fft=True, columns=3, size="large",
)
```

Full fields appear above their matching zooms. Every panel uses the first
result's phase limits, and each title identifies the method and damping strengths.
Use `interactive=False` (the default) for a static Matplotlib figure, or
`interactive=True` for linked interactive inspection. `size` accepts `"small"`,
`"medium"`, `"large"`, and `"xlarge"`; `columns` sets methods per row.
Interactive comparisons require matching pixel grids. These viewers inspect
saved results; `ptycho.show()` opens ShowPtycho for live aberration edits.

`show_fft=True` adds an FFT of each exact displayed crop below the images.
Without a crop, it uses the full field. FFTs remove the mean, apply a Hann
window, and divide magnitude by the window sum. They share a frequency band
and a 60 dB display range relative to the common maximum, preserving relative
signal strengths. The reciprocal scale bar is calibrated from the crop size
and object sampling. FFT appearance is a diagnostic, not an independent
resolution measurement. Interactive image panning does not redefine the saved
crop or its FFT; change `region` and rerun `show` to compare another crop.

Comparison titles identify conventional SSB, regularized phase-only, or
amplitude + phase, including recorded damping strengths. Upsampling
is labeled separately from the phase units. Full fields remain above the
matching marked zooms with a shared contrast range. Choose `region="center"`
(the default), `"top-left"`, `"top-right"`, `"bottom-left"`, or `"bottom-right"`.
This selects only the displayed zoom; it does not crop the acquisition or refit.

The widget and Python calls use the same backend. Changing only strengths
reuses prepared normal equations. Phase updates use a cached eigendecomposition;
complex updates solve the small coupled systems with the two chosen strengths.
Changing optics, rotation, model, sampling, weighting or the smoothness prior
builds a new system. A selected region owns a separate session. The session
retains one prepared system to bound memory. As for conventional SSB, input
arrays must remain unchanged during the session; create a new session after
changing measured data. `force=True` rebuilds the prepared equations.

`reconstruct()` creates independent result arrays. Saving regularized alternatives
uses filenames containing a scientific signature, so changing strengths keeps
the previous saved result. Exact file-backed matches can be restored without
solving again; array inputs record provenance but are recomputed. Model choices
and numerical controls are saved with the result.

The regularized training prediction error is in
`result.reconstruction_diagnostics["training_relative_sse"]`. It is distinct
from conventional SSB's phase-variance `loss`, and neither is an independent
resolution measurement. Compare held-out data and repeatability as well as
images with shared display scales.

## Reconstruct a recorded time series

```python
fit = ptycho.fit_aberrations(search="broad", save_dir="results")
series = ptycho.series(frames, fit, method=("phase", "complex"), save_dir="results")
series.report()
drift = series.align()
series.show(size="large")
```

`series` reads recorded frames (files) one at a time. It fits the full optics once (`fit`, from the first frame) and, for each
later frame, refits only defocus C10 and two-fold astigmatism C12, phi12,
starting from the previous frame. Higher orders, sample tilt and effective
depth stay at `fit`. Each frame is read compressed onto the session's GPU with
the session's calibration and released before the next one. `ptycho.stream(...)`
is the earlier name of `series` and still works with a `DeprecationWarning`;
the result class `SSBSeries` was `SSBStream` (kept as an alias).

Every frame is reconstructed by SSB, and by each method in `method`
(`"ssb"`, `"phase"`, `"complex"` or a tuple) at that frame's optics, with
`reconstruct`'s strengths (`phase_damping=...`). Each frame is registered on its
SSB phase against the last confidently placed frame (lattice-safe band
registration from `quantem.gpu.drift`, Bragg spots from the first frame) and
added in register to `series.combined[...]`. A frame whose pair confidence
falls below 0.04 is measured but not added. In a notebook, `show=True` (the
default) draws the newest frame, the combined image and the drift so far, at
most twice a second.

On a measured 40-frame series (512 x 512 scan, 192 x 192 detector, 1536 x 1536
output, one GPU) a frame costs a median 0.84 s with phase-only and amplitude +
phase: 0.49 s reading the file, 0.05 s refitting, 0.03-0.05 s per method,
0.013 s registering. Registration on SSB, phase-only or
amplitude + phase agrees within 0.05 Å.

- `series.report()`: per frame, the refit optics, evaluations, shift against
  frame 0 (Å), live confidence and seconds per stage.
- `series.align()`: registers every pair up to three frames apart, solves one
  shift per frame by least squares, replaces `shifts_A` and rebuilds every
  combined image. Returns the `Drift` (pairs, closure, σ).
- `series.reconstruct("phase", phase_damping=0.05)`: rebuilds a method
  after the series from each frame's saved optics (`series.fits[k]`) and shifts.
- `series.show(interactive=False)`: the combined image of each method with the
  displacement and C10/C12 traces; `interactive=True` returns an interactive viewer.
  Pixels no frame reaches are masked, not drawn as data.

With `save_dir`, each frame's optics and shift are saved as it finishes. A
later call with the same `fit` and frames rebuilds the recorded frames without
refitting or registering and continues; the result equals an uninterrupted run.
Every frame must have the scan shape of the first.

## Drag previews

A viewer drags one control at a time. `direct_ptycho.direct_workflow.drag` serves those
frames for `method="phase"` and `method="complex"` from a Chebyshev drag
window along the dragged control, on CUDA and MPS, with the window rule of
`direct_ptycho.chebyshev` that native Metal shares: a series of the prepared normal
equations over a range around the current value (C10 ±6.8 nm, C12 ±3.4 nm at
300 kV and 30 mrad), every other input held. `prepare` asks for windows before
a grab and `advance` adds one exact preparation per call between frames; until
a window is ready, and outside its range, frames are exact. A window frame
agrees with the exact image to about 1e-6 of its largest phase. `run`,
`preview` and saved results never come from a window.

## Migration

Use `from quantem.gpu import DirectPtycho` and `ptycho.reconstruct(...)` in new code.
The public methods are `fit_aberrations()`, `reconstruct()`, and `scan_rotations()`.
Update old `find_aberrations()`, `run()`, and rotation-comparison calls to these
names; the renamed methods do not retain aliases. Saved calibration/result
schemas and their reading paths remain unchanged. The series entry point is
`DirectPtycho.series(...)`; there is no `stream(...)` spelling.
The package is `quantem.gpu.direct_ptycho` (for example
`from quantem.gpu.direct_ptycho.persistence import FIT_SCHEMA`); there is no
`quantem.gpu.ssb` alias.
The established `SSBResult` type holds all three direct methods. The older SSB
phase estimators are retired: `phase_estimator="mean_phase"` raises naming
`method="phase"`, and `"complex_wave"` naming `method="complex"`. Results saved
with them still load as saved, with a one-line notice.

Iterative object/probe refinement is the separate `myptycho.IterativePtycho`
workflow, also using `.run(...)`. Slices belong to its object model. Optical
fitting and iterative refinement are different operations even though their
scientific models are related.
