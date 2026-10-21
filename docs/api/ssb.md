# SSB API

Use `SSB` from Python on MPS or CUDA. Find aberrations once, inspect the search,
and reuse those parameters for reconstruction. Start with the
{ref}`Python workflow <reconstruct-phase>` for a short example.
For actual images and saved outputs, run the {ref}`Gold notebook <gold-ssb-reconstruction>`.
This page covers parameters, calibration, ownership, and backend limits.

(ssb-fit-methods)=
## Fit methods: model-based by default

quantem fits SSB aberrations with a physical model of the measurement, not an
image-quality score. For each trial set of aberrations, sample tilt and depth
spread, it predicts the bright-field data G(q, k) the detector should have
recorded, then keeps the parameters whose prediction agrees best with the
measurement (least squares). This model-based, tilt-corrected fit is the
default everywhere, and it is where SSB in quantem is heading.

| Call | Method | Use it for |
|---|---|---|
| `find_aberrations()` | model-based, tilt-corrected | **the default**: every fit |
| `find_aberrations(tilt=False)` | model-based, tilt and depth spread held at 0 | thin, untilted samples, or to see what tilt correction changes |
| `find_aberrations(tilt=False, objective="phase_variance")` | phase variance, the earlier SSB search | a backup: reproducing older results, cross-checking a fit |

Phase variance chooses the aberrations under which the corrected phase images
of all bright-field pixels agree best (the least phase variance across them).
It stays as a backup, not an equal alternative: it scores images instead of
predicting the data, it cannot see sample tilt or depth spread, and it is about
10x slower (gold, both 180° branches: about 75 s against 7.8 s). On simulated
crystals with known aberrations the model-based fit lands about 3x closer to
the true C10. Evidence: [the standard-fit objective](../maintainer/2026-10-20-ssb-standard-fit-objective.md).

`objective` is `"agreement"` (default) or `"phase_variance"`, kept for
reproducing older results and cross-checks; with `tilt=True` the phase
variance raises. Each method searches 200 seeded trials and then refines:
`refinement="auto"` is Newton for the agreement fits and Nelder–Mead for the
phase variance, and for an agreement fit that locks a coefficient, bounds
phi12 or holds a nonzero C12. `result.objective` records the objective and
`report()` shows it; `result.loss` is the phase variance at the fit whatever
the objective. The same names pass through `compare_rotation_branches(...)`
and are part of a saved fit's settings, so a saved phase-variance fit never
answers an agreement call.

| Method | MPS | CUDA | Browser |
|---|---|---|---|
| `find_aberrations()` | yes | yes | no fit |
| `find_aberrations(tilt=False)` | yes | yes | no fit |
| `find_aberrations(tilt=False, objective="phase_variance")` | yes | yes | no fit |

On CUDA, `tilt=False` was the phase-variance search until 2026-10-20; that
search is now `objective="phase_variance"`, the same search as on MPS
(`ssb/phase_variance_fit.py`). The browser engine has no aberration search.

## Inputs and outputs

`SSB.open(path, ...)` loads a supported detector source. `SSB(data, ...)`
accepts an existing `Dataset4dstemGPU` or supported array. Prefer loading once
and inspecting the acquisition before constructing the algorithm:

```python
from quantem.gpu import SSB, io

data = io.load("acquisition.qem")
data.calibration.semiangle_mrad = 30   # set what the file does not record
data  # The notebook displays geometry, calibration, units, and overridden values.
```

In the next cell, pass that same dataset to SSB and reconstruct:

```python
ssb = SSB(data)
result = ssb.reconstruct()   # fits the aberrations once when none are known
```

Both file-backed entry points use the loaded `voltage_kV`, `semiangle_mrad`,
`scan_sampling_A`, and `detector_sampling_mrad` metadata unless explicitly
overridden. The first three remain required when supplying an array without
metadata. Missing physical calibration raises an error. Explicit keyword
overrides affect the session without rewriting `data.metadata`.

`scan_detector_rotation_deg` is the physical scan-to-detector rotation, kept in
[0, 360). Omitted, SSB uses the acquisition's recorded rotation, else estimates
it from the curl of the centre-of-mass field and prints one line. The curl knows
the angle only modulo 180 degrees; for an estimated angle `reconstruct()` and
`find_aberrations(check_rotation=True)` fit both 180-degree branches and keep
the one whose atomic columns have positive phase (see
[Settle the 180° branch](#settle-the-180-branch)). A given or recorded angle is
fitted once and never moved. `scan_detector_rotation_flipped=True` means the
detector's row and column axes are swapped relative to the scan, which no
rotation can produce; it is estimated with the angle, or False when an angle is
given. Reported centers stay in the original detector axes. Give a calibrated
angle to skip the estimate: `SSB(data, scan_detector_rotation_deg=12.3)`.

`reconstruct()` fits the aberrations with `find_aberrations()` defaults (the
tilt-corrected joint fit) when the session has none, then reuses them. Call `find_aberrations(...)` first to
constrain the fit: `search_ranges`, `fixed_aberrations`, `tilt=True`, `seed`.
The joint fit takes a `C10_nm` range or value and a `C12_nm` range from 0, or
holds `C12_nm=0`:
`ssb.find_aberrations(tilt=True, search_ranges={"C10_nm": (-50, 50)}, fixed_aberrations={"C12_nm": 0.0})`.
It searches `phi12_deg` over its whole period, so a `phi12_deg` range or a
nonzero held `C12_nm` alone raises and names what to pass; holding all three
coefficients needs `check_rotation=False`. The joint fit searches depth
spreads in `quantem.gpu.ssb.DEPTH_SPREAD_RANGE_NM` (2 to 60 nm). Assign `ssb.aberrations = {...}` to
state a known calibration, which `reconstruct()` then applies without fitting.
`detector_sampling_mrad` is detector angular sampling in
mrad per pixel, `(row, col)`. Calibrate it as its own step before SSB:

```python
fit = data.calibration.fit_detector_sampling()   # semiangle / bright-field edge radius, recorded on data
fit.show()                                       # the mean pattern with the fitted disk edge, axes in mrad
```

The bright-field disk is the image of the aperture, so its edge sits at the convergence semiangle. The fit
records the detector step with the source **Fitted (bright-field disk)**, together with the disk centre and its
edge radius, and `SSB(data)` reads all of them. When nothing is recorded, CUDA and MPS estimate the step with the
same rule (`quantem.gpu.detector.fit_bright_field_disk`: `semiangle_mrad` divided by the radius in detector
pixels at which the bright-field disk of the full-detector mean pattern falls to half its plateau) and print one
line saying so. A detector step set by hand, on `data.calibration` or as a keyword, always wins. In
`SSB(data, bright_field_center_px=..., bright_field_radius_px=...)`, disk geometry uses
the input detector grid. The session adjusts the center if it takes a
bright-field crop.

`SSB.open` uses the canonical `io.load` path, including its ANS-only GPU
acquisition policy.

`SSB.open` owns the acquisition it loads. On CUDA, for fit-sized scans, it
decodes the bright-field crop and releases the acquisition; large CUDA integer
sources stay encoded until `close()` or context-manager exit. On MPS, integer
acquisitions stay encoded until `close()`: the session decodes their
bright-field columns a few scan rows at a time while it builds G(q, k) and
keeps only G(q, k), so a 512 x 512 scan holds no decoded copy of its crop.
`SSB(data, ...)` borrows the caller-owned dataset or array and never releases
it; on MPS keep the dataset open while the session is in use. `close()` frees
only the session's own crop and Fourier data. Encoded inputs decode only the
bright-field evidence needed by the session.
Treat detector values as fixed for the lifetime of a session: SSB retains their
prepared Fourier data. After editing the array, construct a new `SSB` session.
SSB Fourier-stack preparation and reconstruction are additional work; they
are not included in a detector-viewer loading-time claim.

`find_aberrations()` returns one `SSBResult`. Its primary field is the complex64
`object_wave` with shape `(scan_row, scan_column)`. `phase` and `amplitude` are
derived as `angle(object_wave)` and `abs(object_wave)`. The result also records
the backend, fitted aberrations, rotation, loss, trial counts, the local
`refinement` applied (`"nelder-mead"`, `"newton"` or None), bright-field
geometry, timings, reuse state, and provenance metadata. A fit that compared
the two 180-degree branches records them as `rotation_branches` (a
`RotationBranches`); `reconstruct()` keeps it on its result whenever it applies
that fit's coefficients, so `report()` shows the suggested rotation and
`result.show("rotation")` draws the evidence. Saved results (`save_dir`) are
`quantem.ssb-result/10` records; a `/9` record is computed again. A record
holds no absolute path: it names its source by file name (and its path
relative to the results folder when the data lives under it), and it is reused
when the acquisition's portable identity matches
(`quantem.gpu.ssb.persistence.acquisition_identity`: file name, size and the
SHA-256 of the first MiB of the master and of each data file, the scan and
detector shapes, and how flagged pixels were filled). A results folder moved or
copied with its data is therefore reused; another acquisition is fitted again.

## Shapes, coordinates, dtypes, and units

- detector input: `I[scan_row, scan_column, detector_row, detector_column]`;
- complex result: `object_wave[scan_row, scan_column]`, complex64, on the session's output grid;
- `bright_field_center_px`: `(detector_row, detector_column)`;
- `scan_sampling_A` and `object_sampling_A`: `(row, col)` pairs, in Å; every
  entry point (`SSB`, `SSB.open`, `reconstruct_series`, `parallax.run`) also
  takes one number for both axes and stores and reports the pair;
- `C10_nm` and `C12_nm`: nm; and
- `phi12_deg`: degrees.

For a dense CUDA input, native supported scan shapes are 128×128, 256×256,
512×512 and 1024×1024. Other smaller scan shapes are padded with the mean
diffraction pattern to a supported square. For example, a 32×32 input currently
produces a 128×128 working grid before any requested upsampling. Check
`ssb.scan_shape` and `result.phase.shape`; padded pixels are not additional
measured scan positions; the session prints one line naming the padded size.
Dense CUDA arrays larger than 1024 along either axis are unsupported. Use the
encoded acquisition for the bounded-memory large-scan workflow below.
Computational padding is separate from `reconstruct(..., upsampling_factor=2)`.

The default fit evaluates 200 seeded TPE candidates of the least-squares
agreement of the SSB model with the full active bright field, keeps the best,
and refines it with Newton steps. The standard fit (`tilt=False`) runs the
same search with the sample tilt and depth spread held at 0, on CUDA and MPS;
`objective="phase_variance"` minimizes the phase variance with Optuna TPE and
Nelder–Mead ({ref}`Fit methods <ssb-fit-methods>`). No fit averages optimizer
candidates.

This is the implemented calibration workflow; Levenberg–Marquardt is not an
available refinement mode. The browser SSB engine has no aberration search:
run `find_aberrations()` on CUDA or MPS. The browser never substitutes fewer
trials or a reduced objective.

## Fit a region, reconstruct a large encoded scan

CUDA integer ANS acquisitions with native 1024, 2048 or 4096 scan dimensions
can remain encoded during SSB. Each detector-column batch is decoded exactly,
transformed across **every scan position**, corrected with the existing SSB
kernel, and accumulated before one inverse FFT. Scan tiles are not stitched.
The full source and output must fit on the acquisition's GPU; distributing one
SSB session across GPUs is not supported by this API yet.

Decoded batches use a temporary scan-major uint16 buffer, then a tiled CUDA
transpose writes the float32 FFT input. Counts remain exact in both buffers.
The uint16 scratch uses two bytes per scan position per detector column; at
2048×2048 with 32 columns, it adds 256 MiB. Batch sizing accounts for this
scratch, FFT workspace and the full output arrays before reconstruction.

```python
from quantem.gpu import SSB, io

with io.load("acquisition.qem", backend="cuda") as data:
    with SSB(
        data,
        voltage_kV=300,
        semiangle_mrad=30,
        scan_sampling_A=(0.3, 0.3),
        detector_sampling_mrad=(0.55, 0.55),
        scan_detector_rotation_deg=12,
    ) as ssb:
        fitted = ssb.find_aberrations(
            scan_region=(256, 768, 256, 768),
            tilt=True,
        )
        result = ssb.reconstruct(fitted, compute_loss=False)
        # Native 2048 input produces 2048 output here, with unchanged field of view.
```

Use measured calibration; the numbers above illustrate the units. A session
borrows `data`, so it can also inspect the same counts through detector APIs.
`SSB.open("acquisition.h5", ...)` uses the normal automatic ANS loader and owns
its encoded source. No manual conversion or saved intermediate is required.
Invalid detector pixels are excluded from scientific selection; their original
stored counts remain unchanged.

The fit region is an explicit square of 128, 256, 512 or 1024 scan positions.
Its returned phase, loss and rotation diagnostics describe **that region**.
Reconstruction applies the selected aberrations, tilt and depth spread to the
whole scan. This assumes those optics apply across the field; regional fitting
does not establish that assumption for a specimen with varying thickness or
long-scan drift. `result.fit_scan_region` records the region used.

The bounded path supports a native 1024×1024 scan and even scan dimensions
above 1024 up to 4096, including rectangular grids, uint8/uint16 ANS, C10/C12,
and optional sample tilt/depth. Smaller rectangular scans retain their existing
square computational padding.
It uses complex64 forward spectra, complex128 summation/inverse FFT and
float32 phase. Its default `phase_of_mean` estimator matches the existing CUDA
wave-average path within floating-point FFT rounding, not bit for bit.
Use `compute_loss=False` for the bounded path. Full-field per-detector
phase-variance loss, higher-order aberrations, `mean_phase` and
complex-amplitude reconstruction still use the existing Fourier-stack backend
for 1024 scans, retaining their prior behavior and memory cost. Those operations
are unsupported above 1024. Fit diagnostics remain available on the small
region. A full 1024 fit remains available, but an explicit smaller region avoids
keeping its entire Fourier stack during optimization.

`upsampling_factor="auto"` or an explicit factor changes output sampling and
memory requirements, **not the measured field of view**. A 4096 output from a
512 scan is distinct from a native 4096 scan. Large output arrays can still
exceed available GPU memory. Borrowed resident inputs without a persistent
source identity recompute saved results; file-backed sessions retain exact
saved-result reuse.

(ssb-find-and-reconstruct)=
## Find aberrations, then reconstruct

```python
from quantem.gpu import SSB

ssb = SSB.open(
    "acquisition.qem",
    voltage_kV=300,  # kV
    semiangle_mrad=30,  # mrad
    scan_sampling_A=(0.264, 0.264),  # Å per scan pixel, (row, col)
)
aberrations = ssb.find_aberrations(tilt=True, save_dir="results/aberrations")
aberrations.report()
ssb.show_trials(best=5)
result = ssb.reconstruct(aberrations, save_dir="results/native")
shown = ssb.reconstruct(aberrations, upsampling_factor="auto")
shown.show()
```

The numbers above illustrate the units; use your acquisition's calibration.
SSB here is model-based and tilt-corrected: `find_aberrations()` (`tilt=True`,
the default) jointly estimates specimen tilt and model depth spread as well as
aberrations, and so does the automatic fit of `reconstruct()`. `tilt=False`
holds the tilt at 0, and `tilt=False, objective="phase_variance"` is the older
SSB search ({ref}`Fit methods <ssb-fit-methods>`).
Fits stay on the native scan grid; what you look at is reconstructed at
`upsampling_factor="auto"`, the factor the scan sampling allows (at 300 kV
and 30 mrad: 2× for the 0.264 Å step above, 3× at 0.373 Å, 4× for the 0.5 Å
gold scan), and its title names the factor. `show_trials` and the branch
panels of `compare_rotation_branches` use the same factor. `result.show()`
draws the phase a result holds; for a native fit one line names the auto
factor and the call that shows it.
Keep `ssb` open for the examples below and call `ssb.close()` when finished.

Inspect the best joint fit while the same SSB session is open:

```python
ssb.show()                             # existing interactive ShowPtycho
ssb.show("depth", size="large")         # static focus-depth landscape
ssb.show("tilt", size="medium")         # static row-column tilt landscape
```

The static calls reuse recorded diagnostics or measure them once on the GPU.
If no fit exists, the usual joint fit runs first. `size="small"`, `"medium"`
and `"large"` only change figure dimensions. For explicit dimensions use
`axsize=(6, 5)` instead. Returned figures support `savefig(...)` and render
directly as a notebook cell's final expression.

Select a region in original scan pixels with exclusive stops:

```python
ssb.show("tilt", region=(64, 192, 64, 192), size="small")
```

This jointly fits the region's optics, tilt and depth with one seed, 200 trials
and Newton, keeping the full-field rotation and detector calibration. It
leaves the full-field fit unchanged. Each bounds-based call fits independently;
to inspect multiple views of the same regional fit, retain its session:

```python
region = ssb.select_region((64, 192, 64, 192))
region.show("depth")
```

Then `region.show("tilt")` reuses that fit and its measured slices in the next
cell. Call `region.close()` when finished.

`depth` means fitted **depth spread**, not measured specimen thickness. The
plots are conditional objective landscapes, not spatial depth/tilt maps.
Their marker identifies the joint fit; all unplotted parameters stay fixed.

The explicit measurement API remains available, including plotting from a
saved result without an open GPU session:

```python
ssb.measure_parameters(aberrations)
aberrations.show()  # phase beside the saved parameter slices
aberrations.show("parameters")
aberrations.show(show_parameters=False)  # phase alone
```

This evaluates up to 201 additional parameter settings on the selected GPU
without changing the fitted optics or phase. The saved axes use nm and mrad;
agreement is normalized to the fitted point. These are conditional objective
cuts, not uncertainty estimates or proof of a global optimum. If the fit was
saved, the measurements are saved with it and reopening performs no new
calculations. Measure a regional fit in the same regional session, rather than
against the full acquisition. Joint fitting requires measured spatial-frequency
support in its fitting band; unsupported scans raise a corrective error instead
of returning a zero-agreement fit.

`find_aberrations()` searches on the native scan grid.
`reconstruct()` applies the supplied parameters without
searching. CUDA and MPS use `phase_of_mean` at every output factor, including
all 14 higher-order terms. Both return phase-only complex waves with unit amplitude,
recorded as `amplitude_estimated=False`. That amplitude is not a measured
specimen transmission. Use explicit `phase_estimator="complex_wave"` for a
native thin-sample complex-object reconstruction when amplitude is needed.

`result.upsampling_factor` and `result.object_sampling_A` record the output factor and
`(row, col)` pixel spacing. Input sampling stays in the saved signature. Changing the
estimator, sampling, or scientific parameters invalidates saved-result reuse.

### Inspect and replay search trials

```python
ssb.show_trials(first=5)
ssb.show_trials(last=5)
ssb.show_trials(best=5)
aberrations.trials  # DataFrame, indexed by stable trial ID
aberrations["C10_nm"]  # defocus, nm
aberrations["C12_nm"]  # twofold astigmatism, nm
aberrations["phi12_deg"]  # astigmatism angle, degrees
aberrations.sample_tilt_mrad  # sample tilt (row, column), mrad; None if not fitted
aberrations.depth_spread_nm  # model depth spread, nm; None if not fitted
result.phase  # radians
result.object_sampling_A  # Å per output pixel, (row, column)
```

Coefficients use dictionary indexing; each key carries its unit, nm for
`C10_nm`/`C12_nm` and degrees for `phi12_deg`. `dict(aberrations)` makes a separate coefficient dictionary. Phase,
tilt, rotation and trial history remain properties of the fitted result.
The report table labels its angle column `phi12 (deg)`.

Supply exactly one positive selector. Images share phase contrast and include
scale bars and a parameter table. They are reconstructed at
`upsampling_factor="auto"` by default (the title names the factor); the search
losses in the table are native-grid values. Trials are replayed from their
recorded settings rather than stored as a large image stack. Browsing preserves the
active reconstruction. `best` ranks the latest search only; losses from
different objectives (the phase variance and the agreement) are not mixed.
`result.objective` names the fit's objective, and an agreement trial table its
`objective` column. Local refinement is separate from the search trials and
its final result appears in `report()`.

```python
attempts = ssb.reconstruct(aberrations, trial_ids=aberrations.trials.tail(5).index)
attempts.phase  # radians; axes (trial, row, column)
adjusted = ssb.reconstruct(aberrations, aberrations={"C10_nm": 12.5})  # nm
```

Overrides preserve unspecified coefficients, tilt, depth spread and rotation.
They leave the supplied result unchanged. Replay uses each trial's original
rotation branch: after a branch comparison the trials of both searches are kept,
`search` 0 at the first angle and 1 at the angle plus 180 degrees. Saved search
results retain this history.

## Finer preview sampling

Fit at native sampling, then reuse the fitted parameters for a finer preview:

```python
result = ssb.reconstruct(aberrations, upsampling_factor="auto")
```

`"auto"` selects the smallest supported factor (1, 2, 3, 4 or 8) whose output
Nyquist frequency covers the ideal circular-aperture SSB bandwidth,
`2 * semiangle / wavelength`, on both scan axes. Equivalently, output spacing
must be at most `wavelength / (4 * semiangle)`, with semiangle in radians.
This uses the calibrated voltage, convergence semiangle and scan spacing.
It is a numerical grid criterion, not a measured resolution: detector coverage,
noise, coherence, aberrations and alias separation can reduce useful bandwidth.
See the [direct-ptychography upsampling paper](https://arxiv.org/abs/2507.18610).
More than 8× raises an explicit limit error. CUDA and MPS support all 14
aberration terms at every listed factor, with or without specimen tilt and
depth correction. Both use `phase_of_mean` automatically. Explicit
`phase_estimator="mean_phase"` retains the historical diagnostic comparison.
The aberration search and loss remain on the native grid. Upsampling excludes
replicas of the scan mean while retaining the normalization at the true origin.

For an explicit sampling comparison:

```python
result_2x = ssb.reconstruct(aberrations, upsampling_factor=2)
result_4x = ssb.reconstruct(aberrations, upsampling_factor=4)
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

`find_aberrations()` has no `upsampling_factor` argument: the search stays at 1×.
The {ref}`Gold comparison <gold-upsampling>` shows 1× and 4× outputs with shared
contrast and a matched crop. That Gold result has visible bands at 4×;
the example does not establish improved resolution.

## Fit a region and reconstruct the full scan

```python
aberrations = ssb.find_aberrations(
    scan_region=(128, 384, 128, 384),  # row start/stop, column start/stop; scan pixels
    tilt=True,
)
result = ssb.reconstruct(aberrations, upsampling_factor="auto")
result.show()
```

The stop indices are exclusive. Only the selected region contributes to the
aberration search; reconstruction still uses the full scan. `scan_region=None`
fits the full scan. The detector calibration remains tied to the full input.
Regional fitting currently requires a dense 4D scan source and a square region
of 128, 256, 512 or 1024 pixels per side, without implicit padding or cropping.
The result records `fit_scan_region`, and saved-result matching includes it.
Its saved search phase and branch comparison describe that region (the
dark-field columns are located on the region of the full scan's dark field). Applying
those parameters elsewhere assumes the optics and specimen tilt are shared;
a locally good fit does not establish that the whole specimen is uniform.

## Settle the 180° branch

The centre-of-mass curl fixes the scan-detector rotation only modulo 180°: the
angle and the angle plus 180° reverse every CoM vector and both leave the field
curl-free. SSB fits both equally well, the second as the conjugate object with
C10 of the opposite sign. For a rotation the session estimated, `reconstruct()`
and `find_aberrations()` therefore fit twice, at the estimate and at the
estimate plus 180°: two independent searches with the same settings (tilt,
ranges, trials, seed), neither started from the other. The branch is decided on
tilt-corrected SSB, by positive phase at the atomic columns located on the dark
field, with the phase tail as the fallback: correcting the sample tilt and depth
spread makes the column contrast clearer, while both branches still fit equally
well. Each fit's phase is reconstructed at the native scan sampling, and the
branch is chosen in two steps:

1. **Atomic columns.** The columns are located on the dark field of the same
   scan, `detector.df` from 1.1 times the bright-field radius to the detector
   edge. Dark field is incoherent, so the positions are the same on both
   branches. The image is smoothed by a 1-pixel gaussian and high-passed by
   subtracting a 2.1 Å gaussian. Its local maxima at least 2 Å apart are
   ranked by prominence (height above the window's minimum) and the top 20% are
   the columns (an 8-pixel high pass and a 17 × 17 window at 0.264 Å). The phase
   minus its mean, high-passed the same way, is sampled at the columns:
   `z = mean / (std / sqrt(n))` and `effect = mean / std`. Atoms push the phase
   forward, so when either branch has `|z| >= 3` and `|effect| >= 0.25` the
   branch with the more positive columns is kept. The effect floor keeps a
   negligible shift over many columns from deciding.
2. **Phase tail.** When the column test is not decisive, or the session has no
   dark field, the branch with the larger mean-centred third-moment ratio
   `S = mean(p³) / mean(p²)^1.5`, `p = phase − mean(phase)`
   (`quantem.gpu.ssb.results.phase_skewness`) is kept: a few bright columns
   over a flatter background.

One line says which branch was kept and why, for example
`SSB: kept 339.4°: atomic columns positive (z +10.8, effect +0.70, 75% of 241 columns); 159.4° negative.`
The comparison costs a second full search: two joint fits took 9 to 25 s in
all on 512 × 512 scans on an M-series Mac, two MPS standard fits
(`tilt=False`) 3 to 10 s, two phase-variance fits about 90 s.

```python
branches = ssb.compare_rotation_branches()  # both fits; the session ends on the kept one
branches.report()                           # aberrations, loss, column z and effect, positive columns, S
branches.show()                             # both fits at the auto factor, then the evidence
result = ssb.reconstruct()                  # the kept fit, no third search
result.show("rotation")                     # the same figure from the result
```

`compare_rotation_branches(**find_aberrations_options)` takes every
`find_aberrations` keyword except `check_rotation`. A rotation given to
`SSB(...)` or recorded with the acquisition is the user's calibration:
`reconstruct()` and `find_aberrations()` fit it once and never move it, and
`compare_rotation_branches()` on it reports the evidence and keeps the given
angle unless called with `apply=True`. `find_aberrations(check_rotation=False)`
fits once at the session's rotation, estimated or not.

`find_aberrations(refine_rotation=True)` (off by default) also refines an
estimated angle after the branch decision. G(q, k) does not depend on the
rotation, only the bright-field k vectors do, so each angle costs one polish of
the fit from the nearest angle's fit (`set_rotation` retargets the k geometry
without rebuilding G): 1-degree steps from the estimate until the agreement
falls on both sides, at most 10 degrees away, then bounded Brent to 0.05
degrees, 7 to 12 polishes in all. One line reports it, for example
`SSB: refined rotation 358.31° → 1.24° (+2.93°, agreement +9.82%; 11 polishes, 12.5 s)`
on the 512 × 512 gold scan (M5 Max). `result.rotation_estimate_deg` keeps the
estimate, `result.rotation_refined` says whether the angle moved,
`result.rotation_refinement` records the agreement at both angles and the
angles evaluated, and `report()` shows both angles. A given or recorded angle
is never refined. Use it when the centre-of-mass estimate is suspect (weak curl
contrast, thick or tilted crystals) or before iterative ptychography. It finds
the fit's best angle within its model, which need not improve the SSB image: on
gold the phase image at the refined angle correlates 0.98 with the image at the
estimate, with the same column width, roundness and contrast. On a simulated
crystal (64 × 64, noise free) it recovers a 2-degree error to within
0.01 degrees.

Bright columns are a physical assumption, not proof of the absolute
orientation. The column test assumes the dark-field image peaks on the
columns. A dark field that collects only low angles from a strongly defocused
probe can break that: in a noise-free 64 × 64 simulation (4 Å lattice, dark
field 33 to 55 mrad, tilt-corrected fits) the column test chose the wrong
branch at C10 = −5 and −8 nm and the right one at −3 nm, while the phase tail
was right in all three.
In the figure, the top row is each branch's fit reconstructed at
`upsampling_factor="auto"` by the session (its own rotation, aberrations,
tilt and depth spread; 0.4 s for both 4× panels of the 512 × 512 gold scan),
with the dark-field columns marked at scan pixels times the factor. The
bottom row is the evidence, from the native phases the decision used: the
phase tails (left) and the high-passed phase at the columns (right).
`branches.show(panels=False)` draws the evidence alone, without the GPU.

## View the phase and model probe

```python
result = ssb.reconstruct(aberrations, upsampling_factor="auto")
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

CUDA and MPS support factors 1, 2, 3, 4 and 8 with all 14 aberration terms and optional tilt/depth
correction. The field of view stays fixed. The search and diagnostic loss stay
on the native scan grid; changing the factor does not refit parameters or
interpolate detector measurements. More output pixels do not guarantee more
resolved specimen detail. These Python backend guarantees do not qualify the
separate browser WebGPU or native Swift implementations.

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
- `num_trials` must be non-negative and `refinement` is `"auto"`,
  `"nelder-mead"`, `"newton"` (tilt fits) or `None`. With `tilt=True` and
  `num_trials=0` the session's previous joint fit is refined. Use
  `reconstruct()`, or assign `ssb.aberrations = {...}`, to apply known
  parameters without a search.
- To save results from an in-memory array, provide `source_path` for provenance.
  Direct arrays never reuse results from disk automatically: a path cannot
  identify which crop or edited array the scientist supplied.

## Provenance and exact reuse

`save_dir` names a results directory. `find_aberrations` writes
`ssb-find_aberrations.npz` and `.json` there, `reconstruct` writes
`ssb-reconstruct.npz` and `.json`; a path ending in `.npz` or `.json` is
refused. It writes the complex object and a readable signature containing source
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
    batch_aberrations = batch_ssb.find_aberrations(save_dir="results/aberrations")
    result = batch_ssb.reconstruct(batch_aberrations, save_dir="results/native")
```

The context manager closes the session after the block. In a still-open
notebook session, use `reconstruct()` when aberrations are known and no
optimizer should run:

```python
result = ssb.reconstruct(
    aberrations={"C10_nm": 12.5, "C12_nm": 3.0, "phi12_deg": 15.0},
    save_dir="results/fixed-ssb",
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

Taking a phase is nonlinear: `angle(mean(waves))` is not
`mean(angle(waves))`. Taking each angle first discards amplitude information
and can compress reconstructed contrast. Combining the corrected spectra
before one inverse FFT gives the same wave sum as summing separate inverse
FFTs, because both operations are linear. The diagnostic loss remains the
native-grid per-detector phase variance. Stronger contrast alone does not
establish quantitative phase accuracy.

QuantEM's `DirectPtychography` SSB implementation uses a related but distinct
weak-phase convention: multiply by `-1j * conj(gamma) / abs(gamma)`, inverse
transform, take the real part, normalize by BF weight, and sum contributions.
It does not average per-contribution phase angles. Do not describe our
DC-referenced `angle(mean(waves))` as numerically identical to that output.

CUDA and MPS default to phase after complex-wave averaging, including higher orders:

```python
phase, native_loss = ssb.preview(
    dict(aberrations),
    sample_tilt_mrad=aberrations.sample_tilt_mrad,
    depth_spread_nm=aberrations.depth_spread_nm,
    upsampling_factor=4,
    phase_estimator="phase_of_mean",
)
```

Both estimators use the same supplied aberration polynomial and depth weights, the
same measured diffraction patterns and the supplied calibration. Fitting never
runs inside `preview`. Diagnostic loss retains the original native-grid
per-detector phase variance, even for wave averaging. Use
`compute_loss=False` when only the image is needed.

The estimator choice applies at **every** output factor, including 1x. CUDA
and MPS previews use wave averaging at 1x as well; request `mean_phase`
explicitly to reproduce historical preview output. To compare 1x/2x/3x/4x
scientifically, use the same estimator at each factor. Switching from legacy
1x to wave-average 2x introduces an estimator change as well as a sampling change.

Reduced contrast compression from wave averaging does not establish
quantitative phase accuracy. Known-phase controls still show amplitude
attenuation, and improvement is not uniform across tested noise/sampling cases.
There is no fitted gain, display normalization or detector interpolation in
this option. Higher-order corrections, tilt/depth and automatic sampling can
be combined on both backends. The fit still searches C10/C12 and specimen
geometry; higher-order coefficients are supplied corrections, not extra fitted parameters.

This estimator does not change the native aberration-search objective.
It is also accepted by `reconstruct`, where it participates in saved-result reuse.
Live uses the same public preview default on both backends.
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
`find_aberrations()` (`tilt=True`, the default) fits the aberrations together with the sample tilt and a depth
spread, in one search with the same trial budget as the standard fit (`tilt=False`):

```python
import pandas as pd

standard = ssb.find_aberrations(tilt=False)  # C10_nm, C12_nm, phi12_deg; same objective, tilt held at 0
tilted = ssb.find_aberrations()              # + specimen tilt and depth spread, jointly

tilted.sample_tilt_mrad   # (row, col) mrad, scan frame
tilted.depth_spread_nm    # model depth spread, not a measured thickness
tilted.tilt_fit_gain      # least-squares agreement relative to standard SSB (> 1: tilt explains more)
pd.concat([standard.report(), tilted.report()])   # one row per fit
```

After a tilt fit, `C10_nm` is the defocus at mid-depth, so it can differ from the
standard fit's. The search is joint on purpose: fitting the tilt after a
standard fit leaves C10 behind and stalls (0.91-0.99 of the best agreement on
two acquisitions, every seed). With 200 trials every seed reached the same tilt
within 0.02 mrad; 300 and 400 trials gave the same answer.

`preview(aberrations, sample_tilt_mrad=(row, col), depth_spread_nm=d)` renders the
thick-sample model for interactive viewers; with no depth spread the tilt has
no effect. CUDA and MPS both implement it.

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
`save_dir` names the folder of per-acquisition records (QuantEM Live's
`<output>/live/screen`; default `<source>/live/screen`),
`scan_detector_rotation_flipped` gives the flip (None: the saved fit's, else
unflipped for a recorded rotation), and an acquisition with no recorded
rotation is estimated by `SSB.open`. `scan_sampling_A` may be one number.

`quantem.gpu.ssb.series.screen_record_settings(record)` returns the
`SSB.open` keywords a QuantEM Live screening record's stored fit used (beam,
`(row, col)` scan and detector sampling, physical rotation and flip,
bright-field centre and radius, aberrations), and
`series.open_screen_record(record, source=None, backend="auto")` opens the
session at that fit, so `ssb.show()` opens ShowPtycho there. The fit is the
record's tilt-corrected fit when it has one (`series.screen_record_fit`:
`computed.ssb_tilt` beside an ordinary fit, else `computed.ssb`), and the
session also takes its sample tilt and depth spread
(`series.screen_record_sample`).
`quantem.gpu.formats.dataset_yaml.file_entry(document, master)` finds an
acquisition's `files` entry by QuantEM Live's rules (its master name, a
`.qem` copy's master, then the trailing number or a range key such as
`"06-24"` that names no other master), and
`acquisition_scan_sampling_A(document, master, ...)` resolves its step.

A fit stored in `live/screen/<dataset>/config.json` is read only when its
`computed.ssb` block carries `"schema": "quantem.ssb-fit/7"`
(`quantem.gpu.ssb.persistence.FIT_SCHEMA`), which `reconstruct_series` writes
with every fit. A block with another schema or none is treated as absent: its
aberrations, rotation and recorded settings are not read or converted, the
probe reference is fitted again, and the physical settings come from
`dataset.yaml` or the keyword arguments. Evidence:
[stored fit schema](../maintainer/2026-10-11-stored-fit-schema.md).

That `config.json` is QuantEM Live's screening record, which Live reads only
with `"schema": "quantem.screen-acquisition/2"` at its top level
(`quantem.gpu.ssb.persistence.SCREEN_ACQUISITION_SCHEMA`). A record that
`reconstruct_series` creates carries that key. In a record that exists it
changes nothing outside `computed.ssb`; a record with another top-level
schema, or none, raises `ValueError` before anything is written and names
QuantEM Live's `tools/rewrite_screen_config.py`.

`dataset.yaml` is read only at schema `quantem.dataset/2`: the `microscope`
voltage and semi-angle, the file entry's `scan_detector_rotation_deg` (a
per-file `0.0` counts) else `microscope.scan_detector_rotation_deg`, and the
`[row, col]` step of the file entry, its `calibrations` row or
`calibrations.default`. Another schema raises and names QuantEM Live's
`tools/rewrite_dataset_yaml.py`.

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

### Inspect the launch calibration

Load first so acquisition metadata remains available independently of SSB:

```python
from quantem.gpu import SSB, io

data = io.load("scan.qem")
data
```

Voltage, convergence semiangle and scan step are read from the loaded metadata.
For scan-detector rotation, explicitly use the recorded angle, or provide an
independently calibrated angle when the file does not record one:

```python
ssb = SSB(data, scan_detector_rotation_deg=data.metadata["scan_detector_rotation_deg"])
ssb.calibration_table()
```

The metadata lookup intentionally raises if rotation is missing. Replace it
with your known calibrated angle; do not silently substitute zero. Omitting
rotation estimates it from the centre-of-mass curl, labelled **Estimated
(centre-of-mass curl)**, not a measured calibration.

`ssb.calibration_table()` returns a DataFrame with **File value**, **File unit**,
**Value used**, **Unit used**, and **Source**, using the session's current values.
File values include saved calibration overrides. Source labels are **Recorded**,
**Overridden** (set on `data.calibration`), **Fitted (bright-field disk)** (`data.calibration.fit_detector_sampling()`),
**Explicit override**, **Estimated**,
**Estimated (centre-of-mass curl)** and **Default**, from the one label set of
`quantem.gpu.io.metadata_display`. Later
session changes are labeled **Modified in session**. Explicit keywords remain
overrides even when they equal stored values. Labels describe provenance,
not validation or confidence. Reading the table does not reconstruct or alter
source data; detector estimates remain **Pending probe fit** until preparation.

Detector sampling is optional: omit `detector_sampling_mrad` to reuse the
loaded acquisition's saved calibration. If missing, SSB estimates sampling
from the full-field probe disk edge and the known convergence semiangle.
The working session retains the estimate for reconstruction, regional sessions
and persistence. The source QEM and its recorded metadata remain unchanged.
An explicit numerical keyword remains available for an independent calibration.
The convergence semiangle must be known; disk radius alone cannot set an
absolute angular scale.
