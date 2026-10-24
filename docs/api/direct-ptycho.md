# Choose a direct reconstruction

`DirectPtycho` owns acquisition preparation, SSB-based optical fitting and
three direct reconstruction methods. The default remains conventional SSB.
Regularized phase and complex reconstruction are experimental CUDA methods;
MPS continues to support conventional SSB. This API does not imply that a
linear weak-object model is accurate for every specimen.

Use the paired development sources (`quantem.gpu` 0.0.1rc18.dev0 and
`quantem.widget` 0.0.1rc44.dev0). These are source candidates, not a package-index
release; use branch `oct-24-direct-iterative-api` in both repositories and
install both checkouts, or install the matching supplied wheels. An older
widget does not provide the method and regularization controls.

```python
from quantem.gpu import DirectPtycho

ptycho = DirectPtycho.open("scan.qem")
optics = ptycho.find_aberrations(tilt=True)

ssb = ptycho.run(optics, upsampling_factor="auto")
phase = ptycho.run(
    optics, method="phase", phase_regularization=0.03,
    upsampling_factor="auto", save_dir="results",
)
complex_result = ptycho.run(
    optics, method="complex", phase_regularization=0.03,
    amplitude_regularization=0.1, upsampling_factor="auto", save_dir="results",
)
```

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
  + s * phase_regularization * phi* D phi
  + s * amplitude_regularization * eta* D eta
```

`eta` and its term are absent in the phase method. `s` normalizes the penalty
by weighted aperture power. Both strengths are absolute dimensionless controls;
the amplitude value is **not a multiplier** of the phase value. Larger values
suppress poorly measured components more strongly, and can also suppress real
features. The default strength is 0.03 for each applicable channel.

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
phase.show()
phase.reconstruction_options
phase.reconstruction_diagnostics
ptycho.show(view="regularization", upsampling_factor="auto")
```

The widget and Python calls use the same backend. Changing only strengths
reuses prepared normal equations. Phase updates use a cached eigendecomposition;
complex updates solve the small coupled systems with the two chosen strengths.
Changing optics, rotation, model, sampling, weighting or the smoothness prior
builds a new system. A selected region owns a separate session. The session
retains one prepared system to bound memory. As for conventional SSB, input
arrays must remain unchanged during the session; create a new session after
changing measured data. `force=True` rebuilds the prepared equations.

`run()` creates independent result arrays. Saving regularized alternatives
uses filenames containing a scientific signature, so changing strengths keeps
the previous saved result. Exact file-backed matches can be restored without
solving again; array inputs record provenance but are recomputed. Model choices
and numerical controls are saved with the result.

The regularized training prediction error is in
`result.reconstruction_diagnostics["training_relative_sse"]`. It is distinct
from conventional SSB's phase-variance `loss`, and neither is an independent
resolution measurement. Compare held-out data and repeatability as well as
images with shared display scales.

## Migration

Use `from quantem.gpu import DirectPtycho` and `ptycho.run(...)` in new code.
`SSB` remains a compatibility alias to the same class and `.reconstruct(...)`
forwards to `.run(...)` with a deprecation warning. Saved calibration/result
schemas and their reading paths remain unchanged. The series entry point is
`DirectPtycho.run_series(...)`; the previous spelling remains readable.
The established `SSBResult` type holds all three direct methods.

Iterative object/probe refinement is the separate `myptycho.IterativePtycho`
workflow, also using `.run(...)`. Slices belong to its object model. Optical
fitting and iterative refinement are different operations even though their
workflow verbs are consistent.
