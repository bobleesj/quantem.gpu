# Neural phase for a scan series

Fit a model on a representative counting scan. Reuse its frozen weights for
later scans of the same field. Each scan supplies its own diffraction data;
previous images guide registration, not the current neural phase. This
workflow uses CUDA. No pretrained specimen model or simulated training data
is required.

## Train and check one scan

Start with a calibrated counting dataset. The everyday notebook, kept in the
private spectra repository, includes the optical fit, staged training and
direct/neural phase comparison.

```python
from quantem.gpu import DirectPtycho

with DirectPtycho(reference_scan) as ptycho:
    optics = ptycho.fit_aberrations(show=False)
    model = ptycho.train_neural_phase(optics)
    scores = model.score(ptycho, optics, partition="detector")
```

Default training uses complementary bright-field detector blocks on one
scan patch, then selects the staged U-Net, annular readout and bounded heads
on a separate patch. `model.show()` displays selection curves. The detector
score above evaluates a third unused patch in both directions, conditional
on shared optics and the measured annular guide. Lower error means better
prediction under this linear model; it does not alone establish resolution.

`model.save("phase-model.pt")` optionally writes the fitted model for another
session. The current session can reuse `model` directly. The previous
electron-half recipe remains available with `recipe="count_split", steps=60,
seed=42`; `score(partition="electron")` retains its three-way comparison.
The advanced notebook, in the same private repository, covers damping, probe
phase and measured-pattern exploration.

The public fitting method is `DirectPtycho.fit_aberrations()`. Older examples used
`find_aberrations()`, which is no longer a public method. Saved fit files
retain the historical `ssb-find_aberrations` prefix. The native Swift
entry point is `DirectPtychoModelFit.findAberrations`.

## Reuse the model

```python
model = DirectPtycho.load_neural_phase("phase-model.pt")
with DirectPtycho(reference_scan) as ptycho:
    fit = ptycho.fit_aberrations(show=False)
    series = ptycho.series(fit=fit, method="phase", neural_model=model,
                           max_frames=2, show=False)

series.add(completed_scan)
reconstruction = series.latest
reconstruction.show()
```

Pass a resident acquisition or its file path to `add`. A receiver can use
`process=series.add`; it owns the input through the call. A sequence of files
can go straight into `ptycho.series(paths, ...)`, as in the
[multiple-scan notebook](../examples/reuse-phase-across-scans.ipynb). By
default, each scan refits defocus and twofold astigmatism at fixed higher-order
optics. These fits never retrain the network. Use `refit="none"` for a
fixed-optics comparison.

- `reconstruction.phase` is the phase-only direct solution in radians.
  `neural_phase` adds the frozen, measurement-conditioned correction.
  `complex_phase` and `amplitude` are available when `"complex"` was requested.
- `reconstruction.annular_counts` is the measured virtual-detector image on
  the native scan grid. `annulus_mrad` records the inclusive inner and
  exclusive outer angles. The calibrated center, row and column angular
  sampling and detector validity mask determine which pixels contribute.
- `reconstruction.aberrations` is a copy of the aberration coefficients applied to that scan. `displacement_A`
  is the measured (row, column) shift relative to the first scan. `placed`
  tells whether registration accepted it for the combined image.

`series.latest` is `None` before the first scan. An unrequested image is
`None`. A rejected scan still has its individual images. Hold the returned
snapshot to keep those GPU arrays after a rolling window advances. The series
keeps only the latest annular image. `reconstruction.show()` pairs that
scan’s phase and annular counts with separate contrast and calibrated scale bars.
`series.show()` compares images combined over accepted scans.

## Check what the annular detector contributes

The default displayed band is `model.annulus_mrad`. To compare network on
and off, use the same reference fit, specimen and band. Here `scan` is one
measured acquisition; both series use the same optical refit policy.

```python
with DirectPtycho(reference_scan) as ptycho:
    fit = ptycho.fit_aberrations(show=False)
    control = ptycho.series([scan], fit=fit, method="phase",
                            annulus_mrad=model.annulus_mrad, show=False)
    learned = ptycho.series([scan], fit=fit, method="phase",
                            neural_model=model, show=False)

control.latest.show()  # direct phase + measured annular counts
learned.latest.show()  # neural phase + the same measured counts
```

You can inspect another band by passing `annulus_mrad=(inner, outer)` on a
neural series. Only the measured view changes; the learned guide still uses
the checkpoint’s angles. With neither a model nor angles, no annular image
is requested. This control tests the total effect of refinement. Isolating the
guide’s contribution requires fitting matched models with and without it.

## What the correction can support

The U-Net proposes a residual from normalized direct phase. A separate learned
readout uses phase features and annular intensities. Both proposals are filtered
by the current scan’s diffraction equations. A bounded final Fourier update
uses the remaining mismatch and tapers to zero between 1 and 1.5 Å⁻¹. This
taper applies only to the final update. Annular brightness can help locate
scattering contrast; it does not establish chemistry, improved resolution or
layer separation.

Version-one checkpoints require uniform detector weighting, phase damping
0.03 and a 1 Å⁻¹ smoothness scale. Loading verifies the schema; applying the
model checks voltage, convergence semiangle, phase sampling and upsample.
The saved record includes the seed, training and validation patches, history
and selected step. Series provenance includes the checkpoint SHA-256.
Use a region supported by the regularized direct solver if a full field
exceeds its resident limit.
