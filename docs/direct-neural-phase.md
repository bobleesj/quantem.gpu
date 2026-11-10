# Neural phase in a direct reconstruction

Use one completed acquisition and the regular `DirectPtycho.run` call:

```python
optics = ptycho.aberrations(tilt=True, show=False)
model = ptycho.train_neural_phase(optics)
reconstruction = ptycho.run(
    optics, method="phase", neural_model=model, phase_damp=0.03, upsample="auto"
)
reconstruction.show(show_fft=True, cmap="gray")
phase = reconstruction.phase
neural_phase = reconstruction.neural_phase
```

The existing direct result owns both arrays. The direct wave and phase are
unchanged by adding the model. No series, callbacks, acquisition lifecycle,
transition flags, or separate reconstruction wrapper are needed. Existing
`series(..., neural_model=model)` callers remain supported for repeated scans.

## Fixed inference architecture

1. A four-level residual U-Net reads normalized direct phase and a fixed noise
   channel. Its output is a zero-mean phase proposal.
2. Eighteen learned readout weights combine spectral phase bands, local phase
   features and a measured annular guide. Hardware-flagged detector pixels do
   not contribute interpolated counts to that guide.
3. Current-acquisition normal equations constrain the proposed residuals.
4. The checkpoint's spectral heads produce an averaged correction bounded by
   25% of the anchor coefficient magnitude, with a cosine taper from 1 to
   1.5 inverse Angstrom. This is a prior bound, not a resolution guarantee.

The versioned checkpoint fixes U-Net width, learned weights, number of heads,
annular collection angles, voltage, convergence angle and phase sampling.
Reconstruction applies the fitted weights without further training. The
preceding `fit` call learns from measured diffraction, with no simulated
reference images. Its validity still depends on the weak-object model,
calibration and generalization of the supplied weights.

The neural input uses phase damping 0.03. A different `phase_damp` changes
only the displayed direct baseline; the required input is solved using the
same prepared diffraction equations. Titles state both strengths. The model
requires unit smoothness and uniform detector weighting.

## Calibration and persistence

`scan_basis=B` optionally supplies an independently calibrated physical scan
basis for direct CUDA models: physical positions equal B times nominal
positions, in (row, col) order. Frequencies therefore transform by the inverse
transpose of B. This leaves the measured counts and nominal output grid intact;
it does not estimate the basis or rectify the displayed image. Both direct and
neural paths use the same optics and geometry. Omit it for the existing model.

`save_dir` persists the direct wave, neural phase, measured annular counts,
model identity, calibration, numerical settings and timings. Reuse requires an
exact signature and both neural arrays. A missing array triggers recomputation.
The portable checkpoint contains learned weights; scientific data and original
training artifacts belong outside the public repository.

## Reproduce and evaluate

The everyday notebook, kept in the private spectra repository, runs raw-count
loading, a fresh optical fit, staged training, full-field phase/zoom/FFT comparison
and a held-region check. Only the acquisition and recorded microscope settings
are required. `fit` returns an in-memory model. `model.save("phase-model.pt")`
is optional for reuse; it is not a prerequisite.

The advanced notebook, in the same repository, adds live
Show4DSTEM and ShowPtycho controls, a wrapped model-probe phase view,
underdamping/overdamping examples, damping selection and stage validation curves.
The widgets need a live Jupyter kernel. They browse measured diffraction and
the optical model; they do not yet provide a linked predicted-pattern/residual
browser. Keep the acquisition open while exploring.

### Training and stopping

The default `recipe="staged"` trains a width-eight residual U-Net, fits an
18-weight annular readout, then trains two bounded spectral heads. It uses
two disjoint spatial patches. Within each patch, complementary 4 × 4
bright-field detector blocks predict each other. The first patch supplies
optimizer gradients; the second selects weights and readout regularization.
Each stage includes the unchanged input as a candidate. The final two heads
are averaged at inference.

`steps` limits U-Net updates (default 400); `direction_steps` limits each head
(default 2400). The U-Net retains its best checked validation step. Each head
stops after five hundred updates without a significant validation improvement,
after at least six hundred updates, and retains its best checked weights.
`model.show()` plots the recorded selection curves without retraining.

`model.score(ptycho, optics, partition="detector")` evaluates a third unused
patch. An explicitly supplied patch overlapping training or selection is
rejected. This tests predictions conditional on shared full-data optics, dose
normalization and a disjoint annular guide; it is not completely independent
calibration validation. Full-field inference uses a lower noise-level channel
than detector-half training. That extrapolation also needs validation.

The previous one-head electron-half training recipe remains available via
`recipe="count_split", steps=60, seed=42`. Existing frozen model files and
inference behavior are preserved. A newly fitted model need not reproduce
earlier weights or image quality. Cleaner FFTs alone do not certify new
resolution, atom positions or depth separation.
