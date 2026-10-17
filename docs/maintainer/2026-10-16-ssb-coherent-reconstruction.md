# Coherent SSB reconstruction and scan-mean aliases

C10/C12 reconstruction now uses `phase_of_mean` on both Python MPS and CUDA.
The corrected complex waves are summed before phase is extracted. Explicit
`phase_estimator="mean_phase"` retains the previous per-detector phase average;
higher-order reconstruction retains its supported estimator. This changes
MPS displayed phases intentionally. It is not a bit-exact speed optimization.

## Why this choice

For corrected detector waves `w_k`, the coherent estimate is
`arg(sum_k w_k)`. Averaging `arg(w_k)` is a different nonlinear estimate.
They can agree in the weak-phase limit and differ when phases or amplitudes
vary. Larger contrast alone does not establish accurate projected potential,
particularly for a thick specimen. No universal absolute-phase accuracy is
claimed by this change.

A constant-scan control exposed an upsampling artifact on both backends.
Unfolding scan-frequency aliases also repeated the scan mean into nonzero
frequencies. Those copies can fall inside the probe support at coarse scan
sampling. Reconstruction now excludes all nonzero replicas of the scan mean
while keeping the existing real normalization at the true origin. The native
grid, nonzero measured Fourier terms, aperture, depth/tilt model and storage
precision are unchanged. The correction applies to both phase estimators.

## Validation

- A spatially constant 128 by 128 scan with a nonuniform detector remains flat
  within 1e-7 rad at factors 1, 2, 3, 4 and 8, with both phase estimators and
  zero/nonzero depth, on physical MPS and CUDA.
- A nonconstant control agrees within the existing 2e-5 rad reference bound
  with independent GPU mean subtraction followed by real-space zero insertion
  and FFT. This checks that the fix preserves the remaining spatial signal.
- Existing GPU reconstruction oracles and tolerances are retained. Tests for
  the historical mean-phase kernel request that estimator explicitly.
- Full 512 by 512 by 192 by 192 acquisitions A and B retain exactly the saved
  native diagnostic loss, fitted aberrations, tilt, depth and selected angle
  from the previous workflow on each respective backend. This is a matched
  workflow check, not a replacement for unavailable frozen acquisition pins.
- The native Metal workflow gate passes; it does not certify this Python
  default in a Swift application. The separate frozen fit gate remains
  unavailable because its external acquisition fixture is missing.

The 200-trial search and refinement objective are unchanged. Fit timings now
separate reference comparison, coarse search, refinement, Fourier preparation,
native diagnostic and final native reconstruction. Timers observe existing
completed-result boundaries and do not add synchronization to inner kernels.

## Saved results

Reconstruction policy is part of the exact result signature:
`default_c10_c12_estimator="phase_of_mean"` and
`upsampling_dc_policy="single_origin"`. A previous image cannot silently pass
exact-image reuse. The fit schema remains valid because units, fitted-parameter
meaning and search objective have not changed. Historical saved arrays remain
readable. Live's Auto cache has its own version increment for regenerated images.

Private evidence identifier: `20261016-ssb-coherent-arrival`.
Acquisition names, images, manifests and raw benchmark logs remain outside this
public repository. Timings from shared hardware are observations, not an
isolated speedup or a guarantee of microscope frame rate.
