# Resident joint Direct Ptycho fitting

`DirectPtycho.find_aberrations(search="primary", tilt=True)` runs the six-parameter
adaptive search and refinement on the session's MPS or CUDA device. The model
jointly fits defocus, two-fold astigmatism and angle, both sample-tilt components,
and depth spread. No extra notebook optimizer is needed.

The resident path preserves multivariate TPE's groups of eight, 24 proposal
candidates, startup budget, and three distinct Newton starts. Its probability
model and random stream follow Optuna 4.9.0, independently of the installed
Optuna version. The Newton climbs on the exact gradient and Hessian of the agreement
(`_exact_newton`, `LowOrderDerivatives`) with trust-region updates, boundary handling
(fixed aberrations and a standard fit's tilt and depth spread are held coordinates),
convergence checks and start pruning; the finite-difference stencils are retired.
The requested trial count is not reduced. ShowPtycho's regional workflow still
runs all five seeds with 200 trials and Newton for each seed.

MT19937 state updates remain ordered on the GPU, while independent 53-bit
uniform conversions run in parallel. This preserves the exact seeded stream.
Metal schedules independent four-candidate model tiles together with a 64 MiB
partial-sum target, selected from the actual frequency-band size and retaining
at least one tile per dispatch. Search and
Newton therefore use the available parallelism without changing each thread's
arithmetic or bright-field reduction order. The hardware tests compare combined
and serial tiles exactly for eight-point search and 84-point Newton batches.

`ptycho.select_region((row_start, row_stop, col_start, col_stop))` creates an
independent fitting session over the full-field session's resident measured
counts. It retains detector calibration and source provenance; closing or
fitting a region does not change its parent. ShowPtycho uses this method rather
than reopening and decoding an already-loaded acquisition after every drag.
Full-field reconstruction still precedes cropping the corrected regional view.

Numerical candidates, objective scores, probability densities, Hessians,
eigensolves, and trust-region decisions stay on the GPU. Python supplies fixed
configuration and dispatches kernels; active-start/design-size flags may return
for scheduling. Completed fit records return for display and persistence.
`SSBResult.optimizer` records the execution device, numerical precision, and
per-start iteration/evaluation counts; `timings` records search and refinement
separately. The phase-variance search (`objective="phase_variance"`) is
retired. The standard fit
(`fit_standard_resident`, CUDA and MPS) is this search and Newton with the
sample tilt and depth spread held at 0; its Newton kernels are derived from the
joint ones by fixed text edits, so the joint fit's generated source is
unchanged on both backends.

## Numerical policy

The full measured data and active bright-field pixels remain unchanged. Model
kernels retain their float32 arithmetic. CUDA uses float64 for search, Newton,
and final objective reduction. Metal uses compensated three-float expansions
for those operations, preserving float64-level accuracy over the bounded fit
range. This is not general IEEE float64 emulation: the exponent range is still
float32. Probability routines are tested over the fit's operating range.

Search and Newton arithmetic can change a trajectory when objective values are
nearly tied. Qualification therefore checks trial sequence against fixed score
history, recovery of a known joint optimum, fixed-point model agreement, and
same-device experimental fit/image parity. Tolerances and frozen pins must not
be silently updated to accommodate a mismatch.

## Verification

Run `tests/hardware/test_ssb_resident_joint.py` on each physical backend. It
checks the full 200-trial sequence, scalar precision and random stream, a known
six-parameter optimum, and public model fitting with CPU optimizer calls
blocked. CUDA-only or MPS-only skips do not certify the other backend.

Record cold compilation separately from warmed refits. Measure actual widget
moves/resizes from the submitted request through the displayed result, including
all five repeats, preparation, and full-field corrected reconstruction. A
solver-only timing is not an end-to-end refit or an image update rate.
