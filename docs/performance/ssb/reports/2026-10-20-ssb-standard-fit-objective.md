# The MPS standard SSB fit uses the agreement objective (2026-10-20)

## Question

The joint fit (`tilt=True`, the default) maximizes the least-squares agreement of
the thick-sample SSB model with G(q, k). At a depth spread of 0 that model is
ordinary SSB exactly (every depth weight is 1). The standard fit (`tilt=False`)
minimized a different objective, the phase variance over bright-field pixels,
with its own search, kernels and buffers. Can the standard fit run the joint
fit's search with the sample tilt and depth spread held at 0, so that one
search is maintained, without losing accuracy?

## Change

On MPS, `find_aberrations(tilt=False)` now runs the resident multivariate TPE
of the joint fit (200 trials, 10 startup trials) with the tilt and depth spread
held at 0, then Newton (`refinement="auto"`) or Nelder-Mead (`"nelder-mead"`, or
`"auto"` when a coefficient is locked, phi12 is bounded or a nonzero C12 is
held). The search box is unchanged: C10 ±40 nm, C12 0 to 10 nm, phi12 over its
period. `search_ranges`, `refine_lock` and `fixed_aberrations` constrain it.
The phase variance is computed once, at the fit, as `result.loss`. The fused
objective kernel skips the depth weights when the depth spread is 0. The
joint fit is unchanged bit for bit (default fits on the public gold scan and
on another 512 × 512 acquisition: every coefficient, tilt, depth spread, loss,
agreement, gain and branch).

**Phase variance kept as an explicit option** (same day). The phase-variance
search is not deleted: `find_aberrations(tilt=False, objective="phase_variance")`
runs it (Optuna TPE in candidate pairs on the exact loss, then Nelder–Mead), and
`objective="agreement"` is the default everywhere:

| Call | Method |
| --- | --- |
| `find_aberrations()` | default: model agreement, tilt-corrected |
| `find_aberrations(tilt=False)` | model agreement, tilt and depth spread held at 0 |
| `find_aberrations(tilt=False, objective="phase_variance")` | the older SSB search, minimum phase variance |

With `tilt=True` the phase variance raises: it cannot see the tilt. On MPS its
files (`ssb/mps/loss.py`, the pair kernels, the 512 × 512 pair-pack policy,
`bf_storage_chunk_packs`, the search in `mps/optimizer.py` and
`scripts/benchmark_ssb_mps_scaling.py`) are restored unchanged, so the
restored search reproduces the removed one bit for bit: on gold (both branches,
358.31° kept) C10 −17.7557 nm, C12 2.6289 nm, phi12 −40.312°, and on the other
512 × 512 acquisition C10 10.0741 nm, C12 2.0805 nm, phi12 80.138° at 345.828°,
with identical losses, branch records, phases and trial records, and the same
live peak memory (12.50 GiB of MLX arrays on gold; 18.56 GiB of driver memory
above the encoded data in most runs, the overnight 18.6 GiB;
`2026-10-20-mps-performance.md`). The default
and `tilt=False` fits are unchanged bit for bit.
CUDA runs the same standard fit since the same day
(`2026-10-20-ssb-cuda-standard-fit.md`); its phase-variance fit is the same
search, in `ssb/phase_variance_fit.py`.

## Evidence

**Synthetic truth.** 64 × 64 scans of a periodic phase object, 300 kV, 30 mrad,
C10 −20, −8, −3, +5 and +15 nm × C12 0, 2.5 and 5 nm at phi12 20°: 15 crystals,
noise-free and with Poisson noise at 1e5 and 1e4 electrons per pattern. Errors
against the truth, mean (worst); C12 as the astigmatism vector:

| Dose | phase variance, C10 | agreement, C10 | phase variance, C12 | agreement, C12 |
| --- | --- | --- | --- | --- |
| noise-free | 0.078 (0.127) nm | 0.024 (0.046) nm | 0.011 (0.040) nm | 0.005 (0.018) nm |
| 1e5 | 0.077 (0.127) nm | 0.024 (0.044) nm | 0.008 (0.020) nm | 0.006 (0.019) nm |
| 1e4 | 0.085 (0.128) nm | 0.026 (0.046) nm | 0.017 (0.049) nm | 0.010 (0.019) nm |

On a 15 nm thick crystal tilted by (3, −4) mrad, both standard objectives miss
the mid-depth C10 by 3.2 nm; the joint fit recovers it within 0.35 nm and the
tilt within 0.05 mrad.

**Half-split repeatability.** Each scan was split into its (even row, even
column) and (odd row, odd column) positions: two 256 × 256 inputs at twice the
step, with the same field of view and disjoint electrons, each fitted at the
full scan's rotation. On the public gold scan:

| | phase variance | agreement |
| --- | --- | --- |
| half-to-half C10 | 0.157 nm | 0.008 nm |
| half-to-half C12, phi12 | 0.032 nm, 0.05° | 0.013 nm, 0.02° |
| FRC of the halves' phases, mean 0.2 Å⁻¹ to Nyquist | 0.761 | 0.776 |
| FRC = 1/7 | 0.99 Å⁻¹ | 1.24 Å⁻¹ |

On three further crystalline 512 × 512 acquisitions (0.26 to 0.5 Å steps)
both objectives repeated within 0.11 nm in C10 (the agreement within 0.01 nm),
with half-split FRCs within 0.02 of each other and lattice-peak sharpness
within 4 %. Every fit kept
the same 180° branch with either objective.

**Gold, full scan, both branches** (warm, under the shared GPU lock on an
M5 Max with other GPU clients): the phase variance fitted C10 −17.756 nm, C12
2.629 nm, phi12 −40.31° in about 75 s with an 18.6 GiB peak; the agreement fits
C10 −18.467 nm, C12 2.662 nm, phi12 −39.51° in 7.8 s with a 9.4 GiB peak. The
joint fit gives C10 −18.441 nm. The phase-variance loss at the agreement fit is
within 0.1 % of the phase-variance minimum (0.04451 vs 0.04450).

**Objective parity.** At depth spread 0 the fused kernel equals the torch
reference `tests/parity/torch_ssb.py` `fit_power` to 8e-8 (relative; the
float32 bound is 1e-5), and a nonzero tilt with depth spread 0 gives the same
value bit for bit.

## Conclusion

The agreement objective lands about three times nearer the synthetic truth,
repeats as well or better on real data, keeps the same branch, and fits in a
tenth of the time and half the memory, so it is the default of every fit. The
standard fit's coefficients move: on gold C10 −17.756 → −18.467 nm. The
phase-variance search stays as `objective="phase_variance"`, for comparison
with earlier results.

CUDA follow-up, done the same day (`2026-10-20-ssb-cuda-standard-fit.md`): the
standard fit on `fit_standard_resident`, with the phase-variance search kept as
`objective="phase_variance"`.
