# CUDA standard fit on the agreement objective, and one phase-variance search (2026-10-20)

> **Historical:** describes the code as of 2026-10-20; the phase-variance search (`objective="phase_variance"`) and its Nelder–Mead refinement are retired. See the [current fit methods](../../../api/ssb.md).

## Question

`2026-10-20-ssb-standard-fit-objective.md` moved the MPS standard fit
(`find_aberrations(tilt=False)`) to the joint fit's objective, the
least-squares agreement of the SSB model with G(q, k), at sample tilt and
depth spread 0. CUDA still minimized the phase variance. Do CUDA and MPS give
the same result for all three fits once CUDA runs the same code?

| Call | Objective |
|---|---|
| `find_aberrations()` | agreement, sample tilt and depth spread fitted (unchanged) |
| `find_aberrations(tilt=False)` | agreement, sample tilt and depth spread held at 0 |
| `find_aberrations(tilt=False, objective="phase_variance")` | phase variance |

## Change

- `cuda/optimizer.py::fit_standard` runs `_resident_search.fit_standard_resident`
  with `ThickSample.fit_resident` (the fused `thick_fit_batch_kernel`) as the
  objective, with the MPS start point, bounds and refinement rules (Newton by
  default; Nelder-Mead when a coefficient is locked, phi12 is bounded or a
  nonzero C12 is held). `fixed_aberrations` works in CUDA standard fits.
- The CUDA kernel already skipped the depth weights at depth spread 0
  (`if (t > 0.0f)`), so the tilt has no effect there, bit for bit
  (`test_ssb_standard_agreement.py::test_depth_zero_objective_matches_torch_reference`).
- `objective="phase_variance"` runs `direct_ptycho/phase_variance_fit.py`, the search
  MPS used before 2026-10-20 moved into a backend-neutral module: the start
  point (C10 0, C12 5 nm, phi12 0) evaluated first, Optuna TPE asked two
  trials at a time, then Nelder-Mead with step floors of 2 Angstrom in C12 and
  0.04 rad in phi12, `fatol` 3e-6, 80 iterations. CUDA's own search
  (`batch_optimize`, four trials per ask, Nelder-Mead without the floors,
  `xatol` 0.25, `fatol` 2e-6) is removed.

## Setup

A private 512 x 512 crystal scan, 192 x 192 detector, 300 kV, 30 mrad, the
detector step from `fit_detector_sampling()`. CUDA: RTX PRO 6000 Blackwell
Max-Q, CuPy 14.2, Optuna 4.9, idle device. MPS: the reference run of the same
branch on an Apple GPU; its phase-variance numbers are those of the MPS search
(`mps/optimizer.optimize_phase_variance`, restored in 33a6790 unchanged). Evidence: experiment
`2026-10-20-ssb-cuda-parity` in the private archive.

## Results

The same reference script on both backends: the default fit with both rotation
branches, `tilt=False` at the kept rotation, `objective="phase_variance"`, and
reconstructions at fixed parameters (the MPS fits) at 1x and 3x.

| Fit | Quantity | MPS | CUDA | difference |
|---|---|---|---|---|
| default | kept branch | 345.83 deg (columns) | 345.83 deg (columns) | same |
| default | C10, C12 (nm) | 8.84192, 0.90483 | 8.84189, 0.90483 | 2.3e-5, 2.5e-6 |
| default | phi12 (deg) | 59.4988 | 59.4986 | 2.3e-4 |
| default | tilt (mrad) | (-0.70370, -3.00519) | (-0.70372, -3.00527) | 1.9e-5, 8.0e-5 |
| default | depth spread (nm) | 13.60125 | 13.60129 | 4.5e-5 |
| tilt=False | C10, C12 (nm) | 9.89919, 2.13771 | 9.89916, 2.13770 | 3.4e-5, 1.7e-5 |
| tilt=False | phi12 (deg) | 81.7053 | 81.7053 | 6.3e-5 |
| tilt=False | agreement | 2.04825e14 | 2.04825e14 | 2.1e-7 relative |
| phase variance | C10, C12 (nm) | 10.07410, 2.08055 | 10.07410, 2.08055 | 2.6e-7, 2.0e-7 |
| phase variance | phi12 (deg) | 80.1381 | 80.1381 | 3.5e-7 |

The phase variance at the fixed parameters agrees to 1.5e-7 relative (0 for
the other two parameter sets). The reconstructed phases at the fixed
parameters correlate to better than 0.999999999 at 1x and 3x; the largest
difference is 2.3e-5 rad (tilt-corrected, 3x) and 2.8e-6 rad at depth 0.

Before this change CUDA's phase-variance search stopped at C10 10.0465,
C12 2.0817 nm, phi12 80.18 deg, loss 0.19469926; the MPS search's point has
the lower loss 0.19469902. The difference was the search, not the loss: the
MPS search run on the CUDA loss reproduces the MPS coefficients and loss to
every printed digit, so the seeded Optuna stream is the same on both
backends.

Timings on CUDA (median of 3; previews median of 20 with C10 changing every
call, full bright field, `compute_loss=False`):

| Stage | CUDA |
|---|---|
| default fit, both branches | 3.43 s |
| `tilt=False` (agreement), one branch | 0.72 s |
| `objective="phase_variance"`, one branch | 9.9 s |
| tilt-corrected preview 1x / 2x / 3x | 32.9 / 87.8 / 52.0 ms |
| standard preview 1x / 2x / 3x | 27.5 / 77.2 / 46.5 ms |
| tilt-corrected preview 1x with the loss | 253 ms |

The MPS default fit took 17 to 20 s and the tilt-corrected drag 104 ms on the
same scan.

## Conclusion

CUDA and MPS give the same fits for all three calls, far inside the parity
tolerances (C10 and C12 0.01 nm, phi12 0.2 deg, tilt 0.02 mrad, depth spread
0.05 nm, agreement 1e-4 relative), and keep the same branch by the same
column test. The phase-variance search now lives in one backend-neutral module,
so the two backends cannot drift apart again by search settings.

## Open

- The 2x preview is slower than 3x on CUDA (87.8 against 52.0 ms). One
  difference: at 3x the wave sum evaluates the Hermitian half plane
  (`ThickSample._wave_columns`, the 3x Nyquist lies beyond the double-overlap
  support) and at 2x the full plane, but the output counts are similar
  (1024 x 1024 against 1536 x 769). Not investigated further.
- The phase-variance search evaluates one candidate at a time on CUDA
  (`SSBEngine.reconstruct_with_loss`); a batched loss would shorten its 9.9 s.
