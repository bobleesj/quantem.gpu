# Calibration speed-up on main: the tilt fit's Newton polish and resident sessions - 2026-10-13

> **Historical:** describes the code as of 2026-10-13; `refinement="newton"` with finite-difference steps and the Nelder–Mead polish are retired; fits refine on exact derivatives. See the [current fit methods](../api/direct_ptycho.md).

## Question

A ptychography pipeline calibrates each acquisition on its full native scan: beam centre, DPC rotation, then SSB's
joint tilt fit and phase images, once on the whole scan and again on the region it reconstructs. A branch of 11-12 Oct,
built before the package restructure and never merged, cut that from 15.1 s to 5.9 s on a 512 x 512 x 192 x 192
acquisition (same card, its own timing table): the Nelder-Mead polish of the tilt fit made about 1100 sequential passes
over G, a Newton polish makes 5 or 6 batched ones; the acquisition was read from its file once and handed to SSB; the
standard-SSB search that only feeds `tilt_fit_gain` was skipped. Does the same work, rewritten on main's names, give
the branch's numbers?

## What moved

| branch feature | on main |
|---|---|
| `fit(tilt=True, refinement="newton")`, Newton polish with bounded trust steps | `find_aberrations(tilt=True, refinement="newton")`, `thick_sample_fit.newton_polish` |
| `compare_standard=False` | same keyword; `tilt_fit_gain` is None |
| `fit(tilt=True, trials=0)` refines the previous scan's fit | `num_trials=0` from the session's `aberrations`, `sample_tilt_mrad`, `depth_spread_nm` |
| `SSB.open(acquisition, scan_region=, mean_pattern=)` | `SSB(dataset, scan_region=, mean_pattern=)`; main's constructor already took a dataset |
| saved fit of a resident acquisition identified by its data | same identity (shape, dtype, mean-pattern hash, scan region), `source_kind="resident"` |
| `preview(output="native")` | same keyword |
| `io.load.bin` | same path; integer input sums in a 64-bit integer (the branch summed CuPy counts in uint32 and Torch uint16 in float32), and `edge="partial"` applies to 2D and 3D input too |

Not moved, because main already has its own form: the 180-degree branch taken from the first joint fit by symmetry
(`find_aberrations(check_rotation=True)` reuses the fit on CUDA), fused FFT kernels built per device, a deterministic
fused sum for the wave-averaged thick preview (`ThickSample._reconstruct_wave`), and the preview context that follows a
rotation change. The branch's threaded Nelder-Mead batching is not moved: main evaluates one point per launch so that
every value repeats bit for bit.

## Setup

One real Arina acquisition, 512 x 512 x 192 x 192, 300 kV, 30 mrad, 0.373 A scan step, loaded once onto an RTX PRO 6000
Blackwell Max-Q shared with other jobs. The pipeline's calibration ran four ways, each through the same calls: the
whole scan (200 trials, seed 42, Newton polish, 180-degree check), a 128 x 128 region fit, a 192 x 192 phase preview
around it, the region fit again with a through-focus curve, and one frame of a second acquisition of the series fitted
on its own scan. Runs:

- A1, A2: the branch's code, Optuna 5.0, twice.
- B1: this code, in the same environment as A (Optuna 5.0).
- C1, C2: this code with Optuna 4.9, which the `[cuda]` extra requires (`optuna<5`).

## Numbers

Whole-scan fit (C10, C12 nm; phi12 rad; tilt mrad, scan frame; depth spread nm):

| run | C10 | C12 | phi12 | tilt row | tilt col | depth |
|---|---|---|---|---|---|---|
| A1 | -16.09362 | 8.50438 | -0.751438 | -2.15466 | 1.31668 | 15.7168 |
| A2 | -16.09365 | 8.50441 | -0.751438 | -2.15482 | 1.31659 | 15.7168 |
| B1 | -16.09364 | 8.50443 | -0.751439 | -2.15487 | 1.31660 | 15.7168 |
| C1 = C2 | -16.09358 | 8.50443 | -0.751463 | -2.15495 | 1.31682 | 15.7140 |

Largest difference from A1 over every fitted value of the five fits (whole scan, region, region with focus curve,
second acquisition), and the phase images:

| pair | tilt, mrad | C10, nm | depth, nm | phase images, worst r |
|---|---|---|---|---|
| A1 vs A2 (branch against itself) | 3.7e-4 | 1.0e-4 | 1.9e-4 | 0.99995 |
| A1 vs B1 (this code, same environment) | 5.9e-4 | 2.0e-4 | 2.1e-4 | 0.999995 |
| A1 vs C1 (this code, Optuna 4.9) | 2.7e-3 | 1.3e-3 | 3.4e-3 | 0.99994 |
| C1 vs C2 | 0 | 0 | 0 | identical |

Beam centre, descan, disk radius, detector sampling, DPC rotation and phase, the mean pattern and the virtual
bright- and dark-field images are identical bit for bit in every run.

## Conclusion

The port reproduces the branch within the branch's own call-to-call spread when the environment is the same (A1 vs B1
against A1 vs A2): the branch's tilt-fit kernel adds its bright-field slices with atomic additions, so it does not repeat
itself, while main adds them in a fixed order (1d443719) and repeats bit for bit (C1 = C2). With Optuna 4.9 the search proposes other starting points and the
polish ends up to 2.7e-3 mrad, 1.3e-3 nm of C10 and 3.4e-3 nm of depth spread away. Each of those is inside the polish's
own stopping step (`NEWTON_TOLERANCE`: 0.005 mrad of tilt, 0.005 nm of C10, 0.05 nm of depth spread) and far inside
the fit's uncertainty measured on the branch (scan halves: 1.1 mrad of tilt and 0.5 nm). No timing is claimed here: the
card was shared with other jobs throughout.

## Rejected

- Running the branch's code alongside main: the two histories share no commit and the branch's names (`det_sampling`,
  `rotation_angle_deg`, `trials`, `tilt_mrad`) were renamed on main; every caller would need both spellings.
- Bit-exact equality with the branch as the gate: the branch's own repeats differ by up to 3.7e-4 mrad, so no
  implementation can match one of its runs bit for bit.
