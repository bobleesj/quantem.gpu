# Initial fitting before fifth-order refinement

> **Historical:** describes the code as of 2026-10-22; the finite-difference Newton and Nelder–Mead refinements it compares are retired. See the [current fit methods](../../../api/direct_ptycho.md).

## Decision and scope

Use 200 TPE trials for the initial defocus, astigmatism, tilt and depth fit,
including its existing Newton polish, then jointly refine through fifth order.
Keep broader fifth-order TPE exploration optional. This is an empirical workflow
choice for the tested calibrated acquisitions, not proof of a global optimum.

The shared API is `fit = ssb.find_aberrations()`. `search="broad"` adds a
400-trial fifth-order exploration before the same refinement. `search="primary"`
retains the initial-only model for constrained fits and cross-checks. A scalar
`trials` controls the initial budget for standard and the additional budget for
broad. Both routes preserve the bounded Newton and Nelder–Mead alternatives.

## Frozen CUDA experiment

The initialization study used three full 512 × 512 scans, all valid bright-field
pixels, native-grid fitting, and Auto reconstruction. Acquisitions A and B had
192 × 192 detectors; C had a 12 × 12 detector. Do not compare C's runtime to the
other acquisitions as if the detector workloads were equal.

Hardware: RTX PRO 6000 Blackwell Max-Q Workstation Edition. Scientific objective
and reconstruction: float32/complex64 CUDA. Candidate generation, finite-difference
Newton control and its small linear solve: CPU float64. This is a hybrid optimizer.

Median seconds from three fresh processes per route:

| Acquisition | Initial fit → Newton: fit / workflow | Newton from zero: fit / workflow | Valid BF pixels | Auto |
| --- | ---: | ---: | ---: | ---: |
| A | 7.920 / 8.920 | 8.954 / 9.998 | 8,807 | 3× |
| B | 11.995 / 13.039 | 19.358 / 20.386 | 8,930 | 4× |
| C | 0.144 / 1.674 | 0.080 / 1.348 | 101 | 1× |

Workflow includes loading/upload, preparation, Fourier transform, fitting,
native and Auto reconstruction, host handoff and array saving. Imports,
provenance audits and report plotting are excluded. Filesystem/compiler caches
were uncontrolled; unrelated work shared host resources. Three samples give
descriptive percentiles, not a latency guarantee. C's fit saving is 64 ms;
loading variation also contributes to its larger workflow difference.

The initial fit cost 1.37 s on B but reduced fifth-order refinement from
19.30 to 10.59 s and unique GPU objective evaluations from 4,195 to 2,317.
Starting closer to the useful basin saved more work than initialization cost.
The speed ordering survived a control using the same objective normalization.

All 18 main runs and three normalization controls were finite and bounded.
Nine initialized controls reproduced their frozen native/Auto phase arrays,
coordinates and numerical refinement traces exactly. Zero-start versus initialized
phases were close, **not identical**: median-centered phase RMS differences were
0.000140, 0.000262 and 0.000018 rad for A/B/C; correlations were above 0.999996.
Raw objective differences were below 0.000018%, around float32 score granularity.
These differences are not evidence of improved resolution.

Private evidence: experiment `1022-07-newton-scratch`; API integration acceptance:
`1022-08-standard-api`. The frozen implementation was GPU commit
`8100dee31a08bb99781232c72186ab4b2cb69427`. Raw data, images and manifests remain
outside this repository.

## Optimization lessons

- **Separate budgets.** TPE trials, Newton objective evaluations, accepted steps
  and unique GPU evaluations are different counts. Cached stencil points count
  toward the solver budget but need not execute on the GPU again.
- **Do not mistake a staircase for instability.** Finite-difference batches and
  rejected trust-region proposals produce flat sections. The best recorded point
  can come from a stencil as well as an accepted center step.
- **Report why a run stopped.** A small bounded step, exhausted trust radius and
  an evaluation cap are different outcomes. Several useful results reached the
  cap; none thereby establishes convergence or a global optimum.
- **Compare raw objectives across starts.** Each route normalizes by its initial
  objective. Comparing independently normalized scores can reverse the apparent
  ranking. Scaling can also change late stopping near a float32 plateau.
- **Preserve physics while optimizing execution.** Reuse geometry and exact
  evaluations, batch stencils, retain every valid BF pixel and keep final score
  reductions on the GPU. Do not trade detector support, precision or estimator
  for a faster benchmark. Validate complete arrays and the optimizer trajectory.
- **Keep fitting and display distinct.** Fit on the native scan grid; reconstruct
  with Auto sampling. Auto can legitimately be 1× when native sampling already
  covers the model support. More output pixels do not prove more information.
- **Measure outside the optimizer too.** Loading, preparation, fitting,
  reconstruction, handoff and browser painting need separate clocks. Saved
  convergence plots consume recorded scalars and must not rerun objectives.
- **Check contention and compiler state.** A busy GPU can look like a regression.
  Cold/warm compilation and mixed-cache discrepancies need separate controls;
  an unexplained cache discrepancy is not cleared by one fresh passing run.
- **Keep broader exploration available.** A good initialized local fit on three
  calibrated scans does not establish robustness to unknown rotation, calibration
  errors or all specimens. Preserve the selected empirical references when
  evaluating new search strategies; they are not independent physical ground truth.

## MPS integration

The Metal objective shares the coefficient polynomial and parameter encoding
with CUDA. It retains all BF pixels, reuses detector geometry, batches four
candidate rows, and targets 64 MiB for queued partial buffers. The minimum
four-candidate tile can exceed that target on larger grids. Scores reduce on
the GPU; candidate generation and the float64 optimizer remain on the CPU.

On an Apple M5 Max (128 GiB), three fresh processes per acquisition gave these
**shared-device observations**, including overlap with other small MPS fixtures:

| Acquisition | Fit p50 / p95 / max (s) | Workflow median (s) | Peak active MLX memory |
| --- | ---: | ---: | ---: |
| A | 53.081 / 61.183 / 62.083 | 57.386 | 9.7 GiB |
| B | 114.356 / 115.260 / 115.361 | 118.000 | 10.6 GiB |
| C | 0.406 / 0.730 / 0.766 | 0.588 | 1.8 GiB |

Initial-fit first-call times were 4.20, 10.28 and 0.44 s. They exclude browser
painting. Notebook `show=True` exposes that initial Auto phase while refinement
continues, then updates one display. `show=False` avoids both display passes.
Fit timing excludes this handoff; total timing includes it. This does not make
the high-order solve faster or establish a dedicated-device latency limit.

Fixed-coordinate full native/Auto arrays pass the existing CUDA tolerance
(`rtol=atol=2e-4`). Auto maximum errors are 2.95e-5, 4.68e-5 and 2.89e-8 rad.
Three MPS fits per acquisition produced bit-identical coordinates and arrays.
Independent CUDA and MPS optimization trajectories are not bit-identical;
Auto phase correlations exceed 0.999996 on these inputs.

A full shifted-geometry cache consumed another 6.23 GB for a timing change
within run variation; rejected. Coefficient staging was slightly slower and
changed score bits; rejected. Larger tiles and partial buffers gave no robust
gain. The remaining cost is high-order optical evaluation across every BF
pixel, fit frequency and candidate; no theoretical-limit claim is justified.

## Qualification limits

The shared default and optional broad CUDA routes reproduce their frozen full
arrays and numerical optimizer traces. MPS has separate physical-device parity
and repeated-fit evidence. Notebook display tests cover one initial/final view,
read-only handoff and restoration of the native session result after rendering.
These results do not certify every calibration, convergence, microscope sample
or a global optimum. No release qualification follows from this study.
