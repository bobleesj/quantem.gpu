# Exact aperture bounds for interactive MPS SSB

## Question and change

Can full-resolution coherent reconstruction avoid work without changing phase?
The MPS alias-correction kernel now rejects frequencies outside conservative
bounds of both shifted probe apertures before evaluating geometry and phase.
It also rejects entire BF groups when neither aperture can contribute.

The bounds include the soft aperture edge, unequal detector pixel angles and
an outward float32 margin. True DC and its aliases follow the existing rules.
Contributing terms retain their original arithmetic and accumulation order.
No BF pixels, measured frequencies or higher-order coefficients are removed.
The estimator remains `phase_of_mean`; fitting and native loss are unchanged.

## Matched measurements

These first measurements describe the isolated bounds change at `2792ed5a`.
The ordered-group integration below has a separate paired comparison.

Two full 512 × 512 × 192 × 192 acquisitions use all 8,807 and 8,930 active BF
pixels. Auto sampling produces 1536² and 2048² phases, respectively. Each row
contains 20 changed-parameter, warm requests on the same MPS device. Times
include reconstruction and localhost binary phase transfer, but exclude
loading, preparation, fitting, browser FFT and presentation.

| Acquisition and change | Before p50 (ms) | After p50 / p95 / max (ms) |
| --- | ---: | ---: |
| A, C10 with tilt/depth | 277.6 | 150.6 / 158.8 / 176.6 |
| A, tilt | 288.5 | 153.6 / 176.5 / 179.0 |
| A, ordinary SSB | 275.4 | 124.3 / 127.6 / 148.1 |
| B, C10 with tilt/depth | 543.2 | 274.4 / 310.7 / 312.0 |
| B, tilt | 599.7 | 292.9 / 316.7 / 390.2 |
| B, ordinary SSB | 600.1 | 254.4 / 304.1 / 310.2 |

Prepared MLX allocations were approximately 9.28 and 9.42 GB, with process
MLX peaks of 19.83 and 20.35 GB. Bounds add only five floats per BF group.
These peaks include preparation; they are not incremental redraw memory.

## Validation and limits

- All 28 complete frozen real-data phase arrays were bitwise identical.
  Comparisons cover ordinary, tilt/depth, native and Auto reconstruction,
  plus requests with loss. Recorded loss values were identical.
- All 33 MPS upsampling tests passed, including higher orders, partial BF
  groups, repeated execution and a new anisotropic-aperture check at 3×.
  The existing independent GPU reference tolerance remains 2e-5 rad.
- The native Metal reconstruction and saved-workflow gate passed. This does
  not certify a native-app implementation of the Python change.
- The separate frozen fit-trajectory gate could not run because its external
  acquisition fixture was unavailable. Its pins were not regenerated.
- CUDA was not changed or benchmarked in this experiment.

The speedup does not establish 60 FPS: 16.7 ms is still much shorter than a
complete exact reconstruction. Browser timing must include queueing, transfer
and presentation, and distinguish final loss updates from new phase images.
No theoretical hardware limit or universal bitwise guarantee is claimed.

Private evidence identifier: `ssb-slider-latency-20261017`. Source data,
phase arrays, profiles, browser captures and timing logs remain private.

## Integration with ordered coherent accumulation

The bounds were reconciled with the ordered BF accumulation and guarded
half-spectrum implementation from `5e176105`. Group bounds follow the existing
chunk boundaries, including incomplete final groups and batches. The shared
per-detector correction also retains the bounds for the mean-phase estimator.
No accumulation order, sampling policy, or half-spectrum guard was changed.

The combined implementation is bitwise identical to `5e176105` for all 28
full-source cases above and three additional 1024-square phases from a third
acquisition. Native losses and two fixed-optics regional fits are unchanged.
The 28 earlier slider arrays were frozen from `6aa323b5` and also match
`2792ed5a` exactly. They differ from both `5e176105` and the combined version
by at most 3.52e-6 rad because the pre-integration detector accumulation order
differs. This is tolerance-based parity across those lineages, not bitwise
parity; the existing 2e-5 bound is unchanged. The third acquisition retains
its separate 1e-7 frozen-oracle
bound, with maximum error 8.94e-8 rad. Eight synthetic coherent cases at
native sizes 128 through 1024 and factors 1 and 2 are bitwise unchanged, as
are four native mean-phase and loss comparisons.

Matched warm phase-only measurements use the same prepared data and parameters
before and after adding the bounds. Shared desktop load was not controlled.
GPU completion and the returned host phase are included; browser transfer,
display, loss and fitting are excluded from these timings.

| Output / all BF pixels | Parameter | Before p50 (ms) | Combined p50 (ms) |
| --- | --- | ---: | ---: |
| 1024-square / 4697 | Three tilt settings | 33.1 | 26.4 |
| 1536-square / 8807 | C10 | 202.3 | 163.1 |
| 1536-square / 8807 | Tilt | 228.1 | 181.5 |
| 2048-square / 8930 | C10 | 518.1 | 445.2 |
| 2048-square / 8930 | Tilt | 592.9 | 468.6 |

The reconstruction-focused suite passed 81 tests, including the existing
strict coherent comparison, higher orders, anisotropic sampling, scan-mean
handling, group tails and repeated execution. No tolerance was relaxed.
Four rectangular-region tests and seven regional-precision tests also passed
on MPS. The native Metal reconstruction and saved-workflow gate passed; the
external frozen fit-trajectory fixture remains unavailable.
The integration checks did not restart or replace active notebook runtimes.
These measurements still do not establish 60 FPS or a browser-paint rate.

Private evidence identifier: `regional-aperture-reconciliation-20261017`.
