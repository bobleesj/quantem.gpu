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
