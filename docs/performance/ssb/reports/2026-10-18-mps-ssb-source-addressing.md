# Reuse source addresses in MPS SSB reconstruction

> **Historical:** describes the code as of 2026-10-18; the `mean_phase` estimator is retired. See the [current fit methods](../../../api/ssb.md).

## Change

The coherent Metal kernel calculates the native Fourier alias address once
per output frequency and reuses it across BF pixels. It also starts the
source load before the optical arithmetic. Address calculations use 32-bit
integers when the complete chunk fits, with a compile-time 64-bit fallback.
This does not change floating-point precision, optical equations, BF support,
summation order or the phase estimator. No additional geometry volume is stored.

The existing resident Fourier data and probe cache are still reused between
slider requests. Larger trial caches increased memory traffic without a useful
speedup; a vectorized optical trial changed rounding and was rejected.

## Matched application measurements

Two full `512 × 512 × 192 × 192` acquisitions used all calibrated BF pixels,
Auto sampling, and their fitted tilt and depth on an Apple M5 Max. Old and new
Live servers were called alternately and never concurrently. Each row below
contains 20 changed-parameter requests after the first call.

| Output | Control | Reconstruction p50, before → after | Response p50, before → after | After response p95 / max |
|---|---|---:|---:|---:|
| 1536² | C10 | 90.1 → 86.4 ms | 96.0 → 92.9 ms | 94.5 / 97.4 ms |
| 1536² | Tilt | 90.3 → 86.6 ms | 95.9 → 93.4 ms | 97.8 / 99.5 ms |
| 2048² | C10 | 151.2 → 144.0 ms | 160.0 → 152.6 ms | 154.7 / 158.0 ms |
| 2048² | Tilt | 156.9 → 150.2 ms | 165.5 → 159.3 ms | 170.1 / 173.4 ms |

The reconstruction improvement is about 4–5%. The full request includes
resident-source lookup, reconstruction, output handoff and local HTTP transfer.
It excludes initial loading and fitting. These measurements do not establish
displayed FPS or a theoretical lower bound. The 60 FPS budget is 16.7 ms.

The isolated trial's MLX peak allocation, including preparation, was unchanged
at 19.83 GB and 20.35 GB respectively. This is allocator memory, not total
process or shared-device memory. Timing outliers remain in the evidence.

## Validation

- All 222 complete application comparisons were byte-identical to the old
  GPU implementation. Coverage includes native and Auto sampling, ordinary
  and tilt/depth-corrected SSB, C10/tilt changes, and higher-order optics.
- 61 MPS hardware tests passed. Two optional simulation tests were skipped
  because their external fixture was unavailable; they are not counted as passes.
- The session-reuse test now covers both phase estimators while changing optics,
  tilt, depth, rotation and sampling.
- The frozen GPU fit gate and native Metal reconstruction/saved-result gate
  passed without changing pins or tolerances.
- The application checks exercised the real Live API. Browser rendering was
  not measured in this run because browser control was disabled.

Private scripts, full timing distributions, rejected trials and phase hashes
are archived under experiment `ssb-optical-reuse-20261018`.
