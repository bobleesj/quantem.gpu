# MPS SSB preview reuse

The reconstruction kernel now skips aberration and depth calculations for a
sideband term when its aperture weight is exactly zero. It retains every
selected bright-field pixel and the original expression and accumulation order
for nonzero contributions. Precision, estimator, calibration, Fourier aliases,
and output dimensions are unchanged.

Each prepared session also retains the most recent frequency grid, aperture
bounds, and probe. Tilt and depth changes reuse the probe; aberration changes
replace it. Rotation changes invalidate both caches. The cache retains one
state, approximately 0.2 MB for the evaluated acquisitions.

## Paired reconstruction timing

Both acquisitions have 512 × 512 scan positions and 192 × 192 detector pixels.
Case A uses 8807 valid BF pixels and a 1536 × 1536 output; case B uses 8930 BF
pixels and a 2048 × 2048 output. The estimator is `phase_of_mean`.

Times are milliseconds, p50 / p95 / maximum, across 20 fresh parameter calls
after two warmups. Baseline and candidate order was randomized for each pair.
Each call includes GPU completion and host phase handoff. Loading, preparation,
fitting, browser transfer, and loss diagnostics are excluded. Measurements were
taken on an active desktop, so they do not establish a hardware limit.

| Case | Changed parameter / model | Baseline | Candidate |
| --- | --- | --- | --- |
| A | C10, tilt/depth model | 180 / 197 / 208 | 152 / 167 / 170 |
| A | Tilt, tilt/depth model | 184 / 206 / 213 | 154 / 170 / 172 |
| A | C10, ordinary model | 166 / 168 / 169 | 150 / 156 / 162 |
| B | C10, tilt/depth model | 307 / 386 / 460 | 257 / 290 / 298 |
| B | Tilt, tilt/depth model | 303 / 338 / 338 | 255 / 278 / 291 |
| B | C10, ordinary model | 257 / 273 / 390 | 229 / 239 / 241 |

Peak active MLX allocation during these calls was 9.69 GB for case A and
10.22 GB for case B. This is accelerator allocation, not total process RSS.

## Validation

- 188 paired complete-array comparisons against the frozen pre-change GPU
  implementation were bit-identical: 56 diagnostic cases and 132 timing cases,
  including timing warmups. Available previously saved phase arrays were also
  checked without updating them.
- Cases cover both models, C10, tilt, depth, higher-order aberrations, rotation
  reversal and restoration, and output sampling changes. Six phase-plus-loss
  comparisons had identical phase and loss.
- 50 focused hardware/contract tests and five repository privacy tests passed.
- The consuming browser application was exercised with fresh C10 and tilt
  changes, full BF, automatic sampling, visible phase/FFT, and settled loss.
- The native benchmark built, but the broader fit-trajectory gate could not run
  because its required local real-data fixture was absent. Fitting and native
  Swift sources were not modified. CUDA was not evaluated in this change.

Alternative algebraic depth calculations exceeded the existing phase tolerance
and were rejected. Parallel BF accumulation did not give a repeatable gain and
was not adopted. This change does not claim 60 FPS.

Private experiment id: `2026-10-17-mps-preview-reuse`.
