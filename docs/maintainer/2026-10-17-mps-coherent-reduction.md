# Coherent SSB reduction on Metal

## Question and change

Can repeated full-field aberration previews avoid writing a dense corrected
spectrum for every bright-field pixel? The MPS coherent estimator now sums
groups of 32 corrected spectra in registers and writes one spectrum per group.
The final Fourier transform and phase extraction are unchanged. Batches retain
all BF pixels, including incomplete groups and single-pixel tails.

This changes the floating-point reduction order. It does not change the
estimator, float32/complex64 precision, calibration, scan resolution, aperture
support or sampling rule. The native-grid mean-of-phase loss still evaluates
the individual BF waves. Higher-order aberrations retain the full complex
spectrum; the existing symmetry guard is unchanged.

## Matched measurements

Two complete 512 × 512 × 192 × 192 acquisitions were tested on Apple Silicon
through the Python MPS backend. Case A used 8,807 BF pixels and 1536 × 1536
output; case B used 8,930 BF pixels and 2048 × 2048 output. The frozen baseline
was `3362fd55`. Ten new phase evaluations per row alternated baseline and
candidate on the same prepared data, including zero and both focus signs and
both rotation candidates. Times include GPU completion and phase return, but
exclude loading, initial preparation, fitting and the separate native loss.

| Case / model | Before p50 / p95 / max (s) | After p50 / p95 / max (s) |
| --- | ---: | ---: |
| A ordinary | 0.753 / 0.879 / 0.928 | 0.434 / 0.614 / 0.650 |
| A tilt/depth | 0.746 / 0.778 / 0.793 | 0.432 / 0.457 / 0.458 |
| B ordinary | 1.359 / 1.494 / 1.548 | 0.770 / 0.865 / 0.883 |
| B tilt/depth | 1.411 / 1.466 / 1.471 | 0.809 / 0.891 / 0.918 |

Median speedup was 1.7–1.8×. The complete interleaved validation took 72.70 s,
including preparation and comparisons. Peak MLX allocation was 9.70 → 9.67 GB
for A and 9.92 → 10.17 GB for B; these figures include resident prepared data.
These are observed interactive-system measurements, not a 60 Hz claim or a
complete acquisition-processing benchmark.

## Numerical and workflow checks

- Forty complete-array comparisons in the interleaved test passed the existing
  absolute tolerance of 2e-5 rad, with maximum error 3.52e-6 rad.
- Twelve additional frozen full-array comparisons covered ordinary, tilt/depth
  and higher-order reconstruction at native and automatic sampling. Maximum
  error was 3.49e-6 rad; all twelve native loss values were unchanged.
- All 31 MPS upsampling tests passed, including factors 1, 2, 3, 4 and 8, the
  symmetry guard and a new incomplete-BF-group regression. Repeated candidate
  results were identical.
- The legacy frozen-fit script built the native Metal target, then could not
  run because its external fixture `strict-parity/arina-128-full-disk/case.json`
  was unavailable. This is an unexecuted gate, not a pass. Native Swift and CUDA
  execution are unchanged by this MPS-only patch.

Private evidence: experiment `20261017-ssb-coherent-reduction`. The application
acceptance and HTTP timing are recorded there separately from these prepared
GPU measurements.
