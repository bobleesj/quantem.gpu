# Direct phase reconstruction and viewer handoff

## Question and scope

Can a full-field regularized phase reconstruction feed the interactive viewer
without preparing identical equations again?

Development source `5e2a153b53f4` plus the live-convergence and unchanged-rotation
cache changes was exercised in a real Jupyter workflow. Private evidence id:
`20261024-direct-live-full-field`. The evidence retains the executed notebook,
calibration, source fingerprint, package profile and original comparison array.
No acquisition images or raw data are included here.

The input has 512 × 512 scan positions and native 192 × 192 detector pixels;
300 kV, 30 mrad aperture, 0.373046875 Å scan sampling and
0.5515329888475119 mrad detector sampling. Fifth-order optics, tilt and effective
depth were reused from the archived fit. Auto output is 1536 × 1536 over
19.1 nm. The phase solve uses regularization 0.01, uniform detector weighting
and smoothness crossover 1 Å⁻¹. No new binning, cropping of scan positions,
optics search or diagnostic loss was introduced.

## Observations

CUDA ran on an NVIDIA RTX PRO 6000 Blackwell Max-Q with other GPU jobs active.
These are single notebook observations, not isolated benchmark medians or
cold-disk measurements. Regularization timing includes stream synchronization;
viewer timing ends after Python creates its object, before browser painting.

| Operation | Before cache fix | After cache fix |
|---|---:|---:|
| First phase solve | 11.2 s | 11.1 s |
| Cached regularization change | 1.258 ms | 1.298 ms |
| Create viewer after the phase solve | 10.8 s | 88.7 ms |

The viewer reapplied the same rotation through `set_rotation`, which discarded
the prepared direct equations. An unchanged rotation now preserves that cache;
a changed rotation still invalidates it. Reconstruction equations and precision
are unchanged. Defocus changes still need new preparation and are slower than
regularization changes.

The current full-field phase correlates with the archived phase at 0.999999,
with RMS difference 0.000066 rad. This comparison is not bit-exact: historical
and current bad-pixel handling differ for one pixel inside the aperture. It is
a replay check, not independent physical resolution validation.

The browser used JavaScript fallback display, while reconstruction used CUDA.
These timings do not demonstrate 60 fps. The live convergence chart was checked
separately on a public experimental acquisition; it reuses optimizer values,
throttles rendering and excludes display work from the numerical fit clock.

## Verification

Both executed notebooks completed without errors. Independent review confirmed
the unchanged scientific calculation and timing separation. The contract suite
reported 497 passed, 57 skipped and 5 deselected; skips are not passes. Changed
rotation invalidation and unchanged rotation reuse are covered explicitly.

## 2026-10-24: regularized methods in `stream`

Question: can `stream(frames, fit, method=("phase", "complex"))` reconstruct both regularized methods
on every frame live, without changing the registration?

Setup: a real experimental 4D-STEM time series, 40 frames (512 x 512 scan, 192 x 192 detector) after
the reference frame, output 1536 x 1536 at `upsampling_factor="auto"`, GPU 0, `show=False`, a fresh
scratch `save_dir` (every frame refit). Each frame runs SSB (for registration), then phase-only, then
amplitude + phase, each with `force=True, compute_loss=False`.

| Per frame, s (frames 1-39) | median | max |
|---|---|---|
| Load | 0.483 | 0.603 |
| Setup | 0.170 | 0.311 |
| Refit | 0.047 | 0.096 |
| SSB | 0.033 | 0.068 |
| Phase-only | 0.037 | 0.103 |
| Amplitude + phase | 0.048 | 0.096 |
| Align | 0.078 | 0.165 |
| Total | 0.898 | 1.442 |

Frame 0: phase-only 0.121 s, amplitude + phase 0.074 s (first preparation). SSB alone: total median
0.837 s, wall 34.4 s for 40 frames; with both methods 36.9 s.

- Shifts and confidences with both methods equal the SSB-only run exactly (max difference 0.0 Å):
  registration is measured on the SSB phase whatever the methods.
- All 40 frames placed; one frame flagged `Near threshold` (confidence within 10% above 0.04).
- Memory: each frame's SSB engine and its thick-sample model reference each other, so `del` alone
  left about 10 GB a frame until Python's cycle collector ran (process peak 66 GB SSB-only, 95.5 GB
  with both methods). `stream` now calls the session's `close()` after each frame, which breaks that
  cycle: process peak 40.5 GB (SSB) and 43.1 GB (both methods), 24-26 GB above the reference
  session (one frame in flight), with the cycle collector disabled for the whole run and no session
  or engine left afterwards. Stacks kept: 0.45 GB (SSB) and 1.56 GB (SSB, phase-only, amplitude +
  phase, amplitude), preallocated; torch keeps 0.03 GB after the series.
- Fix-round rerun (same setup): phase-only median 0.039 s, amplitude + phase 0.050 s, SSB 0.034 s,
  total 0.919 s a frame (max 0.967 s) against 0.861 s for SSB alone; shifts identical again.
- The combined phase, complex phase and amplitude show sharp atomic columns and the moiré at full
  resolution; no smearing from the shared shifts.
