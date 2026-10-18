# MPS coherent SSB dispatch

The coherent Auto reconstruction now uses a `16 × 8` Metal threadgroup
instead of `32 × 8`. This changes scheduling only. The optical calculation,
BF contribution order, float32 precision and phase estimator are unchanged.
The native fitting and loss paths are unchanged.

## Measured result

Two measured acquisitions retained their full `512 × 512 × 192 × 192`
inputs, all calibrated BF pixels, fitted tilt and depth, and automatic output
sampling. Median resident reconstruction time in five interleaved C10 trials
changed from approximately 93 to 88 ms for a `1536 × 1536` output, and from
152 to 146 ms for a `2048 × 2048` output on an Apple M5 Max.

In the consuming browser application, median C10 response changed from
122 to 119 ms and from 191 to 187 ms respectively. These measurements include
request scheduling, reconstruction, transfer and the application's frame
presentation marker. They are not physical scanout measurements or a claim
of sustained 10 FPS. Timing varies with other GPU activity.

## Validation

- Complete phase arrays matched frozen same-device results exactly for 122
  application calls, covering ordinary and tilt/depth-corrected reconstruction.
- Existing MPS reconstruction and infrastructure checks: 77 passed; two
  optional simulation-fixture tests were skipped and are not counted as passes.
- The frozen GPU fit trajectory, repeat determinism and native saved-result
  workflow passed without updating reference pins or tolerances.
- Hardware browser checks exercised C10 and tilt with Auto sampling and full
  BF support. The remaining cost is primarily optical correction.

CUDA launch variants did not show a consistent improvement under a shared
GPU workload; CUDA kernels remain unchanged. Private commands, timings,
arrays and application evidence are retained as `ssb-explorer-100ms-20261018`.
