# MPS slider correction and interruptible diagnostics

> **Historical:** describes the code as of 2026-10-17; the `mean_phase` estimator and the phase-variance search are retired. See the [current fit methods](../../../api/ssb.md).

The fine-grid correction evaluates the frequency-dependent tilt shift and
aperture bound once per coherent BF group. Previously the helper repeated these
expressions for each BF contribution. The expressions, BF order, precision,
calibration, output sampling and `phase_of_mean` estimator are unchanged.

## Paired reconstruction measurements

Both private acquisitions contain 512 × 512 scan positions and 192 × 192
detector pixels. A uses all 8807 BF pixels and a 1536² output; B uses all 8930 BF
pixels and a 2048² output. Each mode has 20 fresh parameter calls after two
warmups, with randomized baseline/candidate order. Times include GPU completion
and host phase handoff; they exclude loading, fitting and browser display.

| Case | Control / model | Before p50 / p95 / max (ms) | After p50 / p95 / max (ms) |
| --- | --- | --- | --- |
| A | C10, tilt/depth | 111 / 115 / 115 | 102 / 105 / 105 |
| A | Tilt, tilt/depth | 113 / 120 / 124 | 104 / 112 / 114 |
| A | C10, ordinary | 103 / 105 / 109 | 92 / 96 / 100 |
| B | C10, tilt/depth | 211 / 215 / 215 | 190 / 197 / 200 |
| B | Tilt, tilt/depth | 212 / 216 / 216 | 189 / 195 / 200 |
| B | C10, ordinary | 189 / 199 / 202 | 168 / 173 / 177 |

Peak active MLX allocation was approximately 9.72 GB / 10.21 GB. Complete
comparison runs took 26.31 / 43.54 seconds. These are active-desktop measurements,
not hardware limits or whole-acquisition fit timings.

## Cancellation without partial scientific output

`SSB.preview(cancel=predicate)` is an optional advanced application control.
MPS checks it at completed GPU batch boundaries. Cancellation raises
`concurrent.futures.CancelledError`; it returns no partial image or loss.
Other backends check before and after computation. Existing calls are unchanged.

For cancellable 512² native loss, the row/column kernels process 256 BF pixels
per completion boundary. They carry the original per-pixel phase sums and all
eight independent per-thread square sums into the next tile. The original BF
group boundaries and final square-sum reduction remain unchanged. Combining
the eight square sums earlier would change rounding, so this is deliberately
avoided.

A consuming viewer that already holds the completed coherent phase can request
native `mean_phase` with loss and discard that diagnostic phase. This avoids a
second coherent image that the viewer would not use. The displayed estimator
and native per-BF phase-variance objective remain unchanged.

In paired completed-loss calls, removing the unused image reduced medians from
269 to 191 ms (A) and 280 to 191 ms (B). In eight cancellation trials per case,
the maximum remaining time after the cancellation event was 5.33 / 4.57 ms.
These kernel-level cancellation times exclude HTTP detection and queueing.
Applications must signal cancellation when a new image arrives; browser
disconnect notification alone is not a reliable low-latency signal.

## Validation and scope

- 188 complete-array comparisons against frozen pre-change GPU source were
  bit-identical, covering ordinary and tilt/depth models, both rotations,
  sampling changes, signed C10, tilt, depth and higher-order terms.
- Another 24 full native diagnostic phase/loss comparisons were bit-identical
  against frozen reconstruction and column-kernel source. Cancellation followed
  by reconstruction also retained the complete reference phase.
- Sixty focused MPS hardware, contract and privacy tests passed, exercising
  cancellation and recovery, alternate tile lengths, ordinary/tilted loss and
  higher-order reconstruction.
- The consuming dashboard verifies C10/tilt movement, diagnostic interruption,
  full BF, Auto sampling, WebGPU display and recovery of settled loss.
  All 28 consuming-API arrays matched frozen references exactly. Sixteen real
  browser interruptions yielded to the next image; maximum queue waits were
  8.64 / 5.53 ms. Direct pointer drags and rapid input reached their final
  requested settings. These are application tests, separate from GPU timings.
- The native harness builds, but the broader frozen fit-trajectory gate remains
  blocked by an absent local real-data fixture. Native Swift and fitting source
  are unchanged. This work does not qualify CUDA or establish 60 FPS.

Parallel BF lanes, alternate threadgroups, rectangular support dispatch and
individual sideband geometry pruning gave no reliable improvement and were
not adopted. Private experiment id: `ssb-slider-preemption-20261017`.

## Integration with ordered coherent reconstruction

The correction reuse also works with the newer ordered BF accumulation and
bounded Fourier-support dispatch. It uses their shared transfer function and
keeps their group boundaries and summation order. The timings above describe
the earlier isolated optimization, not an additional integration speedup.

The combined implementation passed 140 complete phase comparisons against
frozen pre-integration GPU source, plus 40 independent native phase/loss and
public preview checks on the two full-size acquisitions. These include
ordinary and tilted models, both rotation branches, signed defocus, sampling,
higher-order terms, and completed calls with cancellation enabled. Every
comparison was bit-identical. Private experiment id:
`ssb-slider-rotation-integration-20261017`.
