# MPS fine-grid optical calculations, 2026-10-17

## Change

Fine-grid reconstruction now uses conservative bounds to avoid evaluating the
soft aperture edge where the shifted aperture is certainly closed or fully
open. At the edge, it uses the original precise geometry calculation. It also
submits four dependent correction batches before waiting for completion.
Cancellation is checked after the queued batches complete.

BF selection, group boundaries, float32 summation order, calibration, precision
and the phase estimator are unchanged. No background keepalive, global memory
policy or persistent Fourier-data duplicate was added.

## Paired measurements

Two full 512 × 512 × 192 × 192 acquisitions were reconstructed on Apple Metal
with 8,807 and 8,930 BF pixels. Their automatic output sizes were 1536² and
2048². Each row contains 20 new settings, alternating the baseline and candidate
on the same prepared engine. Times include reconstruction and output handoff;
they exclude loading, fitting and browser transport.

| Output | Change | Baseline median (ms) | Candidate median (ms) | Candidate p95 / max (ms) |
| --- | --- | ---: | ---: | ---: |
| 1536² | Joint C10 | 112.1 | 99.8 | 123.8 / 125.6 |
| 1536² | Joint tilt | 121.9 | 105.8 | 128.1 / 128.7 |
| 1536² | Ordinary C10 | 107.3 | 91.8 | 108.6 / 109.0 |
| 2048² | Joint C10 | 174.3 | 152.1 | 167.6 / 169.7 |
| 2048² | Joint tilt | 176.3 | 155.4 | 281.9 / 315.7 |
| 2048² | Ordinary C10 | 152.8 | 129.6 | 144.1 / 144.4 |

Higher-order controls remained bit-exact but showed little speed benefit:
221.2 → 223.4 ms and 401.3 → 394.9 ms median. All 160 full phase-array pairs
were bit-identical, including signed-zero bits. Ordinary controls had zero
tilt and depth with fixed optics; they were not separately refitted.

Peak active MLX allocation remained 9.47 GB and 9.75 GB; temporary allocation
above the resident starting state remained approximately 187 MB and 332 MB.
Allocator cache is separate. These are not process-wide peak-memory values.

## Idle latency

A Metal trace localized a prepared first-update stall before GPU execution.
The first correction batch took 158 ms, of which driver command preparation
occupied 143 ms. A 62 ms memory-residency event was inside that interval.
The first GPU compute segment took 6 ms. These intervals overlap and must not
be added together. The rest of the driver interval remains unattributed.

After the native diagnostic and six seconds idle, four paired measurements
per version gave a complete-redraw median of 262.2 → 226.3 ms. This does not
eliminate the idle stall. Global wiring, allocator trimming and an extra tiled
copy of the retained Fourier data did not give a repeatable improvement.

The 16.7 ms budget for 60 scientific frames per second remains unmet. Smooth
display refresh is not equivalent to a newly computed phase image per frame.

## Verification

- 47 hardware MPS upsampling tests passed, including anisotropic soft-edge
  boundary cases and cancellation after queued batches, followed by recovery.
- Frozen GPU fit trajectory and optimum matched their unchanged pin.
- Native reconstruction and saved-result round-trip checks passed.
- Eight complete consuming-application phase responses and native loss values
  matched the baseline exactly. Both datasets were exercised through actual
  C10 and tilt controls with full BF and automatic sampling.

Private evidence identifier: `ssb-first-update-20261017`. Raw acquisition data,
trace files, screenshots and application logs are kept outside this repository.
