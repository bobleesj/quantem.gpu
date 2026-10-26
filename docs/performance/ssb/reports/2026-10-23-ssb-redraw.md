# Full-field SSB redraws, 2026-10-23

This comparison retains a 512 × 512 scan, 192 × 192 detector, all 9,275
bright-field pixels, the same fitted optics through fifth order, sample tilt,
depth spread and `phase_of_mean` reconstruction. It measures changed defocus,
not repeated returns of a cached image. Fitting is not timed here.

## CUDA correction

Higher-order coefficients now enter the coherent correction kernel as a
per-launch parameter struct. CUDA can broadcast these 28 float32 values
through its constant cache. There is no mutable module-level coefficient
array, so concurrent sessions retain independent optics. The correction
arithmetic, compensated accumulation, complex128 inverse FFT and float32
phase output are unchanged.

The hardware is an RTX PRO 6000 Blackwell Max-Q Workstation Edition. The
backend baseline is `e8750f3e3652`; the candidate changes
`direct_ptycho/cuda/kernels/engine.py` and `direct_ptycho/cuda/thick.py` on that checkout.
The browser frontend baseline is `5dc84772dc4b` with the existing ShowPtycho
upsampling and histogram changes. Both runs use the same full acquisition and
saved calibration.

| Upsampling | Output | Before, ms | After, ms |
| --- | --- | ---: | ---: |
| 1 | 512 × 512 | 36.58 | 31.31 |
| 2 | 1024 × 1024 | 128.68 | 109.39 |
| 3 | 1536 × 1536 | 248.39 | 213.80 |
| 4 | 2048 × 2048 | 338.58 | 294.16 |
| 8 | 4096 × 4096 | 508.95 | 363.51 |

These are synchronized warm medians of ten changed-defocus observations,
including host delivery; preparation and the first observation are separate.
At 8×, wall times were variable: the candidate p90 was 422.86 ms and the
maximum was 928.92 ms. Paired interleaved trials support a roughly 10–14%
CUDA gain at 1×–4×; the separately measured 8× median is not a stable
percentage-speedup promise.

Full output arrays are bit-identical to the pre-change CUDA oracle at every
factor. Alternating before/after full-data trials also compare exactly. The
hardware regression suite passed 46 tests, and an additional concurrent-stream
regression verifies independent higher-order coefficients for two sessions.

## Browser display

The Mac frontend uses hardware WebGPU for the complete FFT and colormap,
including 4096 × 4096 images, and a canvas blit for presentation. Display
preparation, magnitude and histogram work still run in a worker; this is not
an entirely GPU-resident end-to-end pipeline.

The display changes reuse one bounded FFT workspace and remove intermediate
centering, split-spectrum and shift arrays. They preserve the existing
float32 rounding before windowing and `log1p`. Real-data browser comparisons
match every output bit, including overlapping transforms, inverse transforms
and changes of size. The histogram reuses the already measured finite range.

On Apple M5 Max, the isolated 4096² FFT stage fell from 50.9 to 41.2 ms in
an alternating 12-observation comparison. Colormapping plus the GPU blit is
about 9–11 ms. This display-only timing is not scientific reconstruction FPS.
One cached 4096² FFT workspace retains approximately 256 MiB of GPU buffers;
changing dimensions replaces it rather than retaining a cache per size.
Concurrent calls use separate temporary workspaces. Intermediate CPU arrays
are eliminated, but overall process peak memory was not inferred from this
allocation accounting.

## Actual notebook updates

With the same 4–6 nm C10 sweep requested at 60 Hz and both phase and FFT shown,
the final CUDA notebook delivered these **new scientific images**, measured
using passive paint counters. Repaints for zoom or contrast do not count.
These are complete development notebook measurements, not a released-build
performance guarantee.

| Upsampling | CUDA phase fps | MPS phase fps |
| --- | ---: | ---: |
| 1 | 28.93 | 7.01 |
| 2 | 8.69 | 1.51 |
| 3 | 4.38 | 0.47 |
| 4 | 3.21 | 0.52 |
| 8 | 1.60 | 0.47 |

These final rates use elapsed time between the canvas's own scientific-paint
counters and timestamps, including observed stalls. Automation-call wall
clocks are deliberately excluded: their asynchronous boundaries overstated
some rates, especially at high upsampling. Raw snapshots retain both clocks
for audit. These are single observation intervals, not steady-state guarantees.
The phase and FFT panels agree within approximately 0.01 fps. The MPS
measurements are mixed across factors; no MPS numerical kernel change was
accepted and no uniform MPS speedup is claimed.

The 4K CUDA kernel-to-host path during the final sweep had a median of 395 ms
(p10 366 ms, p90 550 ms); the corresponding MPS median was 1737 ms.
Reconstruction remains the main limit. Full-model 60 fps was **not achieved**.
A fresh kernel on each backend restored the saved full-field optics,
rotation, tilt and depth before measuring. The Apple GPU was M5 Max.

## Reproduce

Restore the reviewed full-field calibration in an existing `SSB` session,
then run:

```python
from quantem.gpu.benchmark import run_ssb_preview

report = run_ssb_preview(ssb, upsampling=(1, 2, 3, 4, 8), runs=11)
```

The report records hardware, source digest, full scientific settings,
preparation, first/warm distributions and output hashes. It explicitly does
not measure browser paint. Hashes identify outputs; independently retained
oracles establish parity. Do not compare different upsampling factors as a
quality-preserving speedup: they represent different reconstructed frequency
coverage. Native-grid or simpler-optics historical 60-fps measurements are
not evidence of 60 fps for this full fifth-order model.

MPS 4× and 8× were rechecked after multiple new frames arrived: 15 images / 28.9165 seconds = 0.519 fps, and 13 images / 27.3691 seconds = 0.475 fps. These replace the first MPS intervals for those factors. The workload and full output shape were verified again; the original joint fit was restored at 4× afterward.
