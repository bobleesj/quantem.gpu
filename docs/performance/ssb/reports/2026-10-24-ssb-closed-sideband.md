# CUDA SSB closed-sideband correction and host transfer

A full 512 × 512 × 192 × 192 acquisition with 8,807 bright-field pixels was
replayed with all 14 aberration terms, 12 nonzero higher-order magnitudes,
sample tilt and depth. No input positions, detector samples, precision or
nonzero aperture contributions were removed. The change affects fine-grid
paired CUDA wave reconstruction, not joint fitting or its objective.

## Counter evidence

Nsight Compute 2025.4.1 profiled the first correction/summation launch on an
RTX PRO 6000 Blackwell Max-Q Workstation Edition (300 W). Default profiler
clock control held the measured SM clock near 1 GHz. These launch durations
are not application reconstruction timings or browser FPS.

| Counter | 2× before | 2× after | 4× before | 4× after |
| --- | ---: | ---: | ---: | ---: |
| Profiled launch, ms | 14.20 | 11.86 | 15.49 | 13.01 |
| Executed warp instructions, billions | 8.286 | 6.629 | 8.824 | 7.140 |
| Compute throughput, % | 78.59 | 74.83 | 76.39 | 73.11 |
| DRAM throughput, % | 20.16 | 24.25 | 28.58 | 34.23 |
| Registers per thread | 68 | 69 | 68 | 69 |
| Achieved occupancy, % | 47.76 | 47.78 | 47.30 | 47.52 |
| Local spilling requests | 0 | 0 | 0 | 0 |

Instruction work, rather than DRAM saturation, was the main limitation.
When either sideband's soft aperture is exactly closed, its contribution is
zero. The kernel now omits that term's higher-order polynomial, depth sinc
and trigonometry. The other term, normalization, summation order and full
complex inverse FFT are unchanged. No GPU cache or additional GPU buffer
was introduced.

## Matched warm reconstruction

Alternating reference/candidate runs used a fresh process, the same saved
joint parameters and changing defocus/higher-order coefficients. Each median
uses 18 synchronized warm observations. This boundary includes reconstruction
and the native output copy, but excludes host transfer and browser rendering.

| Sampling | Before median / p90, ms | After median / p90, ms |
| --- | ---: | ---: |
| 1×, unchanged kernel | 18.44 / 19.59 | 18.44 / 18.91 |
| 2× | 37.13 / 37.18 | 31.90 / 31.95 |
| 3× | 41.53 / 41.67 | 36.05 / 36.13 |
| 4× | 45.90 / 45.96 | 40.30 / 40.40 |

Register caps alone were neutral or slower; tighter caps changed output.
Splitting ordinary and boundary frequencies did not provide a worthwhile gain.
Those variants were not adopted. A register cap combined with the chosen
change was slightly faster on this card, but the default compiler allocation
was retained rather than imposing an unverified policy on other CUDA cards.

## Actual notebook interaction

The same existing notebook, input, fit and WebGPU renderer were used before
and after. C10 swept at a requested 60 Hz; counts below are actual new
scientific canvas paints over 10–19-second warm intervals. Both phase and FFT
remained visible. The browser used a physical Apple Metal WebGPU adapter,
with remote CUDA reconstruction on the card identified above.

| Sampling | Phase before / after, fps | FFT before / after, fps |
| --- | ---: | ---: |
| 2× | 24.7 / 28.1 | 24.7 / 28.1 |
| 3× | 22.0 / 23.8 | 22.0 / 23.8 |
| 4× | 18.8 / 20.2 | 18.8 / 20.2 |

The kernel-only candidate initially exposed intermittent host-preview stalls:
4× browser replays fell to 12.1–12.9 fps despite faster CUDA reconstruction.
Explicit owned pinned host buffers removed those stalls in the measured replay.
Over 511 prototype preview calls, median/p90/max were 37.6/38.4/50.3 ms. The
final implementation transfers directly into an independently owned NumPy
array backed by the pinned allocator. Later previews do not overwrite earlier
arrays. The returned float32 pixels and public API remain unchanged.

The phase images are 1024², 1536² and 2048² respectively. The 3× display FFT
uses its existing 2048² zero-padded transform; no display-resolution reduction
was introduced. Direct C10 keyboard input, FFT contrast and restoring the saved
joint controls were also exercised. No browser warnings or errors were observed.

## Validation and limits

- 80 full-acquisition comparisons across factors 1, 2, 3, 4 and 8, varied
  rotation, depth, tilt and higher-order coefficients matched bit-for-bit.
- The final kernel plus host transfer matched the frozen previous output
  bit-for-bit at 2×, 3× and 4×.
- 40 checks passed: CUDA reconstruction and joint-fit regressions, independent
  preview ownership across sampling changes, and repository privacy checks.
- These measurements do not establish 60 fps or a theoretical hardware ceiling.
  The correction kernel remains the main compute cost; notebook delivery also
  contributes. They do not qualify MPS, another CUDA architecture or a release.

Experiment: `20261024-ssb-cuda-counters`. Raw counter reports, launch/source
records, rejected candidates, full-data parity records and browser frame
counts are retained in the private evidence archive.

Kernel source SHA-256 before:
`68819da1e42d95d808c17510ee2d56398c632ba551097959bb9f91facc648c0d`.
After: `32225632c94df818da972fe8718b6467d6a9ae9cd5534cd312e10d1215df3677`.
The checkout was based on `e8750f3e3652` with pre-existing uncommitted work;
this is a source-checkout measurement, not an installed-release qualification.
