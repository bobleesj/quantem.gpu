# CUDA SSB compact support and boundary dispatch

The fine-grid correction still dominated interactive reconstruction after the
closed-sideband optimization. This pass retains the full 512 × 512 × 192 × 192
input, all 8,807 bright-field pixels, 14 aberration terms, sample tilt and depth.
It changes scratch storage and launch ownership, not the correction formulas,
BF order, precision, fit objective or output sampling.

## Implementation

A GPU-generated frequency map retains every frequency that can intersect the
full soft-aperture support. Its radius is rounded outward; the correction still
uses its original support predicate. The map is cached by grid and calibration,
not fitted aberrations. It stores the ordinary frequency pairs first and the
independent native-alias/Nyquist entries last.

Two specializations of the same correction source process those two ranges.
Ordinary pairs no longer carry the boundary branch through every BF iteration.
The existing compensated sums and complex128 group reductions operate on the
compact scratch plane. Results are scattered into a full complex128 spectrum
before the unchanged full-size inverse FFT. No interpolation, binning, detector
subsampling or reduced-resolution interaction is introduced.

At 4×, reduction scratch storage decreased from 1,024 MiB to 264.95 MiB. This is
one buffer, not total resident acquisition memory. At 2× it decreased from
256 MiB to 236.52 MiB. The coordinate map adds a small cached index allocation.
The BF group count still follows the previous full-plane workspace budget so
that reduction grouping and rounding do not change.

## Matched warm reconstruction

Physical device: RTX PRO 6000 Blackwell Max-Q Workstation Edition, 300 W.
A fresh process alternated frozen reference and candidate, with the same saved
joint model and changing defocus/higher-order coefficients. Each entry uses 18
synchronized warm observations; it includes the native output copy but excludes
host transfer, notebook delivery and browser rendering.

| Sampling | Before median / p90, ms | After median / p90, ms |
| --- | ---: | ---: |
| 1×, unchanged dispatch | 18.98 / 19.02 | 18.99 / 19.03 |
| 2× | 31.50 / 31.59 | 29.59 / 29.64 |
| 3× | 35.46 / 35.54 | 30.55 / 30.66 |
| 4× | 39.51 / 39.55 | 31.02 / 31.05 |

## Hardware counters

Nsight Compute 2025.4.1 used its default clock control, near 1 GHz. It captured
the first dense launch before and the first compact interior launch after;
the latter is followed by a small boundary launch. These counters identify the
main-kernel change, not total application latency. End-to-end reconstruction
above includes both launches and the full inverse FFT.

| Metric | 2× before / after | 4× before / after |
| --- | ---: | ---: |
| Profiled launch duration, ms | 11.87 / 10.18 | 13.11 / 10.46 |
| Compute throughput, % | 75.03 / 83.25 | 73.10 / 82.82 |
| DRAM throughput, % | 24.34 / 27.85 | 33.98 / 27.72 |
| Registers per thread | 69 / 70 | 69 / 70 |
| Achieved occupancy, % | 47.76 / 47.52 | 47.47 / 47.15 |
| Local spill requests | 0 / 0 | 0 / 0 |

The remaining correction is still primarily arithmetic work, not saturated DRAM
traffic. These counters do not establish a theoretical hardware ceiling.

## Actual notebook image updates

The existing notebook ran the saved full-field higher-order model. The C10
sweep requested 60 updates/s, with phase and FFT visible. Rates below count new
scientific canvas paints, not requested updates or browser animation callbacks.
The browser used an Apple Metal WebGPU adapter; reconstruction ran on the remote
CUDA device above. Intervals were 17–21 seconds after startup.

| Sampling | Before phase / FFT, fps | After phase / FFT, fps |
| --- | ---: | ---: |
| 2× | 28.93 / 28.93 | 30.56 / 30.54 |
| 3× | Not repeated in this pass | 27.92 / 27.93 |
| 4×, first pass | 21.03 / 21.03 | 22.28 / 22.24 |
| 4×, passive callback timers | - | 24.41 / 24.40 |
| 4×, final uninstrumented repeat | - | 21.91 / 21.92 |

The 4× variation is retained rather than reporting only the fastest interval.
Across 522 warm instrumented 4× requests, preview including host transfer was
29.46 ms median / 29.98 ms p90; the full Python widget callback was
31.52 / 32.41 ms. Delivery and browser updates contribute to the remaining
interval. The phase and FFT retain 2048² pixels at 4×; the existing 3× FFT uses
2048² padding for the 1536² phase image. No FFT quality change was made.

Direct C10 keyboard changes, FFT histogram adjustment, sampling switches and
restoring the saved joint fit were verified. No browser warnings or errors
were observed. Temporary timing hooks were removed, and the notebook was left
paused at 4× with the saved fit restored. The calibration file was not saved
over during testing.

## Validation and limits

Eighty full-acquisition comparisons across 1×, 2×, 3×, 4× and 8×, varied rotation,
defocus, astigmatism, higher-order terms, tilt and depth matched the frozen GPU
reference bit-for-bit. The native fit and loss path are unchanged. Dedicated
CUDA tests also cover anisotropic sampling, alias/Nyquist preservation, compact
workspace reuse, output ownership and changing output dimensions. The final
CUDA reconstruction, joint-fit regression, preview-ownership and repository
privacy run passed all 41 tests in 145.94 seconds.

Rejected experiments include cooperative sideband lanes (exact but slower),
staged polynomial evaluation (changed rounding without a useful speed gain),
and register caps/normalization shortcuts (small or inconsistent gains). The
private prototype needed an argument-list correction before its acceptance run;
only the corrected, independently tested production path was used in the browser.

60 fps remains unmet. These results do not qualify another CUDA architecture,
MPS, an installed wheel, or a release. No package publication is part of this pass.

Experiment: `20261024-ssb-cuda-single-sideband`. Raw counter reports, frozen
sources, rejected variants, parity records and notebook frame counts are in the
private evidence archive. Source checkout base: `e8750f3e3652`, with pre-existing
uncommitted work preserved.

Kernel source SHA-256 before:
`32225632c94df818da972fe8718b6467d6a9ae9cd5534cd312e10d1215df3677`.
After: `d38d2eb354011403790e000dd4dca9c680da2cf11e55c312de77f248292295aa`.
