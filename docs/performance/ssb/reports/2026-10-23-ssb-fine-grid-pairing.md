# CUDA SSB fine-grid correction

## Scope and reference

This extends the validated native-frequency reuse to finer output grids for a
512 × 512 scan. The real input remains 512 × 512 × 192 × 192, with all 9,275
active BF pixels, unchanged hot-pixel preparation, fifth-order aberrations,
sample tilt and depth. The quantity is the phase of the coherently averaged
object wave at fixed calibration. Search, Newton refinement, native-grid loss,
MPS and WebGPU compute are unchanged.

Reference: checkout `e8750f3e3652` with the previously validated native pairing
implementation. Frozen reference and candidate sources, calibration, logs,
profiler captures and numerical comparisons are retained in private experiment
`20261023-ssb-fine-grid`. No input arrays or derived detector-spectrum caches
are published with this summary.

Final source fingerprints:

- `cuda/kernels/engine.py`: `c49b7cfb2fe301f36f0143ce781dff37ec603623525a5fa7a2e9c97d81c47241`.
- `cuda/thick.py`: `cd1fc688f511191f9198dcb2728733422846a5e74782bff76dd17bed4041c320`.

## Why this path was slow

On the secondary Blackwell Max-Q device, synchronized stage measurements put
about 98% of the 4× reconstruction inside the correction kernels. Avoiding
shared preparation for entirely empty blocks was essentially neutral. Skipping
individual zero-aperture polynomial terms changed output and was rejected.

Opposite spatial frequencies share shifted probe, polynomial and depth work.
The new guarded variant reuses those intermediates while independently
normalizing both complex corrections. It retains the original BF order,
compensated accumulation, complex128 group reduction and complex128 inverse
FFT. It does not impose Hermitian symmetry on the corrected wave.

Fine grids need additional edge handling: the entire output Nyquist row is
evaluated directly, and repeated native real-FFT boundary columns read their
own stored partner values. A conservative, outward-rounded geometric bound
skips only pairs for which both soft apertures are exactly closed. The default
partial workspace budget remains 128 MiB; no large cache is introduced.

One shared CUDA kernel source serves the variants. Automatic dispatch enables
the new path for finer grids from a 512-square, real-FFT higher-order session.
Other input layouts retain their previous path. A 64/128/256/512-thread sweep
retained the existing 128-thread fine-grid launch; larger launches did not help.

## Matched GPU timing

Hardware: RTX PRO 6000 Blackwell Workstation Edition. Three variants alternate
in the same prepared session, with reversed order on successive rounds and
changed C10 values. Medians exclude two warmup rounds and retain twelve rounds
per factor. GPU synchronization bounds reconstruction; copies and numerical
comparison occur afterward. Clocks and caches are not forcibly controlled.

| Output factor | Reference | Frequency reuse | Reuse plus geometric bound | Time reduction |
| --- | ---: | ---: | ---: | ---: |
| 1× | 14.08 ms | 14.06 ms | 14.08 ms | unchanged |
| 2× | 80.44 ms | 58.54 ms | 57.98 ms | 27.9% |
| 3× | 159.11 ms | 119.34 ms | 117.48 ms | 26.2% |
| 4× | 216.80 ms | 165.32 ms | 159.70 ms | 26.3% |
| 8× | 271.03 ms | 218.88 ms | 211.26 ms | 22.1% |

These are backend reconstruction timings, not notebook FPS. Each factor
preserves its own complete reference output; different factors evaluate
different frequency coverage and are not interchangeable quality settings.
At 4× and 8×, reaching 60 fps would still require roughly 9.6× and 12.7× more
backend throughput, before transfer and display. These measurements do not
establish a hardware lower bound.

## Hardware counters

A matched 4× correction-launch capture compares the reference with frequency
reuse before the extra geometric bound. Nsight replay times are distinct from
normal reconstruction timings above.

| Counter | Reference | Frequency reuse |
| --- | ---: | ---: |
| Executed warp instructions | 67.33 billion | 50.81 billion |
| Profiled correction launch | 41.81 ms | 32.19 ms |
| SM throughput | 92.18% | 79.97% |
| DRAM throughput | 7.72% | 11.16% |
| Achieved occupancy | 72.26% | 52.94% |
| Registers per thread | 53 | 63 |

This is primarily an instruction-cost problem; bandwidth is not saturated.
The reuse variant executes about 25% fewer instructions, at the cost of more
registers. The separate launch sweep rules out simply increasing block size
as a further gain for this workload.

## Verification and limits

The geometric-bound candidate matched all forty real-data joint-parameter
cases exactly across factors 1, 2, 3, 4 and 8, rotations 0, 17, 177 and 270
degrees, depths 0–50 nm, and varied tilt/defocus/astigmatism. All matched timing
outputs also equal their reference byte for byte. A corresponding forty-case
sweep of frequency reuse alone passed on the other CUDA device.

The installed-source run passed 31 CUDA tests, including full-plane parity at
24 combinations of upsampling, rotation, depth and square/anisotropic detector
calibration. The standard `run_ssb_preview` runner compared the installed
candidate with the frozen reference: all 55 output hashes match at 1×–8×.
The public-repository privacy checks also passed (5 tests).

The standard runner used the secondary RTX PRO 6000 Blackwell Max-Q device;
these sequential measurements are separate from the alternating GPU timing
above. They measure complete phase delivery to the host, not browser paint.

| Output factor | Warm reconstruction median | Warm reconstruction p90 | Warm download median | Warm delivery median |
| --- | ---: | ---: | ---: | ---: |

| 1× | 18.45 ms | 19.76 ms | 0.17 ms | 18.62 ms |
| 2× | 83.13 ms | 83.33 ms | 0.42 ms | 83.55 ms |
| 3× | 169.67 ms | 169.90 ms | 0.73 ms | 170.41 ms |
| 4× | 229.21 ms | 229.55 ms | 3.34 ms | 232.55 ms |
| 8× | 285.57 ms | 285.74 ms | 25.08 ms | 310.49 ms |

At 8×, host download had large outliers: its p90 was 183.24 ms and complete
phase-delivery p90 was 469.17 ms. Those are retained rather than describing
reconstruction alone as interactive FPS. Browser interaction is not requalified
here, and existing Python kernels must reload the updated backend before using
it. No package release or push was performed.
