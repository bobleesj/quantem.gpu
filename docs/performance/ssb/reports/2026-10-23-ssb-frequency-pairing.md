# Native 512 SSB frequency pairing

## Question and workload

Can the paired-frequency prototype remove duplicate work while retaining the
previous GPU result? The earlier instruction-reduction experiment was faster
but failed numerical acceptance. This follow-up preserves its frozen reference
and enables a corrected implementation only for native 512-square, higher-order
coherent reconstruction with real-FFT half-plane storage.

The real acquisition is 512 × 512 × 192 × 192, with all 9,275 active BF pixels,
the same recorded hot-pixel preparation, fifth-order optics, tilt and depth.
Neither input data nor precision, tolerances or summation order change. These
are fixed-calibration redraws, not search/refinement timings.

Hardware: RTX PRO 6000 Blackwell Workstation Edition, CUDA GPU 0. Additional
numerical checks use the Blackwell Max-Q device. The checkout is `e8750f3e3652`
plus the previously tested coefficient-struct and launch-layout changes.
Final source SHA256:

- `cuda/kernels/engine.py`: `0ea3ad99f6d7395c1d945b4c0a0d840746da9588f343e3a63b8417cd3695e59f`
- `cuda/thick.py`: `22f23a6e4ba3990597d575c6503b96297c8d3f16da3fb35c3f23212af8d9c504`

Private evidence: `20261023-ssb-pair-parity`. Frozen sources, parameter cases,
timing samples, hashes and profiler captures are retained there.

## Implementation

At opposite spatial frequencies, the shifted probe evaluations exchange roles.
One shared correction helper reuses their polynomial, geometry and depth
intermediates, while evaluating and normalizing both complex outputs. It does
not assume Hermitian symmetry of the corrected higher-order reconstruction.
The BF order, compensated sums, complex128 reduction and inverse FFT remain
unchanged. The existing 128 MiB default partial workspace is retained; there
is no acquisition-sized cache.

Explicit rounding of the depth shift and gamma normalization preserves the
reference arithmetic. This fixes the earlier mismatch: the corrected version
matches all 109 real-data parameter cases exactly. The two exceptional columns
are scheduled together so normal warps avoid mixed edge/pair branches; the
Nyquist row still receives direct corrections. This remains one shared kernel
source, with a guarded launch variant. Other shapes, fine grids and MPS retain
their existing paths.

## Matched warm comparison

Alternating reference/candidate observations use eight C10 settings in the
same prepared session. GPU synchronization bounds each reconstruction; output
copies and comparisons are outside timing. The first two rounds are excluded
from these medians.

| Implementation | Median reconstruction |
| --- | ---: |
| Frozen full-plane reference | 21.93 ms |
| Paired corrections with original frequency layout | 15.95 ms |
| Paired corrections with grouped edge frequencies | 14.36 ms |

The accepted layout reduces time by about 35%, or 1.53× throughput. A separate
32-round launch-size comparison measured 14.31 ms with 256 threads. The 512
thread result was effectively tied, so the existing 256-thread configuration
was retained. This is approximately 70 backend reconstructions per second,
not measured notebook frames per second.

## Standard preview runner

`quantem.gpu.benchmark.run_ssb_preview` also ran against the frozen reference
and the final checkout, with 11 changed-C10 observations per factor. The
following candidate results separate reconstruction and host delivery. First
observation does not mean cold disk or cold compilation. Sequential runner
passes are descriptive; the alternating experiment above isolates the speedup.

| Upsampling | First reconstruction | Warm median | Warm p90 | Median download | Median phase delivery |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1× | 15.12 ms | 13.42 ms | 14.61 ms | 0.14 ms | 13.56 ms |
| 2× | 81.43 ms | 83.76 ms | 84.29 ms | 0.39 ms | 84.15 ms |
| 3× | 165.23 ms | 164.83 ms | 166.08 ms | 0.77 ms | 165.58 ms |
| 4× | 229.05 ms | 223.13 ms | 223.62 ms | 3.21 ms | 226.33 ms |
| 8× | 279.50 ms | 274.86 ms | 275.62 ms | 13.52 ms | 288.12 ms |

The 8× host copy had outliers, including 244 ms; its delivery p90 was 357 ms.
Fine-grid paths were not optimized here. All 55 complete output hashes match
the frozen reference. Changing upsampling changes the evaluated frequency
coverage, so these rows are different workloads, not quality-equivalent
speedups.

## Hardware counters and verification

A quiet Nsight Compute capture of the middle correction launch reports:

| Counter | Previous reference | Accepted implementation |
| --- | ---: | ---: |
| Executed warp instructions | 6.689 billion | 3.998 billion |
| Profiled launch duration | 4.27 ms | 2.83 ms |
| Instruction issue slots busy | about 92% | 83.10% |
| DRAM throughput | about 32% | 48.31% |
| Achieved occupancy | about 65% | 64.92% |

The accepted kernel uses 64 registers per thread and has no local-memory
spilling. Instructions fall by about 40%; profiler replay duration is separate
from normal warm reconstruction. Clocks and caches were not forcibly controlled.
These counters do not establish an algorithmic or hardware lower bound.

- 109 real-data joint-parameter comparisons match the frozen GPU phase exactly.
- 24 synthetic calibration combinations cover rotation, depth and unequal
  detector sampling, with full-plane GPU comparison in regression tests.
- Three full-data GPU 1 checks compare both complex spectra and final phase
  byte for byte, including zero depth and a large defocus/depth change.
- The final CUDA wave-average, upsampling, higher-order and thick-sample suite
  passes all 45 tests. An initial run passed the scientific assertions but
  failed one stale error-message expectation (`num_trials=0` versus the current
  `trials=0`); the expectation was corrected and the complete suite rerun.

The native CUDA path is enabled in the working checkout. No commit, release or
push is part of this experiment. MPS and notebook/browser image rate have not
been requalified by these measurements.
