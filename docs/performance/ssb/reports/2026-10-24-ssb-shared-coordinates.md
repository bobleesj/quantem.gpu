# CUDA SSB: shared BF coordinates and inactive blocks

## Workload and change

Full 512 × 512 × 192 × 192 input, all 8,807 active BF pixels, saved fifth-order
optics, sample tilt and depth. This measures coherent phase reconstruction
(`phase_of_mean`) during parameter changes, not optimization or loss evaluation.

The wave-sum kernel stages BF coordinates beside each tile's probe geometry,
replacing repeated global-coordinate reads with shared-memory broadcasts. A
block-wide vote skips probe-tile preparation when no lane can contribute.
Empty blocks still write their original DC or zero outputs. BF accumulation
order, output coverage, precision and reconstruction estimator are unchanged.
The shared allocation grows by 1 KiB per block; there is no additional resident
dataset cache. The common kernel serves native and finer output grids.

## Isolated matched timing

On an RTX PRO 6000 Blackwell Max-Q Workstation Edition, the reference and
candidate alternate within one freshly loaded, isolated CUDA process. Twenty
rounds change C10 and C30; medians exclude two warmups. Each call synchronizes
before and after reconstruction. Host delivery and browser paint are excluded.
The other GPU may be active; clocks are not locked. These are workload timings,
not theoretical hardware limits.

| Output | Reference | Candidate | Time reduction |
|---|---:|---:|---:|
| 1× | 19.24 ms | 18.95 ms | 1.5% |
| 2× | 39.06 ms | 38.12 ms | 2.4% |
| 3× | 43.96 ms | 42.63 ms | 3.0% |
| 4× | 48.81 ms | 46.92 ms | 3.9% |

## Actual notebook image updates

The existing CUDA notebook was driven in the in-app browser. Apple Metal
WebGPU rendered phase and FFT side by side. Counters track new scientific
image paints rather than animation frames; the sweep target remained 60 FPS.
The full saved model retained twelve nonzero higher-order magnitudes, tilt and
depth. The quiet window had no competing compute job on the measured GPU.

| Output sampling | Phase FPS | FFT FPS |
|---|---:|---:|
| 1× | 49.39 | 49.39 |
| 2× | 24.62 | 24.64 |
| 3× | 21.99 | 14.10 |
| 4× | 18.88 | 12.60 |

These are single candidate observation intervals, not matched browser speedup
claims against an earlier day or a guarantee of sustained rate. The kernel
speedup above is small; 60 FPS remains unmet. C10 sweeps were measured at each
sampling. A C30 sweep at 4× additionally verified parameter switching, followed
by restoration of the saved joint optics, tilt and depth. The calibration file
was not overwritten.

At 3×, the 1536-square phase produces a padded 2048-square display FFT; at 4×,
both are 2048 square. Phase and FFT have different update rates. The inspected
widget path does window preparation in a worker, WebGPU FFT, spectrum readback,
and worker magnitude/histogram processing before WebGPU painting. Thus WebGPU
rendering does not imply a wholly GPU-resident FFT display pipeline. No stage
profile was taken here to apportion that additional FFT latency, and no widget
implementation change is claimed.

## Correctness and rejected candidates

The isolated process compared 80 full phase arrays over 1×, 2×, 3×, 4× and 8×,
eight rotations, five depths including zero, and changing tilt and aberrations.
All were bit-identical to the frozen previous GPU kernel. All 31 existing CUDA
wave-average/thick-sample tests passed, including independent references and
anisotropic detector calibration.

An initial experiment inside the live notebook session produced three large
mismatches; neither targeted forward/reverse replays nor a full replay
reproduced them. That live-session run is retained as failed evidence, not
silently discarded. The isolated-process comparison is the acceptance basis;
the origin of the transient discrepancy is not established.

Closed-aperture polynomial shortcuts and pointer-restriction variants failed
strict numerical parity. Larger BF accumulation groups changed some pixels at
finer sampling. Larger thread blocks and unrolled loops were generally slower.
None of those candidates is enabled.

The previous kernel file SHA-256 is
`3197bf5fd16049d8876d412bbae2d5375aa9eca4019b6dd3da64054e3fe64bb5`.
The candidate SHA-256 is
`68819da1e42d95d808c17510ee2d56398c632ba551097959bb9f91facc648c0d`.
Raw evidence is archived privately.
This is a local source change, not a package release. MPS and fitting paths
were not modified or requalified by this experiment.
