# Higher-order SSB: group alias-edge work on CUDA

## Workload and change

The target is a new scientific phase image every 16.7 ms at 2× output sampling,
with full 512 × 512 × 192 × 192 data, fifth-order aberrations, sample tilt and
depth retained. This is parameter interaction after calibration, not refitting.

Fine-grid paired correction has special native FFT alias columns that require
independent mirror reads. Previously those lanes were scattered through the
ordinary interior. They split warps inside every bright-field iteration.
The new index mapping groups the special columns after the ordinary interior.
It changes thread ownership only: all output pixels, independent Nyquist
corrections, arithmetic, BF accumulation order and precision are retained.
There is no new cache or workspace and no per-size kernel implementation.

The existing guard restricts this path to higher-order fine-grid reconstruction
from a 512-square real FFT. Native output, other scan layouts, fitting, Newton
refinement, loss diagnostics and MPS are unchanged.

## Matched reconstruction measurements

RTX PRO 6000 Blackwell Max-Q Workstation Edition. Reference and candidate
alternate in one prepared session over twelve rounds, excluding two warmups.
The parameter sequence changes C10, C12, astigmatism angle, C30, a higher-order
azimuth, and sample tilt/depth. Synchronization bounds reconstruction; host
copy and browser rendering are excluded. The acquisition retains all 8,807
active BF pixels and its saved joint fit.

| Output sampling | Reference median | Candidate median | Time reduction |
|---|---:|---:|---:|
| 1× | 18.21 ms | 18.20 ms | unchanged |
| 2× | 41.98 ms | 37.68 ms | 10.2% |
| 3× | 46.57 ms | 42.36 ms | 9.0% |
| 4× | 51.11 ms | 47.20 ms | 7.6% |

The installed `run_ssb_preview` runner separately measured complete warm phase
reconstruction plus host delivery at 16.72, 37.68, 41.93 and 46.34 ms for
1×, 2×, 3× and 4×. These sequential measurements are not the causal speedup
comparison and are not notebook frame rates.

A second full acquisition, with 9,275 active BF pixels and different sampling,
matched exactly at 1×, 2×, 3×, 4× and 8× over five rotation/depth settings.
Its initial matched 2× measurements were 143.10 versus 126.76 ms. Another GPU
was active during that interval; later quieter measurements were substantially
faster for both variants. Do not interpret those different runtime intervals
as an additional software gain or rank acquisitions by data dimensions alone.

## Live notebook interaction

The CUDA notebook was executed in a fresh kernel and its real widget was driven
in the in-app browser. Apple Metal WebGPU rendered phase and FFT side by side.
Both counters measured newly painted scientific data frames, using the elapsed
time between the first and last recorded paint. The target sweep rate was 60;
the measured rates below are not the control setting or animation-only FPS.
The saved joint fit retained twelve nonzero higher-order magnitudes, tilt and
depth. The source scan and detector dimensions remained unchanged.

| Sampling and sweep | Reference phase FPS | Candidate phase FPS | Candidate FFT FPS |
|---|---:|---:|---:|
| 1× C10 | not measured | 50.48 | 50.50 |
| 2× C10 | 22.31 | 24.87 | 24.86 |
| 2× C30 | not measured | 24.36 | 24.35 |

The 2× C10 rate improved approximately 11.5%. Each row is one live observation
interval, not a long-duration reliability result. The previous kernel and the
candidate were selected within the same prepared session for the 2× comparison.
No parameters were re-fitted, no saved calibration was overwritten, and the
saved fit was restored after the sweeps.

**The 60 FPS target is not met.** At 2×, roughly 38 ms of reconstruction already
exceeds the 16.7 ms frame budget. Browser delivery adds overhead, but cannot
explain away this backend limit. This is a measured limit of the current code
and workload, not a theoretical hardware bound. MPS interaction was not changed
or re-measured in this experiment.

## Verification

All 48 matched first-acquisition outputs and all 25 second-acquisition joint
parameter cases are bit-identical to the frozen reference. Existing CUDA
wave-average and thick-sample tests passed: 31 tests, including independent
unpaired references, anisotropic detectors, and alias/Nyquist edges. Five
public-repository checks passed.

Aggressive register caps were rejected: the 48- and 56-register variants
changed the phase by up to 6.61e-6 radians. Smaller shared-memory tiles did
not improve the measured workload. A 64-register cap was exact in its probe
but added little benefit on top of the layout change and is not enabled.

The frozen reference kernel file has SHA-256
`026f16815ffe7e8312a42437863e939fcfd45546b469019e210591fce9d97822`.
Raw evidence and before/candidate sources are retained privately under
`20261023-ssb-alias-layout`. No package publication is claimed.

The candidate kernel file has SHA-256
`3197bf5fd16049d8876d412bbae2d5375aa9eca4019b6dd3da64054e3fe64bb5`.
