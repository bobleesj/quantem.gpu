# Skip closed-aperture loads in fine-grid SSB

## Scope

The paired fine-grid CUDA wave sum already identifies contributions outside
both soft probe apertures as zero. Move that same outward-rounded geometric
test into the BF loop, before loading the spectrum and accumulating zeros.
Keep all nonzero corrections in their original order, with the same
compensated float32 sums, complex128 group reduction and inverse FFT.

This changes the existing 512-square, real-FFT, higher-order fine-grid path.
It introduces no new cache, public API or kernel family. Native reconstruction,
joint search, Newton refinement, native loss, MPS and browser kernels retain
their previous implementations. No data, detector support or precision is
reduced.

Reference kernel fingerprint:
`c49b7cfb2fe301f36f0143ce781dff37ec603623525a5fa7a2e9c97d81c47241`.
Candidate fingerprint:
`6f5621e3db38aa5e75f300e89ae5b19c1124609417a960924410bd352e8478c9`.
Sources, calibration, raw measurements, rejected candidates and parity evidence
are retained privately in experiment `20261023-ssb-zero-contributions`.

## Matched reconstruction timing

RTX PRO 6000 Blackwell Max-Q Workstation Edition; full 512 × 512 × 192 × 192
measured acquisition, fitted fifth-order optics, tilt and depth, and complete
BF support. Reference and candidate alternate in one prepared session, with
reversed order on successive rounds. C10 changes each round. Medians exclude
two warmup rounds and retain eight rounds. CUDA synchronization bounds each
reconstruction; transfer, parity checks and browser work are excluded.

| Upsampling | Reference | Candidate | Time reduction |
| --- | ---: | ---: | ---: |
| 2× | 50.63 ms | 43.88 ms | 13.3% |
| 4× | 61.94 ms | 52.76 ms | 14.8% |
| 8× | 98.34 ms | 88.77 ms | 9.7% |

These are reconstruction timings, not measured notebook frame rates. They do
not establish a hardware lower bound or a 60 fps result. Sampling factors
evaluate different output grids; each is compared with its own reference.

## Remaining costs

Before this change, CUDA event measurements at 4× attributed 50–55 ms to
correction, about 4.3 ms to group reduction, and about 1 ms to the final inverse
FFT and phase conversion. The standard preview runner measured about 1.3 ms
for copying the 2048-square float32 output to the host. Releasing a slider also
requests a separate native-grid phase-variance diagnostic, measured at roughly
445 ms; this diagnostic remains unchanged.

A separate HTTP payload probe through the notebook tunnel transferred 1, 4
and 16 MiB in warm medians of about 2, 6 and 12 ms. This is an independent
transport control, not a widget WebSocket or browser-presentation measurement.
Browser slider, FFT, WebGPU-adapter and presentation-rate verification was
blocked by disabled browser automation. No frontend FPS claim is made.

## Parity and rejected experiments

All 30 matched timing outputs are array-identical to the reference. A further
80 full-data cases across two acquisitions are array-identical at factors
1, 2, 3, 4 and 8, with varied rotation, defocus, astigmatism, tilt and depths
from zero through 60 nm. The shared hardware test also covers anisotropic
detector sampling and independently evaluated alias/Nyquist edges.
The installed implementation passed 31 CUDA tests, including joint-fit and
native-loss checks, in 123.19 seconds. Five repository privacy checks passed.

A fused group-reduction candidate preserved the tested outputs but did not
improve 4×. A group-size/thread-count sweep did not yield a reliable 4× gain;
some grouping changes also produced tiny nonzero errors. Neither experiment
is adopted. Existing group sizes, launch sizes and memory budgets are retained.

## Installed notebook replay

The maintained advanced CUDA tutorial completed all 12 code cells from a fresh
kernel, including aberration search, rotation comparison and ShowPtycho creation.
A subsequent prepared preview replay changed C10 twelve times per factor.
All 36 output hashes matched the corresponding pre-change outputs.

| Upsampling | Reconstruction median | Host copy median | Combined median |
| --- | ---: | ---: | ---: |
| 1× | 17.26 ms | 0.18 ms | 17.44 ms |
| 2× | 43.92 ms | 0.42 ms | 44.33 ms |
| 4× | 50.56 ms | 1.29 ms | 51.80 ms |

These separate-run medians exclude the first observation and use eleven warm
observations. Use the alternating matched experiment above for the causal
speed comparison. Notebook execution and preview replay do not verify browser
presentation or interactive FFT behavior.
