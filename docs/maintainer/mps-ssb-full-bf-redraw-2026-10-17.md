# Full BF MPS redraw comparison, 2026-10-17

## Change and scope

Zero-depth reconstruction now specializes the identically-one thickness weights.
It retains precise trigonometry, calibration, estimator, all BF pixels and the
original summation order. Joint tilt/depth reconstruction keeps the general
optical model. This builds on the exact aperture shortcuts and bounded queued
batches in `327337dc`; those existing changes account for the joint-mode gain.

## Matched measurements

Two complete 512 × 512 × 192 × 192 acquisitions were tested on Apple M5 Max.
Acquisition A has 8,807 BF pixels and a 1536² automatic output; B has 8,930 BF
pixels and a 2048² output. Each row uses 20 new parameter settings, alternating
the frozen deployed implementation (`10e0de54`) and this implementation on the
same prepared engine. Both use `phase_of_mean` and identical tilt and depth.
Ordinary controls set tilt and depth to zero without refitting.

Times include synchronized reconstruction and phase handoff to the caller.
They exclude loading, fitting, the native loss diagnostic and browser delivery.
First use of a specialization is included in the distribution.

| Acquisition | Mode | Baseline p50 / p95 / max (ms) | Candidate p50 / p95 / max (ms) |
| --- | --- | ---: | ---: |
| A | Joint C10 | 103.88 / 105.48 / 105.77 | 89.89 / 91.93 / 92.29 |
| A | Joint tilt | 102.99 / 104.20 / 104.94 | 89.28 / 90.61 / 91.02 |
| A | Ordinary C10 | 86.99 / 89.52 / 90.04 | 71.13 / 75.83 / 134.12 |
| A | Higher order | 235.79 / 238.59 / 241.97 | 237.90 / 245.03 / 285.80 |
| B | Joint C10 | 167.21 / 171.73 / 174.17 | 146.80 / 149.72 / 150.47 |
| B | Joint tilt | 177.77 / 186.93 / 193.00 | 155.50 / 162.23 / 164.34 |
| B | Ordinary C10 | 163.52 / 170.42 / 170.69 | 134.63 / 142.40 / 200.12 |
| B | Higher order | 534.69 / 624.32 / 649.35 | 543.79 / 639.84 / 702.85 |

All 160 complete phase-array comparisons were bit-identical. Higher-order
controls did not become faster. The first ordinary specialization calls took
134 and 200 ms; subsequent maxima were 73 and 139 ms. Kernel specialization
does not eliminate first-use compilation or idle residency latency.

Peak active MLX allocation was 8.82 / 9.09 GiB. Process peak RSS was
2.96 / 1.50 GiB, a separate metric that does not include all device allocation.
The alternating benchmark harness took 34.13 / 46.30 s including loading,
preparation, both implementations and array validation. These are not
single-workflow times. Loading took 2.20 / 2.39 s; first preparation and
reference reconstruction took 10.51 / 1.65 s in those processes.

## Historical 60 FPS claim

The recorded 63.9 / 66.8 FPS result in `ssb-performance.md` is a synthetic
CUDA object-redraw benchmark with 8,809 BF pixels on a 512² native grid.
It does not establish 60 FPS for Metal with automatic sampling and joint
tilt/depth correction. Replaying the historical Metal native ordinary kernel
on A and B gave 35.0 / 36.6 ms, using all BF pixels. That older kernel lacks
the matched joint/automatic mode and uses different trigonometric precision;
it is not a drop-in equivalent for the present scientific contract.

No matching historical Metal 60 FPS evidence was found. The unrelated
2,824-BF benchmark is not a baseline for these acquisitions.

## Bottleneck and rejected approaches

Controlled diagnostic ablations implicate optical calculations more than source
reads alone. They deliberately omit calculations and are not valid outputs or
additive timing components. Separable trigonometric caches, different thread
groups, larger batches and larger sum groups did not produce a sufficient
repeatable gain while preserving the requested behavior. They were not kept.

The 16.7 ms scientific-frame budget remains unmet. These measurements do not
establish a theoretical hardware limit. Idle driver preparation can add latency
before the GPU executes; warm redraw measurements do not measure that delay.

## Verification

- 77 focused hardware tests passed for ordinary, joint, coherent and higher-order
  reconstruction, including exact zero-depth specialization and cancellation.
- The frozen GPU fit trajectory, optimum and loss matched the unchanged pin.
- Native reconstruction and saved-result workflow checks passed.
- Eight application comparisons retained every phase bit and identical native
  loss values. Actual C10 and tilt controls were exercised on both acquisitions
  with full BF and automatic sampling.

An additional 24 application phase-array pairs were bit-identical. Warm HTTP
responses, including full binary output delivery but excluding browser paint,
improved from 115.24 to 102.36 ms median for A and 193.57 to 171.22 ms for B
(eight new settings each). Candidate p95 / max were 107.42 / 108.73 ms and
173.41 / 174.00 ms. Queue waits were below 0.2 ms.

After a native loss diagnostic and six seconds idle, two measurements per
acquisition gave medians of 253.05 to 198.44 ms and 367.96 to 273.48 ms.
This small sample confirms that the idle cost remains; it does not qualify
rare latency tails. Dataset opening/preparation was measured separately and
is not included in those warm response distributions.

Private evidence identifier: `ssb-full-bf-redraw-20261017`. Raw data, arrays,
screenshots, logs and machine paths remain outside this repository.
