# Faster broad SSB search with identical results

> **Historical:** describes the code as of 2026-10-22; its fits refined with the finite-difference Newton, since replaced by exact derivatives, so its fit timings are not current. See the [current fit methods](../../../api/direct_ptycho.md).

## What changed

The accepted change batches the independent Optuna proposal calculations and reuses completed trial history. RNG draw order, Gaussian models, float64 density ranking and each parameter's inverse-CDF stopping rule remain unchanged. Scientific objective evaluation and reconstruction remain CUDA; Optuna and Newton orchestration remain CPU. This is a hybrid solver.

The optimized sampler is qualified for Optuna 4.9.0. Other versions retain stock Optuna. No CUDA reconstruction kernel was changed.

## Matched workflow comparison

Three alternating fresh-process pairs per dataset and scan size on physical GPU1 (RTX PRO 6000 Blackwell Max-Q, 96 GiB). All valid BF pixels and the complete detector are retained. The 256² rows explicitly select the upper-left native scan region; 512² rows use the full acquisition. No detector binning or precision change. Source-page and compiler disk caches were not evicted, so these are not cold-disk measurements.

Workload: fixed recorded rotation; 200 primary trials plus 400 fifth-order broad trials, then Newton with the unchanged 3,000-evaluation cap. Fit on the native grid, reconstruct with Auto. Auto is 3× for A, 4× for B, and 1× for C because its existing aperture/sampling calls for 1×. The ordinary benchmark below separately tests estimated rotation branches. These broad timings do not include rotation discovery.

Workflow timing includes loading/upload, preparation, Fourier construction, fit, Auto reconstruction, host handoff and saving complete NumPy arrays. Imports and final cleanup are excluded. Each row has only three samples; p95 is descriptive, not a reliable tail-latency bound.

| Dataset | Scan | All BF | Auto output | Before p50 / p95 / max (s) | After p50 / p95 / max (s) | Reduction in median |
|---|---:|---:|---:|---:|---:|---:|
| A | 256² | 8807 | 768² | 10.888 / 11.080 / 11.101 | 6.869 / 7.279 / 7.324 | 36.9% |
| A | 512² | 8807 | 1536² | 17.527 / 18.229 / 18.307 | 13.954 / 13.979 / 13.982 | 20.4% |
| B | 256² | 8930 | 1024² | 12.217 / 12.844 / 12.913 | 8.904 / 8.921 / 8.923 | 27.1% |
| B | 512² | 8930 | 2048² | 23.964 / 24.082 / 24.095 | 20.456 / 20.456 / 20.456 | 14.6% |
| C | 256² | 101 | 256² | 8.108 / 8.375 / 8.405 | 4.955 / 5.856 / 5.956 | 38.9% |
| C | 512² | 101 | 512² | 9.292 / 9.387 / 9.398 | 5.555 / 5.563 / 5.564 | 40.2% |

The baseline is the same in-development broad API with stock Optuna, not the standard-search API on main.

## After-change stages and memory

Median seconds. Whole-device sampled peaks include other resident processes and the additional ordinary parity reconstruction. CuPy reserved is sampled at stage boundaries.

| Dataset | Scan | Load | Preparation | Fourier | Fit | Auto reconstruction | Host handoff + array save | Total workflow | Process wall | Peak device used (GiB) | Peak CuPy reserved (GiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dataset A | 256² | 0.671 | 0.098 | 0.053 | 6.022 | 0.021 | 0.012 | 6.869 | 9.298 | 15.55 | 8.19 |
| dataset A | 512² | 0.684 | 0.053 | 0.126 | 12.929 | 0.077 | 0.051 | 13.954 | 16.491 | 25.39 | 15.46 |
| dataset B | 256² | 0.625 | 0.100 | 0.053 | 8.007 | 0.035 | 0.023 | 8.904 | 11.305 | 16.14 | 8.46 |
| dataset B | 512² | 0.618 | 0.055 | 0.126 | 19.464 | 0.139 | 0.050 | 20.456 | 22.955 | 25.93 | 16.35 |
| dataset C | 256² | 1.109 | 0.000 | 0.041 | 3.796 | 0.000 | 0.010 | 4.955 | 7.366 | 4.03 | 0.23 |
| dataset C | 512² | 1.719 | 0.000 | 0.046 | 3.662 | 0.000 | 0.018 | 5.555 | 7.942 | 4.28 | 0.49 |

## Verification

- All 18 matched pairs reproduce complete native/Auto phase arrays, ordinary zero-tilt/depth phase arrays, all 400 proposals, coarse objective scores, Newton iterates/scores and final coordinates bit for bit. Elapsed-time columns are excluded from numerical comparisons.
- The production helper independently reproduces the frozen dataset A, dataset B and dataset C 512² arrays. dataset A/dataset B production-port checks ran on physical GPU0 (RTX PRO 6000 Blackwell Workstation Edition); repeated timings above use GPU1. The GPUs are not treated as interchangeable performance baselines.
- The shipped standard SSB benchmark passes on dataset A and dataset B against origin/main, including full-field fitting, regional fits, movement and resizing. Every returned phase hash matches. It explicitly requests `search="standard"` to preserve its 200-trial-per-branch plus Newton workload despite the broad API default. The existing 127×128 resize test reports its padding to 128².
- 21 focused sampler, search-mode, benchmark and privacy checks pass. Three adaptive synthetic objectives/seeds preserve every proposal across 400 trials; extreme-tail and boundary quantiles match stock Optuna. Boundary warnings occur in both implementations.
- Fresh dataset A notebook: CUDA broad fit plus Auto output; complete phase and all proposals match the frozen reference. ShowPtycho renders the phase/FFT and restores the fitted C10 after interaction. Browser rendering/interactive shader work is WebGPU; no CUDA slider-FPS claim is made.

## GPU reconstruction profile: what remains

Synthetic resident, all 8,809 BF pixels, 192² detector geometry, tilt/depth enabled, 4× reconstruction, 10 warm samples on physical GPU0. These are reconstruction-only timings, not full acquisition loading or fitting; 1024² real-data fitting remains untested.

| Native scan | Output | p50 / p95 / max (ms) | CuPy reserved (GB) |
|---:|---:|---:|---:|
| 256² | 1024² | 6.04 / 6.09 / 6.10 | 7.09 |
| 512² | 2048² | 21.47 / 21.72 / 21.73 | 9.83 |
| 1024² | 4096² | 91.36 / 91.66 / 91.74 | 37.91 |

At 1024², optical correction/BF accumulation takes about 74.5 ms and FFT about 2.3 ms. GPU-event profiling identifies the expensive stage, but hardware instruction/memory counters were unavailable; this does not establish a bandwidth or compute roofline or a theoretical minimum.

## Rejected approaches

- Reusing the standard-search Gaussian sampler for broad search was slower: 8.83 s versus 6.45 s for the CPU replay. Rejected.
- Skipping wholly inactive aperture blocks preserved phase but improved reconstruction only around 1%. Kept private; not integrated.
- Avoiding calculations for closed individual apertures preserved phase but slowed reconstruction 1–2%. Rejected.
- Shared FFT compiler-cache discrepancy remains held. Specialized 512/1024 schedules and accumulation order are unchanged.

## Limits and next target

The speedup removes CPU proposal overhead. It does not improve the fitted solution, lower trial budgets, certify physical ground truth, or prove a global minimum. More GPU speed requires reducing the remaining optical/BF work without changing its arithmetic. The current kernels are retained because the tested replacements did not justify integration.

Existing ShowPtycho explanatory text still refers to a primary-only automatic fit, and its inherited loss/trial summary does not fully describe the new broad objective. The numeric phase/proposal acceptance above is independent of that text. No widget source change or release is included in this optimization.

## Source and private evidence

Base `7029520bb8aff854b391585eb32908c221e1baa9`; local candidate `ssb-search-modes` with the ongoing broad API. This is a source-checkout study, not installed-wheel or release qualification. Dependency versions: optuna 4.9.0, numpy 2.4.6, scipy 1.18.0, cupy-cuda12x 14.2.0, numba 0.66.0.

Private experiment ID: `1022-exact-independent-tpe`. Complete source fingerprints, frozen arrays, timings and process manifests remain in the private archive. Sampler source SHA-256: `8631c9b01801025a82c6c4e69ba1c8600e3ce9f23144055d900fb70e4cd3f19a`.
