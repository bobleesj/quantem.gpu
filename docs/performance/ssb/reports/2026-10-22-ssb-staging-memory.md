# Lower CUDA SSB staging memory without changing numerical results

> **Historical:** describes the code as of 2026-10-22; the Nelder–Mead refinement it checks is retired. See the [current fit methods](../../../api/ssb.md).

## Change

Allocate corrected per-BF image staging only when a consumer needs it. Coherent
reconstruction does not use this array. Preserve an existing allocation across
rotation changes so phase-mean/loss previews reuse it. Reduce Fourier-preparation
staging from a 2 GiB target to 256 MiB, retaining the same specialized FFT kernels.

No detector pixels, scan positions, precision, optics terms, search trials or
accumulation order changed. Decoded counts remain available for regional fits
and export. The stored Fourier half-plane was already exploiting exact Hermitian
symmetry; this patch does not change its representation.

## Matched measurement

Physical RTX PRO 6000 Blackwell Workstation Edition. Three alternating quiet
fresh-process pairs per row. Filesystem and compiler disk caches were not
evicted; these are not cold-disk timings. Two additional contended pairs are
retained for numerical comparison only. Three observations provide descriptive
p95, not a tail-latency guarantee.

Real acquisitions: A/B use the complete 192² detector; C uses the complete 12²
detector. The 256² cases explicitly select that native scan region from a 512²
acquisition. BF counts remain 8,807 / 8,930 / 101. Fixed recorded rotation,
200 primary trials plus 400 broad fifth-order trials, then Newton with the
existing 5,000-evaluation ceiling and normal early stops. Auto output is 3× / 4× /
1× respectively; fitting stays on the native grid. CUDA evaluates the objective
and reconstruction; Optuna and the Newton step calculation remain CPU work.

Workflow includes loading/upload, Fourier preparation, fitting, Auto reconstruction,
host handoff and saving arrays. Imports and final cleanup are excluded. Retained
memory below is live CuPy allocation after output handoff, not whole-process peak.

| Acquisition | Native scan | Live GPU memory before → after (GiB) | Before p50 / p95 / max (s) | After p50 / p95 / max (s) |
|---|---:|---:|---:|---:|
| C | 512² | 0.460 → 0.263 | 4.447 / 4.815 / 4.855 | 4.484 / 4.608 / 4.622 |
| A | 256² | 8.100 → 3.800 | 6.586 / 6.647 / 6.654 | 6.504 / 6.529 / 6.532 |
| A | 512² | 15.165 → 15.165 | 12.996 / 13.287 / 13.319 | 13.003 / 13.119 / 13.132 |
| B | 256² | 8.308 → 3.948 | 8.525 / 8.539 / 8.541 | 8.295 / 8.504 / 8.527 |
| B | 512² | 15.757 → 15.757 | 20.079 / 20.226 / 20.242 | 20.052 / 20.236 / 20.257 |

There is no consistent workflow slowdown in these samples. A/B at 256² retain
4.30/4.36 GiB less GPU memory; C at 512² retains 202 MiB less. A/B at 512² do not
retain less after handoff because their old code already avoided the full staging
allocation. Their FFT scratch falls by 1.75 GiB, but another stage still determines
the overall peak. Do not report this scratch reduction as an end-to-end peak reduction.

Sampled whole-device peaks include unrelated resident processes and the additional
ordinary parity reconstruction. The stable quiet observations were approximately
16.26 → 11.96 GiB (A, 256²), 16.87 → 12.51 GiB (B, 256²), 5.01 → 4.82 GiB (C,
512²), and unchanged at 26.13/26.66 GiB for A/B at 512². These are whole-device
observations, not isolated process memory requirements.

## Numerical and workflow checks

- All 17 paired broad workflows reproduce complete native/Auto phase arrays,
  ordinary zero-tilt/depth output, all proposals, coarse objective scores,
  numeric Newton trajectory and final coordinates exactly. Elapsed-time columns
  are excluded from equality checks.
- Three direct source-import checks reproduce the frozen real-data results with
  no baseline/candidate method replacement. A separate Nelder–Mead check preserves
  the complete phase and trajectory with its unchanged 3,000-evaluation cap.
- Synthetic 128²/256²/512²/1024² cases preserve every stored Fourier value and
  complete ordinary, tilt/depth and higher-order phase/object arrays, plus native
  phase-mean/loss output. These use an 8² detector and 60 BF pixels. This is not
  real full-detector 1024² acceptance or a 1024² performance claim.
- The shipped benchmark passes on A/B at full 512², including both estimated
  rotation branches, 256²/128² regions, region movement and 127×128 resizing.
  Twelve operations with two repeats each preserve complete phase hashes,
  fit parameters, scores and optimizer records against the frozen buffer code.
  It retains 200 trials per branch plus Newton and full-field Auto output.
- Nineteen focused hardware checks, nine explicit-standard workflow checks,
  and 23 search-contract/privacy checks pass. They cover close/release,
  rotation changes, upsampling, higher order,
  device residency, region selection, saved geometry, search-mode contracts and
  privacy. Four existing legacy workflow tests fail under the unrelated new
  broad default on both the frozen and changed buffer code; their intended
  explicit-standard versions pass. This patch does not alter that API contract.

No fresh browser frame-rate or MPS performance claim is made. This is a
source-checkout qualification, not an installed-wheel release qualification.

## Source and evidence

Base: `7029520bb8aff854b391585eb32908c221e1baa9`, with the same ongoing broad-search
candidate on both sides. Private experiment ID: `1022-02-memory`. Full manifests,
frozen source/arrays, raw timings and logs remain private. The shared-FFT
compiler-cache experiment remains held and was not integrated.

Frozen `engine.py` SHA-256: `941c51d9b4eb23445c64d010470dd944e3bc16c7f873dbf635d88962462f167e`.
Changed `engine.py` SHA-256: `8863d06bc80d17ec0a445ef06f1e2b29e31c0c8fac37a169ff5fe9ee86b15cf3`.
Frozen `brightfield.py` SHA-256: `140accdbf145ba8dc181e90de6b4c0e0f9c3d0256750afc285cb401c4aa82904`.
Changed `brightfield.py` SHA-256: `5a4b498bc907a5f1e90d2c91079051cb0467918056ca1905c898593f99cdaa2e`.
