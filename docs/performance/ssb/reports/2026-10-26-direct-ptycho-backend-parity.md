# Direct Ptycho backend parity: CUDA, Python MPS and native Metal, 2026-10-26

## Question

CUDA, Python MPS and native Swift/Metal implement one Direct Ptycho model. Do they fit the same
aberrations and reconstruct the same images on measured data, is each fit repeatable, and how long does
the low-order fit take on each backend after the exact-derivative searches landed?

## Method

- **Fixtures**: five measured 512 × 512 scans at 300 kV and 30 mrad: MoS₂ bilayer A, MoS₂ bilayer B, MoS₂
  bilayer C, a MoS₂ monolayer and a thick oxide crystal.
- **Fits**: 180 in all: 3 backends × 5 fixtures × 2 higher-order boxes (±2 and ±20) × 6 runs (seed 42, seed
  42 repeated in a fresh process, seeds 1–4). Each run is the low-order fit followed by the fifth-order fit.
  The rotation is fixed per fixture, so every backend fits the same branch.
- **Code**: quantem.gpu d3eeffa7. CUDA on an NVIDIA RTX PRO 6000 Blackwell Max-Q; native Metal and Python MPS
  on an Apple M5 Max.
- **σ**: the full-data uncertainty of each fitted parameter, estimated from four binomial count splits in
  the [25 October study](2026-10-25-ssb-exact-derivative-fit.md); the estimate itself is good to about
  ±35 %. Differences below are given in physical units and as |difference| / σ.
- **Images**: every seed-42 fifth-order endpoint is reconstructed by one reconstructor (CUDA DirectPtycho
  at d3eeffa7), so an image difference can come only from the fitted parameters. Upsampling is Auto (3× for
  bilayer A, bilayer B and the oxide, 4× for bilayer C and the monolayer), full field, no crop or bin.

All 180 fits converged; none stopped at an evaluation cap.

## Fit parity

Seed 42, fifth-order fit. Each cell is the difference, then |difference| / σ. Δ agreement is the difference
in the fraction of band power the fit explains.

### Box ±2

| Fixture | Pair | C10 (nm) | C12 (nm) | φ12 (°) | Tilt r (mrad) | Tilt c (mrad) | Depth (nm) | Δ agreement |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Bilayer A | native−CUDA | +2.1e-03 (0.4σ) | -2.7e-04 (0.2σ) | -2.0e-04 (0.0σ) | -4.6e-04 (0.1σ) | +6.3e-04 (0.1σ) | +4.7e-04 (0.1σ) | +7.5e-10 |
| Bilayer A | MPS−CUDA | +1.5e-03 (0.3σ) | -3.5e-04 (0.2σ) | -4.7e-05 (0.0σ) | -5.1e-04 (0.1σ) | +1.7e-04 (0.0σ) | +4.6e-04 (0.1σ) | +7.5e-10 |
| Bilayer A | native−MPS | +6.4e-04 (0.1σ) | +8.0e-05 (0.0σ) | -1.6e-04 (0.0σ) | +5.0e-05 (0.0σ) | +4.6e-04 (0.1σ) | +1.5e-05 (0.0σ) | +0.0e+00 |
| Bilayer B | native−CUDA | -8.6e-04 (0.2σ) | -1.1e-03 (0.1σ) | -1.9e-04 (0.0σ) | +6.1e-04 (0.1σ) | -2.6e-04 (0.0σ) | -8.2e-04 (0.2σ) | -5.1e-09 |
| Bilayer B | MPS−CUDA | -5.6e-04 (0.1σ) | -1.0e-03 (0.1σ) | +2.0e-04 (0.0σ) | +3.9e-04 (0.0σ) | +5.5e-05 (0.0σ) | -2.4e-04 (0.1σ) | -5.1e-09 |
| Bilayer B | native−MPS | -2.9e-04 (0.1σ) | -4.8e-05 (0.0σ) | -3.9e-04 (0.1σ) | +2.2e-04 (0.0σ) | -3.1e-04 (0.0σ) | -5.9e-04 (0.1σ) | +0.0e+00 |
| Bilayer C | native−CUDA | -5.6e-05 (0.0σ) | -1.8e-04 (0.1σ) | +2.2e-04 (0.1σ) | +2.9e-04 (0.1σ) | -6.2e-05 (0.0σ) | -3.7e-04 (0.1σ) | +4.3e-09 |
| Bilayer C | MPS−CUDA | -2.6e-04 (0.1σ) | +2.0e-04 (0.1σ) | +4.6e-04 (0.1σ) | +3.6e-05 (0.0σ) | +1.1e-04 (0.0σ) | +1.3e-05 (0.0σ) | +5.3e-10 |
| Bilayer C | native−MPS | +2.1e-04 (0.1σ) | -3.8e-04 (0.3σ) | -2.4e-04 (0.1σ) | +2.5e-04 (0.0σ) | -1.7e-04 (0.0σ) | -3.9e-04 (0.1σ) | +3.8e-09 |
| Monolayer | native−CUDA | +3.2e-04 (0.1σ) | -3.2e-04 (0.1σ) | +1.5e-03 (0.3σ) | -1.7e-03 (0.3σ) | +1.6e-03 (0.2σ) | -1.5e-03 (0.3σ) | +1.6e-09 |
| Monolayer | MPS−CUDA | +1.5e-05 (0.0σ) | -6.6e-04 (0.3σ) | +8.2e-04 (0.1σ) | -2.6e-04 (0.0σ) | +2.0e-03 (0.3σ) | -7.9e-04 (0.1σ) | +6.5e-10 |
| Monolayer | native−MPS | +3.1e-04 (0.1σ) | +3.4e-04 (0.1σ) | +7.2e-04 (0.1σ) | -1.4e-03 (0.2σ) | -4.3e-04 (0.1σ) | -6.9e-04 (0.1σ) | +9.7e-10 |
| Oxide | native−CUDA | +1.9e-05 (0.0σ) | -1.1e-04 (0.3σ) | +2.6e-03 (0.6σ) | +2.0e-04 (0.1σ) | -3.5e-04 (0.2σ) | -3.8e-04 (0.2σ) | +5.7e-08 |
| Oxide | MPS−CUDA | +1.1e-04 (0.1σ) | -6.2e-05 (0.2σ) | -3.7e-04 (0.1σ) | -3.9e-05 (0.0σ) | +6.4e-05 (0.0σ) | -4.6e-04 (0.2σ) | +5.7e-08 |
| Oxide | native−MPS | -9.6e-05 (0.1σ) | -5.2e-05 (0.2σ) | +2.9e-03 (0.7σ) | +2.4e-04 (0.1σ) | -4.1e-04 (0.3σ) | +7.7e-05 (0.0σ) | +0.0e+00 |

### Box ±20

| Fixture | Pair | C10 (nm) | C12 (nm) | φ12 (°) | Tilt r (mrad) | Tilt c (mrad) | Depth (nm) | Δ agreement |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Bilayer A | native−CUDA | -1.7e-03 (0.5σ) | -6.4e-04 (0.4σ) | -8.9e-04 (0.1σ) | +4.5e-06 (0.0σ) | +5.5e-04 (0.1σ) | +1.4e-04 (0.0σ) | +7.1e-09 |
| Bilayer A | MPS−CUDA | +2.5e-04 (0.1σ) | +7.5e-04 (0.4σ) | +7.1e-04 (0.1σ) | -5.3e-04 (0.2σ) | +3.7e-04 (0.1σ) | +2.0e-04 (0.1σ) | +4.1e-09 |
| Bilayer A | native−MPS | -1.9e-03 (0.5σ) | -1.4e-03 (0.8σ) | -1.6e-03 (0.2σ) | +5.3e-04 (0.2σ) | +1.8e-04 (0.0σ) | -5.9e-05 (0.0σ) | +3.0e-09 |
| Bilayer B | native−CUDA | -1.8e-03 (0.5σ) | -1.8e-03 (0.7σ) | -6.9e-03 (1.1σ) | +3.9e-04 (0.2σ) | -1.2e-04 (0.0σ) | -1.8e-04 (0.1σ) | +7.1e-09 |
| Bilayer B | MPS−CUDA | -1.1e-03 (0.3σ) | -1.2e-03 (0.4σ) | -7.5e-03 (1.2σ) | +2.6e-04 (0.1σ) | -2.4e-04 (0.0σ) | -2.3e-04 (0.1σ) | +1.0e-08 |
| Bilayer B | native−MPS | -7.9e-04 (0.2σ) | -6.3e-04 (0.2σ) | +6.3e-04 (0.1σ) | +1.3e-04 (0.1σ) | +1.2e-04 (0.0σ) | +5.2e-05 (0.0σ) | -3.0e-09 |
| Bilayer C | native−CUDA | -8.9e-04 (0.3σ) | -5.6e-04 (0.3σ) | -1.4e-03 (0.3σ) | +3.3e-04 (0.1σ) | +1.6e-03 (0.3σ) | -3.3e-04 (0.1σ) | +1.2e-08 |
| Bilayer C | MPS−CUDA | -1.4e-03 (0.4σ) | +1.7e-04 (0.1σ) | -3.2e-03 (0.7σ) | -7.0e-04 (0.1σ) | +1.4e-03 (0.3σ) | +2.4e-04 (0.1σ) | +2.6e-09 |
| Bilayer C | native−MPS | +4.7e-04 (0.2σ) | -7.3e-04 (0.4σ) | +1.9e-03 (0.4σ) | +1.0e-03 (0.2σ) | +2.9e-04 (0.1σ) | -5.7e-04 (0.2σ) | +9.5e-09 |
| Monolayer | native−CUDA | -2.4e-04 (0.0σ) | -5.7e-05 (0.0σ) | +3.7e-04 (0.0σ) | +1.6e-04 (0.0σ) | -6.3e-04 (0.1σ) | -9.5e-05 (0.0σ) | +2.1e-09 |
| Monolayer | MPS−CUDA | -3.8e-05 (0.0σ) | +3.8e-04 (0.2σ) | +1.4e-03 (0.1σ) | +1.1e-05 (0.0σ) | -4.4e-04 (0.1σ) | +1.4e-05 (0.0σ) | +1.2e-09 |
| Monolayer | native−MPS | -2.1e-04 (0.0σ) | -4.3e-04 (0.2σ) | -1.0e-03 (0.1σ) | +1.5e-04 (0.0σ) | -1.9e-04 (0.0σ) | -1.1e-04 (0.0σ) | +9.7e-10 |
| Oxide | native−CUDA | -1.4e-03 (1.4σ) | -1.6e-04 (0.2σ) | +2.5e-03 (0.3σ) | +3.6e-05 (0.0σ) | -8.5e-05 (0.1σ) | +3.8e-05 (0.0σ) | +4.1e-08 |
| Oxide | MPS−CUDA | -1.2e-03 (1.2σ) | +1.2e-04 (0.1σ) | +3.7e-03 (0.4σ) | +1.4e-04 (0.1σ) | +6.3e-05 (0.1σ) | -3.4e-04 (0.1σ) | +2.5e-08 |
| Oxide | native−MPS | -2.8e-04 (0.3σ) | -2.8e-04 (0.3σ) | -1.2e-03 (0.1σ) | -1.1e-04 (0.1σ) | -1.5e-04 (0.2σ) | +3.8e-04 (0.1σ) | +1.6e-08 |

The largest difference is 0.7σ at ±2 (oxide φ12, native − MPS) and 1.4σ at ±20 (oxide C10, native − CUDA,
1.4e-3 nm). Typical differences are ≤ 0.3σ at ±2 and ≤ 0.5σ at ±20. The largest C10 difference is
2.1e-3 nm. Agreement differences are 1e-9 to 6e-8. The number of coordinates at a box bound is identical
across backends on every fixture (±2: 9/7/6/8/6 over the five fixtures; ±20: none).

At ±20 the half-band check passes on the bilayers and the monolayer (0.16–0.35 rad) and fails on the oxide
(1.33 rad), which therefore shows its ±2 fit, on every backend.

## Repeatability

Seed 42 and its fresh-process repeat are bitwise identical on all backends, fixtures and boxes. Spread over
seeds 42 and 1–4 within one backend (max − min, worst parameter):

### Box ±2

| Fixture | Native | CUDA | MPS | Repeat bitwise (native/CUDA/MPS) |
|---|---|---|---|---|
| Bilayer A | 0.8σ (C10) | 1.3σ (C10) | 0.2σ (tilt r) | yes/yes/yes |
| Bilayer B | 0.4σ (depth) | 0.9σ (φ12) | 0.2σ (depth) | yes/yes/yes |
| Bilayer C | 0.4σ (C12) | 1.0σ (φ12) | 0.4σ (C12) | yes/yes/yes |
| Monolayer | 0.3σ (tilt c) | 0.5σ (C12) | 0.3σ (tilt r) | yes/yes/yes |
| Oxide | 1.0σ (φ12) | 0.4σ (depth) | 0.4σ (φ12) | yes/yes/yes |

### Box ±20

| Fixture | Native | CUDA | MPS | Repeat bitwise (native/CUDA/MPS) |
|---|---|---|---|---|
| Bilayer A | 0.7σ (C12) | 1.3σ (C12) | 0.4σ (C12) | yes/yes/yes |
| Bilayer B | 0.5σ (φ12) | 1.7σ (φ12) | 0.5σ (φ12) | yes/yes/yes |
| Bilayer C | 1.7σ (φ12) | 0.7σ (C10) | 0.6σ (φ12) | yes/yes/yes |
| Monolayer | 0.5σ (C10) | 0.5σ (C12) | 0.8σ (C12) | yes/yes/yes |
| Oxide | 0.7σ (C12) | 1.9σ (C10) | 1.3σ (tilt c) | yes/yes/yes |

Seed spread within a backend (≤ 1.3σ at ±2, ≤ 1.9σ at ±20) is the same size as the gaps between
backends: the backends differ by no more than a backend differs from itself under another seed.

## Image parity

Phase images after removing each image's mean; % of std is rms / std of the second image. Context
columns: CUDA's own seed 1 − seed 42 difference, and the CUDA endpoint with its parameters rounded to
float32. The same point reconstructed twice is bitwise equal.

| Fixture | Box | Method | Upsampling | Largest backend-pair rms (rad) | % of std | Lowest corr | CUDA seed 1 − seed 42 rms (rad) | CUDA z → float32 rms (rad) |
|---|---|---|---|---|---|---|---|---|
| Bilayer A | ±2 | SSB | 3× | 1.03e-04 (native−CUDA) | 0.238 | 0.9999972 | 6.83e-05 | 6.32e-07 |
| Bilayer A | ±2 | Phase | 3× | 6.78e-06 (MPS−CUDA) | 0.012 | 1.0000000 | 5.95e-06 | 1.14e-08 |
| Bilayer A | ±20 | SSB | 3× | 7.88e-05 (native−MPS) | 0.150 | 0.9999989 | 7.06e-05 | 1.72e-06 |
| Bilayer A | ±20 | Phase | 3× | 1.37e-05 (native−MPS) | 0.021 | 1.0000000 | 1.68e-05 | 1.50e-08 |
| Bilayer B | ±2 | SSB | 3× | 9.95e-05 (native−CUDA) | 0.217 | 0.9999977 | 1.06e-04 | 2.51e-07 |
| Bilayer B | ±2 | Phase | 3× | 2.33e-05 (MPS−CUDA) | 0.041 | 0.9999999 | 2.46e-05 | 9.17e-09 |
| Bilayer B | ±20 | SSB | 3× | 6.10e-05 (native−CUDA) | 0.111 | 0.9999994 | 8.38e-05 | 4.55e-06 |
| Bilayer B | ±20 | Phase | 3× | 1.05e-05 (MPS−CUDA) | 0.016 | 1.0000000 | 8.21e-06 | 3.39e-08 |
| Bilayer C | ±2 | SSB | 4× | 9.66e-05 (native−MPS) | 0.196 | 0.9999981 | 2.51e-04 | 1.90e-06 |
| Bilayer C | ±2 | Phase | 4× | 9.87e-06 (native−MPS) | 0.012 | 1.0000000 | 1.29e-05 | 2.02e-08 |
| Bilayer C | ±20 | SSB | 4× | 2.20e-04 (native−CUDA) | 0.392 | 0.9999923 | 2.14e-04 | 3.49e-06 |
| Bilayer C | ±20 | Phase | 4× | 3.48e-05 (native−CUDA) | 0.043 | 0.9999999 | 2.32e-05 | 3.86e-08 |
| Monolayer | ±2 | SSB | 4× | 2.10e-04 (MPS−CUDA) | 0.678 | 0.9999770 | 2.01e-04 | 6.48e-06 |
| Monolayer | ±2 | Phase | 4× | 1.28e-05 (native−CUDA) | 0.029 | 1.0000000 | 1.04e-05 | 1.17e-08 |
| Monolayer | ±20 | SSB | 4× | 1.27e-04 (native−CUDA) | 0.375 | 0.9999930 | 1.23e-04 | 6.05e-06 |
| Monolayer | ±20 | Phase | 4× | 1.44e-05 (native−CUDA) | 0.031 | 1.0000000 | 1.01e-05 | 2.42e-08 |
| Oxide | ±2 | SSB | 3× | 1.02e-04 (native−MPS) | 0.082 | 0.9999997 | 1.05e-05 | 1.02e-06 |
| Oxide | ±2 | Phase | 3× | 2.73e-05 (native−MPS) | 0.013 | 1.0000000 | 9.38e-08 | 3.07e-08 |
| Oxide | ±20 | SSB | 3× | 8.28e-05 (native−MPS) | 0.066 | 0.9999998 | 1.12e-04 | 6.40e-06 |
| Oxide | ±20 | Phase | 3× | 1.66e-05 (MPS−CUDA) | 0.008 | 1.0000000 | 2.60e-05 | 4.11e-08 |

- **SSB**: pairwise rms 4.7e-5 to 2.2e-4 rad, 0.04–0.68 % of the phase std; correlation ≥ 0.99998; median
  lattice SNR over the 40 strongest reflections within 0.3 %.
- **Phase**: ≤ 0.043 % of std; correlation ≥ 0.9999999; lattice SNR identical to three decimals.
- The differences match CUDA's own seed 1 − seed 42 spread (on the oxide at ±2, CUDA's two seeds land
  closer together than the backends do) and sit 5–30× above the reconstructor's float32 floor for SSB
  (5.4e-6 to 8.7e-6 rad against a float64 oracle, 25 October). They are fit spread, not reconstruction
  noise and not a backend effect: visible only at 200–550× amplification, with no lattice structure.

The earlier image acceptance (25 October) still stands: native vs CUDA Phase and Complex 3.6e-8 to
1.7e-7 rad rms and SSB 6.1–6.7e-6 rad at the same point; MPS vs CUDA and the float64 oracle 456 of 456 within
2× CUDA's float32 error (2e-8 to 1.5e-7 rad).

## Speed

Low-order fit, warm, MoS₂ bilayer A:

| CUDA (RTX PRO 6000 Blackwell Max-Q) | Native Metal (M5 Max) | Python MPS before, d3eeffa7 (M5 Max) | Python MPS now, e82c22ff (M5 Max) |
|---:|---:|---:|---:|
| 0.57–0.73 s | 2.2–2.4 s | 6.9 s | 2.5 s |

Per stage on bilayer A at e82c22ff:

| Stage | Python MPS | Native Metal |
|---|---:|---:|
| Standard comparison | 0.84 s | 0.70 s |
| Joint TPE (42 ms per batch on both) | 1.13 s | 1.0–1.07 s |
| Newton | 0.50 s | 0.47–0.50 s |
| Preparation (not part of the fit time) | 1.84 s | 0.9–1.1 s |

The search is the resident multivariate TPE: 200 joint and 200 standard-comparison trials in groups of 8
(25 + 25 rounds). On a busy GPU the MPS / native ratio over all five fixtures is about 1.0–1.05. The
speed-up did not move the MPS endpoints: C10 within 1.5e-6 nm of before, with new derivative tests against
central differences.

CUDA wall time per fit across the five fixtures: low order 0.5–1.4 s, fifth order 0.15–0.5 s at ±2 and
0.7–1.8 s at ±20 (including the half-band check). The 60 CUDA runs take about 2 min.

## What changed since 25 October

In the [25 October study](2026-10-25-ssb-exact-derivative-fit.md) the CUDA path still used the
finite-difference Newton, and at ±20 the oxide differed between backends by 35–264σ (C10 0.26 nm). With
the finite-difference Newton retired on every backend, that gap is gone: the oxide now differs by at most
1.4σ (1.4e-3 nm in C10).

## Not measured / open

- Native Metal does not run the standard comparison for the fifth-order fit by default (Python does), so
  native fifth-order fits show no tilt gain. Not yet decided.
- Python MPS preparation takes 1.8 s against about 1 s native; a single-threaded uint8 to float32 column
  copy costs 0.7 s.
- The float64 polish was not rerun at this revision (25 October: both starts polish to one optimum,
  z ≤ 7e-8).
- The capped-4× Auto upsampling notice has not been checked in a headed native app; the widget readout is
  covered by unit tests only.
