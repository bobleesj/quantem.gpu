# Coma, trefoil and spherical aberration in the model-based SSB fit (2026-10-21)

## Question

The joint fit (`find_aberrations()`) maximizes the least-squares agreement of the
thick-sample SSB model with G(q, k) over C10, C12, phi12, the sample tilt and a
depth spread. The model's probe phase can carry any Krivanek term. Can the same
objective also recover coma (C21), trefoil (C23) and spherical aberration (C30),
jointly with the tilt and depth spread, accurately and fast? Coma and a sample
tilt both move bright-field images, so the first question is whether they can be
told apart.

The probe phase of the first sideband must be evaluated at k − q for the odd
orders; every backend had it at q − k, which `2026-10-21-ssb-odd-aberration-sign`
fixes and verifies. With the old sign, the model's agreement at the true
coefficients was 38% below the corrected model's, and the fit found no coma.

## Change

`find_aberrations(higher_order=("C21", "C23", "C30"))` (any subset; MPS and
CUDA) runs the usual fit: search, both 180-degree branches, optional rotation
refinement. Then one trust-region Newton polish runs from that fit. It moves
C10, C12 (as a vector), the tilt, the depth spread and the higher orders (each as
a Cartesian vector, C30 as a signed number), starting with the higher orders
at 0 (`ssb/higher_order_fit.py`). The objective is the joint fit's. A per-term
kernel evaluates it, computing the probe phase from the Krivanek polynomial
(`ssb/mps/higher_order_agreement.py`, `ssb/cuda/higher_order_agreement.py`); the
joint fit's angle-sum kernel needs a quadratic χ. `tilt=False` holds the tilt
and depth spread at 0.

The agreement evaluations and reconstruction run on the selected GPU backend.
The optional higher-order polish uses NumPy on the CPU for its Newton steps,
finite differences and Hessian calculations. It is not a GPU-resident optimizer.
The result records `host_numerical_search=True`; its `higher_order` record
identifies `optimizer_backend="numpy"` and the `agreement_backend`.
This optional path does not change the default fit optimizer.

The result's `aberrations` carry `C21_nm`, `phi21_deg`, `C23_nm`, `phi23_deg` and
`C30_nm`, so `reconstruct(result)` applies them. `result.optimizer["higher_order"]`
records the agreement before and after, the evaluations, the time, and the
guard below: `trusted` and its numbers in `guard`.

| kernel check | MPS | CUDA |
| --- | --- | --- |
| higher orders 0, against the joint fit's kernel | 1.1e-6 relative | 1.9e-6 |
| C21, C23, C30, against a float64 full-plane model (k − q) | 2.6e-7 (q − k: 6–40% off) | 2.6e-7 (q − k: 5–38% off) |
| ms per evaluation, 3,900 BF pixels, batch of 78 | 0.11 | 0.31 |

The CUDA timing was taken on a shared card. CUDA's 128 × 128 FFT kernels have no
higher-order path; a preview with higher orders and no depth spread now
reconstructs there through the depth-aware reference path at zero depth, which
is the same model and estimator.

## Evidence

Multislice simulations of an SrTiO3-type perovskite [001] supercell (6 × 6
cells, 23.4 Å; Gaussian atomic potentials, one frozen-phonon configuration),
300 kV, 30 mrad, 64 × 64 scan at 0.366 Å, 0.84 mrad detector pixels. Truth:
C10 −5 nm (mid-depth), C12 2 nm at 30°, C21 50 nm at 60°, C23 40 nm at −20°,
C30 3 µm, tilt (3, −4) mrad. The potential was scaled to separate the model's
identifiability from its validity: ×0.02 or ×0.1 (weak phase, where the SSB
model is exact), ×0.25 (a light-element crystal) and ×1 (SrTiO3).

**Identifiability.** At the fitted maximum, the correlations from the inverse
Hessian of the agreement (11 parameters) are:

| | weak phase, 15.6 nm, tilted | light, 8 nm, tilted | SrTiO3, 1 nm |
| --- | --- | --- | --- |
| coma with tilt | ≤ 0.07 | ≤ 0.23 | ≤ 0.24 (tilt not determined at 1 nm) |
| coma with C10, C12 | ≤ 0.26 | ≤ 0.15 | ≤ 0.05 |
| trefoil with anything | ≤ 0.19 | ≤ 0.18 | ≤ 0.06 |
| C30 with C10 | −0.85 | −0.86 | −0.93 |

The tilt enters the model only through the real depth weights, coma through the
phase of the transfer function, so they separate. Direct controls on the weak
15.6 nm crystal: data with the tilt and no aberrations above C12 fitted to a
coma of 1.3 nm, and holding the tilt at 0 left the agreement 16% below its
maximum with coma 2.7 nm (coma does not absorb the tilt); data with coma and no
tilt gave a tilt of (0.0, 0.0) mrad with or without coma in the fit (the tilt
does not absorb coma). C30 trades against C10 (balanced spherical aberration)
but is still determined.

**Accuracy, noise-free** (fitted magnitude; angles within 2° unless noted):

| crystal | C21 (50 nm) | C23 (40 nm) | C30 (3 µm) | tilt (3, −4) |
| --- | --- | --- | --- | --- |
| weak phase, one slice | 49.9 | 40.0 | 2.98 | not determined |
| weak phase, 15.6 nm | 50.3 | 39.9 | 3.08 | (3.0, −4.0) |
| light, 2 nm | 49.1 | 40.1 | 2.74 | (2.6, −2.3) |
| light, 8 nm | 46.9 | 39.2 | 2.81 | (3.1, −3.8) |
| light, 15.6 nm | 44.8 (15° off) | 38.5 | 1.04 | (3.0, −3.9) |
| SrTiO3, 0.4 nm | 48.7 | 39.9 | 2.80 | not determined |
| SrTiO3, 1 nm | 46.5 | 39.4 | 2.31 | not determined |
| SrTiO3, 2 nm | 39.8 | 37.5 | 1.80 | not determined |
| SrTiO3, 4 nm | 23.0 (9° off) | 32.4 | 0.45 | (−0.4, −1.7) |
| SrTiO3, 15.6 nm | 64.8 (25° off) | 19.2 | −3.45 | (3.4, 0.6) |

Where the crystal scatters weakly the fitted terms are the probe's to a few
percent. In strongly scattering crystals they are not: an SrTiO3 crystal of
15.6 nm with no aberrations above C12 fitted to coma 55 nm and C30 −7 µm, while
the agreement rose 11%. The terms then fit dynamical scattering the model does
not contain. A coma of 100 nm, trefoil of 100 nm and C30 of 10 µm (light, 4 nm)
were recovered as 98 nm, 100 nm and 9.7 µm from a joint fit whose tilt was
8 mrad off; 10 nm, 10 nm and 1 µm as 9.8 nm, 10.2 nm and 0.95 µm.

Fitting the higher orders also corrects the low orders: with the truth above,
the joint fit alone was 0.6 to 1.4 nm off in C10 (C30 balanced by defocus) and
1.2 mrad off in tilt (light, 8 nm); with them, 0.2 nm and 0.2 mrad.

**Noise.** Poisson noise at 1e5, 1e4 and 1e3 e/Å² (three seeds each) on the
64 × 64 scans scattered the fitted coma by 0.1 to 2 nm (standard deviation) and
C30 by 0.01 to 0.2 µm; a 10 nm coma's azimuth, 5° off without noise, was
18 ± 7° off at 1e3 e/Å². For these crystals the bias above, not the dose,
limits the accuracy. On the weak-phase 15.6 nm crystal (little
signal) 1e4 e/Å² scattered the coma by 3.5 nm and 10°.

**Cost.** One polish: 150 to 1,400 evaluations in 4 to 21 batched passes. On the
64 × 64 scans (3,900 bright-field pixels) 0.04 to 0.4 s, on top of a joint fit
of 0.4 to 2 s. On another 512 × 512 acquisition (8,807 bright-field pixels) the
polish took 7.2 s (437 evaluations, 8 passes, 16 ms each) after a two-branch
joint fit of 10.3 s; with the rotation given the whole fit went from 7.3 s to
13.0 s (C21, C23, C30), 12.0 s (C21, C23) and 8.5 s (C30). On 256 × 256 regions
it added 1.7 to 2.4 s. M5 Max.

## CUDA and MPS give the same fit

The same 128 × 128 scans (the 64 × 64 simulations above, tiled; one period, so
neither backend pads) were fitted on both backends with `tilt=True`:

| crystal | backend | C21 | C23 | C30 | tilt (mrad) | polish |
| --- | --- | --- | --- | --- | --- | --- |
| weak phase, one slice | MPS | 50.08 nm @ 60.00° | 40.09 nm | 2911.8 nm | (0.33, 0.07) | 0.39 s |
| | CUDA | 50.08 nm @ 60.00° | 40.09 nm | 2911.5 nm | (0.33, 0.06) | 0.28 s |
| weak phase, 15.6 nm | MPS | 50.30 nm @ 61.68° | 39.93 nm | 3080.1 nm | (2.98, −4.03) | 0.20 s |
| | CUDA | 50.30 nm @ 61.68° | 39.93 nm | 3080.0 nm | (2.98, −4.03) | 0.20 s |
| light, 8 nm | MPS | 46.87 nm @ 58.14° | 39.23 nm | 2822.3 nm | (3.12, −3.80) | 0.22 s |
| | CUDA | 46.87 nm @ 58.13° | 39.23 nm | 2825.1 nm | (3.12, −3.80) | 0.08 s |

The agreements at the two maxima agree to 1e-7 relative. With `tilt=False`
(weak lattice) C21 was 49.97 nm @ 60.00° on both backends. On the measured 512 × 512
acquisition (8,807 bright-field pixels, both 180-degree branches) the two
backends returned the same terms to 0.01 nm and 0.01° (C21 104.45 against
104.46 nm, C30 5276.6 nm) and the same tilt and depth spread to 1e-5. The CUDA
polish took 4.9 s (437 evaluations in 8 passes, as on MPS) and the whole fit 12.1 s, against
7.2 s without the higher orders; on MPS, 7.1–7.4 s, 17.7–20.9 s and 11.3 s.
CUDA ran on an RTX PRO 6000 Blackwell card that other jobs kept at 99%
utilisation; MPS ran on an M5 Max.

## When to trust the terms

A fitted term is the probe's only where the SSB model describes the data. Five
guards were tested on 280 fits of new 64 × 64 multislice simulations, each
noise-free and at 1e5, 1e4 and 1e3 e/Å² with three Poisson draws:

- **Thick nulls:** 12 crystals with no aberration above C12 in the truth.
  SrTiO3-type at 2, 4 (tilted and not), 8 (two settings), 15.6 (tilted and
  not) and 30 nm; the potential ×1.5 at 4 and 8 nm; ×0.25 at 15.6 and 30 nm.
- **Weak-phase cases:** 9 crystals, potential ×0.1 or ×0.02, 0.2–30 nm. They
  carry coma, trefoil and C30 of 50/40 nm and 3 µm, of 10/10 nm and 1 µm, of
  25/20 nm and −2 µm, coma alone (30 nm), and none.
- **Thin series:** the light-element and SrTiO3 crystals of the earlier study
  (2–8 nm light, 0.4–1 nm SrTiO3).

A fit counts as accurate when coma and trefoil are within max(5 nm, 15%)
(vector error) and C30 within max(0.5 µm, 25%) of the truth. Without a guard,
100 of the 120 null fits are wrong: the terms absorb dynamical scattering (coma
up to 200 nm, C30 to −15 µm). So are 30 of the 90 weak-phase fits, all at
1e3–1e4 e/Å² on the weakest specimens, where noise sets the error.

| guard | thick nulls passed | weak phase passed (wrong among them) | accurate weak passed |
| --- | --- | --- | --- |
| none | 120 / 120 | 90 / 90 (30) | 60 / 60 |
| agreement gain over the fit without higher orders ≥ 1% | 51 / 120 | 59 / 90 (24) | 35 / 60 |
| depth spread inside the depth model's validity range | 118 / 120 | 90 / 90 (30) | 60 / 60 |
| scatter over the three noise draws ≤ 2 nm (coma) | 84 / 120 | 36 / 90 (0) | 36 / 60 |
| coma and trefoil agree within 5 nm between two q bands | 13 / 120 | 22 / 90 (0) | 22 / 60 |
| coherence ≥ 0.9 | 0 / 120 | 58 / 90 (7) | 51 / 60 |
| **coherence ≥ 0.9 and edge-phase error ≤ 0.1 rad** | **0 / 120** | **45 / 90 (0)** | **45 / 60** |

The depth model's validity range is 50 nm for Si, 30 nm for SrTiO3 and 20 nm for
BaTiO3, taken as ×0.25, ×1 and ×1.5 here. Neither the agreement gain nor that
range separates the nulls: a thick null gains as much from the extra terms as a
small true coma does, and the higher orders fail far inside the depth model's
range (SrTiO3 from 4 nm). Scatter over noise draws and the two-band split do not
see a bias that every draw shares.

**Coherence** is the share of G's signal power over the fit band that the fitted
model explains:

ρ = (A − F_A) / (P − F_P), with P = Σ_q w_q Σ_k |G(q, k)|².

A is the agreement and A ≤ P. F_A and F_P are their noise floors; for counted
electrons the noise of G(q, k) has variance G(0, k). A weak-phase object gives
ρ ≈ 1, because its G is the model's. Dynamical scattering puts power where the
model cannot follow, and the extra terms then fit part of it. Noise-free,
the weak-phase crystals reached 0.96–1.00 and the thin series 0.90–0.99. No
null whose fit was wrong exceeded 0.80: light 15.6 nm reached 0.80, SrTiO3 4 nm
0.77, 8 nm 0.58 and 15.6 nm 0.25. The 2 nm SrTiO3 null, whose fit returned no
spurious terms, reached 0.87.

**Precision.** The covariance of the fitted coordinates is 2σ²(−∇²A)⁻¹, taken
from the polish's own Hessian. A / (2σ²) is the profile log-likelihood for
Gaussian noise of variance σ² on G over the independent half of the q plane.
The guard asks that each term's standard error change the probe phase at the
aperture edge by ≤ 0.1 rad: (2π/λ) σ_C α^(n+1)/(n+1) ≤ 0.1, which at 30 mrad
is 3.5 nm for coma and trefoil and 0.15 µm for C30. Across 79 (crystal, dose)
groups, the coma's scatter over three draws divided by this standard error had
a median of 1.00 (quartiles 0.74–1.38). The very noisy weak cases are the
exception: their fits leave the quadratic region, the scatter exceeds the
prediction, and the guard rejects them anyway.

With both conditions, no wrong fit of the 280 passed. Weak-phase fits passed
9/9 noise-free, 21/24 accurate ones at 1e5 e/Å², 15/19 at 1e4 and 0/8 at 1e3.
The thin series passed 51/70, all accurate. The guard is conservative: the 20
accurate null fits (SrTiO3 2 nm, and 4 nm untilted) fail it at ρ = 0.74–0.87.

On the measured 512 × 512 acquisition (a thick crystalline film) the polish is
statistically precise (coma ±0.07 nm), but the model explains 0.27 of the
signal, so the guard rejects it. That fits the earlier finding that the coma
there varies by ±50 nm between quadrants that share one probe.

`find_aberrations(higher_order=...)` records `coherence`, `standard_error_nm`,
`edge_phase_error_rad` and `trusted` in `result.optimizer["higher_order"]["guard"]`.
An untrusted fit prints one line saying why and keeps the terms.
Input that is not integer counts gets no noise floor (F = 0), which lowers the
coherence of noisy data, and no precision estimate.

## Conclusion

Coma and trefoil are identifiable and separable from the sample tilt; C30 is
identifiable but strongly correlated with C10. The fit is accurate where the
SSB model holds (thin or weakly scattering specimens) and returns non-probe
terms in thick, strongly scattering crystals. It is therefore an option for
calibrating the probe on suitable specimens, not a default. The coherence and
precision guard tells the two apart without the truth: it passed no wrong fit
in these simulations, and it passes most weak-phase fits at 1e4–1e5 e/Å².
