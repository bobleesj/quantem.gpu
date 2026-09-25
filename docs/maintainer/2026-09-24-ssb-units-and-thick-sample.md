# SSB aberration units (nm) and the thick-sample (tilt + thickness) model - 2026-09-24

## 1. C10 / C12 were Angstrom under an nm label

**Question.** The API, docs and ShowPtycho label SSB C10 / C12 in nm. Are they?

**Setup.** abTEM 4D-STEM of a thin crystal (one BaTiO3 unit cell, 4 A, so thickness cannot move the apparent focus), 300 kV,
30 mrad, 64 x 64 scan at 0.25 A, probe defocus set explicitly (abTEM C10 = -defocus). `SSB.fit(trials=200,
refinement="nelder-mead")`. Script: study folder `c10_unit_test.py` (same setup as `tests/hardware/cuda/test_ssb_units.py`).

| abTEM defocus | true C10 | fit before | fit after |
|---|---|---|---|
| +100 A | -10.0 nm | -100.26 "nm" | -10.03 nm |
| -150 A | +15.0 nm | +150.25 "nm" | +15.03 nm |

**Cause.** Every backend evaluates chi = (pi / lambda[A]) alpha^2 (C10 + C12 cos 2(phi - phi12)) (CUDA `engine._factor`,
MPS `engine.py` factor, WebGPU `makeParams` f[12], Swift `MetalSSBKernels`) with no conversion, so the numbers are Angstrom.
The 14-coefficient magnitudes follow the same convention. Sign convention matches abTEM.

**Fix.** The public `SSB` session (`ssb/workflow.py`) converts at its boundary: nm in -> x10 to the engine; results,
trial histories and `fit_sample` outputs /10 back to nm (`_ENGINE_PER_NM`, `ABERRATION_UNIT = "nm"`). Kernels, backend-level
parity tests and fixtures keep Angstrom unchanged. `_persistence.SCHEMA = 2` (schema-1 saved results are not reused). Live
fit records written by `_write_series_fit_metadata` carry `"aberration_unit": "nm"`; records without it are read /10.
ShowPtycho (quantem.widget) and quantem.live normalise the same way; the browser WebGPU engine is called through an nm->A
helper in ShowPtycho. **Not fixed:** the native Swift app (`c10Nanometers` is Angstrom); quantem.live reads its artifacts /10.

**Consequence found.** The quantem.live dashboard multiplied SSB "nm" by 10 to seed ptychography defocus
(`LaunchPtychoDialog.tsx:66`, `PtychoLauncherDialog.tsx:262`, `trial_launch.py:109`), so SSB-seeded launches used 10x the
defocus. With true nm the x10 is correct.

Test: `tests/hardware/cuda/test_ssb_units.py`.

## 2. Thick-sample SSB: sample tilt and thickness

**Model.** Slice at depth z (from mid-depth) sees defocus C10 + z and sits shifted by z theta. Averaging each SSB term over
the depth gives real weights: gamma = w1 t1 - w2 t2, t1 = P(q-k) conj P(k), t2 = conj P(q+k) P(k), w = sinc(rate t / 2),
rate1 = -factor(alpha_m^2 - alpha_k^2) - 2 pi q.theta, rate2 = +factor(alpha_p^2 - alpha_k^2) - 2 pi q.theta. Thickness 0 ->
weights 1 -> standard SSB exactly.

**Objective.** The phase-variance loss used by `fit` does not see tilt (sim tilted (3,-4) -> (5.5, 0.4); untilted ->
(-2.7, -4.3)): it applies a phase-only correction. The least-squares fit of G = Psi gamma, sum over q in 0.2-0.9 /A of
|sum_k G conj(gamma)|^2 / sum_k |gamma|^2 (`SSBEngine.thick_fit`), does. Search: multivariate TPE over C10, C12, phi12, tilt,
thickness jointly (C10 must move to the mid-depth focus; tilt-only from the standard optimum stops at (8.6, -6.3)), then
Nelder-Mead.

| data | truth | CUDA | MPS (phil) |
|---|---|---|---|
| sim BaTiO3 15.2 nm tilted | (3, -4) mrad | (3.0, -4.1) | (3.12, -4.19) |
| sim untilted control | (0, 0) | (-0.3, -0.1) | (0.002, 0.000) |
| logic device (Si), clean-crystal corner, 128 x 128 | ptycho (-11.6, +2.8) object frame (256 crop) | (-10.3, +4.7) scan = (-10.9, +3.1) object frame: 2.4 deg, 5 % | - |
| BaTiO3/SrTiO3 film ~15 nm, 128 crop | ptycho ~5 mrad | no stable optimum (grid edge) | - |

Thickness is a model depth spread, not a measurement (15.2 nm sim -> 10-13 nm).

**Frame.** SSB tilt is in the scan frame; quantem.thick's object frame is the scan rotated by the same rotation_deg
(p_obj = M p_scan, no sign change, transpose only if the reconstruction transposed). Details: study notes (frame check).

**Detector orientation matters.** On the logic device the standard fit barely separates the axis-swapped detector orientation
(fit 37.4 vs 35.7); the tilt model on the wrong orientation gives a ridge and streaked images. Choose orientation with the
thick fit (or trust the calibrated rotation).

**Performance.** CUDA 128 crop (8889 BF): thick preview 29-61 ms, `fit_sample` 30-55 s (1500 trials). MPS M5: preview
~20 ms after 1 s compile, fit ~65 s.

**API.** `SSB.preview(aberrations, sample={"tilt_row_mrad", "tilt_col_mrad", "thickness"})` (nm, mrad),
`SSB.fit_sample()`, `SSB.supports_sample`. CUDA and MPS backends. WebGPU (`reconstruct(..., {sample: {tiltRowMrad, tiltColMrad, thickness}})`, a separate thick shader so the thin path is bit-identical) drives ShowPtycho exports: logic crop in Mac Chrome (Metal) loss 0.170941 vs CUDA 0.170950, phase correlation 0.99999997; fitting stays in Python.
Tests: `tests/hardware/cuda/test_ssb_thick_sample.py`, `tests/hardware/mps/test_ssb_thick_sample_mps.py`, `tests/webgpu/ssb-thick-sample.ts` (+ `run_ssb_thick_sample.py`, weights vs CUDA: max rel 5e-4, median 1.5e-7).

## 3. cuFFT import order

`ImportError: libcufft.so.11` in every notebook that imports quantem.widget (torch) before SSB: torch (conda CUDA 13) loads
its cuFFT first and cuda.pathfinder then skips CuPy's CUDA 12 wheel library. `quantem.gpu._cuda_libraries.preload()` at
package import dlopens the nvidia-*-cu12 wheel libraries by path (different sonames, both coexist). Test in
`test_ssb_thick_sample.py::test_cupy_fft_loads_after_torch_cufft`.

## Rejected / open
- Rescaling `_factor` x10 inside the kernels: would change every backend-level fixture and Optuna trajectory; the boundary
  conversion keeps them.
- Lattice-SNR score (peak / median background): gamed by noise-free data and by weight normalisation; replaced by the
  least-squares fit.
- BaTiO3/SrTiO3 film: tilt not recovered - candidates scan distortion (2.4 %, 1.3 deg), scale mismatch (~3 %), diffuse background.
- Pre-existing, unrelated: `detector/workflow.py:560-565` sums float data with uint64 (float inputs lose their BF disk on
  MPS); `test_cuda_packed_ssb` fails on the io `representation` policy.
