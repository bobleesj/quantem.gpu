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
`SSB.fit(tilt=True)` (result `.sample`, `.report()`), `SSB.supports_sample`. CUDA and MPS backends. WebGPU (`reconstruct(..., {sample: {tiltRowMrad, tiltColMrad, thickness}})`, a separate thick shader so the thin path is bit-identical) drives ShowPtycho exports: logic crop in Mac Chrome (Metal) loss 0.170941 vs CUDA 0.170950, phase correlation 0.99999997; fitting stays in Python.
Tests: `tests/hardware/cuda/test_ssb_thick_sample.py`, `tests/hardware/mps/test_ssb_thick_sample_mps.py`, `tests/webgpu/ssb-thick-sample.ts` (+ `run_ssb_thick_sample.py`, weights vs CUDA: max rel 5e-4, median 1.5e-7).

## 3. cuFFT import order

`ImportError: libcufft.so.11` in every notebook that imports quantem.widget (torch) before SSB: torch (conda CUDA 13) loads
its cuFFT first and cuda.pathfinder then skips CuPy's CUDA 12 wheel library. `quantem.gpu._cuda_libraries.preload()` at
package import dlopens the nvidia-*-cu12 wheel libraries by path (different sonames, both coexist). Test in
`test_ssb_thick_sample.py::test_cupy_fft_loads_after_torch_cufft`.

## 4. Speed: standard vs thick-sample, CUDA vs MPS (measured 2026-09-24)

**Setup.** `bench_ssb_tilt.py` (study folder): `SSB.from_array` on uint16 counts, 192 x 192 detector, 300 kV, 30 mrad.
Preview = median of 30 `SSB.preview` calls returning the loss (a float, so the device is synchronised); drag = the same
through `preview_context(num_bf // 4)`. Fit = `SSB.fit(trials=200, refinement="nelder-mead")`; tilt fit = the thick-sample search (then
`fit_sample`: 200-trial standard baseline + 300 joint trials + 3-start Nelder-Mead, all on the least-squares objective). CUDA: RTX PRO
6000 Blackwell, idle GPU. MPS: M5 (128 GB). Same committed source on both. Samples: logic device (Si) and BaTiO3/SrTiO3 film.

| scan | sample | backend | preview std / thick (ms) | drag std / thick (ms) | fit std / thick (s) | tilt (mrad, scan frame) |
|---|---|---|---|---|---|---|
| 128^2 | logic | CUDA | 16.4 / 26.2 | 1.8 / 3.0 | 1.9 / 3.3 | (-10.25, +4.68) |
| 128^2 | logic | MPS | 8.8 / 11.4 | 3.0 / 3.4 | 2.9 / 5.8 | (-10.26, +4.68) |
| 128^2 | film | CUDA | 16.3 / 26.2 | 4.1 / 7.7 | 3.5 / 1.2 | (-0.66, -5.35) |
| 128^2 | film | MPS | 9.0 / 11.3 | 2.7 / 3.2 | 2.7 / 2.4 | (-0.65, -5.34) |
| 256^2 | logic | CUDA | 16.5 / 97.9 | 9.0 / 29.4 | 9.5 / 3.9 | (-9.79, +4.80) |
| 256^2 | logic | MPS | 36.4 / 48.1 | 9.6 / 12.5 | 13.3 / 23.4 | (-9.79, +4.80) |
| 256^2 | film | CUDA | 21.1 / 52.4 | 4.2 / 12.9 | 5.3 / 2.5 | (-0.45, -5.40) |
| 256^2 | film | MPS | 40.6 / 93.1 | 23.7 / 32.6 | 19.5 / 9.0 | (-0.45, -5.39) |
| 512^2 | logic | CUDA | 26.7 / 393.9 | 44.3 / 113.6 | 10.8 / 22.3 | (-9.14, +4.03) |
| 512^2 | logic | MPS | 128.1 / 2793 | 33.0 / 713.9 | 59.2 / 93.5 | (-9.13, +4.04) |
| 512^2 | film | CUDA | 60.7 / 242.8 | 44.4 / 118.7 | 13.8 / 7.4 | (-0.09, -5.08) |
| 512^2 | film | MPS | 173.0 / 2861 | 47.1 / 704.8 | 52.3 / 24.0 | (-0.10, -5.06) |

**Conclusions.**
- The fit is not the bottleneck: full-field tilt fit 7-22 s on CUDA, 24-94 s on MPS; CUDA and MPS agree to 0.02 mrad.
- The thick preview is. CUDA's `reconstruct_thick` is a chunked element-wise kernel + cuFFT, outside the fused standard
  kernels: 1.6x the standard preview at 128^2, 15x at 512^2 (394 ms: the ShowPtycho tilt slider is not interactive at
  full field).
- MPS fuses the depth weights into the row-IFFT kernel, but that kernel exists only for 128/256/1024 scans
  (`backends/mps/_thick_sample.py:39`, `engine.py:1587`); 512^2, the common full scan, falls back to the MLX reference graph
  (2.8 s, 22x standard).
- CUDA drag (25 % BF) is slower than the full preview at 512^2 (44 vs 27 ms): the drag path does not scale with the subset.
- Crop-to-full tilt is stable on both samples (logic -10.3/-9.8/-9.1 row; film about 5 mrad along -col).

**Fix candidates.** CUDA: fold w1/w2 into the fused standard kernel (the MPS approach). MPS: add a 512 row-IFFT
specialisation. Both before claiming a real-time full-field tilt slider.

## 5. Fit strategy, trial count, rotation (full 512 x 512 field, CUDA, 2026-09-24)

Scripts `fit_strategies.py`, `trials_rotation.py` (study folder). All candidates scored on the same least-squares
objective; "fit" is relative to the best value found on that acquisition. Seeds 0-2 (strategies), 0-4 (trials).

| strategy | logic device | film | time |
|---|---|---|---|
| standard SSB (no tilt) | 0.55 | 0.78 | 8-9 s |
| standard fit, then tilt + thickness only | 0.91, every seed | 0.99, every seed | 10-11 s |
| same, then Nelder-Mead over all six | 1.00 (2 of 3 seeds exact) | 1.00 | 11-16 s |
| joint TPE 200 + Nelder-Mead | 1.00 | 1.00 | 4-15 s |
| standard baseline 200 + joint 300 + Nelder-Mead (then default) | 1.00 | 1.00 | 3-14 s |
| centre 128 crop search + full-field Nelder-Mead | 1.00 | 0.9993 (row tilt -0.7) | 2-6 s |

Sequential fitting stalls because C10 must move with the tilt (logic device 15.3 -> 4.0 nm mid-depth defocus).

| joint trials | film: seeds at best / tilt spread | logic: seeds at best |
|---|---|---|
| 50 | 3 / 5, up to 1 mrad off | 5 / 5 (one at 0.9988) |
| 100 | 5 / 5 | 4 / 5 (one at 0.9993) |
| 200 | 5 / 5, +-0.02 mrad | 5 / 5 |
| 300, 400 | 5 / 5, same answer | 5 / 5 |

Time does not depend on trials (Nelder-Mead dominates). **Decision:** `fit(tilt=True)` = one joint search of `trials`
(default 200, the same budget as the standard fit) + 3-start Nelder-Mead; `fit_sample()` removed.

Rotation (joint fit at calibrated +- 6 deg): both acquisitions peak at **+1 deg** (logic fit +0.3 %, film +1.1 %); the
tilt moves by < 0.25 mrad; +-4 deg costs 4-10 %. Rotation is well determined but left fixed at the calibration: it is a
per-instrument calibration shared by every acquisition, and freeing it trades off against the tilt direction. The
consistent +1 deg on two datasets is a lead for the rotation calibration itself.

## 6. SSB.open keeps the acquisition ANS encoded

`SSB.open` asked `io.load` for `representation="dense"`, which the GPU loader now refuses ("GPU acquisitions must remain
ANS encoded"). It now loads encoded (film, 512 x 512 x 192 x 192: 2.2 GB resident, 3.0 s), finds the bright-field disk
on the full-detector mean pattern (0.34 s) with the backend's own rule, decodes only that detector crop (112 x 112,
0.22 s) and pins the disk centre (`bf_center`, detector pixels) so the crop selects the same pixels. Session memory 9.6
GB instead of decoding the 19 GB cube first. Against `from_array` on the fully decoded cube: same 8939 BF pixels,
max |phase difference| 6e-8 rad, identical loss (`tests/hardware/cuda/test_ssb_open_encoded.py`).

Also: at the full BF count, the drag-preview context no longer copies G and a result buffer (2 x BF x scan complex64,
37 GB at 512 x 512); CUDA and MPS reuse the session.

## Rejected / open
- Rescaling `_factor` x10 inside the kernels: would change every backend-level fixture and Optuna trajectory; the boundary
  conversion keeps them.
- Lattice-SNR score (peak / median background): gamed by noise-free data and by weight normalisation; replaced by the
  least-squares fit.
- BaTiO3/SrTiO3 film: tilt not recovered - candidates scan distortion (2.4 %, 1.3 deg), scale mismatch (~3 %), diffuse background.
- Pre-existing, unrelated: `detector/workflow.py:560-565` sums float data with uint64 (float inputs lose their BF disk on
  MPS); `test_cuda_packed_ssb` fails on the io `representation` policy.
