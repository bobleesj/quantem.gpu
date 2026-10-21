# Sign of the odd aberrations in SSB reconstruction (2026-10-21)

## Question

For a thin object the bright-field spectrum is

G(q, k) ∝ φ̃(q) [P(k − q) P*(k) − P(k) P*(k + q)]

(Pennycook et al., Ultramicroscopy 151, 2015; probe at p(r − r₀), scan FFT
e^{−2πi q·r₀}). Every backend evaluated the first term's probe phase at q − k.
For C10, C12 and every even order χ(q − k) = χ(k − q); for the odd orders
(C21, C23, C41, C43, C45) the sign reverses on one sideband only. Which one do
the data follow, and does each backend now evaluate it?

## The oracle

An independent float64 model: χ in polar form (α, atan2), the backends' soft
aperture, the depth weights, both estimators (mean phase, phase of the mean), and
either sign. It shares no code with the kernels.

- **Closed form.** On an exact thin-object 4D-STEM simulation (one transmission
  function, probe with C21 50 nm, C23 40 nm, C30 3 µm, 64 × 64 scan), the linear
  part of the simulated G, ½[G(φ) − G(−φ)], matches the k − q form to
  1.7 × 10⁻⁵ (relative, one complex amplitude per q). The q − k form misses
  by 0.59.
- **Data symmetry.** Real intensities give G(−q, k) = conj G(q, k), and the
  k − q model keeps Γ(−q, k) = −conj Γ(q, k) for any probe phase. The q − k
  model does not. Applied to a measured 4D-STEM acquisition (a 128 × 128 crop,
  2,287 bright-field pixels) with C21 104 nm, C23 21 nm and C30 5.3 µm, the
  corrected spectra break the symmetry by 2 × 10⁻⁸ (relative) with k − q and by
  1.05 with q − k (Nyquist row and column excluded). The half-plane fit
  objectives rely on this symmetry.

## Every backend against the oracle

Largest phase difference (rad) from the oracle at either sign: before
(quantem.gpu dc8f0398) and after the fix. "Simulation": a weak periodic lattice,
one slice, the probe above, tiled to the backend's scan size. "Measured": the
crop above. Each backend reconstructs from its own G(q, k) and k, and the
oracle uses the same inputs. A backend has the right sign when it matches k − q
to float32 precision. The two signs differ by 5 × 10⁻³ to 0.13 rad.

| backend, path | data | before: vs k − q | before: vs q − k | after: vs k − q | after: vs q − k |
| --- | --- | --- | --- | --- | --- |
| MPS, phase of mean | simulation | 8.4e-3 | 1.2e-6 | 1.5e-6 | 8.4e-3 |
| MPS, mean phase | simulation | 8.4e-3 | 1.6e-7 | 1.3e-7 | 8.4e-3 |
| MPS, phase of mean | measured | 0.125 | 6.5e-5 | 8.6e-5 | 0.125 |
| MPS, mean phase | measured | 0.110 | 5.4e-4 | 7.5e-5 | 0.110 |
| CUDA, wave sum (phase of mean) | simulation, 256² | 8.5e-3 | 1.3e-7 | 9.2e-8 | 8.5e-3 |
| CUDA, fused FFT (mean phase) | simulation, 256² | 8.4e-3 | 1.6e-7 | 1.3e-7 | 8.4e-3 |
| CUDA, wave sum | measured, 256² | 0.130 | 5.7e-5 | 8.6e-5 | 0.130 |
| CUDA, fused FFT | measured, 256² | 0.118 | 9.1e-4 | 7.7e-5 | 0.118 |
| WebGPU, phase of mean | simulation, 128² | 8.5e-3 | 8.1e-8 | 6.2e-8 | 8.5e-3 |
| WebGPU, mean phase | simulation, 128² | 8.4e-3 | 4.9e-8 | 1.2e-8 | 8.4e-3 |
| WebGPU, both | measured | 0.080–0.092 | 2.7e-6–4.2e-6 | 4.6e-6–4.9e-6 | 0.080–0.092 |
| native Metal (MetalSSBKernels) | simulation, 128² | 8.5e-3 | 3.3e-8 | 2.9e-8 | 8.5e-3 |
| native Metal | measured | 0.092 | 2.7e-6 | 4.3e-6 | 0.092 |

The measured data's float32 floor is set by the low-order terms alone: there
the two signs agree, and every path matches the oracle to 3 × 10⁻⁵ rad (MPS) or
9 × 10⁻⁷ (native). WebGPU ran headless in Chrome on the Apple GPU (adapter
`apple / metal-3`, subgroup size 32, not a fallback adapter). CUDA ran on an
RTX PRO 6000 Blackwell.

The paths include CUDA's fused full-aberration FFT kernels
(`cuda/kernels/common.py`, `gamma_mul_pk_onthefly_full`), which
`preview(..., phase_estimator="mean_phase")` runs with higher orders and no
depth spread on 256, 512 and 1024 scans.

## Effect on the image

With the probe's own coefficients, the reconstructed phase of the simulation
correlates with the object as follows (both estimators, every backend; 64², 128²
and 256² scans of the same periodic specimen):

| | k − q (after) | q − k (before) |
| --- | --- | --- |
| true C10 … C30 | 0.963–0.964 | 0.941–0.942 |
| C10, C12 only | 0.942 | 0.942 |
| odd terms reversed | 0.875 | 0.880–0.881 |

Before the fix, entering the true coma and trefoil did not improve the image.

## Tests

- `tests/test_ssb_higher_order_sign.py` (CPU): the closed form against an exact
  thin-object simulation, and the symmetry of each form.
- `tests/hardware/test_ssb_higher_order_sign.py` (MPS or CUDA): reconstruction of
  a weak-phase simulation, true coefficients against none and against reversed
  odd terms, for both estimators. On CUDA the scan is tiled to 256², so it runs
  the fused FFT path.
- `tests/hardware/cuda/test_ssb_higher_order.py` and
  `tests/hardware/test_ssb_coherent_higher_order.py`: their float64 references
  now use k − q. On the old kernels 9 of their 16 cases fail.

The fused CUDA test's tolerance rises from 5e-6 to 1e-5 rad. Against the
independent oracle, the fused path (float32 Chebyshev χ, all 14 orders active)
is 3.5 × 10⁻⁶ off and the test's own float64-χ reference 3.6 × 10⁻⁶. Before
the fix the two errors partly cancelled (2.0 × 10⁻⁶). With the wrong sign the
difference is 2.6 × 10⁻³.

The saved-result signature's `higher_order_model` becomes `"krivanek14 k-q"`.
Reconstructions saved with odd orders under the old sign are recomputed.
