# Exact-derivative SSB fit in Python on CUDA and MPS, 2026-10-25

> **Historical:** describes the code as of 2026-10-25; `refinement="newton"` (the finite-difference reference) is retired, and the default higher-order box is now ±20 with the half-band check. See the [current fit methods](../../../api/ssb.md).

## Question

The native Metal fit became about 40× faster when its searches moved to exact derivatives of the agreement. Is
that gain algorithmic? The same search was ported to the Python CUDA and MPS backends and checked against a
plain-PyTorch float64 ground truth.

## What changed

- `cuda/exact_derivatives.py`, `mps/exact_derivatives.py`: the exact gradient of the fifth-order agreement
  (variable projection, k-moments shifted by q, BF subsamples as strided views of G) and the exact gradient and
  Hessian of the low-order agreement. Kernels compute in float32 per (q, k) pair; sums over BF chunks and band
  frequencies are float64. Values come from the production objective kernels, with the band sum in float64.
- `_gradient_refinement.py`: the bounded trust-region quasi-Newton (BFGS) of the native search. It starts from
  a Hessian of every 64th BF pixel, applies D scaling and an active set at the box, climbs on every 8th pixel,
  then converges on all of them. `find_aberrations(refinement="auto")` selects it; `"newton"` keeps the
  finite-difference Newton as the reference trajectory.
- `_exact_newton.py`: the low-order trust-region Newton on the exact gradient and Hessian. Trust region,
  pruning and second polish are unchanged; derivatives are computed only where a step uses them. The 200-trial
  search is unchanged.
- `workflow._mapped_branch`: the opposite 180° branch is the exact image of the first fit (C10 and C12 change
  sign, φ12 moves by 90°, the tilt changes sign, depth is kept) instead of a second search.
- `higher_order_limit`: the box of the 22 higher-order coordinates, 2 or 20 rad at the aperture edge. The
  measurements below are at 2, the default when they were taken; the default is now 20 with the half-band
  check (`fifth_order_fit.higher_order_consistency`).
- `torch_agreement.py`: S(z) in ordinary torch arithmetic, on any device and dtype, with autograd gradient
  and Hessian. In float64 on the CPU it is the ground truth for the kernels and searches, and the referee
  between fits.

## Workload and boundary

Five measured 512 × 512 × 192 × 192 acquisitions at 300 kV and 30 mrad, with 8,800–9,300 BF pixels: three at a
0.373 Å scan step and two at 0.525 Å. Calibration and rotation are fixed at each acquisition's kept 180° branch.
`DirectPtycho.find_aberrations()` uses its defaults. Wall clock is measured with device synchronization in fresh
sessions: before and after alternate in two processes of three sessions each, and the table gives the median of
six. Hardware: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 4 CPU threads. Another user's jobs were active
in about a third of the samples; medians are within 1–8 % of minima. Before is `oct-25-native-direct-methods`
1da24e4d. Raw outputs are in the private evidence archive.

## Speed (s)

| Acquisition | Before | After | × | Fifth-order objective passes before → after |
|---|---:|---:|---:|---|
| A (0.373 Å) | 7.01 | 0.56 | 12.5 | 5,000 → 8 (+7 gradients) |
| B (0.373 Å) | 7.77 | 0.77 | 10.1 | 3,456 → 22 (+14) |
| C (0.373 Å) | 10.14 | 0.58 | 17.4 | 5,000 → 15 (+8) |
| D (0.525 Å) | 14.06 | 1.06 | 13.3 | 3,341 → 15 (+9) |
| E (0.525 Å) | 14.60 | 1.25 | 11.7 | 5,000 → 19 (+11) |

`io.load` (0.5–1.1 s) and `run(method, upsampling_factor="auto")` for SSB, Phase and Complex (0.04–0.5 s) are
unchanged. With both 180° branches compared (measured on an RTX PRO 6000 Blackwell Max-Q, quiet), the total drops
from 8.3–19.2 s to 0.70–1.66 s, and every acquisition keeps the same branch by the same test.

## Precision

The torch reference reproduces the float64 reconstruction oracle's model (rotated k, G from the integer counts,
aperture, χ and sidebands): S agrees to 1e-15 on a cropped acquisition. On every acquisition, with the session's
own float32 inputs promoted to float64, the CUDA kernels agree with autograd to:

- value: 1.7e-6 (3.7e-7 at the fits)
- fifth-order gradient: 3.7e-5 of its maximum
- low-order Hessian: 2.1e-6

CPU and CUDA float64 evaluations agree to 7e-16. Scored by the reference from the integer counts, each fit's
distance below the float64 optimum (1 − S/S_opt) is:

| Acquisition | Before | After | After − optimum: C10, C12 (nm) | Tilt (mrad) | Higher orders (rad) |
|---|---:|---:|---|---:|---:|
| A | 5.1e-7 | 8.5e-10 | −0.0001, −0.0000 | 0.0002 | 0.000 |
| B | 1.4e-7 | 5.1e-8 | −0.0018, +0.0003 | 0.0006 | 0.007 |
| C | 9.8e-8 | 6.1e-8 | +0.0006, +0.0009 | 0.0003 | 0.009 |
| D | 3.0e-7 | 3.3e-9 | +0.0000, −0.0000 | 0.0002 | 0.000 |
| E | 1.7e-7 | 7.0e-8 | −0.0001, +0.0006 | 0.0020 | 0.003 |

The float32 objective resolves about 1e-7. The old search stopped at its 5,000-evaluation cap or on a small step
above that level. The new one ends below it, within 2 pm of the float64 optimum in C10 and C12. The low-order fit
also gains (+1.2e-7 to +1.5e-6). Its C10 and C12 move by 1–2e-3 nm, inside the 0.05 Å step at which the
finite-difference Newton stops.

## MPS and the native app

On an Apple M5 Max (Python MPS, fresh sessions, under a shared GPU lock), `find_aberrations()` goes from
67.6–218.9 s to 6.7–20.5 s (6.7–16.4×). The fifth-order stage drops from 57.7–168.7 s to 0.95–4.7 s. On MPS the
200-trial search (2.0–6.3 s) and the low-order Newton (1.7–7.7 s) are now most of the time. The MPS kernels agree
with the float64 reference in the hardware tests at the same tolerances as CUDA.

Scored by the same float64 reference, the new CUDA, MPS and native Metal fits keep the same 180° branch on all
five acquisitions. They lie 7e-8 or less below the float64 optimum and agree with each other to 6.8e-8 in S,
2.1e-3 nm in C10 and 1.1e-3 nm in C12. Every earlier fit (CUDA and MPS) is 7e-8 to 5.1e-7 below. One native
fit is 1.9e-9 above the optimum found by the float64 search, so along the flat directions that optimum is itself
determined to about 2e-9.

## Conclusion

The speed-up is algorithmic and carries over to CUDA unchanged: 10–17× end to end on the same GPU, with a better
optimum. MPS gains 7–16×, and all three backends reach the same optimum. Float32 kernels with float64 reductions suffice; the float64 reference certifies the result.
