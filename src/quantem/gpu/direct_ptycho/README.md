# Direct Ptycho

`DirectPtycho` fits the probe aberrations of a 4D-STEM scan from its bright-field
sidebands, then reconstructs the object directly by SSB, regularized Phase or Complex.

## Three backends, one model

| Backend | Where | Device |
|---|---|---|
| CUDA | [cuda/](cuda/) | NVIDIA GPUs |
| Python MPS | [mps/](mps/) | Apple GPUs from Python |
| Native Swift/Metal | [MetalDirectPtychoKernels](../../../../native/swift/Sources/MetalDirectPtychoKernels/) | Apple GPUs in native apps |

All three implement the same agreement objective, search and reconstruction model.
Shared, device-free code (parameters, bounds, search control, results) lives in this folder.
[torch_agreement.py](torch_agreement.py) evaluates the objective in plain PyTorch float64
and referees between backends.

## Fit design

- **Low-order fit** (C10, C12, φ12, tilt, depth; tilt-corrected): a resident multivariate TPE
  start search, 200 joint and 200 standard-comparison trials in groups of 8
  ([_resident_search.py](_resident_search.py)), then a trust-region Newton on the exact gradient
  and Hessian ([_exact_newton.py](_exact_newton.py)).
- **Fifth-order fit**: bounded trust-region BFGS on exact gradients
  ([_gradient_refinement.py](_gradient_refinement.py), [fifth_order_fit.py](fifth_order_fit.py)).
- **Wide (±20) box by default, then checked**: two half-band refits on alternating |q| rings
  must agree to ≤ 0.5 rad rms of aperture phase; otherwise the ±2 fit is shown and the
  higher orders are marked not trusted.
- **Auto upsampling** follows the scan sampling and is capped at 4×, with a printed notice
  when the cap applies; 8× only when chosen (`max_factor`).

## Current numbers

Five measured 512 × 512 scans (300 kV, 30 mrad): MoS₂ bilayers A, B, C, a MoS₂ monolayer and a
thick oxide crystal. σ is the full-data fit uncertainty from count splits (±35 %).

| Check | Result | Measured |
|---|---|---|
| Cross-backend fit parity (180 fits, fifth order) | ±2 box ≤ 0.7σ, typically ≤ 0.3σ; ±20 box ≤ 1.4σ, typically ≤ 0.5σ; largest C10 difference 2.1e-3 nm | 2026-10-25, d3eeffa7 |
| Repeatability | same seed in a fresh process: bitwise identical on all backends; seed spread within a backend ≤ 1.3σ (±2), ≤ 1.9σ (±20) | 2026-10-25, d3eeffa7 |
| Image parity (one reconstructor) | SSB phase rms 4.7e-5 to 2.2e-4 rad (0.04–0.68 % of std); Phase ≤ 0.043 % of std | 2026-10-25, d3eeffa7 |
| Low-order fit, warm, bilayer A | CUDA 0.57–0.73 s (NVIDIA RTX PRO 6000 Blackwell Max-Q); native Metal 2.2–2.4 s and Python MPS 2.5 s (Apple M5 Max) | CUDA 2026-10-25, d3eeffa7; MPS and native 2026-10-26, e82c22ff |

Details, method and open items: the
[backend parity report](../../../../docs/performance/ssb/reports/2026-10-26-direct-ptycho-backend-parity.md).
Other fitting and reconstruction measurements:
[Direct Ptycho performance](../../../../docs/performance/ssb/index.md).
The public contract is the [Direct Ptycho API](../../../../docs/api/direct_ptycho.md).
