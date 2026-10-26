# Tilt-corrected SSB on Metal: preview and fit speed (2026-10-20)

> **Historical:** describes the code as of 2026-10-20; its fits refined with the finite-difference Newton, since replaced by exact derivatives, so its fit timings are not current. See the [current fit methods](../../../api/ssb.md).

## Question

The default SSB result is tilt-corrected (sample tilt and depth spread fitted with the aberrations), so the
ShowPtycho sliders and `SSB(data).reconstruct()` run the thick-sample model. How fast can both run on an Apple
GPU without changing any result beyond float32 rounding, the 200-trial search, the Newton polish or the
two independent 180-degree branch fits?

## Setup

Apple M5 Max (40-core GPU, 128 GB), macOS 26.4, MLX 0.32. A 512 × 512 thin-film scan, 192 × 192 detector,
8,807 bright-field pixels, 300 kV, 30 mrad, 0.373 Å step (auto display factor 3x). Every GPU run held a shared
lock; the screen was shared, so WindowServer used 20 to 50 % of the GPU throughout, and every comparison below
interleaves the old and new code in one lock hold (medians of 12 calls, C10 changing every call).

## Slider drag: separable depth weights

The coherent (`phase_of_mean`) preview with tilt and depth spread used the per-term kernel: for every
(output frequency, BF pixel) term two soft-edge geometries, two chi values, two precise `sincos` and two
precise `sin` for the depth weights. The two-term transfer was already separable
([2026-10-20-mps-ssb-separable-coherent](2026-10-20-mps-ssb-separable-coherent.md)); the depth weights are too.
With f = factor wl², the depth-phase rates are x1 = C(q) − P(q) + B(q, k) and x2 = C(q) + P(q) + B(q, k), with
P = t f |q|² / 2, C = −π t q·θ and B = t f q·k, so sin x1 = Im(exp(i(C − P)) exp(iB)) and exp(iB) is a row
phasor times a column phasor of each BF pixel, reduced modulo 2π like the transfer tables. The kernel reads one
float4 table entry per row and column (transfer and depth phasor), evaluates sinc as sin x / x above |x| = 1 and
its Taylor series to x^10 below, and keeps the flat classes of the two-term kernel (a lone sideband keeps the sign
of its weight; both sidebands inside the flat aperture share A(k), which cancels; soft edges evaluate the unchanged
geometry with the 1e-8 floor). One thread computes every fine row that shares a source row (up to three), from one
read of each G value and column table.

| Drag, tilt-corrected, no loss | Before | After |
| --- | ---: | ---: |
| native 512² | 89–123 ms | 36 ms |
| 3x auto 1536² | 93–121 ms | 69–76 ms |

Parity on the thin-film scan, against a float64 evaluation of 11 spectrum rows: mean relative error 2.5e-6
(per-term kernel 2.8e-6) at the fitted parameters and 1.2e-5 (per-term 6.5e-5) at a far point
(C10 −15 nm, 4 nm C12, 40 nm depth); p99.9 1.5e-5 against 3.1e-5. Phase difference between the two kernels:
max 1.4e-5 rad, RMS 3.0e-6 rad at native sampling. `tests/hardware/mps/test_ssb_upsampling.py` compares the
separable and per-term kernels with tilt and depth at 1x, 2x and 3x (max 2e-6 rad on the synthetic session) and
checks that one or three rows per thread give the same bits.

Tried and dropped: precise `sin` per term instead of the depth phasor (the table loads are cheaper: 41 → 56 ms
native), hoisting the loads of 2, 4 or 8 BF pixels (no faster within the screen-sharing noise), groups of 8 or 32
BF pixels (no faster).
