# Redraw at slider speed: q, -q symmetry for SSB and a drag window for regularized methods (2026-10-24)

## Questions

1. Can the 3x SSB redraw with a 14-term fit get closer to 30 fps without changing the model?
2. Can the regularized `phase` and `complex` methods redraw while one aberration slider is dragged in a few ms, at float32-level precision?

**1. Yes, 56 -> 30 ms.** Every term of the SSB sum obeys F(-q) = -conj F(q) for any aberration order, so the fine-grid kernel now sums each pair q, -q once with the 14-term probe too. Images differ from the full-plane sum by at most 3.0e-8 rad (one float32 ulp of the largest phase).

**2. Yes, 1.0 ms (phase) and 2.9 ms (complex) per update**, against 34 and 45-48 ms for an exact preparation. Error against a float64 reference: 4.1-5.3e-7 of the largest phase, the exact path's own 4.1-4.9e-7. The cost moves to the grab: 0.45 s (phase) or 0.75 s (complex) of exact preparations, built one per frame while exact frames are shown.

## Setup

- Acquisition A: MoS2 bilayer, 512 x 512 scan, 192 x 192 detector, 300 kV, 30 mrad, 0.373 A step, rotation 176.3 deg, 9298 BF pixels, a broad 14-term fit (tilt (5.92, -0.07) mrad, depth 12.4 nm). `upsampling_factor="auto"` = 3, output 1536 x 1536.
- GPU 0 only (RTX PRO 6000 Blackwell Workstation, 600 W), confirmed by name; GPU 1 (Max-Q) is never compared with it.
- Timing: device-synchronized wall clock; SSB 8 calls with C10 changed each call, first two dropped; drag window 5 calls after one warm-up, median.
- Strengths 0.03 / 0.03, smoothness 1 1/A, uniform weights.
- Source: quantem.gpu `oct-24-direct-iterative-api` at `f1cbe8aa` before, commits `953f9370` (drag window) and `c5b17d86` (SSB pairing) after.

## 1. SSB: the q, -q symmetry

The fine-grid SSB wave sums, for each output frequency q and BF pixel k, G(q, k) conj(gamma / |gamma|) with

```
gamma(q, k) = w1 t1 - w2 t2,  t1 = P(k - q) conj P(k),  t2 = conj P(k + q) P(k)
w1 = sinc(D/2 [-pi lambda (|k - q|^2 - |k|^2) - 2 pi q . theta]),  w2 = sinc(D/2 [+pi lambda (|k + q|^2 - |k|^2) - 2 pi q . theta])
```

Under q -> -q: t1(-q, k) = P(k + q) conj P(k) = conj t2(q, k) and t2(-q, k) = conj t1(q, k) for any chi (no parity of the aberrations is used), and the sinc is even, so w1(-q) = w2(q), w2(-q) = w1(q). Hence gamma(-q, k) = -conj gamma(q, k). The counts are real, so G(-q, k) = conj G(q, k). Each corrected term at -q is therefore -conj of the term at q, and so is the sum: the kernel needs only half the plane (plus the columns without a mirror, which the existing Nyquist guard keeps).

The pairing already existed for two-term optics. It was disabled for higher orders on the assumption that odd orders break it; they do not. Removing that condition is the whole change (`ssb/cuda/thick.py`).

| Preview, 3x, C10 changed per call | before | after |
|---|---|---|
| 14-term fit (full plane before) | 56.1 ms | **29.9 ms** |
| C10, C12, phi12 only (already paired) | 22.6 ms | 22.5 ms (bit-identical) |

Images, 14-term: max |difference| 3.0e-8 rad, 1.3e-7 of the largest phase; 87 % of pixels differ in the last bits because the mirrored corrections are rounded once instead of computed again. Test: `tests/hardware/cuda/test_ssb_wave_pairing.py` (odd orders C21, C23, C41, paired vs forced full plane, tolerance 1e-6 of the largest phase, and an assertion that pairing ran).

What remains in the 30 ms: the 14-term chi is evaluated twice per (q, k) pair; with two terms the same kernel takes 22.5 ms, so about 7 ms is the polynomial.

## 2. Regularized methods: the drag window

Along one control c with everything else held, every Gram entry and right side of the per-frequency normal equations is sum_k f_k exp(i (c - c0) u_k), an entire function of c. On [c0 - h, c0 + h] its Chebyshev series converges like 2 (z/2)^(N+1) / (N+1)!, z = h max|u|. `DragWindow` (`ssb/cuda/direct.py`) prepares the statistics exactly at the N + 1 = 13 Chebyshev nodes and keeps the coefficients c_n = (2/(N+1)) sum_j T_n(x_j) F(x_j). Any c in the window is then one matrix-vector product, F = sum_n c_n T_n(x), followed by a float64 Cholesky solve per frequency, the scatter and one inverse FFT.

max|u| is the spread of d chi / dc over the aperture (chi enters every product as a difference of two pupil points): (2 pi / lambda) alpha^(n+1) / (n + 1) for a magnitude C_nm, doubled for m > 0; m C_nm times that, doubled, for an azimuth. h = 3 / max|u| with N = 12.

| Control | half width h |
|---|---|
| C10 | 2.05 nm |
| C12 | 1.03 nm |
| phi12 (at C12 = 6.9 nm) | 4.2 deg |
| C21 / phi21 | 51 nm / 33 deg |
| C30 | 4.5 um |
| C50 | 7.3 mm |

Window frames against an exact preparation at the same value (max over both edges, the centre, a node and three points between; images as max |difference| over the largest value):

| Control | phase method | complex: phase | complex: log-amplitude | update, phase / complex |
|---|---|---|---|---|
| C10 | 3.5e-7 | 4.4e-7 | 2.5e-6 | 1.0 / 2.9 ms |
| C12 | 3.9e-7 | 4.8e-7 | 2.3e-6 | 1.2 / 3.1 ms |
| phi12 | 3.8e-7 | 4.8e-7 | 2.5e-6 | 1.2 / 3.1 ms |
| C21 | 3.8e-7 | 5.1e-7 | 2.3e-6 | 1.1 / 3.0 ms |
| phi21 | 3.6e-7 | 4.1e-7 | 2.3e-6 | 1.1 / 3.2 ms |
| C30 | 3.4e-7 | 3.9e-7 | 2.5e-6 | 1.2 / 2.9 ms |
| C50 | 3.3e-7 | 4.0e-7 | 2.6e-6 | 1.0 / 3.0 ms |

The log-amplitude column is the quantization of a complex64 wave near |w| = 1 (2**-23 absolute, about 2e-6 of the largest log-amplitude); at the nodes it is 0.

Against an independent float64 reference (transfer, Gram, solve and FFT all complex128) at C10 = -9.618 and -10.364 nm:

| | window | exact preparation |
|---|---|---|
| phase method | 4.1e-7, 4.8e-7 | 4.3e-7, 4.1e-7 |
| complex, phase | 4.9e-7, 5.3e-7 | 4.6e-7, 4.9e-7 |

The window stays within 1.2x of the exact path's own float32 error.

Cost:

| | phase | complex |
|---|---|---|
| exact preparation + solve | 34.3 ms | 45.4-48.6 ms |
| window update | 0.96-1.2 ms | 2.8-3.2 ms |
| window build (13 nodes, eigenbasis skipped) | 0.42-0.47 s | 0.73-0.82 s |
| coefficient memory | 0.58 GB | 2.10 GB |

A complex window holds the phase statistics as its lower-right block (phase from it: 2.1 ms per update, within 3.6e-7 of an exact phase preparation), so one build serves both methods.

How it is used (`ssb/direct_workflow.py`): `drag(session, control, ...)` serves a frame from the prepared system when the optics are those already prepared (a strength drag), from the window when it is complete and covers the value, and otherwise exactly. It starts a new window, centred on the current value, when the dragged control changes, a held input changes, or the value leaves the window. `advance(session)` adds one node and is meant to run between frames, so a drag shows exact frames for its first 13 steps and window frames after. `run()`, the public `preview()` and committed frames never use the window; saved results remain the exact model. Tests: `tests/hardware/cuda/test_direct_drag_window.py` (nodes at 1e-6, between nodes at 2e-6 of the largest phase, phase from a complex window, and the drag/rebuild sequence).

## Rejected or deferred

- **Chebyshev window for SSB.** gamma / |gamma| is not analytic in c: in triple-overlap regions where the two sideband terms have equal weight, gamma = a e^{i c u1} - b e^{i c u2} has zeros on the real axis, so the correction is discontinuous there and a polynomial in c cannot reach 1e-6.
- **FFT correlation for the Gram, optics-free diagonal split, Taylor series, interpolating the image, one window over the full slider range**: see the regularized-method math notes of the same day (the diagonal split leaves the transfer evaluation for the right side, 40-50 ms; a full-range C10 window needs N about 65 and 10 GB for complex).
- **Fused regularized preparation** (a parallel prototype on the SSB kernel structure, 19.5 ms phase / 35.5 ms complex with fast transcendentals): not merged here; it would cut the exact frames during a window build and the build itself.

## Next

- End-to-end frames per second in ShowPtycho per control, once the transport path delivers one float32 image per update.
- SSB: a cheaper 14-term chi (Horner form) for the remaining 7 ms.
