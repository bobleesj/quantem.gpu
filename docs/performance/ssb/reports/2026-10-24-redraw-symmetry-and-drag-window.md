# Redraw at slider speed: q, -q symmetry and drag windows for SSB and regularized methods (2026-10-24)

## Questions

1. Can the 3x SSB redraw with a 14-term fit get closer to 30 fps without changing the model?
2. Can the regularized `phase` and `complex` methods redraw while one aberration slider is dragged in a few ms, at float32-level precision?
3. Can an SSB drag do the same, although its correction gamma / |gamma| is not analytic in the optics?

**1. Yes, 56 -> 30 ms.** Every term of the SSB sum obeys F(-q) = -conj F(q) for any aberration order, so the fine-grid kernel now sums each pair q, -q once with the 14-term probe too. Images differ from the full-plane sum by at most 3.0e-8 rad (one float32 ulp of the largest phase).

**2. Yes, 1.0 ms (phase) and 2.9 ms (complex) per update**, against 34 and 45-48 ms for an exact preparation. Error against a float64 reference: 4.1-5.3e-7 of the largest phase, the exact path's own 4.1-4.9e-7. The cost moves to the grab: 0.45 s (phase) or 0.75 s (complex) of exact preparations, built one per frame while exact frames are shown.

## Setup

- Acquisition A: MoS2 bilayer, 512 x 512 scan, 192 x 192 detector, 300 kV, 30 mrad, 0.373 A step, rotation 176.3 deg, 9298 BF pixels, a broad 14-term fit (tilt (5.92, -0.07) mrad, depth 12.4 nm). `upsampling_factor="auto"` = 3, output 1536 x 1536.
- GPU 0 only (RTX PRO 6000 Blackwell Workstation, 600 W), confirmed by name; GPU 1 (Max-Q) is never compared with it.
- Timing: device-synchronized wall clock; SSB 8 calls with C10 changed each call, first two dropped; drag window 5 calls after one warm-up, median.
- Strengths 0.03 / 0.03, smoothness 1 1/A, uniform weights.
- Source: quantem.gpu `oct-24-direct-iterative-api` at `f1cbe8aa` before; commits `953f9370` (regularized drag window), `c5b17d86` (SSB pairing), `bc438e85` (SSB drag window), `9fdd3c4d` (triple-overlap list), `d0e6ad84` (windows per control, idle prepare).

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

How it is used (`ssb/direct_workflow.py`): `drag(session, control, ...)` serves a frame from the prepared system when the optics are those already prepared (a strength drag), from the window when it is complete and covers the value, and otherwise exactly. It starts a new window, centred on the current value, when the dragged control changes, a held input changes, or the value leaves the window. `advance(session)` adds one node and is meant to run between frames, so a drag shows exact frames for its first 13 steps and window frames after. The session keeps one window per control, valid while every other input is unchanged; `prepare(session, controls, ...)` asks for windows (C10, C12, phi12 typically) after a release, and `advance` builds them while the sliders are idle, the dragged control first, so the next grab starts on window frames (`test_prepared_windows_serve_the_first_frame_of_a_grab`). Moving one control drops the windows of the others, whose held inputs include it. `run()`, the public `preview()` and committed frames never use the window; saved results remain the exact model. Tests: `tests/hardware/cuda/test_direct_drag_window.py` (nodes at 1e-6, between nodes at 2e-6 of the largest phase, phase from a complex window, and the drag/rebuild sequence).

## 3. SSB drags: interpolate the single-sideband terms, sum the triple overlap

Serves: SSB drags of C10, C12, phi12 (or any term) with the 14-term fit held, 3x, full field.

A term (q, k) has one of three supports: neither sideband aperture open (zero), one open, or both open (triple overlap). With one open, gamma = w1 t1 alone and gamma / |gamma| = sign(w1) exp(-i (chi(k - q) - chi(k))): the sinc weight does not depend on the aberrations, so this term is an entire function of the control, like the regularized statistics. With both open, |gamma| depends on the optics and vanishes on the real axis where the two terms cancel, so no polynomial in the control fits it. The kernel now takes a class switch (`SSB_WAVE_PAIRS` 1 = one sideband, 2 = both); their sum is the full spectrum to 1.2e-18 of its largest value (float64 summation order). The scan mean is counted in class 1 only.

`WaveWindow` (`ssb/cuda/thick.py`) keeps the class-1 spectrum as a Chebyshev series (N = 12, the same h as section 2) and sums class 2 exactly every frame. On the test field class 2 holds 3.3e8 of the 2.2e9 nonzero terms (15 %), all at |q| <= R (136k of the 543k frequencies in the support).

| GPU 0, 3x, 14-term fit | exact preview | drag frame | build (13 nodes) | memory |
|---|---|---|---|---|
| C10 | 34.5 ms | **6.5-7.0 ms** | 0.41 s | 0.25 GB |
| C12, phi12, C21, C30 (before the triple list below) | 34.5-35.2 ms | 10.0-10.4 ms | 0.42-0.43 s | 0.25 GB |

Window frame against the exact preview, max |difference| over the largest phase, both edges, centre, node and three points between: 1.5-2.6e-7 for every control. For scale, the float32 probe phase itself puts the exact SSB image 1.7e-5 from the same kernel with chi in float64 (measured here; gamma / |gamma| amplifies chi rounding where the two sidebands nearly cancel).

The class-2 sum first took 9.3 ms of the 10 ms frame. ncu showed DRAM 57 % busy: each of its 10 launches wrote, and then reduced, 16 partial planes of the whole 1536 x 769 stored plane (302 MB) for 136k useful frequencies. Running class 2 over a list of the |q| <= R frequencies (`SSB_WAVE_QLIST`), with BF pixels in compact cells of a quarter BF radius (`SSB_WAVE_ORDER`, 64 per group) and 64 groups per launch: 9.2 -> 5.8 ms, frame 10.2 -> 6.5 ms. Tests: `tests/hardware/cuda/test_direct_drag_window.py` (`test_ssb_window_matches_exact_preview`: first frame bit-identical to the preview, nodes at 1e-6, between nodes at 2e-6 of the largest phase).

A block whose frequencies all miss its BF group now returns before the per-tile BF setup (`__syncthreads_or`): bit-identical, 12.7 -> 12.3 ms of class 2 on GPU 1, kept.

## Rejected or deferred

- **Horner chi for the 14-term SSB kernel** (28 Cartesian monomials, 27 FMAs): chi itself is closer to float64 (rms 6.2e-7 vs 1.4e-6 rad over the aperture), but the image is not (max 3.9e-5 vs 1.7e-5 of the largest phase against a float64-chi kernel, rms equal), and the redraw gained only 13 % (46.7 -> 40.5 ms, GPU 1). Ablation on GPU 1: chi is 15.5 ms of 46.4, the depth sinc 7.5 ms, sincos about 0; the kernel is compute-bound (75.6 % issue slots, IPC 3.0, 76 registers, 40 % occupancy).
- **Compact BF cells for the exact SSB sum**: would cut evaluated pairs 3.9e9 -> 3.0e9 (culling efficiency 0.57 -> 0.73), but culled-in pairs with both apertures closed cost only the two geometries; for class 2 alone the cells gave 9.3 -> 9.2 ms. Not pursued for the exact path.
- **Frequency list for the exact fine-grid sum** (support |q| <= 2R, 46 % of the stored plane): bit-identical, no measurable change (31.4 ms); the exact kernel is compute-bound. Reverted.

- **Chebyshev window over the whole SSB spectrum.** gamma / |gamma| is not analytic in c: in triple-overlap regions where the two sideband terms have equal weight, gamma = a e^{i c u1} - b e^{i c u2} has zeros on the real axis, so the correction is discontinuous there and a polynomial in c cannot reach 1e-6. Section 3 interpolates only the single-sideband class.
- **FFT correlation for the Gram, optics-free diagonal split, Taylor series, interpolating the image, one window over the full slider range**: see the regularized-method math notes of the same day (the diagonal split leaves the transfer evaluation for the right side, 40-50 ms; a full-range C10 window needs N about 65 and 10 GB for complex).
- **Fused regularized preparation** (a parallel prototype on the SSB kernel structure, 19.5 ms phase / 35.5 ms complex with fast transcendentals): not merged here; it would cut the exact frames during a window build and the build itself.

## Next

- End-to-end frames per second in ShowPtycho per control, once the transport path delivers one float32 image per update.
- SSB exact frames (shown while a window builds, and on release) remain 30-35 ms; the depth sinc (16 % of the kernel) is the next ablation-backed target.
