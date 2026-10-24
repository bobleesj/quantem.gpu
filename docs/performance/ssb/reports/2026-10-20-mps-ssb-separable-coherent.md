# Separable C10/C12 coherent SSB on Metal, 2026-10-20

## Question

Can the interactive MPS SSB preview (`phase_of_mean`, C10/C12/phi12 and
rotation sliders) reach 30 frames per second at native sampling on a full
512 × 512 × 192 × 192 acquisition, and how fast can the default 200-trial
aberration search run, without changing results beyond float32 rounding?

## Change

For the two-term probe the aberration function is a quadratic form,
chi(v) = v^T M v. Each transfer phase therefore splits exactly:

    chi(q - k) - chi(k) = Psi(q) - u(q).k,   chi(q + k) - chi(k) = Psi(q) + u(q).k,

with Psi = q^T M q and u = 2 M q. On the scan grid u.k = rf P_row(k) +
cf P_col(k) for integer frequencies rf and cf, so exp(-i u.k) is a product of
one row phasor and one column phasor per bright-field pixel. A corrected term
is G(q, k) exp(-i u.k) w, where w is exp(i Psi) or -exp(-i Psi) when one
sideband is open (its aperture weight cancels in the normalisation) and the
unchanged normalised mixture where both are open.

- The coherent kernel reads two table phasors per term instead of evaluating
  two soft-edge geometries, two chi values and two precise sincos.
- Tables and Psi are reduced modulo 2 pi from exact float32 products of 12-bit
  splits of float64 coefficients, so each phasor carries about 1e-7 rad error.
- Aperture weights, soft edges, the 1e-8 normalisation floor, BF selection
  and the estimator are unchanged. Soft-edge and double-sideband terms keep
  the exact per-term weights; flat regions are proven by conservative margins.
- Native sampling now uses the anti-Hermitian half spectrum as well: the
  Nyquist row, which is its own partner, is computed directly by one extra
  grid row of the same launch (a separate small launch added a serial tail).
- Higher-order and depth-corrected previews keep the per-term kernel.

## Measurements

One full acquisition, 512 × 512 scan, 192 × 192 detector, 8,939 BF pixels,
Apple M5 Max, all GPU work under a shared lock. Preview times are p50 of
interleaved reference/candidate calls in one process; each call changes C10
and returns the float32 phase to the host.

| Preview (C10 drag, no loss) | Before | After |
| --- | ---: | ---: |
| native 512² | 55.7 ms | 22.2 ms |
| 2× 1024² | 37.9 ms | 27.3 ms |

The new kernel is close to memory bound: at native sampling the contributing
G reads alone are about 7 GB per redraw (14 ms at the measured 489 GB/s).

## Parity

- Synthetic sessions (factors 1, 2, 3; C10 between -25 and 35 nm, zero
  aberration, anisotropic detector sampling): against a float64 evaluation of
  the same transfer the new phase error is 1e-7 rad or less, below the old
  kernel's own float32 error (up to 3e-7 rad).
- Full acquisition: new versus old phase, max 9.5e-7 rad (native) and 1.5e-6 rad (2×).
  On 25 spectrum rows against float64 the mean relative error is 1.61e-6
  (new) versus 1.77e-6 (old); p99.9 is 3.0e-5 versus 4.4e-5.
- A second full acquisition (a logic-device scan with only 406 BF pixels
  under a placeholder calibration) against a complete float64 phase: the old
  kernel's error is 4.7e-6 rad maximum (1.4e-6 mean), the new kernel's
  8.1e-7 rad (2.3e-7 mean). The 4.4e-6 rad difference between the two is the
  old kernel's float32 error at low frequencies, where the double-sideband
  normalisation divides two nearly equal phase differences.
- `tests/hardware/mps/test_ssb_upsampling.py` compares the separable and
  per-term kernels and checks the phasor reduction against float64.

## Browser frame rate

With quantem.widget's two-request slider queue the live ShowPtycho drag
(JupyterLab + Chrome, Metal adapter) paints about 34 new phase images per
second at native sampling, up from about 16. At 2× it paints about 24 phase
and 22 FFT images per second, up from about 19; the browser's per-frame
display work, not the reconstruction, is the limit there.

## Aberration search

`find_aberrations()` (200 TPE trials in pairs, Nelder-Mead, rotation check)
takes about 25 s and returns bit-identical coefficients and loss with and
without these changes. 22.8 s of it is the exact phase-variance loss; Optuna
itself takes 0.26 s. A candidate pair costs about 185 ms: the fused
correction and row IFFT writes a 2 MB intermediate per BF pixel and candidate
(about 37.5 GB per candidate) that the column IFFT reads back, which bounds a
candidate at about 77 ms at the measured bandwidth. Smaller packs (8 to 64
planes) did not keep the intermediate in cache and were slower or equal.
The search path was not changed.

Each BF pixel's corrected spectrum is anti-Hermitian, C(-q) = -conj(C(q)),
except at DC and on the Nyquist lines (checked to 3e-7 relative). Its inverse
transform is therefore i times a real image plus a constant and two
separable Nyquist-line terms, so a real-output inverse FFT would halve the
intermediate traffic that bounds the search. That kernel is not written yet.

## Rejected or neutral

Hoisting loads of several pixels in the mixed loop (slower at 2×), culling
groups of 8 or 32 pixels, threadgroups other than 32 × 4, placing the alias
rows q and q + n_source next to each other for cache reuse, and running the
lone native Nyquist column as its own launch (1.7 ms slower) gave no
repeatable gain. Groups whose pixels all share one aperture class use a
simpler loop; it is not faster but keeps the per-pixel classification out of
most groups.
