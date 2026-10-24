# Tilt-corrected SSB on CUDA: fit and preview speed (2026-10-20)

## Question

Where does a tilt-corrected fit spend its time on CUDA, and how close can the
fused agreement kernel, the Newton passes and the previews get to the
hardware, with every fit unchanged within the cross-backend parity
tolerances (`2026-10-20-ssb-cuda-standard-fit.md`)?

## Setup

A private 512 x 512 crystal scan, 192 x 192 detector, 300 kV, 30 mrad, 8,807
bright-field pixels; G(q, k) is 9.3 GB (half plane), and the fit band
(0.2 to 0.9 1/A) holds 44,247 frequencies, so one pass of the agreement kernel
reads 3.1 GB of G. RTX PRO 6000 Blackwell Max-Q (188 SMs), CuPy 14.2, Nsight
Compute 2025.4 (kernel durations below are Nsight Compute's, at its fixed
clocks). Timings follow the shared protocol: warm runs, median of 3 fits or 20
previews. Evidence: experiment `2026-10-20-ssb-cuda-parity` in the private
archive.

## Where the time went

A warm tilt fit (rotation given, 200 trials and Newton) ran 78 passes of
`thick_fit_batch_kernel` (25 standard-comparison batches, 25 search batches,
28 Newton passes), 82 % of its GPU time. Its time grew with the number of
parameter sets per pass (1, 2, 4, 8 sets: 14, 16, 29, 60 ms on a shared GPU),
so the per-set work, not the G reads, set the cost.

## Changes

| Kernel | Change | Before | After |
|---|---|---|---|
| `thick_fit_batch` (8 sets, depth 136 A) | per-set loops unrolled to the compile-time maximum | 25.4 ms | 14.9 ms |
| | division-free probe geometry, linear chi | 14.9 ms | 12.3 ms |
| | `__launch_bounds__(256, 4)` (64 registers, no spills) | 12.3 ms | 10.9 ms |
| `thick_fit_batch` (8 sets, depth 0) | the same three | 24.5 ms | 7.2 ms |
| `thick_wave_sum` (drag preview, 1x) | BF geometry and chi(k) once per block | 2.85 ms | 2.25 ms |
| | aperture denominator constant on square pixels | 2.25 ms | 1.81 ms |
| `ThickSample.reconstruct` (mean phase + loss) | one correction launch from the half plane, no full-plane copy or angle images | 223 ms | 97 ms |

- **Local memory, not arithmetic.** The per-set loop ran to the runtime batch
  size, so the per-set accumulators were indexed dynamically and the compiler
  placed them in local memory: Nsight Compute showed 93 % "memory throughput"
  with DRAM at 15 % and an L1 hit rate of 53 %. Rewriting the arithmetic first
  (linear chi, the tilt shift hoisted out of the pixel loop) changed nothing.
  Loops unrolled over `MAXB` with a uniform `b < B` guard keep them in
  registers.
- **Geometry.** The fit kernel needs only alpha^2 (1, cos 2 phi, sin 2 phi) =
  lambda^2 (|v|^2, dx^2 - dy^2, 2 dx dy) and the soft aperture, whose
  denominator is the constant detector step on square pixels; three IEEE
  divisions and a square root per probe tilt go. The square root that remains
  is the correctly rounded one: the soft edge moves by lambda / ang (about 35)
  per unit of |v|, which magnifies an approximate `rsqrt`.
- **The fit kernel now.** 6.2e9 instructions per pass instead of 13.4e9
  (depth 0: 4.5e9 instead of 8.2e9); the transcendental pipe (sin, cos, sqrt)
  is 78 % busy at depth 136 A, the FMA pipe 49 %, DRAM 34 %: what remains is
  the two `sincos` and two `sin` of each (k, q, set) term. The 8 sets per pass
  are the search's batches of eight, so a larger batch would only shorten the
  Newton passes.
- **Preview.** Nsight Compute reported the FP64 pipe as the busiest in
  `thick_wave_sum` (52 %, its float64 accumulator). Accumulating with a float32
  two-sum instead took the FP64 pipe to 3 % and did not change the time: the
  kernel is issue-bound (3.7 instructions per cycle of 4). Computing each BF
  pixel's own geometry and chi(k) once per block removed a fifth of the
  instructions, bit for bit. The division-free chi of the fit kernel failed the
  float32 reference in `test_ssb_wave_average.py` (1.8e-6 rad against 1e-6):
  chi reaches about 13 rad here, where one float32 step is 1e-6 rad, so a
  reordered chi moves the phase by that much. The preview keeps
  `thick_geometry`'s chi expression and only takes the aperture denominator as
  a constant.
- **Loss path.** The mean-phase reconstruction behind the fit's native phase
  diagnostic and the slider-release loss copied every BF pixel's half plane to
  a full plane (two gathers, a conjugate, a copy), corrected it, ran a
  normalized inverse FFT and stored angle images for two reductions.
  `thick_correct_planes_kernel` writes the corrected full plane straight from
  the stored half plane, the inverse FFT runs unscaled (a power of two, which
  the phase does not see) and `thick_phase_moments_kernel` reduces the angles
  as it reads them. The loss is the same to ten digits.

## Results

Fits after the changes against the MPS reference (same tolerances as the
standard-fit note): every coefficient, tilt, depth spread, agreement and loss
passes; the largest phase difference at fixed parameters is 2.1e-5 rad
(tilt-corrected, 3x; 2.3e-5 before), correlation 0.999999999, and the losses are unchanged.

The shared benchmark protocol on this scan (warm; median of 3 fits or 20 previews; previews move C10 every call,
return host arrays and skip the loss unless stated; GPU idle before and after). "Before" is the CUDA port of the
standard fit with the previous kernels; MPS is the M5 Max run of the same protocol.

| Step | MPS | CUDA before | CUDA after |
|---|---|---|---|
| 1 load and detector-sampling fit (s) | | 0.49 | 0.44 |
| 2 session prepare: bright field, G(q, k) (s) | | 0.16 | 0.17 |
| 3 one tilt fit, rotation given (s) | 9.3 | 1.62 | 0.67 |
| 3b the same, `compare_standard=False` (s) | 6.3 | 1.21 | 0.53 |
| 4 default `reconstruct()`, both branches (s) | 17 to 18 | 3.60 | 1.51 |
| 5 `tilt=False` fit, rotation given (s) | | 0.60 | 0.25 |
| 6 drag preview, tilt-corrected, 1x (ms) | 104 | 33.9 | 23.6 |
| 7 drag preview, tilt-corrected, 3x (ms) | 158 | 53.7 | 36.1 |
| 8 release preview with the loss, 1x (ms) | 414 | 258 | 116 |
| 9 peak memory over steps 2 to 8, CuPy pool in use (GiB) | | 16.6 | 16.6 |

Step 3 by stage, before and after: standard comparison 0.55 and 0.15 s, search 0.48 and 0.20 s, Newton 0.48 and
0.21 s, native phase diagnostic 0.22 and 0.09 s. Each search and comparison batch is one pass of eight sets, about 6
to 8 ms of kernel plus the resident TPE's small kernels; the Newton passes are the same kernel. In the previews the
kernels take 93 to 99 % of the wall time: the wave sum 87 % at 1x (FFT and the host copy under 1 %) and 82 % at 3x
(the 1536 x 1536 host copy 8 %); with the loss, the correction 27 %, its inverse FFTs 43 % and the phase moments
10 %.

## Open

- `thick_wave_sum` keeps the precise `sinf` of the depth weights (as on Metal);
  a fast `__sinf` would remove most of its remaining transcendental work.
- The loss path's inverse FFT of full planes is now most of its time; the
  corrected spectrum of a real image is anti-Hermitian except at DC, so a
  half-plane complex-to-real transform would halve it.
