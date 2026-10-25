# Native Metal SSB redraw precision at the lowest scan frequencies, 2026-10-25

## Question

Is the native interactive SSB redraw (`ssb-redraw`, method `ssb`, 1–4×) within
twice the CUDA reference's own float32 error, measured against a float64
evaluation of the same definition, on every measured acquisition, fitted optics
set and output sampling, at the same drag speed?

## Workload and boundary

- Five measured 512 × 512 × 192 × 192 acquisitions at 300 kV and 30 mrad, with
  8,800–9,300 BF pixels: three at a 0.373 Å scan step (Auto 3×) and two at
  0.525 Å (Auto 4×). Eight fitted optics sets (C10 … C56, sample tilt, depth
  spread), each reconstructed at 1×, 2× and Auto: 24 images.
- References: the CUDA SSB (float32 corrections, error-free float32 sums met in
  float64, complex128 inverse FFT) and an independent float64 evaluation of the
  definition (float64 FFT of the counts, exact chi, complex128 sums and inverse
  FFT). e_C = rms(CUDA − float64), per image, 3.2e-6 to 8.7e-6 rad.
- Runner: `metal-ssb-fit-benchmark --fixture-parity CASE --oracle-dir DIR
  --methods ssb` (rms and maximum of the wrapped phase difference); timing with
  `metal-ssb-redraw-benchmark` as in the 2026-10-24 redraw report (synthetic
  counts on the 0.373 Å geometry, every BF pixel, all 14 terms, tilt, depth;
  20 warm calls with a changed C10). Apple M5 Max, macOS 26.4. Before = the
  same runners built from `f829cab4` with the same release flags; before and
  after alternated run by run under the shared GPU lock. Raw outputs are in the
  private evidence archive.

## Finding

v0.2 was 2.28 and 2.45 e_C from the float64 definition at 1× on the two
0.525 Å acquisitions (1.36–1.51 e_C at 1× on two others). The whole excess sat
at the first scan frequencies (frequency index ±1): 8.1e-6 rad of error power
there against CUDA's 2.5e-6, every other frequency band equal to CUDA's.

There the two sidebands nearly coincide. With
δ− = chi(k − q) − chi(k), δ+ = chi(k + q) − chi(k), μ = (δ+ − δ−)/2 and
ν = −(δ− + δ+)/2, the sideband difference is
gamma = exp(iμ) ((a − b) cos ν + i (a + b) sin ν), so the correction's
direction is set by μ and by the sign of the second difference
chi(k + q) + chi(k − q) − 2 chi(k). At index 1 of a 0.525 Å scan
(q = 0.0037 / Å) that second difference is a few 1e-5 rad when C10 and C12
partly cancel along the scan direction, while chi reaches tens of radians.
v0.2 formed δ± from separately rounded float32 chi values (a row-collapsed
absolute polynomial minus chi(k) from the probe kernel), errors of several
1e-6 rad, so the sign flipped for a sizable fraction of BF pixels, each flip
adding 2 |G(q, k)|. The 0.373 Å scans have twice the q² at index 1 and cancel
less. Trigonometry was not the cause: on the M5 Max `metal::fast::sincos` and
`metal::precise::sincos` return identical values (within 1.3e-7 of double for
|x| < 100), and swapping them changed no output bit. Accumulation order and the
float32 inverse FFT contribute equally to both implementations at every other
frequency.

## Change (v0.3)

- `redraw_taylor` (once per redraw, one thread per BF pixel) shifts the absolute
  7 × 7 chi polynomial to each pixel: δ±(q) = Σ d_ij(k) (±q_row)^i (±q_col)^j,
  stored coefficient-major; it also writes the pixel's probe geometry, replacing
  the probe dispatch. The threadgroup collapses the row power at ±q_row (plus
  and minus from one read) and Horner runs in ±q_col. Rounding is now
  proportional to δ± (about 1e-8 rad at index 1) instead of to chi, and chi(k)
  is no longer subtracted.
- Pairs with both sidebands possible form gamma from μ and ν as above, so the
  rounding of sin and cos is not amplified by 1/|gamma| where the sidebands
  nearly cancel. Single-sideband pairs, accumulation, mirror, origin and FFT are
  unchanged.

## Results

Worst over the optics sets of each acquisition; native − float64 / e_C and
native − CUDA / e_C (rms). The rule is ≤ 2.

| Acquisition | Step | 1× v0.2 | 1× v0.3 | 2× v0.2 | 2× v0.3 | Auto v0.2 | Auto v0.3 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 (three optics sets) | 0.373 Å | 1.20 / 1.08 | 0.95 / 1.11 | 1.13 / 1.00 | 1.07 / 1.01 | 1.13 / 1.00 | 1.08 / 1.01 |
| 2 | 0.373 Å | 1.36 / 1.61 | 0.96 / 1.13 | 1.14 / 1.26 | 0.98 / 1.06 | 1.14 / 1.26 | 0.99 / 1.06 |
| 3 | 0.373 Å | 1.51 / 1.67 | 0.81 / 1.01 | 1.18 / 1.30 | 0.90 / 1.03 | 1.17 / 1.29 | 0.90 / 1.02 |
| 4 | 0.525 Å | 2.28 / 2.20 | 0.74 / 1.07 | 1.45 / 1.40 | 0.93 / 0.97 | 1.34 / 1.25 | 0.95 / 0.93 |
| 5 (two optics sets) | 0.525 Å | 2.45 / 2.53 | 0.62 / 1.11 | 1.57 / 1.62 | 0.82 / 1.00 | 1.40 / 1.44 | 0.85 / 0.97 |

All 24 v0.3 images: native − float64 at most 1.08 e_C, native − CUDA at most
1.13 e_C, maximum absolute differences at most 0.57 × 10 e_C. At index ±1 the
error power fell from 8.1e-6 to 1.5e-7 rad (acquisition 4) and from 9.5e-6 to
4.5e-7 (acquisition 5), 7–17 times below CUDA's; the remaining error is the
high-frequency float32 floor, equal to CUDA's. Synthetic check
(`testRedrawResolvesNearlyCancellingSidebands`, C10 and C12 balanced along the
scan rows, |gamma| about 2e-5 at index 1): v0.2 4.5e-3 rad rms against the
double-precision definition, v0.3 3.7e-8.

Redraw time per frame, median of the per-run medians / median p90, v0.2 → v0.3,
runs alternating on the final builds (other agents' GPU work ran between runs):

| Scan | Order | Output | Rounds | v0.2 | v0.3 | Paired difference |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 512² | fifth | 3× | 12 | 30.5 / 31.3 | 30.8 / 31.4 | +0.1 ms |
| 512² | low | 3× | 8 | 40.3 (loaded; minimum 29.4) | 39.9 (minimum 29.8) | −0.1 ms |
| 256² | fifth | 3× | 8 | 10.1 / 10.3 | 10.3 / 10.7 | +0.3 ms |
| 512² | fifth | 1× | 6 | 23.9 | 23.7 | +0.1 ms |

At 512² the difference is within the run-to-run noise; at 256², where one wave
of threadgroups exposes the per-batch collapse from device memory (constant
memory before), v0.3 costs about 3 %. The 512² fifth-order 3× drag stays
within 33 ms per frame.

## Conclusion

The native SSB redraw now meets the 2× rule everywhere: below CUDA's own error
on 20 of 24 images and within 8 % of it on the rest, at the same drag speed at
512² (about 3 % slower at 256²). v0.3 replaces v0.2 with no fallback. The
in-app headed drag timing is still to be recorded.
