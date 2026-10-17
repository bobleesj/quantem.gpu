# MPS SSB coherent Fourier support — 17 October 2026

Experiment identifier: `mps-ssb-coherent-support-20261017`. Raw measurements,
images and acquisition provenance are retained as private evidence.

The question was whether full-quality interactive phase reconstruction could
avoid work on exactly closed apertures and identically zero Fourier support.
The test used a 512 × 512 scan, 192 × 192 detector, all 4,697 active BF pixels,
2× sampling and a 1024 × 1024 float32 phase-of-mean output on an Apple M5 Max
with 128 GB memory. Prepared source spectra stayed complex64.

The MPS implementation skips sinc and precise sine/cosine evaluation only for
an exactly closed shifted aperture. It stores partial coherent spectra inside
conservative nonzero aperture support, with an outward rounding margin, and
restores the full Fourier grid before the existing inverse transform. Detector
groups and sequential float32 accumulation order are unchanged. This does not
crop source data, discard BF pixels or reduce output sampling.

## Matched measurements

The baseline was `5fba77a2`. Alternating 30 synchronized full-phase evaluations
per version in one process gave:

| Metric | Baseline | Optimized |
| --- | ---: | ---: |
| Median full-phase evaluation | 24.52 ms | 18.49 ms |
| p95 full-phase evaluation | 25.43 ms | 18.88 ms |
| Total partial-spectrum bytes | 941,359,104 | 400,415,232 |
| Notebook distinct images/s, 30 Hz requests | 25.42 | 27.49 |
| Notebook distinct images/s, 60 Hz requests | 18.91–19.04 | 22.68–26.95 |

Notebook measurements used the same full image, visible FFT and a defocus
sweep. They count distinct reconstructed phase revisions drawn to the canvas
at the next animation callback, not physical display scan-out. The faster
request setting did not improve delivered throughput. Neither the notebook nor
standalone reconstruction meets the complete 16.67 ms budget for 60 fps.

Synchronized candidate stage medians were 15.86 ms for coherent alias work,
1.12 ms for the ordered sum and approximately 0.20 ms each for probe and group
bounds. Instrumented stages are diagnostic and cannot be directly summed into
an uninstrumented pipeline timing. Cold loading and preparation were unchanged
and were not rebenchmarked as optimization targets.

## Validation and limits

Ten frozen full phase images across defocus, astigmatism, both tilt axes and
depth were bitwise identical. Three native losses were identical. Two regional
five-seed fits with shared optics produced identical per-seed parameters,
agreement/bound flags and regional phase arrays. Their complete elapsed times
were 5.737 and 5.061 seconds; this is not a fitting-kernel speedup claim.

The focused GPU suite passed 90 tests with two CUDA-only skips. Added exact
comparisons against dense Fourier support cover native scan sizes 128, 256,
512 and 1024, anisotropic detector calibration and higher aberration orders.
The native Metal workflow passed, including its 200-trial fit and saved-result
round trip. Its separate external frozen fit-trajectory gate remains unverified
because the acquisition fixture was unavailable. References and tolerances
were not changed. CUDA and WebGPU performance were not measured.

Avoiding additional aperture geometry showed no useful improvement. Seven
launch shapes preserved exact outputs but did not justify replacing the
existing 32 × 8 threadgroup. Complex scatter was unsupported by MLX, so the
implementation restores zero gaps by concatenation.
