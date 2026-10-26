# Accurate MPS Fourier transforms for iterative waves

## Question

Can a complex64 Metal transform retain the existing multislice derivative
checks without moving waves to the CPU or changing the detector grid?

## Implementation

`quantem.gpu.device.mps_fft.fft2` runs on the PyTorch Metal command stream.
It retains three float32 terms through both Fourier axes, then rounds the
output to complex64. Its GPU adjoint uses the inverse transform and the
normalization required by `torch.fft`.

Supported axis lengths are powers of two or three times a power of two, from
2 to 1024. Rectangular grids, noncontiguous inputs and conjugate views are
covered. No spatial padding, detector binning or precision reduction is used.
Scalar twiddles are fixed metadata; waves and their gradients remain on GPU.
Safe Metal arithmetic and PyTorch 2.8 or newer are required.

This is an optional wave transform. SSB reconstruction, its estimator,
accumulation, calibration and existing backend dispatch remain unchanged.
CUDA retains its existing implementation.

## Validation

- 34 physical MPS cases cover forward/inverse transforms, normalization and
  adjoints, including 192, 512 and 1024 pixel axes.
- Three-slice 48 and 192 pixel checks retain the original normalized
  `atol=1e-6`, `rtol=1e-5` gate for intensities and phase/probe gradients.
- For the 192 pixel derivative checks, the previous transform failed the
  gate. The new transform passes, with maximum phase-gradient differences
  below 0.6 parts per million of the reference peak in the tested cases.
- Real Poisson-loss diagnostics reveal sensitivity near diffraction zeros.
  Neither complex64 backend meets a universal one-part-per-million gradient
  bound there. On two fixed measured inputs the new Metal transform gives
  smaller gradient errors than the existing complex64 CUDA transform against
  a complex128 CUDA control. This does not imply identical iterative paths.

The transform tests use bounded mathematical inputs. Measured reconstruction
and the higher-precision diagnostic run on GPU. Private arrays, original
failures, commands and comparisons belong to evidence `mps-live-fixes-20261018`.
