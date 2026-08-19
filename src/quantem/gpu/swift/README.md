# Native Metal kernel package

The repository root `Package.swift` exposes raw Metal code to native macOS and
iOS applications. Applications import these products instead of copying Metal
sources into their own bundles.

Backend names remain explicit throughout QuantEM:

| Name | Runtime |
| --- | --- |
| CUDA | Native NVIDIA GPU compute |
| MPS | Python-facing Apple GPU compute |
| WebGPU | Browser GPU compute |
| Metal | Native Swift compute for macOS and iOS |

- `MetalDisplayKernels` owns image normalization, colormaps, histograms, and
  display shaders.
- `Metal4DSTEMKernels` owns fused QH5IDX HDF5 decoding, detector products, and
  the detector-word-major kernels used during interactive detector dragging.
- `MetalImageFFT` owns the Browser FFT endpoint
  `MetalImageFFT.logMagnitude`. The contract is
  `fftshift(log1p(abs(fft2(source))))` on a GPU-resident `MTLBuffer`.
- `MetalImageRuntime` owns histogram windows, range, and display contracts.
- `Native4DSTEMIO` owns Python-free HDF5/EMD discovery and QH5 indexing.
- `Metal4DSTEMLoadPlan` and `Metal4DSTEMStreamingPlan` own explicit load and
  scratch geometry. `Metal4DSTEMResidentCacheIO` owns cache integrity and
  provenance validation.
- `Benchmarks` and `Tests` exercise the same package resources imported by an
  application.

A native application keeps SwiftUI, cache policy, and gestures. It must call
these products instead of copying Metal source or launching Python. For the
public endpoint list, 512×512 BF/ADF FFT budget, and Torch MPS comparison, see
[Native Metal image endpoints](../../../../docs/api/metal_image.md).
The native load, audit, and cache contract is documented in
[Native 4D-STEM load and cache contract](../../../../docs/api/native_4dstem_io.md).

Typical Browser BF/ADF scans are `512×512`. Warm `logMagnitude` for that shape
must stay inside 8.33 ms (120 Hz). Measure with:

```bash
swift run metal-image-fft-benchmark 512 512 12
swift test --filter MetalImageFFTTests
```
