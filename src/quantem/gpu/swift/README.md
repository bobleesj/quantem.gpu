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
- `Benchmarks` and `Tests` exercise the same package resources imported by an
  application.

A native application retains platform-specific document and UI handling. These
targets intentionally expose kernels rather than claiming to be a complete
loader. Shared loading, cache, and command orchestration should move into a
separate Metal runtime target before macOS and iOS applications both need it.
