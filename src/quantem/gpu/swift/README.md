# QuantEM GPU Swift package

The repository root `Package.swift` exposes reusable Apple-GPU code to native
applications. Applications import these products instead of copying Metal
sources into their own bundles.

- `QuantEMMetalDisplay` owns image normalization, colormaps, histograms, and
  display shaders.
- `QuantEM4DSTEMMetal` owns fused QH5IDX HDF5 decoding, detector products, and
  the detector-word-major kernels used during interactive detector dragging.
- `Benchmarks` and `Tests` exercise the same package resources imported by an
  application.

A macOS application should retain document handling, cache policy, SwiftUI,
and Metal command orchestration. Reusable kernels and stable entry-point names
belong here so Python, benchmarks, tests, and native clients have one canonical
implementation.
