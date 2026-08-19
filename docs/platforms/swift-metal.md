# Native Swift and Metal

Use the native package when a macOS or iOS application needs direct HDF5 IO,
reusable Metal kernels, or already-transferred image products. Pin the
repository-root Swift package to an exact verified revision.

The package exposes:

- `Native4DSTEMIO` for discovery, inspection, QH5 indexing, and cache integrity;
- `Metal4DSTEMKernels` for decode, detector products, CoM, DPC, and iDPC;
- `MetalDisplayKernels` for range, histogram, colormap, and display math; and
- `MetalImageFFT` and `MetalImageRuntime` for resident 2D products.

The package contains no SwiftUI, AppKit, UIKit, or application state. A client
owns presentation, memory-policy choices, and scheduling but must not copy
`.metal` sources or redefine the scientific math.

```bash
swift test
swift run -c release metal-display-benchmark 512
```

Read [Native 4D-STEM IO](../api/native_4dstem_io.md) and
[native Metal image endpoints](../api/metal_image.md). Backend signoff includes
Swift tests, Metal compilation, frozen numerical parity, and physical-device
end-to-end evidence.
