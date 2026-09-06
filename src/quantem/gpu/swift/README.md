# Native Swift and Metal package

The repository root `Package.swift` exposes raw Metal code to native macOS and
iOS applications. Applications import these products instead of copying Metal
sources into their own bundles.

## Products and source ownership

Run Swift commands from the repository root. The root
[Package.swift](../../../../Package.swift) is the package manifest; there is no
second manifest in this directory.

| Product | Responsibility |
|---|---|
| [Native4DSTEMIO](Sources/Native4DSTEMIO) | Python-free HDF5/EMD discovery, source indexing, value-range audit, packed producer, and cache integrity |
| [Metal4DSTEMKernels](Sources/Metal4DSTEMKernels) | Decode/bin plans and shaders for counts, detector products, packed/ANS consumption, and DPC |
| [Metal4DSTEMStreamingIO](Sources/Metal4DSTEMStreamingIO) | Indexed/sharded loading, packed residents, bounded ANS-array consumers, and explicit buffer ownership |
| [MetalDisplayKernels](Sources/MetalDisplayKernels) | Image normalization, colormaps, histograms, and display shaders |
| [MetalImageFFT](Sources/MetalImageFFT) | Resident `logMagnitude`: `fftshift(log1p(abs(fft2(source))))` |
| [MetalImageRuntime](Sources/MetalImageRuntime) | Histogram windows, range, and display contracts |
| [MetalSSBKernels](Sources/MetalSSBKernels) | SSB reconstruction and optimizer kernels |

```text
swift/
├── Sources/       # product targets, C bridges, and packaged Metal resources
├── Tests/         # native contracts, parity, resources, and opt-in checks
├── Benchmarks/    # executables using those same product targets
└── Vendor/        # bundled HDF5 framework and associated licensing
```

`Metal4DSTEMLoadPlan` and `Metal4DSTEMStreamingPlan` declare load/scratch
geometry. `Metal4DSTEMResidentCacheIO` validates cache integrity and provenance.
Packed counts, dense counts, and derived SSB Fourier storage are different
objects: support for one does not establish support for the others. The new
`MetalANSResidentSource` accepts validated ANS arrays for exact DP and binary
mask sums; reading the canonical standalone ANS file is still pending. See the
[count-IO checklist](../../../../IO-REPRESENTATION-CHECKLIST.md) for qualification.

A native application keeps SwiftUI, cache policy, and gestures. It must call
these products instead of copying Metal source or launching Python. For the
public endpoint list and FFT contract, see
[Native Metal image endpoints](../../../../docs/api/metal_image.md).
The native load, audit, and cache contract is documented in
[Native 4D-STEM load and cache contract](../../../../docs/api/native_4dstem_io.md).

## Build, test, and measure

```bash
swift build -c release
swift test
swift test -c release --filter MetalSSBKernelsTests
```

A `512×512` BF/ADF FFT has a target budget of 8.33 ms for a 120 Hz interaction;
this is a target, not a universal measured result. Measure its package endpoint:

```bash
swift run metal-image-fft-benchmark 512 512 12
swift test --filter MetalImageFFTTests
```

Record load, first-use compilation, resident kernel time, peak memory, and app
presentation separately. Host compilation and skipped opt-in tests are not
physical-device or application acceptance. Latest dated measurements belong in
the [benchmark dashboard](../../../../docs/dashboard.md), not this source guide.
