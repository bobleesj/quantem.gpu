# Native Swift and Metal package

The repository root `Package.swift` exposes raw Metal code to native macOS and
iOS applications. Applications import these products instead of copying Metal
sources into their own bundles.

## Products and source ownership

Run Swift commands from the repository root. The root
[Package.swift](../../Package.swift) is the package manifest; there is no
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
native/swift/
├── Sources/       # product targets, C bridges, and packaged Metal resources
├── Tests/         # native contracts, parity, resources, and opt-in checks
├── Benchmarks/    # executables using those same product targets
└── Vendor/        # bundled HDF5 framework and associated licensing
```

This tree is not part of the Python wheel. Metal sources and data that the
Python package also reads live once inside it, beside the Python that reads
them, and `Package.swift` copies them into the resource bundles:
[colormaps.json](../../src/quantem/gpu/display/colormaps.json) and
[display.metal](../../src/quantem/gpu/display/metal/display.metal) for
`MetalDisplayKernels`; `qh5idx.metal` in
[io/hdf5/mps/kernels](../../src/quantem/gpu/io/hdf5/mps/kernels) and
`runtime_spatial.msl` in
[resident/mps/kernels](../../src/quantem/gpu/resident/mps/kernels) for
`Metal4DSTEMKernels`; `save_uint16.msl` from `io/hdf5/mps/kernels` and
`streamed_counts.msl`, `count_tables.msl`, `hot_pixels.msl` and `precision.msl`
from `resident/mps/kernels` for `MetalCountResources`, whose Swift-only sources
sit in `Sources/MetalCountResources/Resources`.

`Metal4DSTEMLoadPlan` and `Metal4DSTEMStreamingPlan` declare load/scratch
geometry. `Metal4DSTEMResidentCacheIO` validates cache integrity and provenance.
Packed counts, dense counts, and derived SSB Fourier storage are different
objects: support for one does not establish support for the others.
`MetalANSResidentSource` accepts validated ANS arrays for exact DP and binary
mask sums. Saved `.qem` copies open through
`MetalRuntimeANSResidentSource.load(snapshot:device:)`. See the
[ANS acquisition acceptance gate](../../docs/maintainer/ans-io-acceptance.md)
for qualification.

A native application keeps SwiftUI, cache policy, and gestures. It must call
these products instead of copying Metal source or launching Python. For the
public endpoint list and FFT contract, see
[Native Metal image endpoints](../../docs/api/metal_image.md).
The native load, audit, and cache contract is documented in
[Native 4D-STEM load and cache contract](../../docs/api/native_4dstem_io.md).

Original HDF5 detector interactions operate on the lossless packed count
resident. Normal builds do not add a resident detector-region sum cache or
retain a dense 4D copy. Detector outputs and bounded interaction buffers remain
accounted for separately from the packed counts. An optional sum-cache experiment
is available only in explicitly instrumented builds; it is not the default
memory or performance contract.

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
the [benchmark dashboard](../../docs/dashboard.md), not this source guide.
