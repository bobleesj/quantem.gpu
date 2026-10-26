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
| [MetalDirectPtychoKernels](Sources/MetalDirectPtychoKernels) | Direct Ptycho: Fourier preparation, aberration fit, and the SSB, Phase and Complex reconstructions |
| [MetalSSBKernels](Sources/MetalSSBKernels) | Deprecated former name of MetalDirectPtychoKernels: re-exports it with deprecated aliases for the renamed types |

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

## Staged SSB fitting

`DirectPtychoModelFit.findAberrations` runs 200 low-order TPE trials, then a
trust-region Newton on the exact gradient and Hessian of the agreement (C10,
C12, phi12, specimen tilt and depth spread). It then refines the aberrations
through C56 with the tilt and depth spread together: an exact-gradient
trust-region BFGS, first on every 8th BF pixel and then on all of them. Each
of the 22 higher-order coordinates is bounded to ±20 rad of probe phase at the
aperture edge (`DirectPtychoFitOptions.higherOrderLimit`; 2 is the standard box). A ±20
fit is refitted on two halves of the band frequencies and falls back to ±2
when the halves disagree (`checkHigherOrderConsistency`). Rotation is selected
using the low-order fits and held during the fifth-order stage.
`DirectPtychoFitOptions(maximumOrder: 1)` stops after the low-order fit, and
`refineToFifthOrder` refines a shown low-order fit later.

```swift
let fit = try DirectPtychoModelFit.findAberrations(
  model: model, rotationDegrees: rotation,
  darkField: darkField, scanSampling: scanSampling,
  onInitialFit: { initial in
    // Hand this Auto reconstruction to the application's display queue.
    let preview = try model.reconstruct(initial.point, upsamplingFactor: autoFactor)
    publishPreview(preview)
  })
let phase = try model.reconstruct(fit.fit.point, upsamplingFactor: autoFactor)
```

Scientific objectives, their exact derivatives and reconstruction run in Metal
with all selected BF pixels. The controllers keep their small Hessians and
bounded trust steps in host Double arithmetic, and the fifth-order search
accepts and stops on Double-summed agreements. Fitting uses the native scan
grid and the strict `0.2 < |q| < 0.9 Å⁻¹` band. Auto sampling applies to
displayed reconstruction. The fifth-order search stops when a step predicts no
resolvable gain, at 400 full-data objective passes or at 200 iterations. A
lower objective is not the target: this fit maximizes model agreement.

`sampleAgreement` is the fraction of G's power in the fit band that the model
explains, from 0 to 1 (the raw agreement `rawSampleAgreement` over
`bandPower`), as Python's `sample_agreement`; fits saved before it decode with
their raw agreement only. `higherOrderRefinement` records the stopping reason,
the objective and gradient passes, and the best-agreement trace. Divide the raw
agreement by `higherOrderNormalization` to compare with that trace. Native saved joint
results retain the exact Double model point, independent of rounded UI
controls. An imported phase alone does not establish support for editing all of
its originating model parameters.

## Build, test, and measure

```bash
swift build -c release
swift test
swift test -c release --filter MetalDirectPtychoKernelsTests
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
the [verified benchmark results](../../docs/performance/results.md), not this source guide.
