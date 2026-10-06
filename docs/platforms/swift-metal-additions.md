# Native Swift/Metal additions

The [frozen platform overview](swift-metal.md) retains its evidence fingerprint.
`MetalScientificNumerics` additionally provides resident filtering, refined
correlation, translation, and weighted accumulation. See [native encoded inputs
and scaled output](../api/native_resident.md) for its infrastructure contracts.

## Source naming and ownership

Native source files use `Metal` plus the owning type or scientific operation.
Extensions use `Owner+Operation.swift`. This identifies both the API owner and
the work performed without introducing another public wrapper or module.

| File | Responsibility |
| --- | --- |
| `MetalImageOperations.swift` | Image buffers, execution context, FFT and correlation |
| `MetalImageOperations+Filtering.swift` | Gaussian filtering and gradients |
| `MetalImageOperations+TranslatedAccumulation.swift` | Translate and accumulate weighted resident regions |
| `MetalImageOperations+CalibratedDetectors.swift` | Virtual-detector reductions of calibrated intensities |
| `MetalEncodedSource+HDF5.swift` | Load HDF5 acquisitions into encoded residency, in the IO target |

App workflow sequencing, settings and preview/save UI stay in the consumer.
These filenames do not introduce different public types or duplicate kernels.
`loadEncoded` remains a deprecated forwarder for existing consumers; new callers
use `MetalEncodedSource.load(files:indexDirectory:device:)`.

Run `bash scripts/check_metal_scientific_numerics.sh` for the frozen image,
calibrated-detector and encoded-HDF5 checks without a consumer app or XCTest.
The corresponding suites are `MetalImageReferenceTests`,
`MetalCalibratedDetectorTests`, and `MetalEncodedHDF5LoadingTests`.
