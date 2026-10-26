# Native application integration

This guide is for applications that call native products directly. Python users
can use the [Python workflow](../python-workflow.md) and public API without
setting up these products.

## Choose the task

| Task | Contract |
|---|---|
| Discover and load an acquisition | [Native loading](../api/native_4dstem_io.md) |
| Map vendor fields and calibration | [Acquisition formats](../api/native-acquisition-formats.md) |
| Read encoded inputs or write scaled output | [Encoded I/O](../api/native_resident.md) |
| Calculate histograms, ranges, or display FFTs | [Image operations](../api/metal_image.md) |
| Reconstruct SSB and search aberrations | [Native SSB below](#native-swift-and-metal) |
| Implement a portable file reader | [File formats](file-formats.md) |

## Public native products

Native Swift/Metal products for macOS and iOS clients:

- `MetalEncodedSource`, `MetalPrecision`, `MetalHDF5Writer`, and
  `MetalPackedSource` for encoded counts and bounded scaled-output storage;
  `MetalScientificNumerics` for reusable image and sampling operations. See
  [native encoded inputs and scaled output](../api/native_resident.md).

- `MetalImageFFT.logMagnitude` for Browser FFT of an already-transferred 2D
  product. See [Native Metal image endpoints](../api/metal_image.md).
- `MetalImageRuntime` for histogram, range, and display contracts.
- `Native4DSTEMIO` for Python-free HDF5/EMD discovery, prepared QH5 indexes,
  and bounded native frame windows.
- `MetalCompactH5Loader.load(source:device:)` for original HDF5 directly into
  exact packed Metal residency, without a dense 4D allocation. See
  [original HDF5 loading and reload benchmarks](../api/original-hdf5-metal-packing.md).
- `NativeLosslessPackV1Producer` for an authenticated, resource-planned,
  cancellable original-HDF5-to-lossless-pack lifecycle. See the
  [native Lossless Pack Format v1 producer contract](../api/native_lossless_pack_v1_producer.md).
- `Metal4DSTEMStreamingIO` for bounded native QH5 decode, exact `uint64`
  products, source audits, and on-demand full-resolution diffraction frames.
- `Metal4DSTEMLoadPlan`, `Metal4DSTEMStreamingPlan`,
  `Metal4DSTEMResidentCacheIO`, and `Metal4DSTEMResidentSummaryIO` for explicit
  native load, resource, resident-cache, and exact prepared-product provenance.
  See [Native 4D-STEM load and cache contract](../api/native_4dstem_io.md).
- `MetalDirectPtychoEngine` and `MetalDirectPtychoJointModel` for exact native 128×128, 256×256 and
  512×512 SSB: the engine prepares the Fourier evidence, and the joint model fits
  (`DirectPtychoModelFit`) and reconstructs SSB, Phase and Complex. See
  [native SSB](#native-swift-and-metal).

Native clients call these endpoints directly. They are not a local Python
backend.

The dated [experimental native Metal entropy-series SPI](../api/experimental_metal_entropy_series.md)
is a separate opt-in prepared-archive consumer. It documents the 2026-09-08
implementation, exact-count contract, measured limits and missing encoder/API
gates; it is not part of the stable entry points above.

## Browser boundary

WebGPU mirrors reconstruction, phase, and the exact-loss contract asynchronously.
It does not currently implement aberration search. See the [WebGPU backend](../platforms/webgpu.md)
for the supported input and execution contracts. A browser runtime is separate
from the Python MPS and CUDA backends.

(native-swift-and-metal)=
## Native Swift and Metal

`MetalDirectPtychoEngine` consumes exact plane-major BF columns with layout
`[logical_brightfield, scan_row, scan_column]`, source dtype `uint8` by default
(`prepare(brightfield:countType:)` also accepts `uint16` and `uint32`), and square
scan shapes 128×128, 256×256, or 512×512. Geometry must be complete and finite,
and the caller must provide a sufficiently large Metal buffer. Invalid input
raises rather than silently cropping, binning, changing precision, or falling
back to CPU. The engine computes in float32/complex64. It retains every
logical BF term in normalization and skips only the proven-zero aperture union.

```swift
import Metal
import MetalDirectPtychoKernels

let device = MTLCreateSystemDefaultDevice()!
let engine = try MetalDirectPtychoEngine(
  device: device,
  geometry: calibratedGeometry,
  cacheBudgetBytes: availableSSBCacheBytes
)

try engine.prepare(brightfield: planeMajorUInt8Buffer)
let model = try engine.jointModel(scanDetectorRotationDegrees: rotation)
let fit = try DirectPtychoModelFit.findAberrations(
  model: model, rotationDegrees: rotation,
  darkField: darkField, scanSampling: scanSampling)
let phase = try model.reconstruct(fit.fit.point, upsamplingFactor: autoFactor)
```

The engine Fourier-transforms every active BF column once and keeps the
Hermitian spectra; the joint model fits the model agreement on them and
reconstructs at the automatic output sampling. `phase.provenance` records scan
shape, source/compute dtype, scan bin 1, no scan crop, logical/executed/
zero-aperture BF counts, cached/streamed BF counts, and cache bytes. The
staged fit is described in `native/swift/README.md`.

The engine's own non-model reconstruction (`reconstruct(aberrations:)`), its
phase-variance objective (`phaseVariance`), the seeded TPE and Nelder–Mead search
over it (`optimize`, `SSBOptimizer`) and its optional BF sampling are retired.
`DirectPtychoFitOptions(objective: .phaseVariance)` still decodes from saved runs and is
refused as a fit.

`cacheBudgetBytes: nil` requests a complete Hermitian `G(k)` cache. A finite
budget caches whole 32-BF batches and streams the remaining terms exactly.
Cache policy is an application decision: the application must choose and show
the memory policy, while QuantEM.GPU owns the estimator inputs, exact kernels,
and provenance. The Swift package does not own windows, controls, sessions, or
plots.

### Saved native runs

`MetalDirectPtychoJointSavedRun` saves a joint-model result: the phase, the exact model
point, calibration, sampling, region and fit record.

`MetalSSBSavedRun` is the format of the retired non-model engine. It records
both complex64 images, calibrated geometry, aberrations, rotation, fit history,
seed, source identity, backend revision, reconstruction timings and provenance
for a 128×128, 256×256 or 512×512 scan, in a versioned binary property list with
SHA-256 checksums for the two image payloads. Runs saved with it still open:

```swift
let previous = try MetalSSBSavedRun.load(
    from: resultURL, matchingSourceIdentity: sourceSHA256)
let images = try previous.reconstruction(device: device)
let phase = try MetalDirectPtychoEngine(
    device: device, geometry: previous.geometry, cacheBudgetBytes: 0
).phase(of: images)
```

Loading validates the requested source identity and both payloads, restores
the saved calibration, and does not apply current application overrides to an
old result. No optimizer or raw-source decode runs during image restoration.
Recorded reconstruction timings are not reopening timings. Editing such a run
reconstructs it with the joint model.
