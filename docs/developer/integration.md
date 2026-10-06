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
- `MetalSSBEngine` for exact native 512×512 SSB reconstruction,
  phase-variance evaluation, and deterministic 200-trial TPE plus Nelder–Mead
  fitting. See [native SSB](#native-swift-and-metal).

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

`MetalSSBEngine` consumes exact plane-major BF columns with layout
`[logical_brightfield, scan_row, scan_column]`, source dtype `uint8` by default
(`prepare(brightfield:countType:)` also accepts `uint16` and `uint32`), and square
scan shapes 128×128, 256×256, or 512×512. Geometry must be complete and finite,
and the caller must provide a sufficiently large Metal buffer. Invalid input
raises rather than silently cropping, binning, changing precision, or falling
back to CPU. The engine computes in float32/complex64. It retains every
logical BF term in normalization and skips only the proven-zero aperture union.

```swift
import Metal
import MetalSSBKernels

let device = MTLCreateSystemDefaultDevice()!
let engine = try MetalSSBEngine(
  device: device,
  geometry: calibratedGeometry,
  cacheBudgetBytes: availableSSBCacheBytes
)

try engine.prepare(brightfield: planeMajorUInt8Buffer)
let result = try engine.reconstruct(
  aberrations: MetalSSBAberrations(
    c10Nanometers: 72.98,
    c12Nanometers: 14.4,
    phi12Radians: 0.4686
  )
)
```

`result.object` and `result.fourierSum` are row-major complex64 Metal
buffers at the native 128×128, 256×256 or 512×512 scan size. Their
`result.provenance` records scan shape, source/compute dtype, scan bin
1, no scan crop, logical/executed/zero-aperture BF counts, cached/streamed BF
counts, and cache bytes. `phaseVariance(...)` evaluates the same complete
objective. `optimize(...)` defaults to 200 seeded TPE trials followed by
Nelder–Mead and returns the full trial record.

### Optional BF sampling for native optimization

`phaseVariance` and `optimize` accept `brightfieldFraction`, default **1.0**.
This fraction selects detector BF pixels for the objective, not scan positions,
aperture radius, detector binning or the final image resolution. Full aperture
remains the default throughout search and Nelder-Mead refinement.

```swift
let fit = try engine.optimize(
  start: initialAberrations, globalTrials: 200, brightfieldFraction: 0.25)
let final = try engine.reconstruct(aberrations: MetalSSBAberrations(
  c10Nanometers: Float(fit.best.c10Nanometers),
  c12Nanometers: Float(fit.best.c12Nanometers),
  phi12Radians: Float(fit.best.phi12Radians)))
```

Fractions below one intentionally approximate the full objective and may change
the fitted aberrations. They do not produce an equivalent full-BF loss. The
selection is uniform without replacement, fixed at seed 42, sorted into source
order and reused across all objective evaluations. Its size is the rounded
fraction of the logical BF selection, with at least two pixels (or all pixels
if fewer exist). Zero-aperture entries remain part of that logical normalization;
a subset with no active contribution fails with an instruction to increase it.
No intensity-based or central-disk-only selection is made.

The engine skips unselected cached or streamed columns, batches adjacent selected
columns, and reuses the full prepared cache. It does not allocate a second
Fourier-volume cache. Initial full evidence preparation and memory use are not
reduced. `reconstruct` is unchanged and always uses the engine's full selected
aperture, including after a sampled fit. Using fewer objective pixels is not a
kernel speedup or a guarantee of proportional total-workflow acceleration.

`SSBOptimizationResult.brightfieldSampling` stores the policy version, requested
fraction, total BF count and exact selected logical indices. Phase-variance
results carry the same selection and selected-work provenance. Saved native
runs retain this record; applications should label sampled losses explicitly
and retain selection provenance when exporting a phase-only result.

`scripts/check_metal_ssb_bf_sampling.sh` compares sampled loss against explicitly
constructed subset inputs at all three native sizes in cached, streamed and
hybrid modes, checks saved-fit selection, and checks bit-identical full
reconstruction before/after sampling. The unchanged 100% path is additionally
covered by `scripts/check_metal_ssb_scan_sizes.sh` against the frozen CUDA and
independent-equation fixtures. These fixtures validate implementation, not the
scientific quality of fitting a particular experiment with fewer BF pixels.

`cacheBudgetBytes: nil` requests a complete Hermitian `G(k)` cache. A finite
budget caches whole 32-BF batches and streams the remaining terms exactly.
Cache policy is an application decision: the application must choose and show
the memory policy, while QuantEM.GPU owns the estimator inputs, exact kernels,
and provenance. The Swift package does not own windows, controls, sessions, or
plots.
