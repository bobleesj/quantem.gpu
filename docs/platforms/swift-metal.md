# Native Swift and Metal

Use the repository-root Swift package when a macOS or iOS codebase needs direct
HDF5 IO, reusable Metal kernels, or operations on already-resident products.
Pin an exact verified package revision.

## Why Swift is a separate tree

Swift Package Manager requires target-oriented sources, bundled `.metal`
resources, tests, benchmarks, and native library products. This is a packaging
boundary, not separate scientific ownership. The Swift products implement the
same operations, coordinates, dtypes, binning, and provenance as the Python and
WebGPU paths.

## Products and sources

| Product | Responsibility | Source |
|---|---|---|
| `Native4DSTEMIO` | discovery, QH5 indexing, HDF5 access, identities, caches, audits | `src/quantem/gpu/swift/Sources/Native4DSTEMIO` |
| `Metal4DSTEMKernels` | load plans, decode, binning, BF/DF/ADF, CoM, DPC, iDPC primitives | `.../Metal4DSTEMKernels` |
| `Metal4DSTEMStreamingIO` | bounded QH5 decode, exact `uint64` products, source audit, on-demand native frames | `.../Metal4DSTEMStreamingIO` |
| `MetalDisplayKernels` | range, histogram, transfer function, colormap | `.../MetalDisplayKernels` |
| `MetalImageFFT` | FFT operations on resident 2D products | `.../MetalImageFFT` |
| `MetalImageRuntime` | typed resident surface/statistics state | `.../MetalImageRuntime` |
| `MetalSSBKernels` | exact native 512×512 SSB reconstruction, phase-variance objective, and deterministic fitting | `.../MetalSSBKernels` |

The native package does not provide the Python `screening.prepare` cache
contract. It does provide detector, CoM, DPC/iDPC, display, FFT, load, and SSB
primitives that a native client can compose without changing their scientific
meaning.

## Call and resource path

```text
consumer imports one SwiftPM product
  → public Swift value types validate geometry and provenance
  → product creates or reuses Metal pipelines and buffers
  → bundled .metal resource writes a typed destination buffer
  → caller receives the result without an application-framework dependency
```

`Package.swift` is the build graph. Each library target has one matching source
directory under `src/quantem/gpu/swift/Sources`. Metal resources are copied by
SwiftPM and resolved from their owning module bundle; clients do not compile a
second copy.

| Product | Kernel/native dependency | Tests |
|---|---|---|
| `Native4DSTEMIO` | `CNativeHDF5` → vendored `CHDF5.xcframework` and zlib | `Native4DSTEMIOTests` plus real-source benchmarks |
| `Metal4DSTEMKernels` | `detector.metal`, `dpc.metal`, `qh5idx.metal` | `Metal4DSTEMKernelsTests` |
| `Metal4DSTEMStreamingIO` | `Native4DSTEMIO` plus `Metal4DSTEMKernels` | synthetic compressed-fixture parity, overflow, cancellation, and opt-in real-source gates |
| `MetalDisplayKernels` | `display.metal`, packaged colormaps | `MetalDisplayKernelsTests` |
| `MetalImageFFT` | `fft.metal`, MPS/MPSGraph frameworks | `MetalImageFFTTests` |
| `MetalImageRuntime` | `MetalDisplayKernels` | `MetalImageRuntimeTests` |
| `MetalSSBKernels` | custom radix-8 FFT, Hermitian SSB reconstruction, fused exact loss | `MetalSSBKernelsTests` plus `metal-ssb-benchmark` |

The package imports no application UI framework. A client owns presentation,
scheduling, and user-visible memory-policy choices; it must not copy or fork
the `.metal` sources.

## Coordinate and buffer contract

Public geometry uses `(row, column) ≡ (r, c)`. Swift properties such as scan
rows/columns and detector rows/columns preserve
$I[R_r,R_c,k_r,k_c]`, even when a Metal buffer uses a private flattened or
detector-major stride.

Load plans record source/output shapes, half-open regions, scan/detector bins,
source/accumulation/output dtypes, bad-pixel policy, and expected bytes. An
automatic detector-bin choice belongs to client policy and must remain visible
in provenance.

## Exact resident summaries

`Native4DSTEMIO` can persist compact, exact products and sufficient statistics
beside a sealed resident cache. This is a prepared-product boundary: it is not a
compressed-source load, and a client must label it accordingly.

| Public type | Contract |
|---|---|
| `Metal4DSTEMResidentSummaryRole` | Names BF, ABF, ADF, total intensity, detector row/column moments, and selected diffraction artifacts |
| `Metal4DSTEMResidentSummaryMetadata` | Schema, source/resident identity, output shape/dtype, scan region/bin, detector bin, count audit, detector bands, selected scan coordinate, and artifact descriptors |
| `Metal4DSTEMResidentSummary` | Validated metadata plus exact artifact bytes |
| `Metal4DSTEMResidentSummaryIO.write(...)` | Atomically creates a new `quantem.gpu.resident-summary/v1` directory; never overwrites an existing summary |
| `Metal4DSTEMResidentSummaryIO.read(...)` | Fails closed on identity, geometry, dtype, bin, audit, size, or SHA-256 mismatch |

The fused
`detector_products_u16_word_major_with_u64_moments` Metal kernel traverses an
exact detector-bin-4 `uint16` resident volume once. It writes BF/ABF/ADF as
`uint32` and total, detector-row, and detector-column moments as `uint64`; the
wider moment dtype prevents overflow before CoM, DPC, or iDPC derivation.

`quantem.gpu` owns the reusable integer artifacts, the reduction kernel, and
strict provenance validation. A native client owns discovery, admission,
scheduling, eviction, memory-pressure response, and the user-visible reason for
any detector bin or prepared reopen. The client must not describe a summary
reopen as a first source encounter or a binned detector as native resolution.
See [Verified benchmark results](../performance/results.md) for the physical
Phil and 8 GB M2 Air measurements; this platform page does not duplicate them.

`MetalSSBEngine` is intentionally narrower than the Python source loader. It
accepts plane-major lossless `uint8` bright-field columns with shape
`[logical_brightfield, scan_row, scan_column]` for a 512×512 scan. It never
crops or bins scan positions. Every result records the logical, executed,
cached, streamed, and proven-zero BF counts plus the exact cache bytes. A client
chooses `cacheBudgetBytes` from its resource policy; a smaller cache streams the
remaining BF terms without changing the objective or normalization.

## Build, profile, and verify

```bash
swift test
swift run -c release native-4dstem-io-benchmark --help
swift run -c release metal-4dstem-indexed-load-benchmark --help
swift run -c release metal-display-benchmark 512
swift run -c release metal-ssb-benchmark \
  METADATA_JSON FULL_BF_U8 REFERENCE_PHASE_F32 7 full 200 3
```

Profile physical-device wall time and Metal command intervals. For unified
memory, distinguish mapped page-in from explicit copies. Acceptance includes
Swift tests, Metal compilation, frozen CPU/CUDA cross-backend fixtures,
rectangular row/column cases, memory-budget failures, and real-device
cold/warm/prepared timings.

Swift tests verify the package boundary; Python parity fixtures adjudicate
cross-runtime numerical meaning. Passing only one of those layers is not a
complete Swift/Metal signoff.

Read [Native 4D-STEM IO](../api/native_4dstem_io.md) and
[Metal image APIs](../api/metal_image.md) for image and IO types. Read the
[SSB API](../api/ssb.md) for `MetalSSBGeometry`, `MetalSSBEngine`, result
provenance, and the application boundary.
