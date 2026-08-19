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
| `MetalDisplayKernels` | range, histogram, transfer function, colormap | `.../MetalDisplayKernels` |
| `MetalImageFFT` | FFT operations on resident 2D products | `.../MetalImageFFT` |
| `MetalImageRuntime` | typed resident surface/statistics state | `.../MetalImageRuntime` |

The package imports no application UI framework. A client owns presentation,
scheduling, and user-visible memory-policy choices; it must not copy or fork
the `.metal` sources.

## Coordinate and buffer contract

Public geometry uses `(row, column) ≡ (r, c)`. Swift properties such as scan
rows/columns and detector rows/columns preserve
$I[s_r,s_c,q_r,q_c]`, even when a Metal buffer uses a private flattened or
detector-major stride.

Load plans record source/output shapes, half-open regions, scan/detector bins,
source/accumulation/output dtypes, bad-pixel policy, and expected bytes. An
automatic detector-bin choice belongs to client policy and must remain visible
in provenance.

## Build, profile, and verify

```bash
swift test
swift run -c release native-4dstem-io-benchmark --help
swift run -c release metal-display-benchmark 512
```

Profile physical-device wall time and Metal command intervals. For unified
memory, distinguish mapped page-in from explicit copies. Acceptance includes
Swift tests, Metal compilation, frozen CPU/CUDA cross-backend fixtures,
rectangular row/column cases, memory-budget failures, and real-device
cold/warm/prepared timings.

Read [Native 4D-STEM IO](../api/native_4dstem_io.md) and
[Metal image APIs](../api/metal_image.md) for public types.
