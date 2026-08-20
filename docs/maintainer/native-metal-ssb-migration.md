# Native Swift/Metal SSB migration

This record freezes the extraction of reusable native SSB compute from the
earlier iOS implementation into QuantEM.GPU. It does not migrate application
UI, session state, navigation, plots, or cache-policy presentation.

## Source lineage

| Role | Repository state |
|---|---|
| Original native implementation | `Live4DSTEM-iOS` `main` at `55e303a66ce35255cfe1e96d5f7448e99eb09405`, clean and tracking `origin/main` |
| Original SSB sequence | `a949ec71` raw-master calibration → `b279516` parity validation → `124fa856` full-BF acceleration → `adae34ab` fused exact calibration → `55e303a6` architecture refactor |
| QuantEM.GPU base | `platform-parity-profile-integration` at `4ff3db4433a3ca64b25bb705a540078fc8469e54` |
| Extracted compute commit | `native-metal-ssb` at `e1da9bc86a0c1ae6edc60e1205a9966e6826f315` |

The original reusable compute was embedded in an application controller. The
migration moved the FFT, SSB reconstruction, exact objective, and optimizer
into a SwiftPM library while leaving all UIKit and application ownership behind.
`MetalSSBKernels` imports only Foundation and Metal.

## Package boundary

| Public surface | Contract |
|---|---|
| `MetalSSBGeometry` | Complete calibrated BF geometry in logical source order; row and column reciprocal arrays remain explicitly named |
| `MetalSSBAberrations` | `C10` and `C12` in nm; `phi12` in radians |
| `MetalSSBEngine.prepare(brightfield:)` | Plane-major lossless `uint8` BF buffer shaped `[logical_brightfield, 512, 512]` |
| `reconstruct(aberrations:rotationDegrees:)` | Row-major 512×512 complex64 object and Fourier sum |
| `phaseVariance(aberrations:rotationDegrees:)` | Exact full-logical-BF phase-variance objective |
| `optimize(start:rotationDegrees:globalTrials:seed:...)` | Deterministic seeded TPE search, 200 trials by default, then Nelder–Mead |
| `MetalSSBProvenance` | Source/compute dtype, no crop, scan bin 1, logical/executed/proven-zero BF counts, cached/streamed BF counts, and exact cache bytes |

The engine retains all logical BF terms in normalization. The execution union
may omit only BF terms proven to remain outside the aperture. It does not crop
or bin scan positions, change precision, or fall back to CPU.

`cacheBudgetBytes: nil` requests the complete Hermitian cache. A finite budget
caches complete 32-BF batches and streams the remaining terms from the source
buffer with the same objective. Resource-policy choice and user-visible
explanation remain application responsibilities.

## Real-reference acceptance

The retained private fixture is identified publicly as
`native-ssb-fullbf-512-u8-v2`; paths and sample names are intentionally omitted.

| Field | Value |
|---|---|
| Date and revision | 2026-08-19; `e1da9bc86a0c1ae6edc60e1205a9966e6826f315` |
| Device | Apple M5 Max (`Mac17,6`), 40-core GPU, 128 GB unified memory |
| Source | 9,074 plane-major BF images, 512×512, exact `uint8`, 2,378,694,656 bytes |
| Source SHA-256 | `6046f7855b6925aafc86a52cc9ef06156ebf617d63b25c5a2a10fd94762ae3ae` |
| Reference | Independent CUDA-formula 512×512 float32 phase |
| Reference SHA-256 | `7def3b8ae2b781e3f0189ecfb5adbe9442942d8caf8d2e73e8fa672f78f1fa4a` |
| Scientific plan | full 512×512 scan, scan bin 1, no scan crop, detector bin 1, float32/complex64 compute |
| BF policy | 9,074 logical; 2,459 executed; 6,615 proven-zero; full logical normalization |
| Cache state | operating-system page cache warm; not a cold-source claim |

### Complete cache

| Measurement | Result |
|---|---:|
| File mapping | 4.703 ms |
| Engine initialization | 74.342 ms |
| Full cache preparation | 365.238 ms |
| First reconstruction | 8.461 ms wall; 7.906 ms GPU |
| Warm reconstruction | 8.911 ms p50; 9.416 ms p95/max, 7 repetitions |
| First exact-loss call | 51.546 ms including one-time cache-layout change |
| Warm exact loss | 25.120 ms p50; 25.516 ms p95/max, 7 repetitions |
| 200-trial TPE plus Nelder–Mead | 6.061212 s p50; 6.063051 s p95/max, 3 complete fits |
| Fit repeatability | identical fitted parameters and loss in all 3 fits, seed 42 |
| Phase parity | relative L2 `5.86952296e-5`; maximum wrapped error `5.62884106e-6` rad |
| Hermitian cache | 2,588,520,448 bytes |
| Measured peak process footprint | 2,921,529,992 bytes; no process swaps |

The fused Hermitian exact-loss path replaces the generic cached objective. On
the same prepared source it reduced the single loss measurement from about
144 ms to 25.120 ms p50 while retaining the independent phase gate. The generic
streaming path remains the exact low-memory reference.

### Zero cache

| Measurement | Result |
|---|---:|
| Preparation | 0.001 ms |
| First reconstruction | 312.677 ms wall |
| Warm reconstruction | 145.178 ms p50; 147.070 ms p95/max, 7 repetitions |
| Warm exact loss | 263.005 ms p50; 266.441 ms p95/max, 7 repetitions |
| Phase parity | relative L2 `2.30151898e-5`; maximum wrapped error `2.48712759e-6` rad |
| Measured peak process footprint | 310,510,216 bytes; no process swaps |

The two cache policies are not competing scientific modes. They produce the
same full-resolution result with slightly different float32 reduction order;
the focused gate limits cached-versus-streamed loss relative error to `5e-5`.

## Verification

| Check | Result |
|---|---|
| `swift test -c release` | 70 executed, 5 environment-qualified skips, 0 failures |
| Focused debug and release SSB tests | 4/4 passed in each configuration |
| Python SSB contract regression | 45 passed, 4 hardware-qualified skips |
| Full-cache retained log | SHA-256 `9fb88f7cf2a9a429edca7e6e97fcb75210840c3a80a79323df4addace731723a` |
| Zero-cache retained log | SHA-256 `b7ea9a4653799e3db00eee13190ae9fdcf1f87bc3e012d58030d52f6c7e67ece` |

## Limits and next gates

- The native engine currently supports a 512×512 scan. Other scan sizes remain
  unsupported, not inferred from the Python MPS/CUDA implementations.
- The benchmark starts from an exact precomputed BF-column source. Raw HDF5
  selection and BF-column construction are a separate IO/preparation stage.
- No storage-cache reset was performed, so the preparation number is warm
  source evidence rather than cold-source wall time.
- The complete cache fits the measured workstation but is not an 8 GB-device
  support claim. The zero-cache path proves a bounded exact alternative; a
  physical 8 GB app integration still needs its own admission and headed gate.
- The package has no UI framework dependency. A consuming application must own
  controls, progress, memory-policy selection, and visualization.
- The local commits are not pushed, released, or published by this migration.
