# Kernel and benchmark dashboard

This is the one-page technical overview of `quantem.gpu`: what the scientific
kernels compute, where each runtime implements them, how parity is proved, and
what the latest retained measurements actually mean.

```{admonition} Read the state before comparing the number
:class: important
First-process source load, prepared-source load, warm resident compute, and
saved-result reopen are different experiments. A binned or cropped source is
never presented as native resolution. Open the evidence link before using a
number in a design or release decision.
```

**Dashboard review:** 2026-08-19. Every measurement below carries its own date
and code revision; this page does not replace the
[complete benchmark provenance ledger](performance/results.md).

## Platform-first module dashboard

Every table keeps the scientific module as its section and puts the execution
platform in the first column. Empty cells are forbidden:

- **✓ Evidence** — retained real-data or physical-device parity evidence for
  that exact cell.
- **Test only** — deterministic source/test coverage without an equivalent
  retained physical or native-data run.
- **Pending** — implementation exists, but that exact size, bin, or timing
  evidence has not been retained yet.
- **Reference** — CPU correctness adjudication, never a production fallback.
- **—** — unsupported or not a target.

### I/O and first usable product — `quantem.gpu.io`

#### Scan-size evidence

Scan sizes describe the real-space grid; the retained detector is `192x192`
unless a row says otherwise.

| Platform | `128x128` | `256x256` | `512x512` | `1024x1024` | Evidence boundary |
|---|---|---|---|---|---|
| **CUDA** | ✓ Evidence | ✓ Evidence | ✓ Evidence | ✓ Evidence | Real HDF5 crop/full-load parity, including true `1024` no-bin load |
| **Python MPS** | ✓ Evidence | ✓ Evidence | ✓ Evidence | ✓ Evidence | Real crop/full-load parity and chunk-backed true `1024` load |
| **Native Swift/Metal** | Pending | Pending | ✓ Evidence | Pending | Physical `512` full-scan load retained; other size-specific physical runs are not |
| **WebGPU** | ✓ Evidence | ✓ Evidence | ✓ Evidence | Partial: product-first only | True `1024` full-stack no-bin browse remains unsigned off |
| **CPU reference** | Reference | Reference | Reference | Reference | Shape-independent adjudication path; not an accelerator measurement |

#### Detector-bin evidence

Detector binning is exact integer block summation with incomplete edge bins
retained. A pending cell is not permission to claim that bin as measured.

| Platform | Bin 1 | Bin 2 | Bin 4 | Bin 8 | Implementation and next evidence |
|---|---|---|---|---|---|
| **CUDA** | ✓ Evidence | ✓ Evidence | Pending | Pending | Generic count-preserving source path exists; retain real bin 4/8 hardware rows |
| **Python MPS** | ✓ Evidence | ✓ Evidence | Pending | Pending | Fused/source-backed binning exists; retain bin 4/8 physical timing and parity |
| **Native Swift/Metal** | ✓ Evidence | Test only | ✓ Evidence | — | Public load plan supports bins `1/2/4`; physical 8 GB M2 evidence is bin 4 |
| **WebGPU** | ✓ Evidence | ✓ Evidence | ✓ Evidence | ✓ Evidence | Full `512` and true crop `256` bin 2/4/8 checksums are exact on hardware |
| **CPU reference** | Reference | Reference | Reference | Reference | General integer-sum reference with partial-edge semantics |

#### Retained load timing

| Platform | Size and explicit plan | Benchmark boundary | Retained result | Evidence and missing comparison |
|---|---|---|---:|---|
| **CUDA** | `512x512x192x192`, bin 1, audited lossless-low8 output | Warm-source load/decompress median; 946 runs | **0.450 s** | [CUDA-512-LOAD](performance/results.md), 2026-07-20, `b61572e4`; not first encounter |
| **Python MPS** | `1024x1024x192x192` native `uint16`, bin 1 | First observed source; storage cache uncontrolled | **4.617 s** | [MPS-1024-LOAD](performance/results.md), 2026-07-20, `cee0ba5c`; exact selected frames, host model missing |
| **Native Swift/Metal** | `512x512x192x192` `uint16`, full scan, detector bin 4 | First process / first observed source to first complete product; seven isolated runs per fixture | **1.985 / 2.043 s p50** | [M2-AIR-BIN4-E2E](performance/results.md), 2026-08-18, measured `2c047160`, integrated `e662d7fe`; eight products byte-identical, `1.43 GB` peak, zero swap delta |
| **WebGPU** | `512x512x192x192`, bin 1, audited lossless-low8 output | Prepared local-file full-stack soak; 946 cycles | **0.772 s p50** | [WEBGPU-512-FULL](performance/results.md), 2026-07-20, `b61572e4`; prepared, not cold |
| **WebGPU** | Full `512`, explicit detector bins 2 / 4 / 8 | Prepared local-file page profiles | **1.199 / 1.212 / 1.106 s** | [WEBGPU-DET-BIN](performance/results.md), 2026-07-20, `cee0ba5c`; exact corrected-frame checksums |
| **CPU reference** | All sizes/bins | Reference only | **Pending** | No retained comparable accelerator-style load timing |

### Screening and prepared-product caches — `quantem.gpu.screening`

| Platform | Module support | Measured size and plan | Build timing | Evidence or explicit gap |
|---|---|---|---:|---|
| **CUDA** | ✓ Evidence | `1024x1024x192x192`, native `uint16`, bin 1, 12 GB allocator cap | **12.31 s** | [CUDA-CAL-BUILD](performance/results.md), 2026-07-28, `1c5dd03b`; full product build |
| **Python MPS** | ✓ Evidence | `512x512x192x192`, native `uint16`, bin 1, 64-row chunks | **3.96 s** | [MPS-CAL-BUILD](performance/results.md), 2026-07-21, `6c8ca5d0`; integer products exact, CoM max error `7.63e-6` |
| **Native Swift/Metal** | — | — | **Pending** | `quantem.gpu.screening` is not implemented as a native Swift public module |
| **WebGPU** | — | — | **Pending** | `quantem.gpu.screening` is not implemented as a WebGPU public module |
| **CPU reference** | Reference products | Reference fixtures | **Pending** | Correctness adjudication only; no retained build benchmark |

Backend-neutral saved-product reopen is a separate state:
[PRODUCT-CACHE-REOPEN](performance/results.md) retained **6.8-8.0 ms** for five
repeats of full-`1024` BF/DF/CoM/rotation products. It is never represented as
source load or cache construction.

### Virtual images — `quantem.gpu.detector`

| Platform | Mean DP and BF/ABF/ADF/DF | Measured size/bin plan | Latest retained timing | Evidence or placeholder |
|---|---|---|---:|---|
| **CUDA** | ✓ Evidence | Resident `512x512x192x192`, bin 1 | BF **1.35 ms**; ADF **3.86 ms**; DF **1.84 ms** | [CUDA-BF-512](performance/results.md), [CUDA-ADF-512](performance/results.md), and [CUDA-DF-512](performance/results.md), 2026-07-19, `0456e15e`; integer max error `0` |
| **Python MPS** | ✓ Evidence | Full `512` bin-1 products through retained screening parity | **Pending** | No isolated MPS virtual-image timing with complete public device provenance |
| **Native Swift/Metal** | ✓ Evidence | Full `512`, detector bin 4 in physical application parity | **Pending** | Products are byte-identical in the M2 Air gate; isolated kernel timing is not retained |
| **WebGPU** | ✓ Evidence | Full `512`, bin 1, fixed 30 px BF radius | BF page total **0.378 s p50** | [WEBGPU-BF-512](performance/results.md), 2026-07-20, `b61572e4`; prepared selected-block boundary, not isolated kernel time |
| **CPU reference** | Reference | Reference fixtures | **Pending** | Correctness adjudication only |

### Detector moments and phase contrast — `quantem.gpu.dpc`

| Platform | CoM row/column, DPC, rotation, iDPC | Measured size/bin plan | Latest retained timing | Evidence or placeholder |
|---|---|---|---:|---|
| **CUDA** | ✓ Evidence | Resident full `512x512x192x192`, bin 1 | CoM row + column **12.39 ms** | [CUDA-COM-512](performance/results.md), 2026-07-19, `0456e15e`; max error `0` |
| **Python MPS** | ✓ Evidence | Full `512`, bin 1 through retained screening parity | **Pending** | No isolated full-module MPS timing with complete public device provenance |
| **Native Swift/Metal** | ✓ Evidence | Full `512`, detector bin 4 in physical application parity | **Pending** | CoM/DPC/iDPC exports are byte-identical; isolated kernel timing is not retained |
| **WebGPU** | ✓ Evidence | Resident full `512x512x192x192`, bin 1 | DPC row/column/iDPC **14.9 / 13.2 / 13.2 ms p50** | [WEBGPU-DPC-512](performance/results.md), 2026-07-20, `cee0ba5c`; frozen float32 errors retained |
| **CPU reference** | Reference | Reference fixtures | **Pending** | Correctness adjudication only |

### Single-sideband ptychography — `quantem.gpu.SSB`

These are square scan-grid sizes, not detector dimensions.

| Platform | `128x128` | `256x256` | `512x512` | `1024x1024` | Latest retained full-policy result | Evidence or next gap |
|---|---|---|---|---|---:|---|
| **CUDA** | ✓ Evidence | ✓ Evidence | ✓ Real full-BF | ✓ Evidence | **32.2 ms p50** at `512` | [SSB-CUDA-512-FULL](performance/results.md), 2026-07-19, `0456e15e`; production fixed-size kernels and reference checks at all sizes |
| **Python MPS** | ✓ Resized/synthetic | ✓ Resized/synthetic | ✓ Native full-BF | ✓ Resized/synthetic | **537.58 ms p50** at `512` | [SSB-MPS-512-FULL](performance/results.md), 2026-07-28, `e8d49866`; physical runs at all sizes, native acquisition retained at `512` |
| **Native Swift/Metal** | — | — | — | — | **Pending** | No native Swift SSB kernel |
| **WebGPU** | ✓ Real BF30 vs CUDA | Test only | Partial: real interaction | Partial: real load/interaction | **Pending** | Source supports all sizes; comparable frozen real CUDA artifacts remain incomplete at `256/512/1024` |
| **CPU reference** | — | — | Reference fixture | — | **Pending** | Independent `512` adjudication only |

See the [full SSB performance record](maintainer/ssb-performance.md) for the
12-cell redraw matrix, size-specific timings, memory, and rejected experiments.

### Cross-module platform map

| Platform | I/O | Screening | Virtual images | CoM/DPC/iDPC | SSB | Display/movie and other boundaries |
|---|---|---|---|---|---|---|
| **CUDA** | ✓ Evidence | ✓ Evidence | ✓ Evidence | ✓ Evidence | ✓ Evidence | Display ✓; movie via NVENC; parallax CUDA-only |
| **Python MPS** | ✓ Evidence | ✓ Evidence | ✓ Evidence | ✓ Evidence | ✓ Evidence | Display ✓; movie via VideoToolbox; parallax — |
| **Native Swift/Metal** | ✓ Evidence | — | ✓ Evidence | ✓ Evidence | — | `MetalDisplayKernels`/`MetalImageFFT` ✓; native package products |
| **WebGPU** | ✓ Evidence | — | ✓ Evidence | ✓ Evidence | Partial | Display ✓; movie and parallax are not targets |
| **CPU reference** | Reference | Reference products | Reference | Reference | Reference fixture | Explicit adjudication/fallback paths only |

The rows above are intentionally not ranked. A warm resident kernel, prepared
source, first-process application load, and saved-result reopen answer different
questions. See [Benchmark methodology](performance/methodology.md) and
[Verified benchmark results](performance/results.md) before comparing them.

The public scientific array is always

$$
I[R_r,R_c,k_r,k_c],
$$

where $\mathbf R=(R_r,R_c)$ is the probe/scan coordinate and
$\mathbf k=(k_r,k_c)$ is the detector coordinate. Runtime-specific layout,
tiling, fusion, and dispatch are private optimizations; shape, sampling,
precision, calibration, and provenance are shared contracts.

## Where an implementer starts

| Goal | Read first | Then inspect | Acceptance evidence |
|---|---|---|---|
| Change scientific meaning or add an operation | [Scientific contract](concepts/scientific-contract.md) | [Scientific kernels](kernels/index.md) | Operation-specific equation, provenance schema, and independent reference |
| Optimize CUDA | [CUDA implementation](platforms/cuda.md) | Domain `compute/cuda` or `io/backends/cuda` | Real NVIDIA profile plus exact/frozen parity |
| Optimize Python on Apple Silicon | [Python MPS](platforms/mps.md) | Domain `compute/mps` or `io/backends/mps` | Physical Apple device profile plus exact/frozen parity |
| Build a native Apple client/library | [Native Swift and Metal](platforms/swift-metal.md) | `Package.swift` and `src/quantem/gpu/swift/{Sources,Tests}` | `swift test`, physical Metal timing, and cross-language fixtures |
| Optimize a browser client | [WebGPU](platforms/webgpu.md) | Domain WebGPU TypeScript/WGSL resources | Real adapter, headed browser gate, and matching scientific output |
| Deploy CUDA behind a process boundary | [QuantEM.GPU Remote](remote/index.md) | Deployment, protocol, and admission pages | Same array/provenance contract plus transport and capacity checks |
| Add or review a benchmark | [Benchmark methodology](performance/methodology.md) | [Parity](performance/parity.md) and the [optimization ledger](maintainer/backend-optimization-matrix.md) | Date, revision, device, source plan, cache state, memory, wall boundary, and parity artifact |

## Dashboard maintenance rule

Update a dashboard row only after its detailed evidence row is complete. The
detail remains authoritative and must record measurement date, exact source
revision, physical device/runtime, source shape and dtype, cache state,
crop/bin/load plan, benchmark definition, peak memory or swap where available,
and numerical or hash parity. Keep an older result when the newer experiment
changes any of those conditions; label both instead of silently replacing one.

Accepted and rejected experiments remain in the
[optimization ledger](maintainer/backend-optimization-matrix.md), and the
machine-readable evidence fingerprints are in
[`performance/evidence_manifest.json`](performance/evidence_manifest.json).
