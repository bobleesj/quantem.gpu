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

## Module benchmark snapshot

The tables are organized by the public module that owns the work. Each row is
one measurement, not a claim about the speed of the entire runtime.

### I/O and first usable product — `quantem.gpu.io`

| Evidence | Runtime and physical device | Scientific source and plan | State and benchmark boundary | Retained result | Parity and interpretation |
|---|---|---|---|---:|---|
| [M2-AIR-BIN4-E2E](performance/results.md) | Native Swift/Metal; physical 8 GB M2 MacBook Air | `512x512x192x192` `uint16`; full scan, no crop, scan bin 1, explicit exact-sum detector bin 4 to `48x48` | First process / first observed source; seven alternating process-isolated application loads per fixture; OS cache not forcibly evicted | Fixture A **1.985 s** wall p50; Fixture B **2.043 s** wall p50; Metal **1.615-1.618 s** p50; peak process **1.43 GB**, zero swap delta | Eight BF/ABF/ADF/CoM/DPC/iDPC exports byte-identical across frozen/candidate/frozen. Not no-bin or true-cold evidence. 2026-08-18, measured `2c047160`, published integration `e662d7fe`. |
| [CUDA-512-LOAD](performance/results.md) | CUDA; NVIDIA RTX PRO 6000 Blackwell | `512x512x192x192`, source `uint16`, full scan, no crop/bin, audited `uint8` browse output | Warm source; 946-run load/decompress median | **0.450 s** | Selected-frame integer checksums corrected and exact. Not a first encounter. 2026-07-20, `b61572e4`. |
| [WEBGPU-512-FULL](performance/results.md) | WebGPU; Chrome on Apple Metal adapter | `512x512x192x192`, source `uint16`, audited lossless-low8 `uint8` browse output, no crop/bin | Prepared local-file indexes/sidecars; 946-cycle full-stack soak | **0.772 s** p50, `0.726-0.879 s` range | First/middle/last corrected-frame checksums exact to CUDA. Prepared, not cold. 2026-07-20, `b61572e4`. |

### Screening and prepared-product caches — `quantem.gpu.screening`

| Evidence | Runtime and physical device | Scientific source and plan | State and benchmark boundary | Retained result | Parity and interpretation |
|---|---|---|---|---:|---|
| [CUDA-CAL-BUILD](performance/results.md) | CUDA; NVIDIA RTX PRO 6000 Blackwell | `1024x1024x192x192` native `uint16`; full scan/detector, no crop/bin; 12 GB allocator cap | Chunked first product-cache construction | **12.31 s** | Full product build under the stated memory policy. 2026-07-28, `1c5dd03b`. |
| [MPS-CAL-BUILD](performance/results.md) | Python MPS; Apple Metal GPU, exact Mac model not retained | `512x512x192x192` native `uint16`; full scan/detector, no crop/bin; 64-row chunks | First product-cache build; source-cache state not retained | **3.96 s** | Mean DP/BF/DF bit-exact; CoM max error `7.63e-6`. Historical diagnostic. 2026-07-21, `6c8ca5d0`. |
| [PRODUCT-CACHE-REOPEN](performance/results.md) | Backend-neutral local filesystem; host not retained | Persisted BF/DF/CoM/rotation products for a full `1024` scan; raw detector volume not reopened | Saved-result reopen; five repeats | **6.8-8.0 ms** | Derived products only. Never represented as HDF5 source load or cache construction. 2026-07-20, `628214a8`. |

### Virtual images — `quantem.gpu.detector`

| Evidence | Runtime and physical device | Scientific source and plan | State and benchmark boundary | Retained result | Parity and interpretation |
|---|---|---|---|---:|---|
| [CUDA-BF-512](performance/results.md) | CUDA GPU; exact model not retained | GPU-resident full `512x512x192x192`; no crop/bin | Warm resident BF reduction; source load excluded | **1.35 ms** | Integer maximum error `0`. Historical diagnostic because device and mask details are incomplete. 2026-07-19, `0456e15e`. |
| [WEBGPU-BF-512](performance/results.md) | WebGPU; Chrome on Apple Metal adapter | Full `512x512x192x192`; fixed 30 px BF radius | Prepared selected-block page total; not an isolated kernel time | **0.378 s** p50 | Exact to CUDA across 946 cycles. Prepared, not cold. 2026-07-20, `b61572e4`. |

### Detector moments and phase contrast — `quantem.gpu.dpc`

| Evidence | Runtime and physical device | Scientific source and plan | State and benchmark boundary | Retained result | Parity and interpretation |
|---|---|---|---|---:|---|
| [CUDA-COM-512](performance/results.md) | CUDA GPU; exact model not retained | GPU-resident full `512x512x192x192`, no crop/bin | Warm resident-kernel microbenchmark; source load excluded | **200.42 -> 12.39 ms** | CoM row/column max error `0`. Historical diagnostic because the device identity is incomplete. 2026-07-19, `0456e15e`. |
| [WEBGPU-DPC-512](performance/results.md) | WebGPU; headed Chrome on NVIDIA RTX PRO 6000 Blackwell | GPU-resident full `512x512x192x192`, no crop/bin | Warm resident display; source load excluded | DPC row/column/iDPC **14.9/13.2/13.2 ms** p50 | DPC max error `7.63e-6`; iDPC mean/max error `4.70e-6/3.05e-5`. 2026-07-20, `cee0ba5c`. |

### Single-sideband ptychography — `quantem.gpu.SSB`

| Evidence | Runtime and physical device | Scientific source and plan | State and benchmark boundary | Retained result | Parity and interpretation |
|---|---|---|---|---:|---|
| [SSB-CUDA-512-FULL](performance/results.md) | CUDA GPU; exact model not retained | Prepared real `512x512` SSB field, full-BF policy, `float32`/`complex64` | Warm phase+loss kernel; source preparation excluded | **32.2 ms** p50, `33.3 ms` p95 | Same BF disk, aberrations, objective, and loss reference. Historical diagnostic. 2026-07-19, `0456e15e`. |
| [SSB-MPS-512-FULL](performance/results.md) | Python MPS; Apple Silicon GPU, exact model not retained | Prepared real `512x512` Hermitian $G(\mathbf k,\boldsymbol{\nu})$, full active BF, `float32`/`complex64` | Warm phase+loss kernel; source preparation excluded | **537.58 ms** p50, `557.51 ms` p95 | Same full-active BF policy, aberrations, precision, objective, and frozen loss. Historical diagnostic. 2026-07-28, `e8d49866`. |

These rows are intentionally not ranked. They answer different questions.
See [Benchmark methodology](performance/methodology.md) for the required timing
and memory boundaries, and [Verified benchmark results](performance/results.md)
for every retained measurement, including rejected and historical evidence.

## Scientific module and implementation matrix

Status meanings:

- **Verified** — implemented with the required parity gate and retained
  hardware evidence.
- **Partial** — implementation exists, but its full hardware or parity matrix
  is incomplete.
- **Reference** — independent adjudication path, not a production fallback.
- **Not implemented** — intentionally exposed as a gap rather than silently
  falling back.

| Scientific module | Public contract and principal source | CUDA | Python MPS | Native Swift/Metal | WebGPU | CPU reference | Required parity gate |
|---|---|---|---|---|---|---|---|
| **I/O** — load, bitshuffle/LZ4 decode, dtype, crop/bin provenance | [`io.load`](api/io.md); `io/backends/*` | Verified | Verified | Verified: `Native4DSTEMIO`, `Metal4DSTEMKernels` | Verified on hardware | Reference | Exact corrected counts, source/output shape and dtype, half-open regions, bin geometry, bad pixels, and provenance |
| **Screening** — prepared launch products and cache reopen | [`screening`](api/core.md); `screening/*` | Verified | Verified | Not implemented as this public module | Not implemented | Reference products | Exact mean DP/BF/DF integers, frozen CoM error, cache identity, and explicit build-versus-reopen state |
| **Virtual images** — mean diffraction and BF/ABF/ADF/DF | [`detector`](kernels/virtual-detectors.md); `detector/compute/*` | Verified | Verified | Verified: `Metal4DSTEMKernels` | Verified on hardware | Reference | Exact integer accumulation and masks; bad-pixel order fixed |
| **Detector moments and phase contrast** — CoM row/column, DPC, rotation, iDPC | [`dpc`](kernels/com-dpc-idpc.md); `dpc/compute/*` | Verified | Verified | Verified: `Metal4DSTEMKernels` | Verified on hardware | Reference | Same $(k_r,k_c)$ moments, centering, rotation, FFT normalization, and frozen float metric |
| **SSB** — object, phase, loss, aberration fit | [`SSB`](kernels/ssb.md); `ssb/compute/*` | Verified | Verified | Not implemented | Partial | Reference fixture | Same complete BF disk, aberrations, precision, objective, optimizer settings, and frozen phase/loss metric |
| **Display** — transform, histogram, colormap, and FFT | [`display`](kernels/display-export.md); `display/*` | Verified | Verified | Verified: `MetalDisplayKernels`, `MetalImageFFT` | Verified on hardware | Reference | Exact histogram/RGBA integers and frozen `float32` transform/FFT metrics |
| **Movie** — GIF/MP4 frame rendering | [`movie`](api/movie.md); `movie/*` | Verified: NVENC | Verified: VideoToolbox | Native product consumed through package API | Not a target | Explicit fallback | Frame parity and encoded-movie smoke tests |

### SSB square scan-size coverage

These are scan-grid sizes, not detector dimensions. “Verified” means the
stated runtime and evidence class passed; it does not make resized or synthetic
input equivalent to a native acquisition.

| Runtime | `128x128` | `256x256` | `512x512` | `1024x1024` | Evidence qualification |
|---|---|---|---|---|---|
| **CUDA** | Verified | Verified | Verified, real full-BF | Verified | Production fixed-size kernels and reference checks at all four sizes |
| **Python MPS** | Verified, resized/synthetic | Verified, resized/synthetic | Verified, native full-BF | Verified, resized/synthetic | Physical Apple GPU runs at every size; native-acquisition evidence is retained at `512x512` |
| **WebGPU** | Verified, real BF30 vs CUDA | Implemented, synthetic browser reference | Partial, real interaction | Partial, real load/interaction | Source supports all four sizes; frozen real CUDA artifacts remain incomplete at `256/512/1024` |
| **Native Swift/Metal** | Not implemented | Not implemented | Not implemented | Not implemented | No native Swift SSB kernel |
| **CPU reference** | — | — | Reference fixture | — | Independent adjudication only |

See the [full SSB performance record](maintainer/ssb-performance.md) for the
12-cell redraw matrix, timings, memory, and rejected experiments.

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
