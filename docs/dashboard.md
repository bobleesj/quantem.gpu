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

(speed-and-memory-at-a-glance)=
## Speed and memory at a glance

The rows below are deliberately not a leaderboard. Each one keeps its cache
state, scientific plan, device, wall-clock boundary, and memory observation in
the same row.

### Measured load paths

| Implementation | Full source and output plan | Tested device and state | Wall-clock result | Memory and parity | 4/6 GiB reading |
|---|---|---|---:|---|---|
| [**Native Swift/Metal**](platforms/swift-metal.md) | `512x512x192x192 uint16` → full `512x512` scan, audited exact-sum detector bin 4 to `48x48 uint16`; no crop | Physical 8 GB `Mac14,2` M2 Air; first process / first observed source; seven runs per fixture | **1.985 / 2.043 s p50** to first complete product ([M2-AIR-BIN4-E2E](performance/results.md), 2026-08-18, `2c047160`/`e662d7fe`) | **1.43 GB peak process**, zero swap delta; eight products byte-identical | Observed footprint is below 4 GiB, but only the 8 GB unified-memory device is physically signed off |
| [**CUDA**](platforms/cuda.md) | `512x512x192x192 uint16` → audited lossless `uint8`; full scan, no crop/bin | RTX PRO 6000 Blackwell; warm source; 946-run median | **0.450 s** load/decompress ([CUDA-512-LOAD](performance/results.md), 2026-07-20, `b61572e4`) | **9.66 GB decoded resident**; transition peak not retained; selected-frame checksums exact | Full residency does **not** fit 4 or 6 GiB; use a bounded stream or explicit bin |
| [**Python MPS**](platforms/mps.md) | `1024x1024x192x192 uint16`; full scan/detector, no crop/bin, chunk-backed | Apple Metal GPU; first observed source; storage cache uncontrolled; exact Mac model missing | **4.617 s** load ([MPS-1024-LOAD](performance/results.md), 2026-07-20, `cee0ba5c`) | **77.31 GB logical decoded payload**; process peak not retained; selected frames bit-exact | Chunk backing avoids a duplicate host array, but this row is not 4/6 GiB evidence |
| [**WebGPU**](platforms/webgpu.md) | `512x512x192x192 uint16` → audited lossless `uint8`; full scan, no crop/bin | Chrome `apple metal-3` adapter; prepared local-file source; 946 cycles | **0.772 s p50** ([WEBGPU-512-FULL](performance/results.md), 2026-07-20, `b61572e4`) | **9.7 GB decoded payload**; browser peak not retained; selected-frame checksums exact | Full residency does **not** fit 4 or 6 GiB; use product-first or explicit detector binning |

A [saved-product reopen](performance/results.md) can take **6.8-8.0 ms**, but
that state contains only derived products. It is never called a source load.

(dtype-support-and-peak-memory)=
### Dtype support and peak memory

“Source,” “working,” “accumulation,” and “resident” dtype describe different
stages. A `uint8` row is scientifically exact only when the source is already
`uint8` or a complete source audit proves `maximum <= 255` and
`pixelsAbove255 == 0`. Otherwise an explicit `dtype="u8"` load saturates values
above 255 and is a browse representation, not raw-count evidence.

| Implementation | Accelerated compressed source | Exact `uint16` path | `uint8` path | Retained or required peak-memory record |
|---|---|---|---|---|
| [**CUDA**](platforms/cuda.md) | `uint16` ✓; `uint32` ✓; native `uint8` — on the specialized BSLZ4 path | Native `uint16` ✓; detector sums widen when needed | Direct fused saturating output ✓; lossless only with a complete audit | Warm audited-`uint8` row retains **9.66 GB resident**; allocation-transition and total-card peaks are **Pending** |
| [**Python MPS**](platforms/mps.md) | `uint16` ✓; `uint32` ✓; native `uint8` — on the specialized BSLZ4 path | Native `uint16` ✓; guarded `uint32` → `uint16` ✓ | Direct Metal saturating output ✓; audited low-byte working path ✓ | Planner checks output bytes against the Metal working set; measured process/Metal peak is **Pending** for the retained no-bin rows |
| [**Native Swift/Metal**](platforms/swift-metal.md) | Native `uint8` ✓; native `uint16` ✓ | Resident `uint16` ✓; exact sums widen to `uint32` unless a bound/audit permits `uint16` | Native source and audited compact staging ✓; persistent resident cache output is `uint16`/`uint32`, not `uint8` | Physical bin-4 `uint16` output retained **1.43 GB process peak** and zero swap delta on the 8 GB M2 Air |
| [**WebGPU**](platforms/webgpu.md) | Native `uint8`/`uint16`/`uint32` ✓ | Lossless `uint16` decode from `uint16` ✓ | Fused saturating output ✓; audited low-byte variants are separate | Audited-`uint8` row retains **9.7 GB decoded payload**; browser/device peak is **Pending** |
| **CPU reference** | Native `uint8`/`uint16` ✓ | Reference ✓ | Explicit reference conversion ✓ | Host peak is **Pending** and is never accelerator evidence |

The public Python selector is intentionally explicit:

- `dtype="u16"` requests unsigned 16-bit resident counts;
- `dtype="u8"` requests saturating unsigned 8-bit browse counts;
- `dtype="native"` preserves the source dtype; and
- `dtype="auto"` is an advisory convenience, not a substitute for a retained
  complete-source value-range audit.

### What a 4 or 6 GiB budget can hold

This capacity chart fixes the full scan at `512x512` and the native detector at
`192x192`. “Payload” excludes decoder scratch, staging buffers, allocator
reserve, and other GPU users unless the row reports a measured process peak.

| Exact plan | Resident payload or planned raw chunk | 4 GiB | 6 GiB | Product path and current evidence |
|---|---:|---|---|---|
| Native `uint16`, detector bin 1 | **18.00 GiB** | No | No | A full resident mean-DP/BF/ADF/DF/CoM pass needs a larger device |
| Audited lossless `uint8`, detector bin 1 | **9.00 GiB** | No | No | Valid only after the count audit proves every retained value fits `uint8` |
| Exact detector bin 2, general `uint32` result | **9.00 GiB** | No | No | Count-preserving, but not enough reduction for a 4/6 GiB resident stack |
| Exact detector bin 4 | **1.125 GiB** for audited `uint16`; **2.25 GiB** for general `uint32` | Candidate; headroom required | Candidate; headroom required | The audited `uint16` path passed on the physical 8 GB M2 Air at **1.43 GB process peak**; no 4/6 GiB physical-device signoff yet |
| Scan-row stream of native `uint16` | Planner reserves half the budget for the raw chunk | **1.97 GiB per chunk; 10 chunks** (56 rows each) | **2.99 GiB per chunk; 7 chunks** (85 rows each) | `screening.prepare` builds mean DP, BF/DF, CoM, and rotation; DPC/iDPC then use those small maps. The 4/6 GiB plans are code-verified, not physical-memory signoff |

Each `512x512 float32` product map is only **1 MiB**, and one `192x192`
float32 mean diffraction pattern is **144 KiB**. The source working set—not the
final BF/ADF/DF/DPC image—is the capacity problem.

For a full scan with output detector bin $b$ and $w$ resident bytes per value,
the payload alone is

$$
B_{\mathrm{payload}}
=N_{R_r}N_{R_c}
\left\lceil\frac{N_{k_r}}{b}\right\rceil
\left\lceil\frac{N_{k_c}}{b}\right\rceil w.
$$

Peak memory is larger: the benchmark must also report live compressed bytes,
decode and reduction scratch, staging/upload buffers, allocator reserve,
products, concurrent GPU users, and—on unified memory—process pressure and
swap. A calculated payload is never relabeled as a measured peak.

```{admonition} Small-GPU support today
:class: important
The bounded CUDA/MPS screening path covers mean DP, BF, DF, CoM, rotation, and
iDPC without cropping the scan. ADF has an accelerated CUDA/MPS/WebGPU/native
Metal detector kernel, but it is not yet emitted by the single-pass
`screening.prepare` cache. Physical 4 and 6 GiB product-pipeline signoff is
therefore **Pending**, even though the budget planner and kernels exist. See the
{ref}`screening API <screening-products>` before choosing a plan.
```

## Platform-first module dashboard

Every table keeps the scientific module as its section and puts the execution
platform in the first column. Empty cells are forbidden:

- **✓** — verified with retained real-data or physical-device parity evidence.
- **Test** — deterministic source/test coverage without an equivalent
  retained physical or native-data run.
- **Pending** — implementation exists, but that exact size, bin, or timing
  evidence has not been retained yet.
- **Ref** — CPU correctness adjudication, never a production fallback.
- **—** — unsupported or not a target.

### I/O and first usable product — `quantem.gpu.io`

#### Scan-size coverage

Scan sizes describe the real-space grid; the retained detector is `192x192`
unless a row says otherwise.

| Platform | `128x128` | `256x256` | `512x512` | `1024x1024` | Notes |
|---|---|---|---|---|---|
| **CUDA** | ✓ | ✓ | ✓ | ✓ | Real HDF5 crop/full-load parity, including true `1024` no-bin load |
| **Python MPS** | ✓ | ✓ | ✓ | ✓ | Real crop/full-load parity and chunk-backed true `1024` load |
| **Native Swift/Metal** | Pending | Pending | ✓ | Pending | Physical `512` full-scan load retained; other size-specific physical runs are not |
| **WebGPU** | ✓ | ✓ | ✓ | Partial | `1024` is product-first only; full-stack no-bin browse remains unsigned off |
| **CPU reference** | Ref | Ref | Ref | Ref | Shape-independent adjudication path; not an accelerator measurement |

#### Detector-bin coverage

Detector binning is exact integer block summation with incomplete edge bins
retained. A pending cell is not permission to claim that bin as measured.

| Platform | Bin 1 | Bin 2 | Bin 4 | Bin 8 | Notes / next measurement |
|---|---|---|---|---|---|
| **CUDA** | ✓ | ✓ | Pending | Pending | Generic count-preserving source path exists; retain real bin 4/8 hardware rows |
| **Python MPS** | ✓ | ✓ | Pending | Pending | Fused/source-backed binning exists; retain bin 4/8 physical timing and parity |
| **Native Swift/Metal** | ✓ | Test | ✓ | — | Public load plan supports bins `1/2/4`; physical 8 GB M2 evidence is bin 4 |
| **WebGPU** | ✓ | ✓ | ✓ | ✓ | [WEBGPU-DET-BIN](performance/results.md): full `512` and true crop `256` bin 2/4/8 checksums are exact on hardware |
| **CPU reference** | Ref | Ref | Ref | Ref | General integer-sum reference with partial-edge semantics |

### Screening and prepared-product caches — `quantem.gpu.screening`

| Platform | Module support | Measured size and plan | Build timing | Details / gap |
|---|---|---|---:|---|
| **CUDA** | ✓ | `1024x1024x192x192`, native `uint16`, bin 1, 12 GB allocator cap | **12.31 s** | [CUDA-CAL-BUILD](performance/results.md), 2026-07-28, `1c5dd03b`; full product build |
| **Python MPS** | ✓ | `512x512x192x192`, native `uint16`, bin 1, 64-row chunks | **3.96 s** | [MPS-CAL-BUILD](performance/results.md), 2026-07-21, `6c8ca5d0`; integer products exact, CoM max error `7.63e-6` |
| **Native Swift/Metal** | — | — | **Pending** | `quantem.gpu.screening` is not implemented as a native Swift public module |
| **WebGPU** | — | — | **Pending** | `quantem.gpu.screening` is not implemented as a WebGPU public module |
| **CPU reference** | Ref | Reference fixtures | **Pending** | Correctness adjudication only; no retained build benchmark |

Backend-neutral saved-product reopen is a separate state:
[PRODUCT-CACHE-REOPEN](performance/results.md) retained **6.8-8.0 ms** for five
repeats of full-`1024` BF/DF/CoM/rotation products. It is never represented as
source load or cache construction.

### Virtual images — `quantem.gpu.detector`

| Platform | Mean DP and BF/ABF/ADF/DF | Measured size/bin plan | Latest retained timing | Details |
|---|---|---|---:|---|
| **CUDA** | ✓ | Resident `512x512x192x192`, bin 1 | BF **1.35 ms**; ADF **3.86 ms**; DF **1.84 ms** | [CUDA-BF-512](performance/results.md), [CUDA-ADF-512](performance/results.md), and [CUDA-DF-512](performance/results.md), 2026-07-19, `0456e15e`; integer max error `0` |
| **Python MPS** | ✓ | Full `512` bin-1 products through retained screening parity | **Pending** | No isolated MPS virtual-image timing with complete public device provenance |
| **Native Swift/Metal** | ✓ | Full `512`, detector bin 4 in physical application parity | **Pending** | Products are byte-identical in the M2 Air gate; isolated kernel timing is not retained |
| **WebGPU** | ✓ | Full `512`, bin 1, fixed 30 px BF radius | BF page total **0.378 s p50** | [WEBGPU-BF-512](performance/results.md), 2026-07-20, `b61572e4`; prepared selected-block boundary, not isolated kernel time |
| **CPU reference** | Ref | Reference fixtures | **Pending** | Correctness adjudication only |

### Detector moments and phase contrast — `quantem.gpu.dpc`

| Platform | CoM row/column, DPC, rotation, iDPC | Measured size/bin plan | Latest retained timing | Details |
|---|---|---|---:|---|
| **CUDA** | ✓ | Resident full `512x512x192x192`, bin 1 | CoM row + column **12.39 ms** | [CUDA-COM-512](performance/results.md), 2026-07-19, `0456e15e`; max error `0` |
| **Python MPS** | ✓ | Full `512`, bin 1 through retained screening parity | **Pending** | No isolated full-module MPS timing with complete public device provenance |
| **Native Swift/Metal** | ✓ | Full `512`, detector bin 4 in physical application parity | **Pending** | CoM/DPC/iDPC exports are byte-identical; isolated kernel timing is not retained |
| **WebGPU** | ✓ | Resident full `512x512x192x192`, bin 1 | DPC row/column/iDPC **14.9 / 13.2 / 13.2 ms p50** | [WEBGPU-DPC-512](performance/results.md), 2026-07-20, `cee0ba5c`; frozen float32 errors retained |
| **CPU reference** | Ref | Reference fixtures | **Pending** | Correctness adjudication only |

### Single-sideband ptychography — `quantem.gpu.SSB`

These are square scan-grid sizes, not detector dimensions.

| Platform | `128x128` | `256x256` | `512x512` | `1024x1024` | Latest retained full-policy result | Details / next gap |
|---|---|---|---|---|---:|---|
| **CUDA** | ✓ | ✓ | ✓ | ✓ | **32.2 ms p50** at `512` | [SSB-CUDA-512-FULL](performance/results.md), 2026-07-19, `0456e15e`; `512` is real full-BF, with fixed-size kernel/reference checks at all sizes |
| **Python MPS** | ✓ | ✓ | ✓ | ✓ | **537.58 ms p50** at `512` | [SSB-MPS-512-FULL](performance/results.md), 2026-07-28, `e8d49866`; `512` is native full-BF, other sizes resized/synthetic |
| **Native Swift/Metal** | — | — | — | — | **Pending** | No native Swift SSB kernel |
| **WebGPU** | ✓ | Test | Partial | Partial | **Pending** | `128` is real BF30 vs CUDA; `256` is test-only; `512/1024` have real interaction but incomplete frozen CUDA artifacts |
| **CPU reference** | — | — | Ref | — | **Pending** | Independent `512` adjudication only |

See the [full SSB performance record](maintainer/ssb-performance.md) for the
12-cell redraw matrix, size-specific timings, memory, and rejected experiments.

### Cross-module platform map

| Platform | I/O | Screening | Virtual images | CoM/DPC/iDPC | SSB | Display/movie and other boundaries |
|---|---|---|---|---|---|---|
| **CUDA** | ✓ | ✓ | ✓ | ✓ | ✓ | Display ✓; movie via NVENC; parallax CUDA-only |
| **Python MPS** | ✓ | ✓ | ✓ | ✓ | ✓ | Display ✓; movie via VideoToolbox; parallax — |
| **Native Swift/Metal** | ✓ | — | ✓ | ✓ | — | `MetalDisplayKernels`/`MetalImageFFT` ✓; native package products |
| **WebGPU** | ✓ | — | ✓ | ✓ | Partial | Display ✓; movie and parallax are not targets |
| **CPU reference** | Ref | Ref | Ref | Ref | Ref | Explicit adjudication/fallback paths only |

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
