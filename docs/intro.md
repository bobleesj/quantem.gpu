# quantem.gpu

One scientific GPU contract for electron microscopy—from compressed detector
data to exact products—across NVIDIA CUDA, Apple MPS/Metal, native Swift,
WebGPU, and an explicit CPU reference.

```{admonition} Choose how you want to enter
:class: tip
**Whole project:** start with the
[implementation overview](dashboard.md).

**Scientific operation:** start with [Scientific kernels](kernels/index.md).

**Kernel implementation:** start with [Kernel implementations](platforms/index.md).

**Correctness or speed claim:** start with
[Benchmarks and parity](performance/index.md).
```

## Module capabilities and benchmarks

**Overview reviewed:** 2026-08-19. The tables follow public scientific modules,
then compare their runtime implementations. Capability and performance are
separate claims: a missing isolated timing does not mean that a feature is
missing.

Status words have one meaning throughout this page:

- **Verified** — implementation plus retained hardware and parity evidence for
  the stated scope.
- **Implemented** — production source exists, but the full physical or
  native-data evidence is incomplete.
- **Partial** — only part of the module contract has complete evidence.
- **Reference** — independent correctness adjudication, never a silent fallback.
- **Not implemented** — an explicit gap.

Every performance row contains one headline time. Date, device, revision,
scientific plan, and parity evidence stay together in its rightmost column.

### I/O — `quantem.gpu.io`

The I/O module owns discovery, inspection, accelerated bitshuffle/LZ4 decode,
explicit crop/bin geometry, dtype conversion, provenance, and compressed HDF5
saving.

| Feature | CUDA | Python MPS | Swift/Metal | WebGPU | CPU reference |
|---|---|---|---|---|---|
| Load, decode, dtype, and provenance | **Verified** | **Verified** | **Verified** through `Native4DSTEMIO` and `Metal4DSTEMKernels` | **Verified** on a physical browser GPU | **Reference** |
| Explicit detector binning | **Verified** | **Verified** | **Verified** exact-sum binning | **Verified** for bins 2/4/8 | **Reference** |
| Compressed HDF5 save | **Implemented** | **Verified** | Resident-cache writing only; no compressed-HDF5 writer | **Not implemented** | **Reference** writer |

| Implementation | Operation and boundary | Time | Data and scientific plan | Test metadata and evidence |
|---|---|---:|---|---|
| **Swift/Metal**, fixture A | First-process action → first complete product | **1.985 s p50** | `512x512x192x192` `uint16`; full scan; no crop; detector sum bin 4 → `48x48` | 2026-08-18 · physical 8 GB M2 MacBook Air (`Mac14,2`) · candidate `2c047160` · seven isolated runs · peak process `1.43 GB` · zero swap delta · eight product exports byte-identical · [M2-AIR-BIN4-E2E](performance/results.md) |
| **Swift/Metal**, fixture B | First-process action → first complete product | **2.043 s p50** | Same full-scan plan as fixture A | 2026-08-18 · physical 8 GB M2 MacBook Air (`Mac14,2`) · integration [`e662d7fe`](https://github.com/bobleesj/quantem.gpu/commit/e662d7feebf78e7c1513276651d0be55a555cb40) · seven isolated runs · diffraction hash retained · [M2-AIR-BIN4-E2E](performance/results.md) |
| **CUDA** | Warm-source load and decompression | **0.450 s median** | `512x512x192x192` `uint16`; full scan; no crop/bin; audited lossless-low8 `uint8` output | 2026-07-20 · NVIDIA RTX PRO 6000 Blackwell · [`b61572e4`](https://github.com/bobleesj/quantem.gpu/commit/b61572e47058e4b3fa48835541f667d46b762cf0) · 946 runs · selected integer checksums exact · [CUDA-512-LOAD](performance/results.md) |
| **Python MPS** | First-observed load and decompression; storage cache uncontrolled | **4.617 s** | `1024x1024x192x192` chunk-backed `uint16`; full scan/detector; no crop/bin | 2026-07-20 · Apple Metal GPU, exact Mac model not retained · [`cee0ba5c`](https://github.com/bobleesj/quantem.gpu/commit/cee0ba5ca3725b03054ecf5e6a14e304bb93d4ed) · selected frames bit-exact · **historical diagnostic** · [MPS-1024-LOAD](performance/results.md) |
| **WebGPU** | Prepared local-file full-stack load; not cold | **0.772 s p50** | `512x512x192x192` `uint16`; full scan; no crop/bin; audited lossless-low8 `uint8` output | 2026-07-20 · Chrome on Apple `metal-3`, exact Mac model not retained · [`b61572e4`](https://github.com/bobleesj/quantem.gpu/commit/b61572e47058e4b3fa48835541f667d46b762cf0) · 946 cycles · corrected-frame checksums exact to CUDA · [WEBGPU-512-FULL](performance/results.md) |
| **Python MPS** | Best retained compression and HDF5-write sweep; source load excluded | **1.69 s** | Warm resident `512x512x192x192` `uint16`; no crop/bin; batch 2048; bitshuffle/LZ4 | 2026-07-25 · Apple Metal GPU, exact Mac/storage model not retained · [`3061501`](https://github.com/bobleesj/quantem.gpu/commit/30615019cfe293ae9759006ae89c0e378b7065fd) · decoded samples exact · **historical diagnostic** · [MPS-SAVE-U16-512](performance/results.md) |
| **Python MPS** | Public-default compressed-save confirmation; source load excluded | **1.91 s** | Same source and compression plan; output `1.205 GB` | 2026-07-25 · public API [`83bb608`](https://github.com/bobleesj/quantem.gpu/commit/83bb6089e11604b5828e6f94a70d49e487e75929) · decoded samples exact · **historical diagnostic** · [MPS-SAVE-U16-512](performance/results.md) |

### Screening — `quantem.gpu.screening`

`screening.prepare` builds and reopens the small mean-diffraction, BF, DF, CoM,
rotation, and iDPC launch products. It is a separate module from raw I/O.

| Runtime | Feature status | Latest retained measurement | Test metadata and evidence |
|---|---|---:|---|
| **CUDA** | **Verified** chunked product-cache build | **12.31 s** | 2026-07-28 · `1024x1024x192x192` native `uint16` · no crop/bin · 12 GB allocator cap · full product build · [`1c5dd03b`](https://github.com/bobleesj/quantem.gpu/commit/1c5dd03b3ba60b98417449e55a18f0e41a58536b) · [CUDA-CAL-BUILD](performance/results.md) |
| **Python MPS** | **Verified** chunked product-cache build | **3.96 s** | 2026-07-21 · `512x512x192x192` native `uint16` · no crop/bin · exact Mac model not retained · mean DP/BF/DF bit-exact; CoM max error `7.63e-6` · **historical diagnostic** · [MPS-CAL-BUILD](performance/results.md) |
| **Backend-neutral cache** | **Verified** derived-product reopen; raw detector volume is not reopened | **6.8 ms** | 2026-07-20 · fastest of five retained repeats for a full `1024` scan · host/storage not retained · **historical diagnostic** · [PRODUCT-CACHE-REOPEN](performance/results.md) |

### Virtual images — `quantem.gpu.detector`

This module owns mean diffraction and exact masked sums for BF, ABF, ADF, DF,
and arbitrary detector masks.

| CUDA | Python MPS | Swift/Metal | WebGPU | CPU reference |
|---|---|---|---|---|
| **Verified** | **Verified** | **Verified** | **Verified** on hardware | **Reference** |

No isolated MPS or Swift/Metal timing with complete public provenance is
retained; those implementations remain supported through their parity gates.

| Implementation | Operation | Time | Data and parity | Test metadata and evidence |
|---|---|---:|---|---|
| **CUDA** | BF | **1.35 ms** | Warm resident full `512x512x192x192`; integer max error `0` | 2026-07-19 · CUDA GPU, exact model/mask not retained · [`0456e15e`](https://github.com/bobleesj/quantem.gpu/commit/0456e15ebfecd3627794118a930295e55a5a709b) · **historical diagnostic** · [CUDA-BF-512](performance/results.md) |
| **CUDA** | ADF | **3.86 ms** | Same resident source; integer max error `0` | 2026-07-19 · same device/revision · **historical diagnostic** · [CUDA-ADF-512](performance/results.md) |
| **CUDA** | DF | **1.84 ms** | Same resident source; integer max error `0` | 2026-07-19 · same device/revision · **historical diagnostic** · [CUDA-DF-512](performance/results.md) |
| **WebGPU** | BF selected-source page total | **0.378 s p50** | Full `512x512x192x192`; fixed 30 px radius; exact to CUDA | 2026-07-20 · Chrome on Apple `metal-3`, exact Mac model not retained · prepared selected-block source · 946 cycles · [`b61572e4`](https://github.com/bobleesj/quantem.gpu/commit/b61572e47058e4b3fa48835541f667d46b762cf0) · [WEBGPU-BF-512](performance/results.md) |

### Detector moments and phase contrast — `quantem.gpu.dpc`

This module owns CoM row/column, centering, rotation, DPC, and iDPC. All runtime
boundaries preserve `(row, column)` component order.

| CUDA | Python MPS | Swift/Metal | WebGPU | CPU reference |
|---|---|---|---|---|
| **Verified** | **Verified** | **Verified** | **Verified** on hardware | **Reference** |

| Implementation | Operation | Time | Precision and parity | Test metadata and evidence |
|---|---|---:|---|---|
| **CUDA** | CoM row + column | **12.39 ms** | Warm resident full `512x512x192x192`; `float32`; max error `0` | 2026-07-19 · CUDA GPU, exact model not retained · [`0456e15e`](https://github.com/bobleesj/quantem.gpu/commit/0456e15ebfecd3627794118a930295e55a5a709b) · **historical diagnostic** · [CUDA-COM-512](performance/results.md) |
| **WebGPU** | DPC row display | **14.9 ms p50** | Warm resident full `512x512x192x192`; `float32`; max error `7.63e-6` | 2026-07-20 · headed Chrome · NVIDIA RTX PRO 6000 Blackwell · [`cee0ba5c`](https://github.com/bobleesj/quantem.gpu/commit/cee0ba5ca3725b03054ecf5e6a14e304bb93d4ed) · [WEBGPU-DPC-512](performance/results.md) |
| **WebGPU** | DPC column display | **13.2 ms p50** | Same source and precision; max error `7.63e-6` | 2026-07-20 · same device/revision · [WEBGPU-DPC-512](performance/results.md) |
| **WebGPU** | iDPC display | **13.2 ms p50** | Same source; mean/max error `4.70e-6/3.05e-5` | 2026-07-20 · same device/revision · [WEBGPU-DPC-512](performance/results.md) |

### Single-sideband ptychography — `quantem.gpu.SSB`

SSB uses specialized kernels for **square scan grids**. The numbers below are
scan sizes, not detector sizes. “Native” means a retained acquisition at that
scan size; resized or synthetic evidence is labeled explicitly.

| Implementation | `128x128` | `256x256` | `512x512` | `1024x1024` | Evidence boundary |
|---|---|---|---|---|---|
| **CUDA** | **Verified** kernel/reference | **Verified** kernel/reference | **Verified** real full-BF | **Verified** kernel/reference | Fixed-size production registries for all four sizes; CUDA is the only runtime with all object-redraw cells reference-checked. |
| **Python MPS** | **Verified** resized/synthetic | **Verified** resized/synthetic | **Verified** native full-BF | **Verified** resized/synthetic | Physical Apple MPS runs at all four sizes; only `512x512` has native-acquisition evidence in the retained scaling set. |
| **WebGPU** | **Verified** real BF30 vs CUDA | **Implemented** synthetic browser reference | **Partial** real interaction | **Partial** real load/interaction | Production source supports all four sizes. Frozen real CUDA-reference artifacts remain incomplete for `256/512/1024`. |
| **Swift/Metal** | **Not implemented** | **Not implemented** | **Not implemented** | **Not implemented** | There is no native Swift SSB kernel; clients must select a supported external implementation explicitly. |
| **CPU reference** | — | — | **Reference** fixture | — | The retained public reference fixture is `512x512`; CPU is not a production SSB runtime. |

Representative warm phase-and-loss measurements use the complete stated BF
policy and exclude loading/preparation:

| Implementation | Operation | Time | Data and precision | Test metadata and evidence |
|---|---|---:|---|---|
| **CUDA** | `512x512` full-BF phase + loss | **32.2 ms p50** | Real prepared field; `float32`/`complex64` | 2026-07-19 · CUDA GPU, exact model not retained · [`0456e15e`](https://github.com/bobleesj/quantem.gpu/commit/0456e15ebfecd3627794118a930295e55a5a709b) · same BF disk, aberrations, objective, and loss reference · **historical diagnostic** · [SSB-CUDA-512-FULL](performance/results.md) |
| **Python MPS** | `512x512` full-active-BF phase + loss | **537.58 ms p50** | Real prepared Hermitian $G(\mathbf k,\boldsymbol{\nu})$; `float32`/`complex64` | 2026-07-28 · Apple Silicon GPU, exact Mac model not retained · [`e8d49866`](https://github.com/bobleesj/quantem.gpu/commit/e8d49866ea16cc57c0073d734c448cbbf601a5a5) · frozen loss `0.0885396` · **historical diagnostic** · [SSB-MPS-512-FULL](performance/results.md) |

The [SSB performance record](maintainer/ssb-performance.md) contains the full
`128/256/512/1024` timing matrix, native-versus-resized provenance, memory, and
rejected experiments.

### Other public modules

| Module | Supported implementation boundary | Performance status |
|---|---|---|
| `display` | CUDA, Python MPS, Swift/Metal, WebGPU, and CPU reference share transform, histogram, colormap, and FFT parity contracts | Feature/parity evidence retained; use the [display kernel page](kernels/display-export.md) for operation-specific gates |
| `movie` | CUDA/NVENC and Python MPS/VideoToolbox; native clients consume the package products | Movie smoke/parity tests retained; no single cross-runtime headline is comparable |
| `parallax` | CUDA only; other backends fail explicitly | No current top-line public benchmark |
| `optics` and `device` | Backend-neutral physics and explicit device selection | Supporting modules rather than timed scientific kernels |

```{admonition} How to read these numbers
:class: important
Milliseconds for resident kernels are **not loading times**. A warm source, a
prepared index, an application first product, and a saved-result reopen are
different states. A feature marked supported without an isolated time is still
covered by its parity gate.
```

For the complete command, cache state, bin/crop plan, memory record, and parity
artifact behind each row, follow its evidence ID. See the
[complete module dashboard](dashboard.md),
[methodology](performance/methodology.md), and
[revision ledger](performance/changes.md).

## How loading becomes a usable product

```text
START WALL CLOCK
      │
      ▼
HDF5 master + compressed shards
      │  open, metadata, source identity, prepared index lookup/build
      ▼
Verified source geometry: scan shape, detector shape, dtype, calibration
      │  estimate resident + scratch + product memory
      ▼
Explicit load plan: full scan; no automatic real-space crop;
                    detector bin; source/accumulation/output dtype; reason
      │  plan source-aligned chunks and reusable buffers
      ▼
Storage read ──overlap──► GPU bitshuffle/LZ4 decode
                              │  bad-pixel policy + dtype conversion
                              │  + exact detector sum/bin when selected
                              ▼
Resident I[R_r,R_c,k_r,k_c] + complete provenance
      │  fused/reused GPU reductions
      ▼
Mean diffraction, BF/ADF/DF, CoM, DPC, iDPC
      │
      ▼
FIRST COMPLETE USABLE PRODUCT  ← STOP WALL CLOCK
      │
      └── optional cache/finalization, reported separately
```

Detector binning is exact block summation, not interpolation or cropping. It
may run while chunks are decoded so the unbinned 4D volume is never
materialized unnecessarily. The metadata still reports the original detector
shape, selected bin, output shape, accumulation/output dtype, memory estimate,
and policy reason. See [Load, decode, and bin](kernels/load-decode-bin.md) for
the mathematical contract and [Benchmark methodology](performance/methodology.md)
for every timed stage.

## The shared coordinate contract

Every backend interprets 4D-STEM data as

$$
I[R_r,R_c,k_r,k_c],
\qquad (\text{row},\text{column})\equiv(r,c),
$$

where $\mathbf R=(R_r,R_c)$ is the real-space probe/scan coordinate and
$\mathbf k=(k_r,k_c)$ is the detector coordinate. A private device layout may
be flattened, transposed, tiled, packed, or detector-major, but the public
shape, masks, metadata, and results preserve this meaning.

Read [Data model and coordinates](kernels/data-model.md) before implementing a
new kernel.

## Find the operation you are implementing

| Operation | Meaning | Kernel page |
|---|---|---|
| Load/decode/bin | compressed source to typed resident counts | [Load, decode, and bin](kernels/load-decode-bin.md) |
| Virtual detector | BF/DF/ADF and mean-diffraction reductions | [BF, DF, and ADF](kernels/virtual-detectors.md) |
| Detector moments | CoM row, CoM column, DPC, and iDPC | [CoM, DPC, and iDPC](kernels/com-dpc-idpc.md) |
| Ptychography | SSB object, phase, loss, and aberrations | [Single-sideband ptychography](kernels/ssb.md) |
| Scan selection | explicit half-open real-space subsets | [Explicit scan regions](kernels/scan-regions.md) |
| Presentation math | ranges, histograms, colormaps, FFT views, movies | [Display and export kernels](kernels/display-export.md) |

Each page combines the scientific equations, exactness and provenance rules,
optimization model, backend source map, and parity gate. This keeps the math
beside the operation instead of separating it into a generic tutorial.

## Choose the runtime you are implementing

| Runtime | Start here | Primary implementation boundary |
|---|---|---|
| CUDA | [CUDA](platforms/cuda.md) | Python adapters, CuPy, CUDA C/RawKernel |
| Python MPS | [Python MPS](platforms/mps.md) | Python adapters, MLX/PyObjC, Metal kernels |
| Native Swift/Metal | [Native Swift and Metal](platforms/swift-metal.md) | SwiftPM products and bundled Metal resources |
| WebGPU | [WebGPU](platforms/webgpu.md) | TypeScript adapters and WGSL resources |
| CPU reference | [CPU reference](platforms/cpu-reference.md) | independent NumPy/reference implementation |

All runtimes implement the same operation contract. They do not expose
platform-specific scientific workflows.

To run the CUDA implementation as a service, use
[QuantEM.GPU Remote](remote/index.md). Remote access is deployment and
communication, not another kernel runtime.

## What belongs in this package

`quantem.gpu` owns reusable accelerated IO, math, kernels, result contracts,
resource estimation, and scientific provenance. A consuming application owns
presentation, user-visible resource-policy choices, scheduling, and lifecycle.
No application framework or view state is required to build or test this
package.

Read [Kernel architecture](concepts/kernel-architecture.md) for the source tree
and [Kernel development lifecycle](developer/kernel-lifecycle.md) before adding
an implementation.

## How to interpret performance evidence

The [implementation overview](dashboard.md) is the dense one-page view
of implementation coverage and headline measurements. The
[Benchmarks and parity](performance/index.md) section keeps the complete
current and historical evidence with source revision, hardware, data
shape/dtype, cache state, load plan, memory peak, parity artifact, and
benchmark definition.

A cached reopen is not a first source load. A cropped or binned fixture is not
full-resolution evidence. A compile test is not a hardware benchmark. Rejected
experiments remain recorded so kernel developers can avoid repeating known
regressions.

## Start coding

Install the runtime you need, run the smallest relevant parity test, then use
the physical target device for performance evidence:

```bash
python -m pip install -e ".[dev,docs]"
PYTHONPATH=src python -m pytest -q
swift test
```

See [Install](install.md), [API contracts](api/index.md), and
[Contributing](developer/index.md).

## Citing and support

If this package contributed to your research, see
[CITATION.cff](https://github.com/bobleesj/quantem.gpu/blob/main/CITATION.cff).
Questions and bug reports belong in the
[issue tracker](https://github.com/bobleesj/quantem.gpu/issues).
