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

## Speed benchmark overview

**Overview reviewed:** 2026-08-19. Choose the table that matches the work being
timed. Loading, a resident scientific kernel, SSB reconstruction, and saving
have different boundaries and should not be ranked as though they were the same
operation. Each row contains one headline measurement; its test metadata is in
the rightmost column.

### Loading and first usable product

These measurements include storage or prepared-source work. Only the
Swift/Metal rows are application wall time through the first complete product.

| Implementation | Time | What was timed | Data and scientific plan | Test metadata and evidence |
|---|---:|---|---|---|
| **Swift/Metal**, fixture A | **1.985 s p50** | First-process action → first complete product | `512x512x192x192` `uint16`; full scan; no crop; scan bin 1; exact detector sum bin 4 → `48x48`; resident `uint16` | 2026-08-18 · physical 8 GB M2 MacBook Air (`Mac14,2`) · candidate `2c047160` · seven isolated runs · peak process `1.43 GB` · zero swap delta · eight product exports byte-identical · [M2-AIR-BIN4-E2E](performance/results.md) |
| **Swift/Metal**, fixture B | **2.043 s p50** | First-process action → first complete product | Same full-scan plan as fixture A | 2026-08-18 · physical 8 GB M2 MacBook Air (`Mac14,2`) · integration [`e662d7fe`](https://github.com/bobleesj/quantem.gpu/commit/e662d7feebf78e7c1513276651d0be55a555cb40) · seven isolated runs · diffraction hash retained · [M2-AIR-BIN4-E2E](performance/results.md) |
| **CUDA** | **0.450 s median** | Warm-source load and decompression; first product excluded | `512x512x192x192` `uint16`; full scan; no crop/bin; audited lossless-low8 `uint8` output | 2026-07-20 · NVIDIA RTX PRO 6000 Blackwell · [`b61572e4`](https://github.com/bobleesj/quantem.gpu/commit/b61572e47058e4b3fa48835541f667d46b762cf0) · 946 runs · selected integer checksums exact · [CUDA-512-LOAD](performance/results.md) |
| **Python MPS** | **3.96 s** | First product-cache build; source-cache state not retained | `512x512x192x192` native `uint16`; no crop/bin; 64-row chunks | 2026-07-21 · Apple Metal GPU, exact Mac model not retained · [`6c8ca5d0`](https://github.com/bobleesj/quantem.gpu/commit/6c8ca5d0a66bf78d88d6310fba5a1b9a2ea50326) · mean DP/BF/DF bit-exact; CoM max error `7.63e-6` · **historical diagnostic** · [MPS-CAL-BUILD](performance/results.md) |
| **WebGPU** | **0.772 s p50** | Prepared local-file full-stack load; not cold | `512x512x192x192` `uint16`; full scan; no crop/bin; audited lossless-low8 `uint8` output | 2026-07-20 · Chrome on Apple `metal-3`, exact Mac model not retained · [`b61572e4`](https://github.com/bobleesj/quantem.gpu/commit/b61572e47058e4b3fa48835541f667d46b762cf0) · 946 cycles · corrected-frame checksums exact to CUDA · [WEBGPU-512-FULL](performance/results.md) |

### Resident scientific products

Source loading is excluded here. Every row is one warm, GPU-resident product
measurement on the full `512x512x192x192` data volume with no crop or binning.

| Implementation | Product | Time | Precision and parity | Test metadata and evidence |
|---|---|---:|---|---|
| **CUDA** | BF | **1.35 ms** | Integer sum; max error `0` | 2026-07-19 · CUDA GPU, exact model not retained · [`0456e15e`](https://github.com/bobleesj/quantem.gpu/commit/0456e15ebfecd3627794118a930295e55a5a709b) · **historical diagnostic** · [CUDA-BF-512](performance/results.md) |
| **CUDA** | ADF | **3.86 ms** | Integer sum; max error `0` | 2026-07-19 · same device/revision · **historical diagnostic** · [CUDA-ADF-512](performance/results.md) |
| **CUDA** | DF | **1.84 ms** | Integer sum; max error `0` | 2026-07-19 · same device/revision · **historical diagnostic** · [CUDA-DF-512](performance/results.md) |
| **CUDA** | CoM row + column | **12.39 ms** | `float32`; max error `0` | 2026-07-19 · same device/revision · **historical diagnostic** · [CUDA-COM-512](performance/results.md) |
| **WebGPU** | DPC row | **14.9 ms p50** | `float32`; max error `7.63e-6` | 2026-07-20 · headed Chrome · NVIDIA RTX PRO 6000 Blackwell · [`cee0ba5c`](https://github.com/bobleesj/quantem.gpu/commit/cee0ba5ca3725b03054ecf5e6a14e304bb93d4ed) · [WEBGPU-DPC-512](performance/results.md) |
| **WebGPU** | DPC column | **13.2 ms p50** | `float32`; max error `7.63e-6` | 2026-07-20 · same device/revision · [WEBGPU-DPC-512](performance/results.md) |
| **WebGPU** | iDPC | **13.2 ms p50** | `float32`; mean/max error `4.70e-6/3.05e-5` | 2026-07-20 · same device/revision · [WEBGPU-DPC-512](performance/results.md) |

### Single-sideband ptychography

These are warm, prepared phase-and-loss kernel measurements. Loading and source
preparation are excluded.

| Implementation | Time | Data and precision | Test metadata and evidence |
|---|---:|---|---|
| **CUDA** | **32.2 ms p50** | Real `512x512` field; full-BF policy; `float32`/`complex64` | 2026-07-19 · CUDA GPU, exact model not retained · [`0456e15e`](https://github.com/bobleesj/quantem.gpu/commit/0456e15ebfecd3627794118a930295e55a5a709b) · same BF disk, aberrations, objective, and loss reference · **historical diagnostic** · [SSB-CUDA-512-FULL](performance/results.md) |
| **Python MPS** | **537.58 ms p50** | Real `512x512` Hermitian legacy `G_qk`; full active BF; `float32`/`complex64` | 2026-07-28 · Apple Silicon GPU, exact Mac model not retained · [`e8d49866`](https://github.com/bobleesj/quantem.gpu/commit/e8d49866ea16cc57c0073d734c448cbbf601a5a5) · same BF policy, aberrations, precision, objective, and frozen loss · **historical diagnostic** · [SSB-MPS-512-FULL](performance/results.md) |

### Compressed saving

| Implementation | Time | What was timed | Data and precision | Test metadata and evidence |
|---|---:|---|---|---|
| **Python MPS/Metal** | **1.69 s** | Best retained compression/HDF5-write sweep; source load excluded | Warm resident `512x512x192x192` `uint16`; no crop/bin; batch 2048; Bitshuffle/LZ4; output `1.205 GB` | 2026-07-25 · Apple Metal GPU, exact Mac/storage model not retained · [`3061501`](https://github.com/bobleesj/quantem.gpu/commit/30615019cfe293ae9759006ae89c0e378b7065fd) · decoded samples exact · **historical diagnostic** · [MPS-SAVE-U16-512](performance/results.md) |
| **Python MPS/Metal** | **1.91 s** | Public-default confirmation; source load excluded | Same source, precision, and compression plan | 2026-07-25 · public API [`83bb608`](https://github.com/bobleesj/quantem.gpu/commit/83bb6089e11604b5828e6f94a70d49e487e75929) · decoded samples exact · **historical diagnostic** · [MPS-SAVE-U16-512](performance/results.md) |

```{admonition} How to read these numbers
:class: important
Milliseconds for resident kernels are **not loading times**. A warm source, a
prepared index, an application first product, and a saved-result reopen are also
different states. CPU is the independent correctness reference—not a benchmarked
accelerator and never a silent fallback.
```

For the complete command, cache state, bin/crop plan, memory record, and parity
artifact behind each row, follow its evidence ID. See the
[complete benchmark dashboard](dashboard.md),
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
