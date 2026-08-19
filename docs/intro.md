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

**Overview reviewed:** 2026-08-19. This table records the latest retained
measurement for each major runtime/workload represented in the evidence ledger.
“Latest retained” does not mean “measured on the current source tree”: the date
and measured revision identify exactly what ran.

| Stack and workload | Tested date and measured revision | Physical device | Data, precision, and load plan | State and timing boundary | Latest retained speed | Correctness and evidence status |
|---|---|---|---|---|---:|---|
| **Native Swift/Metal — application load + products** | 2026-08-18; candidate `2c047160`, integration [`e662d7fe`](https://github.com/bobleesj/quantem.gpu/commit/e662d7feebf78e7c1513276651d0be55a555cb40) | Physical 8 GB `Mac14,2` M2 MacBook Air | `512x512x192x192` source `uint16`; full scan; no crop; scan bin 1; exact detector sum bin 4 → `48x48`; resident `uint16` | First process, OS cache not forcibly evicted; action → first complete product; seven alternating isolated runs per fixture | Fixture A/B wall **1.985/2.043 s p50**; Metal **1.615-1.618 s p50**; peak process `1.43 GB`, zero swap delta | Eight BF/ABF/ADF/CoM/DPC/iDPC exports byte-identical; diffraction hashes retained. [M2-AIR-BIN4-E2E](performance/results.md) |
| **CUDA — full-source load** | 2026-07-20; [`b61572e4`](https://github.com/bobleesj/quantem.gpu/commit/b61572e47058e4b3fa48835541f667d46b762cf0) | NVIDIA RTX PRO 6000 Blackwell | `512x512x192x192` source `uint16`; full scan; no crop/bin; audited lossless-low8 `uint8` browse output | Warm source; 946-run load/decompress median | **0.450 s**; decoded resident `9.66 GB` | Selected corrected-frame integer checksums exact. Not first-encounter evidence. [CUDA-512-LOAD](performance/results.md) |
| **Python MPS/Metal — product-cache build** | 2026-07-21; [`6c8ca5d0`](https://github.com/bobleesj/quantem.gpu/commit/6c8ca5d0a66bf78d88d6310fba5a1b9a2ea50326) | Apple Metal GPU; exact Mac model not retained | `512x512x192x192` native `uint16`; no crop/bin; 64-row chunks | First product-cache build; source-cache state not retained; raw HDF5 stream + Metal reductions | **3.96 s** build; `3.95 s` stream; `29.3 ms/chunk` reduction p50 | Mean DP/BF/DF bit-exact to CUDA; CoM max error `7.63e-6`. **Historical diagnostic:** incomplete hardware provenance. [MPS-CAL-BUILD](performance/results.md) |
| **WebGPU — full-stack local-file load** | 2026-07-20; [`b61572e4`](https://github.com/bobleesj/quantem.gpu/commit/b61572e47058e4b3fa48835541f667d46b762cf0) | Chrome, Apple `metal-3` adapter; exact Mac model not retained | `512x512x192x192` source `uint16`; full scan; no crop/bin; audited lossless-low8 `uint8` output | Prepared indexes/sidecars; 946-cycle full-stack soak | **0.772 s p50**, `0.726-0.879 s` range | First/middle/last corrected-frame checksums exact to CUDA. Prepared, not cold. [WEBGPU-512-FULL](performance/results.md) |
| **CUDA — resident BF/ADF/DF/CoM kernels** | 2026-07-19; [`0456e15e`](https://github.com/bobleesj/quantem.gpu/commit/0456e15ebfecd3627794118a930295e55a5a709b) | CUDA GPU; exact model not retained | GPU-resident full `512x512x192x192`; no crop/bin; integer sums and float32 CoM | Warm resident-kernel before → after microbenchmarks; source load excluded | BF `4.96→1.35 ms`; ADF `16.16→3.86 ms`; DF `62.64→1.84 ms`; CoM `200.42→12.39 ms` | Integer product max error `0`; CoM max error `0`. **Historical diagnostic:** device/mask provenance incomplete. [CUDA-BF/ADF/DF/COM-512](performance/results.md) |
| **WebGPU — resident DPC/iDPC** | 2026-07-20; [`cee0ba5c`](https://github.com/bobleesj/quantem.gpu/commit/cee0ba5ca3725b03054ecf5e6a14e304bb93d4ed) | Headed Chrome, NVIDIA RTX PRO 6000 Blackwell | GPU-resident full `512x512x192x192`; no crop/bin; float32 DPC/iDPC | Warm resident display p50; source load excluded | row/column/iDPC **14.9/13.2/13.2 ms p50** | DPC max error `7.63e-6`; iDPC mean/max `4.70e-6/3.05e-5`. [WEBGPU-DPC-512](performance/results.md) |
| **CUDA — SSB phase + loss** | 2026-07-19; [`0456e15e`](https://github.com/bobleesj/quantem.gpu/commit/0456e15ebfecd3627794118a930295e55a5a709b) | CUDA GPU; exact model not retained | Prepared real `512x512` field; full-BF policy; `float32`/`complex64` | Warm prepared phase+loss kernel; source preparation excluded | **32.2 ms p50**, `33.3 ms p95` | Same BF disk, aberrations, objective, and loss reference. **Historical diagnostic:** device/BF-count provenance incomplete. [SSB-CUDA-512-FULL](performance/results.md) |
| **Python MPS — SSB phase + loss** | 2026-07-28; [`e8d49866`](https://github.com/bobleesj/quantem.gpu/commit/e8d49866ea16cc57c0073d734c448cbbf601a5a5) | Apple Silicon GPU; exact Mac model not retained | Prepared real `512x512` Hermitian legacy `G_qk`; full active BF; `float32`/`complex64` | Warm prepared phase+loss kernel; source preparation excluded | **537.58 ms p50**, `557.51 ms p95`; loss `0.0885396` | Same full-active BF policy, aberrations, precision, objective, and frozen loss. **Historical diagnostic:** host/BF-count provenance incomplete. [SSB-MPS-512-FULL](performance/results.md) |
| **Python MPS/Metal — compressed save** | 2026-07-25; [`3061501`](https://github.com/bobleesj/quantem.gpu/commit/30615019cfe293ae9759006ae89c0e378b7065fd), API [`83bb608`](https://github.com/bobleesj/quantem.gpu/commit/83bb6089e11604b5828e6f94a70d49e487e75929) | Apple Metal GPU; exact Mac/storage model not retained | Chunk-backed `512x512x192x192` `uint16`; no crop/bin; batch 2048; bitshuffle/LZ4 HDF5 | Warm resident source; compression + HDF5 write; source load excluded | sweep **1.69-1.80 s**; public default `1.91 s`; output `1.205 GB` | Decoded samples exact, mismatches `0`. **Historical diagnostic:** host/storage provenance incomplete. [MPS-SAVE-U16-512](performance/results.md) |
| **CPU — independent reference** | No performance signoff | CPU reference, intentionally hardware-independent | Exact scientific fixtures and widened/reference arithmetic | Correctness adjudication only; never a silent production fallback | **Not benchmarked as an accelerator** | Defines exact integers and frozen floating metrics used to accept CUDA, MPS/Metal, Swift/Metal, and WebGPU. [Parity contract](performance/parity.md) |

These rows are not a platform ranking, and millisecond resident-kernel timings
are not load times. Click the evidence link for the complete command and
artifact record, including cache state, bin/crop plan, and parity artifact. See the
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
