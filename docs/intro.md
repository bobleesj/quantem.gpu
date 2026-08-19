# quantem.gpu

One scientific GPU contract for electron microscopy—from compressed detector
data to exact products—across NVIDIA CUDA, Apple MPS/Metal, native Swift,
WebGPU, and an explicit CPU reference.

```{admonition} Living pre-release draft
:class: important
This site documents an evolving `0.0.1` release-candidate series. Public APIs,
runtime coverage, and recommendations may change between candidates. Python
examples currently pin the exact TestPyPI candidate
`quantem.gpu==0.0.1rc6`; Swift consumers pin an exact verified Git revision.

The documentation is a draft, but retained performance and parity rows are not
draft estimates: each is a frozen historical measurement tied to its stated
date, source revision, device, data plan, cache state, and acceptance rule. A
newer candidate replaces the documented pin only after those gates are rerun.
```

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

Every module table starts with the implementation platform. Cells use the same
evidence vocabulary as the [complete module dashboard](dashboard.md):

- **✓** — verified with retained real-data or physical-device parity evidence.
- **Test** — deterministic tests exist, but equivalent physical evidence
  is not retained.
- **Pending** — the source path exists, but the exact size, bin, or timing row
  still needs evidence.
- **Ref** — independent CPU correctness adjudication, never fallback.
- **—** — unsupported or not a target.

No empty cell implies support. Date, device, revision, scientific plan, and
parity remain attached to every retained performance result.

For the compact wall-time, scan/detector plan, peak-memory, and **4/6 GiB**
capacity view, open
{ref}`Speed and memory at a glance <speed-and-memory-at-a-glance>`.

### I/O — `quantem.gpu.io`

The I/O module owns discovery, inspection, accelerated bitshuffle/LZ4 decode,
explicit crop/bin geometry, dtype conversion, provenance, and compressed HDF5
saving.

| Platform | Load/decode/provenance | Scan sizes | Detector bins | Compressed save | Latest retained result and exact plan |
|---|---|---|---|---|---|
| **CUDA** | ✓ | `128/256/512/1024` ✓ | bin 1/2 ✓; bin 4/8 **Pending** | ✓ | **0.450 s median**, warm `512x512x192x192`, bin 1, audited low8; [CUDA-512-LOAD](performance/results.md), 2026-07-20, `b61572e4` |
| **Python MPS** | ✓ | `128/256/512/1024` ✓ | bin 1/2 ✓; bin 4/8 **Pending** | ✓ | load **4.617 s**, first-observed native `1024`, bin 1; save **1.69 s** sweep / **1.91 s** public default for warm `512` `uint16`; [MPS-1024-LOAD](performance/results.md), [MPS-SAVE-U16-512](performance/results.md) |
| **Native Swift/Metal** | ✓ | `512` ✓; `128/256/1024` **Pending** | bin 1 ✓; bin 2 **Test**; bin 4 ✓; bin 8 — | Resident cache only; HDF5 save — | **1.985 / 2.043 s p50**, first process/observed `512`, full scan, detector bin 4; physical 8 GB M2 Air; [M2-AIR-BIN4-E2E](performance/results.md), 2026-08-18, `2c047160`/`e662d7fe` |
| **WebGPU** | ✓ | `128/256/512` ✓; `1024` partial product-first | bin 1/2/4/8 ✓ | — | no-bin **0.772 s p50** prepared full `512`; bin 2/4/8 **1.199/1.212/1.106 s**; [WEBGPU-512-FULL](performance/results.md), [WEBGPU-DET-BIN](performance/results.md) |
| **CPU reference** | Ref | All sizes Ref | bin 1/2/4/8 Ref | Reference writer | **Pending** — no comparable accelerator-style timing |

### Screening — `quantem.gpu.screening`

`screening.prepare` builds and reopens the small mean-diffraction, BF, DF, CoM,
rotation, and iDPC launch products. It is a separate module from raw I/O.

| Platform | Module support | Size and explicit plan | Latest retained result | Details |
|---|---|---|---:|---|
| **CUDA** | ✓ | `1024x1024x192x192` native `uint16`, bin 1, 12 GB cap | **12.31 s** build | [CUDA-CAL-BUILD](performance/results.md), 2026-07-28, `1c5dd03b` |
| **Python MPS** | ✓ | `512x512x192x192` native `uint16`, bin 1, 64-row chunks | **3.96 s** build | [MPS-CAL-BUILD](performance/results.md), 2026-07-21, `6c8ca5d0`; integer products exact, CoM max error `7.63e-6` |
| **Native Swift/Metal** | — | — | **Pending** | Public `quantem.gpu.screening` module not implemented natively |
| **WebGPU** | — | — | **Pending** | Public `quantem.gpu.screening` module not implemented for WebGPU |
| **CPU reference** | Ref | Reference fixtures | **Pending** | Correctness adjudication only |

Backend-neutral [PRODUCT-CACHE-REOPEN](performance/results.md) is a separate
saved-result state: **6.8 ms** fastest retained repeat for full-`1024` derived
products, never source load or cache construction.

### Virtual images — `quantem.gpu.detector`

This module owns mean diffraction and exact masked sums for BF, ABF, ADF, DF,
and arbitrary detector masks.

| Platform | Mean DP and BF/ABF/ADF/DF | Measured size/bin plan | Latest retained result | Details |
|---|---|---|---:|---|
| **CUDA** | ✓ | Resident full `512x512x192x192`, bin 1 | BF **1.35 ms**; ADF **3.86 ms**; DF **1.84 ms** | [CUDA-BF-512](performance/results.md), [CUDA-ADF-512](performance/results.md), [CUDA-DF-512](performance/results.md); integer max error `0` |
| **Python MPS** | ✓ | Full `512`, bin 1 through screening parity | **Pending** | No isolated timing with complete public device provenance |
| **Native Swift/Metal** | ✓ | Full `512`, detector bin 4 in physical application parity | **Pending** | Products are byte-identical; isolated kernel timing not retained |
| **WebGPU** | ✓ | Full `512`, bin 1, fixed 30 px BF radius | BF page total **0.378 s p50** | [WEBGPU-BF-512](performance/results.md), prepared selected-block boundary, exact to CUDA |
| **CPU reference** | Ref | Reference fixtures | **Pending** | Correctness adjudication only |

### Detector moments and phase contrast — `quantem.gpu.dpc`

This module owns CoM row/column, centering, rotation, DPC, and iDPC. All runtime
boundaries preserve `(row, column)` component order.

| Platform | CoM row/column, DPC, rotation, iDPC | Measured size/bin plan | Latest retained result | Details |
|---|---|---|---:|---|
| **CUDA** | ✓ | Resident full `512x512x192x192`, bin 1 | CoM **12.39 ms** | [CUDA-COM-512](performance/results.md), 2026-07-19, `0456e15e`; max error `0` |
| **Python MPS** | ✓ | Full `512`, bin 1 through screening parity | **Pending** | No isolated full-module timing with complete public device provenance |
| **Native Swift/Metal** | ✓ | Full `512`, detector bin 4 in physical application parity | **Pending** | CoM/DPC/iDPC exports byte-identical; isolated timing not retained |
| **WebGPU** | ✓ | Resident full `512x512x192x192`, bin 1 | DPC row/column/iDPC **14.9/13.2/13.2 ms p50** | [WEBGPU-DPC-512](performance/results.md), 2026-07-20, `cee0ba5c`; frozen float32 errors retained |
| **CPU reference** | Ref | Reference fixtures | **Pending** | Correctness adjudication only |

### Single-sideband ptychography — `quantem.gpu.SSB`

SSB uses specialized kernels for **square scan grids**. The numbers below are
scan sizes, not detector sizes. “Native” means a retained acquisition at that
scan size; resized or synthetic evidence is labeled explicitly.

| Platform | `128x128` | `256x256` | `512x512` | `1024x1024` | Latest retained full-policy result | Details / next gap |
|---|---|---|---|---|---:|---|
| **CUDA** | ✓ | ✓ | ✓ | ✓ | **32.2 ms p50** at `512` | [SSB-CUDA-512-FULL](performance/results.md); `512` is real full-BF, with fixed-size kernel/reference checks at all sizes |
| **Python MPS** | ✓ | ✓ | ✓ | ✓ | **537.58 ms p50** at `512` | [SSB-MPS-512-FULL](performance/results.md); `512` is native full-BF, other sizes resized/synthetic |
| **Native Swift/Metal** | — | — | — | — | **Pending** | No native Swift SSB kernel |
| **WebGPU** | ✓ | Test | Partial | Partial | **Pending** | `128` is real BF30 vs CUDA; `256` is test-only; `512/1024` have real interaction but incomplete frozen CUDA artifacts |
| **CPU reference** | — | — | Ref | — | **Pending** | Independent `512` adjudication only |

The [SSB performance record](maintainer/ssb-performance.md) contains the full
`128/256/512/1024` timing matrix, native-versus-resized provenance, memory, and
rejected experiments.

### Other public modules

| Platform | `display` | `movie` | `parallax` | Details |
|---|---|---|---|---|
| **CUDA** | ✓ | NVENC ✓ | CUDA-only implementation | Operation-specific gates; no single comparable cross-module headline |
| **Python MPS** | ✓ | VideoToolbox ✓ | — | Operation-specific gates; no single comparable cross-module headline |
| **Native Swift/Metal** | `MetalDisplayKernels`/`MetalImageFFT` ✓ | Native package products | — | Display/FFT parity retained; standalone movie headline **Pending** |
| **WebGPU** | ✓ | — | — | Real-adapter display evidence retained |
| **CPU reference** | Ref | Explicit fallback | — | Correctness adjudication only |

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
