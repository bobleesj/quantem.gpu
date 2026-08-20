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

No empty cell implies support. Overview timing rows show the device and test
date. Exact revision, scientific plan, and parity remain in the
[verified benchmark results](performance/results.md).

For the compact wall-time, scan/detector plan, peak-memory, and **4/6 GiB**
capacity view, open
{ref}`Speed and memory at a glance <speed-and-memory-at-a-glance>`.

### I/O — `quantem.gpu.io`

The I/O module owns discovery, inspection, accelerated bitshuffle/LZ4 decode,
explicit crop/bin geometry, dtype conversion, provenance, and compressed HDF5
saving.

One row below is one exact configuration and one headline statistic. The full
[implementation overview](dashboard.md) keeps source/decode/resident dtypes,
memory kind, parity, and unmeasured combinations in separate fields.
Scroll horizontally on narrow screens rather than combining fields.

| Platform | Selected scan | Scan plan | Detector bin | Output detector | Source dtype | Decode dtype | Resident dtype | Cache state | Fixture | Statistic | Time | Device tested | Date tested |
|---|---:|---|---:|---:|---|---|---|---|---|---|---:|---|---|
| **CUDA** | `512x512` | Full | 1 | `192x192` | `uint16` | `uint16` | `uint16` | Warm source | C | p50 | **0.588 s** | NVIDIA RTX PRO 6000 Blackwell, GPU 0 | 2026-08-19 |
| **CUDA** | `512x512` | Full | 2 | `96x96` | `uint16` | `uint16` | `uint32` | Warm source | C | p50 | **0.626 s** | NVIDIA RTX PRO 6000 Blackwell, GPU 0 | 2026-08-19 |
| **CUDA** | `512x512` | Full | 4 | `48x48` | `uint16` | `uint16` | `uint32` | Warm source | C | p50 | **0.553 s** | NVIDIA RTX PRO 6000 Blackwell, GPU 0 | 2026-08-19 |
| **CUDA** | `512x512` | Full | 8 | `24x24` | `uint16` | `uint16` | `uint32` | Warm source | C | p50 | **0.599 s** | NVIDIA RTX PRO 6000 Blackwell, GPU 0 | 2026-08-19 |
| **Python MPS** | `512x512` | Full | 1 | `192x192` | `uint16` | `uint16` | `uint16` | Warm source | C | p50 | **2.164 s** | Apple M5 Max (`Mac17,6`, 40-core GPU, 128 GB) | 2026-08-19 |
| **Python MPS** | `512x512` | Full | 2 | `96x96` | `uint16` | `uint16` | `uint16` | Warm source | C | p50 | **0.691 s** | Apple M5 Max (`Mac17,6`, 40-core GPU, 128 GB) | 2026-08-19 |
| **Python MPS** | `512x512` | Full | 4 | `48x48` | `uint16` | `uint16` | `uint16` | Warm source | C | p50 | **0.586 s** | Apple M5 Max (`Mac17,6`, 40-core GPU, 128 GB) | 2026-08-19 |
| **Python MPS** | `512x512` | Full | 8 | `24x24` | `uint16` | `uint16` | `uint16` | Warm source | C | p50 | **0.575 s** | Apple M5 Max (`Mac17,6`, 40-core GPU, 128 GB) | 2026-08-19 |
| **Python MPS** | `512x512` | Full | 1 | `192x192` | `uint16` | `uint16` | `uint16` | Admission check | C | Guard result | — | Apple M5 MacBook Pro (`Mac17,2`, 10-core GPU, 24 GB) | 2026-08-19 |
| **Python MPS** | `512x512` | Full | 2 | `96x96` | `uint16` | `uint16` | `uint16` | Warm source | C | p50 | **2.224 s** | Apple M5 MacBook Pro (`Mac17,2`, 10-core GPU, 24 GB) | 2026-08-19 |
| **Python MPS** | `512x512` | Full | 4 | `48x48` | `uint16` | `uint16` | `uint16` | Warm source | C | p50 | **1.695 s** | Apple M5 MacBook Pro (`Mac17,2`, 10-core GPU, 24 GB) | 2026-08-19 |
| **Python MPS** | `512x512` | Full | 8 | `24x24` | `uint16` | `uint16` | `uint16` | Warm source | C | p50 | **1.580 s** | Apple M5 MacBook Pro (`Mac17,2`, 10-core GPU, 24 GB) | 2026-08-19 |
| **CUDA** | `512x512` | Full | 4 | `48x48` | `uint16` | `uint16` | `uint32` | First campaign encounter | C | Single run | **2.027 s** | NVIDIA RTX PRO 6000 Blackwell, GPU 0 | 2026-08-19 |
| **Python MPS** | `512x512` | Full | 4 | `48x48` | `uint16` | `uint16` | `uint16` | First campaign encounter | C | Single run | **1.982 s** | Apple M5 Max (`Mac17,6`, 40-core GPU, 128 GB) | 2026-08-19 |
| **Python MPS** | `512x512` | Full | 4 | `48x48` | `uint16` | `uint16` | `uint16` | First campaign encounter | C | Single run | **2.775 s** | Apple M5 MacBook Pro (`Mac17,2`, 10-core GPU, 24 GB) | 2026-08-19 |
| **Native Swift/Metal** | `512x512` | Full | 4 | `48x48` | `uint16` | `uint16` | `uint16` | First process | A | p50 | **1.985 s** | Apple M2 MacBook Air (`Mac14,2`, 8 GB) | 2026-08-18 |
| **Native Swift/Metal** | `512x512` | Full | 4 | `48x48` | `uint16` | `uint16` | `uint16` | First process | B | p50 | **2.043 s** | Apple M2 MacBook Air (`Mac14,2`, 8 GB) | 2026-08-18 |
| **WebGPU** | `256x256` | Explicit crop | 1 | `192x192` | `uint16` | `uint8` | `uint8` | Prepared source | — | p50 | **0.338 s** | Apple Metal-3 adapter (Mac model not retained) | 2026-07-20 |
| **WebGPU** | `256x256` | Explicit crop | 2 | `96x96` | `uint16` | `uint8` | `float32` | Prepared source | — | p50 | **0.774 s** | NVIDIA RTX PRO 6000 Blackwell | 2026-07-20 |
| **WebGPU** | `256x256` | Explicit crop | 4 | `48x48` | `uint16` | `uint8` | `float32` | Prepared source | — | p50 | **0.755 s** | NVIDIA RTX PRO 6000 Blackwell | 2026-07-20 |
| **WebGPU** | `256x256` | Explicit crop | 8 | `24x24` | `uint16` | `uint8` | `float32` | Prepared source | — | p50 | **0.733 s** | NVIDIA RTX PRO 6000 Blackwell | 2026-07-20 |
| **WebGPU** | `512x512` | Full | 1 | `192x192` | `uint16` | `uint8` | `uint8` | Prepared source | — | p50 | **0.772 s** | Apple Metal-3 adapter (Mac model not retained) | 2026-07-20 |
| **WebGPU** | `512x512` | Full | 2 | `96x96` | `uint16` | `uint8` | `float32` | Prepared source | — | Single profile | **1.199 s** | NVIDIA RTX PRO 6000 Blackwell | 2026-07-20 |
| **WebGPU** | `512x512` | Full | 4 | `48x48` | `uint16` | `uint8` | `float32` | Prepared source | — | Single profile | **1.212 s** | NVIDIA RTX PRO 6000 Blackwell | 2026-07-20 |
| **WebGPU** | `512x512` | Full | 8 | `24x24` | `uint16` | `uint8` | `float32` | Prepared source | — | Single profile | **1.106 s** | NVIDIA RTX PRO 6000 Blackwell | 2026-07-20 |
| **WebGPU** | `512x512` | Full | 2 | `96x96` | `uint16` | `uint16` | `float32` | Prepared source | — | Single profile | **2.651 s** | NVIDIA RTX PRO 6000 Blackwell | 2026-07-20 |

Fixture C is the current full-scan, native-`uint16`, 27-shard real fixture. The
three machines have identical hashes for the master and every shard. No C row
uses a real-space crop; scan bin is 1 and detector bin is explicit. “First
campaign encounter” is not cold because the operating-system storage cache was
not reset. The complete p50/p95/max, memory, revision, and parity record is in
the [implementation overview](dashboard.md) and
[verified results](performance/results.md).

Native Swift/Metal release executables were also rebuilt from `8c47a466` and
run against the identical fixture. Catalog and index work is metadata
preparation, not detector decode:

| Device tested | Catalog only | First index build | Prepared index p50 | Prepared index p95 | Date tested |
|---|---:|---:|---:|---:|---|
| Apple M5 Max (`Mac17,6`, 40-core GPU) | **12.397 ms** | **1.370 s** | **3.669 ms** | **3.865 ms** | 2026-08-19 |
| Apple M5 MacBook Pro (`Mac17,2`, 10-core GPU) | **20.198 ms** | **1.239 s** | **2.375 ms** | **2.651 ms** | 2026-08-19 |

The `256x256` rows are explicit crop experiments, never automatic real-space
cropping or substitutes for full-scan evidence. Compressed save is a separate
operation: [MPS-SAVE-U16-512](performance/results.md) retains **1.69 s** for the
sweep winner and **1.91 s** for the public default confirmation, tested
2026-07-25 on an Apple Metal GPU whose exact Mac model was not retained.

### Screening — `quantem.gpu.screening`

`screening.prepare` builds and reopens the small mean-diffraction, BF, DF, CoM,
rotation, and iDPC launch products. It is a separate module from raw I/O.

| Platform | Support | Scan grid | Detector | Source dtype | Detector bin | Chunk plan | Statistic | Time | Device tested | Date tested |
|---|---|---:|---:|---|---:|---|---|---:|---|---|
| **CUDA** | ✓ | `1024x1024` | `192x192` | `uint16` | 1 | 12 GB allocator cap | Single build | **12.31 s** | NVIDIA RTX PRO 6000 Blackwell | 2026-07-28 |
| **Python MPS** | ✓ | `512x512` | `192x192` | `uint16` | 1 | 64 scan rows | Single build | **3.96 s** | Apple Metal GPU (model not retained) | 2026-07-21 |
| **Native Swift/Metal** | — | — | — | — | — | — | — | — | — | — |
| **WebGPU** | — | — | — | — | — | — | — | — | — | — |
| **CPU reference** | Ref | — | — | — | — | Reference fixtures | — | **Pending** | — | — |

Backend-neutral [PRODUCT-CACHE-REOPEN](performance/results.md) is a separate
saved-result state: **6.8 ms** fastest retained repeat for full-`1024` derived
products on 2026-07-20. The host model was not retained. This is never source
load or cache construction.

### Virtual images — `quantem.gpu.detector`

This module owns mean diffraction and exact masked sums for BF, ABF, ADF, DF,
and arbitrary detector masks.

| Platform | Operation | Scan grid | Detector | Detector bin | Input state | Statistic | Time | Device tested | Date tested |
|---|---|---:|---:|---:|---|---|---:|---|---|
| **CUDA** | Mean diffraction | `512x512` | `48x48` | 4 | First resident pass | Single run | **6.102 ms** | NVIDIA RTX PRO 6000 Blackwell, GPU 0 | 2026-08-19 |
| **CUDA** | BF exact sum | `512x512` | `48x48` | 4 | First resident pass | Single run | **1.637 ms** | NVIDIA RTX PRO 6000 Blackwell, GPU 0 | 2026-08-19 |
| **CUDA** | DF exact sum | `512x512` | `48x48` | 4 | First resident pass | Single run | **1.452 ms** | NVIDIA RTX PRO 6000 Blackwell, GPU 0 | 2026-08-19 |
| **Python MPS** | Mean diffraction | `512x512` | `48x48` | 4 | First resident pass | Single run | **76.788 ms** | Apple M5 Max (`Mac17,6`, 40-core GPU) | 2026-08-19 |
| **Python MPS** | BF exact sum | `512x512` | `48x48` | 4 | First resident pass | Single run | **2.245 ms** | Apple M5 Max (`Mac17,6`, 40-core GPU) | 2026-08-19 |
| **Python MPS** | DF exact sum | `512x512` | `48x48` | 4 | First resident pass | Single run | **2.310 ms** | Apple M5 Max (`Mac17,6`, 40-core GPU) | 2026-08-19 |
| **Python MPS** | Mean diffraction | `512x512` | `48x48` | 4 | First resident pass | Single run | **59.063 ms** | Apple M5 MacBook Pro (`Mac17,2`, 10-core GPU) | 2026-08-19 |
| **Python MPS** | BF exact sum | `512x512` | `48x48` | 4 | First resident pass | Single run | **6.757 ms** | Apple M5 MacBook Pro (`Mac17,2`, 10-core GPU) | 2026-08-19 |
| **Python MPS** | DF exact sum | `512x512` | `48x48` | 4 | First resident pass | Single run | **7.131 ms** | Apple M5 MacBook Pro (`Mac17,2`, 10-core GPU) | 2026-08-19 |
| **Native Swift/Metal** | Virtual-image module | `512x512` | `48x48` | 4 | Physical application parity | — | **Pending** | Apple M2 MacBook Air (`Mac14,2`, 8 GB) | 2026-08-18 |
| **WebGPU** | BF | `512x512` | `192x192` | 1 | Prepared selected blocks | p50 | **0.378 s** | Apple Metal-3 adapter (Mac model not retained) | 2026-07-20 |
| **CPU reference** | Virtual-image module | — | — | — | Reference fixtures | — | **Pending** | — | — |

The current detector-bin-4 mean diffraction and exact total/BF/DF products are
byte-identical on both Apple devices and MJGOAT GPU 0. The numbers above are
single first-resident passes, not source-load medians.

### Detector moments and phase contrast — `quantem.gpu.dpc`

This module owns CoM row/column, centering, rotation, DPC, and iDPC. All runtime
boundaries preserve `(row, column)` component order.

| Platform | Operation | Scan grid | Detector | Detector bin | Input state | Statistic | Time | Device tested | Date tested |
|---|---|---:|---:|---:|---|---|---:|---|---|
| **CUDA** | CoM row and column | `512x512` | `48x48` | 4 | First resident pass | Single run | **2.531 ms** | NVIDIA RTX PRO 6000 Blackwell, GPU 0 | 2026-08-19 |
| **CUDA** | Fixed-orientation iDPC | `512x512` | `48x48` | 4 | First resident pass | Single run | **15.852 ms** | NVIDIA RTX PRO 6000 Blackwell, GPU 0 | 2026-08-19 |
| **Python MPS** | CoM row and column | `512x512` | `48x48` | 4 | First resident pass | Single run | **5.258 ms** | Apple M5 Max (`Mac17,6`, 40-core GPU) | 2026-08-19 |
| **Python MPS** | Fixed-orientation iDPC | `512x512` | `48x48` | 4 | First resident pass | Single run | **10.429 ms** | Apple M5 Max (`Mac17,6`, 40-core GPU) | 2026-08-19 |
| **Python MPS** | CoM row and column | `512x512` | `48x48` | 4 | First resident pass | Single run | **14.681 ms** | Apple M5 MacBook Pro (`Mac17,2`, 10-core GPU) | 2026-08-19 |
| **Python MPS** | Fixed-orientation iDPC | `512x512` | `48x48` | 4 | First resident pass | Single run | **10.327 ms** | Apple M5 MacBook Pro (`Mac17,2`, 10-core GPU) | 2026-08-19 |
| **Native Swift/Metal** | Phase-contrast module | `512x512` | `48x48` | 4 | Physical application parity | — | **Pending** | Apple M2 MacBook Air (`Mac14,2`, 8 GB) | 2026-08-18 |
| **WebGPU** | DPC row | `512x512` | `192x192` | 1 | Warm resident | p50 | **14.9 ms** | NVIDIA RTX PRO 6000 Blackwell | 2026-07-20 |
| **WebGPU** | DPC column | `512x512` | `192x192` | 1 | Warm resident | p50 | **13.2 ms** | NVIDIA RTX PRO 6000 Blackwell | 2026-07-20 |
| **WebGPU** | iDPC | `512x512` | `192x192` | 1 | Warm resident | p50 | **13.2 ms** | NVIDIA RTX PRO 6000 Blackwell | 2026-07-20 |
| **CPU reference** | Phase-contrast module | — | — | — | Reference fixtures | — | **Pending** | — | — |

Detector-bin-4 CoM passes the frozen cross-backend `1e-5` gate, but iDPC does
not: its maximum CUDA-versus-MPS error is `2.84e-5`. Native-detector MPS also
has an approximate automatic interaction sidecar. Those paths remain blocked
in the detailed parity ledger instead of receiving a check mark here.

### Single-sideband ptychography — `quantem.gpu.SSB`

SSB uses specialized kernels for **square scan grids**. The numbers below are
scan sizes, not detector sizes. “Native” means a retained acquisition at that
scan size; resized or synthetic evidence is labeled explicitly.

| Platform | Scan grid | Source kind | BF policy | State | Statistic | Time | Device tested | Date tested |
|---|---:|---|---|---|---|---:|---|---|
| **CUDA** | `128x128` | Fixed-size parity | Frozen fixture | ✓ | — | **Pending** | — | — |
| **CUDA** | `256x256` | Fixed-size parity | Frozen fixture | ✓ | — | **Pending** | — | — |
| **CUDA** | `512x512` | Native real acquisition | Full active BF | ✓ | p50 | **32.2 ms** | NVIDIA RTX PRO 6000 Blackwell | 2026-07-19 |
| **CUDA** | `1024x1024` | Fixed-size parity | Frozen fixture | ✓ | — | **Pending** | — | — |
| **Python MPS** | `128x128` | Resized/synthetic | Fixed-size fixture | ✓ | — | **Pending** | — | — |
| **Python MPS** | `256x256` | Resized/synthetic | Fixed-size fixture | ✓ | — | **Pending** | — | — |
| **Python MPS** | `512x512` | Native real acquisition | Full active BF | ✓ | p50 | **537.58 ms** | Apple M5 MacBook Pro (`Mac17,2`, 10-core GPU) | 2026-07-28 |
| **Python MPS** | `1024x1024` | Synthetic | 8,809 BF | ✓ | p50 | **669.1 ms** | Apple M5 MacBook Pro (`Mac17,2`, 10-core GPU) | 2026-07-28 |
| **Native Swift/Metal** | `128x128` | — | — | — | — | — | — | — |
| **Native Swift/Metal** | `256x256` | — | — | — | — | — | — | — |
| **Native Swift/Metal** | `512x512` | — | — | — | — | — | — | — |
| **Native Swift/Metal** | `1024x1024` | — | — | — | — | — | — | — |
| **WebGPU** | `128x128` | Real BF30 parity | Radius 30 px | ✓ | — | **Pending** | — | — |
| **WebGPU** | `256x256` | Deterministic test | Test fixture | Test | — | **Pending** | — | — |
| **WebGPU** | `512x512` | Real interaction | Incomplete frozen reference | Partial | — | **Pending** | — | — |
| **WebGPU** | `1024x1024` | Real interaction | Incomplete frozen reference | Partial | — | **Pending** | — | — |
| **CPU reference** | `128x128` | — | — | — | — | — | — | — |
| **CPU reference** | `256x256` | — | — | — | — | — | — | — |
| **CPU reference** | `512x512` | Independent adjudication | Frozen fixture | Ref | — | **Pending** | — | — |
| **CPU reference** | `1024x1024` | — | — | — | — | — | — | — |

Native Swift/Metal has no native SSB kernel. Untimed CUDA and MPS sizes retain
fixed-size parity coverage; WebGPU rows still need the physical timing or
reference gate represented by **Pending**.

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

Current release-mode native display and FFT diagnostics are kept separate from
load time:

| Platform | Operation | Shape | First execution | Warm p50 | Warm p95 | Device tested | Date tested |
|---|---|---:|---:|---:|---:|---|---|
| **Native Swift/Metal** | Float32 FFT | `512x512` | **7.566 ms** | **0.213 ms** | **0.391 ms** | Apple M5 Max (`Mac17,6`, 40-core GPU) | 2026-08-19 |
| **Native Swift/Metal** | Float32 FFT | `512x512` | **7.001 ms** | **0.551 ms** | **0.865 ms** | Apple M5 MacBook Pro (`Mac17,2`, 10-core GPU) | 2026-08-19 |
| **Native Swift/Metal** | UInt32 statistics | `512x512` | **3.679 ms** | **0.296 ms** | **0.408 ms** | Apple M5 Max (`Mac17,6`, 40-core GPU) | 2026-08-19 |
| **Native Swift/Metal** | UInt32 statistics | `512x512` | **2.141 ms** | **0.727 ms** | **0.975 ms** | Apple M5 MacBook Pro (`Mac17,2`, 10-core GPU) | 2026-08-19 |

The display benchmark also retained exact range `0:4095` and histogram sum
`262144`; linear-render GPU medians were `0.0143 ms` on the M5 Max and
`0.0471 ms` on the M5. These are resident 2D kernels, not wall-to-wall 4D-STEM
loading.

```{admonition} How to read these numbers
:class: important
Milliseconds for resident kernels are **not loading times**. A warm source, a
prepared index, an application first product, and a saved-result reopen are
different states. A feature marked supported without an isolated time is still
covered by its parity gate.
```

For the complete command, source revision, cache state, bin/crop plan, memory
record, and parity artifact behind each row, see the
[complete module dashboard](dashboard.md),
[methodology](performance/methodology.md), and
[verified benchmark results](performance/results.md).

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
