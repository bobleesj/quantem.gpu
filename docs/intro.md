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

For the compact wall-time, scan/detector plan, peak-memory, **6 GiB CUDA**, and
**8 GB laptop** capacity view, open
{ref}`Speed and memory at a glance <speed-and-memory-at-a-glance>`.

### Minimum-memory release gates

The minimum CUDA target is 6 GiB of dedicated VRAM. The minimum WebGPU target
is a physical laptop with 8 GB of total system RAM, including the browser and
operating system. ✓ means the complete physical-device pipeline is retained;
a small calculated payload alone remains **Pending**.

| Platform | Minimum device | Selected scan | Detector bin | Output detector | Resident dtype | Resident payload | Gate | Device tested | Date tested |
|---|---|---:|---:|---:|---|---:|---|---|---|
| **CUDA** | 6 GiB VRAM | `512x512` | 4 | `48x48` | `uint32` | **2.25 GiB** | **Pending** | — | — |
| **Python MPS** | 8 GB unified RAM | `512x512` | 4 | `48x48` | `uint16` | **1.125 GiB** | **Pending** | — | — |
| **Native Swift/Metal** | 8 GB unified RAM | `512x512` | 4 | `48x48` | `uint16` | **1.125 GiB** | **✓** | Apple M2 MacBook Air (`Mac14,2`, 8 GB) | 2026-08-18 |
| **WebGPU** | 8 GB total RAM | `512x512` | 4 | `48x48` | `float32` | **2.25 GiB** | **Pending** | — | — |

The detailed {ref}`minimum-device table <minimum-device-memory-gates>` also
shows the configurations that are already **No**: full-scan WebGPU bins 1 and 2
exceed the entire 8 GB floor from resident payload alone, while bins 4 and 8
remain physical-test candidates.

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
| **CUDA** | `512x512` | Full | 1 | `192x192` | `uint16` | `uint16` | `uint16` | Warm source | D | p50 | **0.386 s** | NVIDIA RTX PRO 6000 Blackwell Max-Q, GPU 1 | 2026-08-19 |
| **CUDA** | `512x512` | Full | 2 | `96x96` | `uint16` | `uint16` | `uint32` | Warm source | D | p50 | **0.396 s** | NVIDIA RTX PRO 6000 Blackwell Max-Q, GPU 1 | 2026-08-19 |
| **CUDA** | `512x512` | Full | 4 | `48x48` | `uint16` | `uint16` | `uint32` | Warm source | D | p50 | **0.390 s** | NVIDIA RTX PRO 6000 Blackwell Max-Q, GPU 1 | 2026-08-19 |
| **CUDA** | `512x512` | Full | 8 | `24x24` | `uint16` | `uint16` | `uint32` | Warm source | D | p50 | **0.381 s** | NVIDIA RTX PRO 6000 Blackwell Max-Q, GPU 1 | 2026-08-19 |
| **Python MPS** | `512x512` | Full | 1 | `192x192` | `uint16` | `uint16` | `uint16` | Warm source | C | p50 | **2.273 s** | Apple M5 Max (`Mac17,6`, 40-core GPU, 128 GB) | 2026-08-19 |
| **Python MPS** | `512x512` | Full | 2 | `96x96` | `uint16` | `uint16` | `uint16` | Warm source | C | p50 | **0.707 s** | Apple M5 Max (`Mac17,6`, 40-core GPU, 128 GB) | 2026-08-19 |
| **Python MPS** | `512x512` | Full | 4 | `48x48` | `uint16` | `uint16` | `uint16` | Warm source | C | p50 | **0.605 s** | Apple M5 Max (`Mac17,6`, 40-core GPU, 128 GB) | 2026-08-19 |
| **Python MPS** | `512x512` | Full | 8 | `24x24` | `uint16` | `uint16` | `uint16` | Warm source | C | p50 | **0.586 s** | Apple M5 Max (`Mac17,6`, 40-core GPU, 128 GB) | 2026-08-19 |
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
| **WebGPU** | `512x512` | Full | 1 | `192x192` | `uint16` | `uint8` | `uint8` | Warm OS cache | D | p50 | **0.824 s** | Chrome 151, Apple M5 Max Metal-3 | 2026-08-19 |
| **WebGPU** | `512x512` | Full | 2 | `96x96` | `uint16` | `uint16` | `float32` | Warm OS cache | D | p50 | **1.281 s** | Chrome 151, Apple M5 Max Metal-3 | 2026-08-19 |
| **WebGPU** | `512x512` | Full | 4 | `48x48` | `uint16` | `uint16` | `float32` | Warm OS cache | D | p50 | **1.044 s** | Chrome 151, Apple M5 Max Metal-3 | 2026-08-19 |
| **WebGPU** | `512x512` | Full | 8 | `24x24` | `uint16` | `uint16` | `float32` | Warm OS cache | D | p50 | **0.979 s** | Chrome 151, Apple M5 Max Metal-3 | 2026-08-19 |
| **WebGPU** | `512x512` | Full | 2 | `96x96` | `uint16` | `uint16` | `float32` | Prepared source | — | Single profile | **2.651 s** | NVIDIA RTX PRO 6000 Blackwell | 2026-07-20 |

Fixtures C and D are independent full-scan, native-`uint16`, 27-shard real
fixtures. No row uses a real-space crop; scan bin is 1 and detector bin is
explicit. CUDA and WebGPU use D; Python MPS and the physical Swift/Metal
application rows use C. The operating-system storage cache was not reset, so no
row is called cold. The complete p50/p95/max, memory, revision, and parity
record is in the [implementation overview](dashboard.md) and
[verified results](performance/results.md).

Native Swift/Metal release executables were also rebuilt from `8c47a466` and
run against the identical fixture. Catalog and index work is metadata
preparation, not detector decode:

| Device tested | Catalog only | First index build | Prepared index p50 | Prepared index p95 | Date tested |
|---|---:|---:|---:|---:|---|
| Apple M5 Max (`Mac17,6`, 40-core GPU) | **13.448 ms** | **1.437 s** | **5.339 ms** | **5.760 ms** | 2026-08-19 |
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
| **Python MPS** | ✓ | `512x512` | `192x192` | `uint16` | 1 | 170 rows, 4 chunks, exact fallback pass | Single build | **6.711 s** | Apple M5 Max (`Mac17,6`, 40-core GPU) | 2026-08-19 |
| **Python MPS** | ✓ | `512x512` | Derived products | `float32` | 1 | Validated screening-v3 | Warm reopen p50 | **20.803 ms** | Apple M5 Max (`Mac17,6`, 40-core GPU) | 2026-08-19 |
| **Native Swift/Metal** | — | — | — | — | — | — | — | — | — | — |
| **WebGPU** | — | — | — | — | — | — | — | — | — | — |
| **CPU reference** | Ref | — | — | — | — | Reference fixtures | — | **Pending** | — | — |

The exact MPS build accumulates the full-scan detector sum in `uint64`. Because
the provisional and final masks differed, it transparently made a second BF/DF
pass. The earlier faster one-pass candidate failed full-scan parity and is not
promoted. Backend-neutral [PRODUCT-CACHE-REOPEN](performance/results.md) is a separate
saved-result state: **6.8 ms** fastest retained repeat for full-`1024` derived
products on 2026-07-20. The host model was not retained. This is never source
load or cache construction.

### Virtual images — `quantem.gpu.detector`

This module owns mean diffraction and exact masked sums for BF, ABF, ADF, DF,
and arbitrary detector masks.

| Platform | Operation | Scan grid | Detector | Detector bin | Input state | Statistic | Time | Device tested | Date tested |
|---|---|---:|---:|---:|---|---|---:|---|---|
| **CUDA** | Mean diffraction | `512x512` | `192x192` | 1 | Warm resident | p50 | **18.392 ms** | NVIDIA RTX PRO 6000 Blackwell Max-Q, GPU 1 | 2026-08-19 |
| **CUDA** | BF exact sum | `512x512` | `192x192` | 1 | Warm resident | p50 | **3.768 ms** | NVIDIA RTX PRO 6000 Blackwell Max-Q, GPU 1 | 2026-08-19 |
| **CUDA** | ADF exact sum | `512x512` | `192x192` | 1 | Warm resident | p50 | **5.586 ms** | NVIDIA RTX PRO 6000 Blackwell Max-Q, GPU 1 | 2026-08-19 |
| **CUDA** | DF exact sum | `512x512` | `192x192` | 1 | Warm resident | p50 | **3.747 ms** | NVIDIA RTX PRO 6000 Blackwell Max-Q, GPU 1 | 2026-08-19 |
| **Python MPS** | Mean diffraction | `512x512` | `48x48` | 4 | Warm resident | p50 | **74.805 ms** | Apple M5 Max (`Mac17,6`, 40-core GPU) | 2026-08-19 |
| **Python MPS** | BF exact sum | `512x512` | `48x48` | 4 | Warm resident | p50 | **2.502 ms** | Apple M5 Max (`Mac17,6`, 40-core GPU) | 2026-08-19 |
| **Python MPS** | ADF exact sum | `512x512` | `48x48` | 4 | Warm resident | p50 | **4.404 ms** | Apple M5 Max (`Mac17,6`, 40-core GPU) | 2026-08-19 |
| **Python MPS** | DF exact sum | `512x512` | `48x48` | 4 | Warm resident | p50 | **2.642 ms** | Apple M5 Max (`Mac17,6`, 40-core GPU) | 2026-08-19 |
| **Python MPS** | Mean diffraction | `512x512` | `48x48` | 4 | First resident pass | Single run | **59.063 ms** | Apple M5 MacBook Pro (`Mac17,2`, 10-core GPU) | 2026-08-19 |
| **Python MPS** | BF exact sum | `512x512` | `48x48` | 4 | First resident pass | Single run | **6.757 ms** | Apple M5 MacBook Pro (`Mac17,2`, 10-core GPU) | 2026-08-19 |
| **Python MPS** | DF exact sum | `512x512` | `48x48` | 4 | First resident pass | Single run | **7.131 ms** | Apple M5 MacBook Pro (`Mac17,2`, 10-core GPU) | 2026-08-19 |
| **Native Swift/Metal** | Virtual-image module | `512x512` | `48x48` | 4 | Physical application parity | — | **Pending** | Apple M2 MacBook Air (`Mac14,2`, 8 GB) | 2026-08-18 |
| **WebGPU** | Mean diffraction | `512x512` | `192x192` | 1 | Warm resident | p50 | **50.9 ms** | Chrome 151, Apple M5 Max Metal-3 | 2026-08-19 |
| **WebGPU** | BF exact sum | `512x512` | `192x192` | 1 | Warm resident | p50 | **5.5 ms** | Chrome 151, Apple M5 Max Metal-3 | 2026-08-19 |
| **WebGPU** | ADF exact sum | `512x512` | `192x192` | 1 | Warm resident | p50 | **15.0 ms** | Chrome 151, Apple M5 Max Metal-3 | 2026-08-19 |
| **WebGPU** | DF exact sum | `512x512` | `192x192` | 1 | Warm resident | p50 | **43.4 ms** | Chrome 151, Apple M5 Max Metal-3 | 2026-08-19 |
| **CPU reference** | Virtual-image adjudication | `512x512` | `192x192` | 1 | One independent traversal | Single run | **31.08 s** | Apple M5 Max CPU | 2026-08-19 |

Current integer and mean-DP rows pass their independent CPU references. The
numbers above are warm resident kernels, not source-load medians.

### Detector moments and phase contrast — `quantem.gpu.dpc`

This module owns CoM row/column, centering, rotation, DPC, and iDPC. All runtime
boundaries preserve `(row, column)` component order.

| Platform | Operation | Scan grid | Detector | Detector bin | Input state | Statistic | Time | Device tested | Date tested |
|---|---|---:|---:|---:|---|---|---:|---|---|
| **CUDA** | CoM row and column | `512x512` | `192x192` | 1 | Warm resident | p50 | **13.002 ms** | NVIDIA RTX PRO 6000 Blackwell Max-Q, GPU 1 | 2026-08-19 |
| **CUDA** | Fixed-orientation iDPC | `512x512` | `192x192` | 1 | CPU small-field integration | p50 | **21.272 ms** | NVIDIA RTX PRO 6000 Blackwell Max-Q, GPU 1 | 2026-08-19 |
| **Python MPS** | CoM row and column | `512x512` | `48x48` | 4 | Warm resident | p50 | **4.637 ms** | Apple M5 Max (`Mac17,6`, 40-core GPU) | 2026-08-19 |
| **Python MPS** | Fixed-orientation iDPC | `512x512` | `48x48` | 4 | CPU small-field integration | p50 | **12.678 ms** | Apple M5 Max (`Mac17,6`, 40-core GPU) | 2026-08-19 |
| **Python MPS** | CoM row and column | `512x512` | `48x48` | 4 | First resident pass | Single run | **14.681 ms** | Apple M5 MacBook Pro (`Mac17,2`, 10-core GPU) | 2026-08-19 |
| **Python MPS** | Fixed-orientation iDPC | `512x512` | `48x48` | 4 | First resident pass | Single run | **10.327 ms** | Apple M5 MacBook Pro (`Mac17,2`, 10-core GPU) | 2026-08-19 |
| **Native Swift/Metal** | Phase-contrast module | `512x512` | `48x48` | 4 | Physical application parity | — | **Pending** | Apple M2 MacBook Air (`Mac14,2`, 8 GB) | 2026-08-18 |
| **WebGPU** | DPC row | `512x512` | `192x192` | 1 | Warm cached CoM | p50 | **0.9 ms** | Chrome 151, Apple M5 Max Metal-3 | 2026-08-19 |
| **WebGPU** | DPC column | `512x512` | `192x192` | 1 | Warm cached CoM | p50 | **0.7 ms** | Chrome 151, Apple M5 Max Metal-3 | 2026-08-19 |
| **WebGPU** | iDPC | `512x512` | `192x192` | 1 | Explicit 0-degree rotation | p50 | **1.4 ms** | Chrome 151, Apple M5 Max Metal-3 | 2026-08-19 |
| **CPU reference** | Rotation and iDPC adjudication | `512x512` | `192x192` | 1 | CPU reference | Single run | **177.6 ms** | Apple M5 Max CPU | 2026-08-19 |

Current CUDA and MPS CoM rows pass their independent references. The historical
same-fixture CUDA/MPS detector-bin-4 iDPC comparison remains blocked at
`2.84e-5`. WebGPU per-pixel CoM/DPC/iDPC arrays were not retained in the current
run, so those float-parity cells remain unproven.

### Single-sideband ptychography — `quantem.gpu.SSB`

SSB uses specialized kernels for **square scan grids**. The numbers below are
scan sizes, not detector sizes. “Native” means a retained acquisition at that
scan size; resized or synthetic evidence is labeled explicitly.

Current `512x512` compute, excluding source preparation and UI paint:

| Platform | Operation | Detector plan | Boundary | Statistic | Time | Device tested | Date tested |
|---|---|---|---|---|---:|---|---|
| **CUDA** | Complex object | Native `192x192` | Warm resident GPU | p50 | **13.883 ms** | NVIDIA RTX PRO 6000 Blackwell Max-Q, GPU 1 | 2026-08-19 |
| **CUDA** | Exact phase and loss | Native `192x192` | Warm resident GPU | p50 | **32.335 ms** | NVIDIA RTX PRO 6000 Blackwell Max-Q, GPU 1 | 2026-08-19 |
| **Python MPS** | Exact phase and loss | Explicit detector bin 2 | Single synchronized reconstruction | Single run | **497.187 ms** | Apple M5 Max (`Mac17,6`, 40-core GPU) | 2026-08-19 |
| **Native Swift/Metal** | Complex object | Native-detector exact BF columns | Warm complete Hermitian cache | p50 | **8.911 ms** | Apple M5 Max (`Mac17,6`, 40-core GPU) | 2026-08-19 |
| **Native Swift/Metal** | Exact phase-variance loss | Native-detector exact BF columns | Warm complete Hermitian cache | p50 | **25.120 ms** | Apple M5 Max (`Mac17,6`, 40-core GPU) | 2026-08-19 |
| **WebGPU** | Complex object | Native `192x192` companion | Readback-complete compute wall | p50 | **32.5 ms** | Chrome 151, Apple M5 Max Metal-3 | 2026-08-19 |
| **WebGPU** | Exact phase | Native `192x192` companion | Readback-complete compute wall | p50 | **102.1 ms** | Chrome 151, Apple M5 Max Metal-3 | 2026-08-19 |
| **WebGPU** | Exact phase and loss | Native `192x192` companion | Readback-complete compute wall | p50 | **189.4 ms** | Chrome 151, Apple M5 Max Metal-3 | 2026-08-19 |

The raw MPS detector-bin-2 phase passes its CUDA reference to `1.2815e-6`
wrapped radians maximum, and the loss differs by `7.45e-9`. The prepared
companion candidate is rejected because its stored columns do not match its
declared detector-bin coordinate grid.

| Platform | Calibration | Refinement | Statistic | Time | Result | Device tested | Date tested |
|---|---|---|---|---:|---|---|---|
| **CUDA** | Seeded Optuna TPE, 200 trials | Nelder–Mead | p50 of 3 | **11.168 s** | Byte-deterministic | NVIDIA RTX PRO 6000 Blackwell Max-Q, GPU 1 | 2026-08-19 |
| **Python MPS** | 200-trial TPE | Nelder–Mead | — | **Pending** | Compatible current source not profiled | — | — |
| **Native Swift/Metal** | Seeded TPE, 200 trials | Nelder–Mead | p50 of 3 | **6.061 s** | Deterministic | Apple M5 Max (`Mac17,6`, 40-core GPU) | 2026-08-19 |
| **WebGPU** | — | — | — | — | Unsupported | — | — |
| **CPU reference** | — | — | — | — | Reference only | — | — |

Levenberg–Marquardt is not implemented. The earlier CUDA atomic-objective fit
was faster (`8.096 s` p50) but produced two fitted minima under an identical
seed, so it remains rejected rather than replacing the deterministic result.

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
| **Native Swift/Metal** | `512x512` | Native real acquisition | 9,074 logical / 2,459 executed BF | ✓ | p50 | **8.911 ms** | Apple M5 Max (`Mac17,6`, 40-core GPU) | 2026-08-19 |
| **Native Swift/Metal** | `1024x1024` | — | — | — | — | — | — | — |
| **WebGPU** | `128x128` | Real BF30 parity | Radius 30 px | ✓ | — | **Pending** | — | — |
| **WebGPU** | `256x256` | Deterministic test | Test fixture | Test | — | **Pending** | — | — |
| **WebGPU** | `512x512` | Real interaction | Incomplete frozen reference | Partial | — | **Pending** | — | — |
| **WebGPU** | `1024x1024` | Real interaction | Incomplete frozen reference | Partial | — | **Pending** | — | — |
| **CPU reference** | `128x128` | — | — | — | — | — | — | — |
| **CPU reference** | `256x256` | — | — | — | — | — | — | — |
| **CPU reference** | `512x512` | Independent adjudication | Frozen fixture | Ref | — | **Pending** | — | — |
| **CPU reference** | `1024x1024` | — | — | — | — | — | — | — |

Native Swift/Metal now has a package-owned exact 512×512 implementation. Other
native Swift scan sizes remain unsupported rather than inferred from CUDA/MPS.
Untimed CUDA and MPS sizes retain fixed-size parity coverage; WebGPU rows still
need the physical timing or reference gate represented by **Pending**.

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
| **Native Swift/Metal** | Float32 FFT | `512x512` | **15.622 ms** | **0.291 ms** | **0.584 ms** | Apple M5 Max (`Mac17,6`, 40-core GPU) | 2026-08-19 |
| **Native Swift/Metal** | Float32 FFT | `512x512` | **7.001 ms** | **0.551 ms** | **0.865 ms** | Apple M5 MacBook Pro (`Mac17,2`, 10-core GPU) | 2026-08-19 |
| **Native Swift/Metal** | UInt32 statistics | `512x512` | — | **0.353 ms** | **0.515 ms** | Apple M5 Max (`Mac17,6`, 40-core GPU) | 2026-08-19 |
| **Native Swift/Metal** | UInt32 statistics | `512x512` | **2.141 ms** | **0.727 ms** | **0.975 ms** | Apple M5 MacBook Pro (`Mac17,2`, 10-core GPU) | 2026-08-19 |

The display benchmark also retained exact range `0:4095` and histogram sum
`262144`; linear-render GPU medians were `0.0313 ms` on the M5 Max and
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
