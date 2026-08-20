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

### Measured load configurations

One row is one exact configuration and one headline statistic. Source, decode,
and resident dtype are separate columns. A second fixture, bin, dtype path,
cache state, or timing statistic requires another row; slash-delimited bundles
are not permitted. **Pending** means the combination is tracked but has no
retained timing with complete provenance.

The table scrolls horizontally on narrow screens so fields remain separate;
the timing, memory, parity, and evidence columns are to the right.

| Platform | Selected scan | Scan plan | Source detector | Detector bin | Output detector | Source dtype | Decode dtype | Resident dtype | Cache state | Wall boundary | Fixture | Statistic | Time | Memory kind | Memory | Parity | Date | Revision | Device | Evidence |
|---|---:|---|---:|---:|---:|---|---|---|---|---|---|---|---:|---|---:|---|---|---|---|---|
| [**CUDA**](platforms/cuda.md) | `512x512` | Full | `192x192` | 1 | `192x192` | `uint16` | `uint8` | `uint8` | Warm source | Load/decompress | — | Median | **0.450 s** | Decoded resident | **9.66 GB** | Exact | 2026-07-20 | `b61572e4` | RTX PRO 6000 | [CUDA-512-LOAD](performance/results.md) |
| [**Python MPS**](platforms/mps.md) | `1024x1024` | Full | `192x192` | 1 | `192x192` | `uint16` | `uint16` | `uint16` | First observed source | Load/decompress | — | Single run | **4.617 s** | Logical payload | **77.31 GB** | Exact | 2026-07-20 | `cee0ba5c` | Apple GPU; model missing | [MPS-1024-LOAD](performance/results.md) |
| [**Native Swift/Metal**](platforms/swift-metal.md) | `512x512` | Full | `192x192` | 4 | `48x48` | `uint16` | `uint16` | `uint16` | First process | First complete product | A | p50 | **1.985 s** | Process peak | **1.43 GB** | Byte-identical | 2026-08-18 | `2c047160` | 8 GB `Mac14,2` M2 Air | [M2-AIR-BIN4-E2E](performance/results.md) |
| [**Native Swift/Metal**](platforms/swift-metal.md) | `512x512` | Full | `192x192` | 4 | `48x48` | `uint16` | `uint16` | `uint16` | First process | First complete product | B | p50 | **2.043 s** | Process peak | **1.43 GB** | Byte-identical | 2026-08-18 | `2c047160` | 8 GB `Mac14,2` M2 Air | [M2-AIR-BIN4-E2E](performance/results.md) |
| [**WebGPU**](platforms/webgpu.md) | `256x256` | Explicit crop | `192x192` | 1 | `192x192` | `uint16` | `uint8` | `uint8` | Prepared source | Full-stack load | — | p50 | **0.338 s** | Browser peak | **Pending** | Exact | 2026-07-20 | `b61572e4` | Apple Metal-3 | [WEBGPU-256-CROP](performance/results.md) |
| [**WebGPU**](platforms/webgpu.md) | `256x256` | Explicit crop | `192x192` | 2 | `96x96` | `uint16` | `uint8` | `float32` | Prepared source | Page profile | — | p50 | **0.774 s** | Browser peak | **Pending** | Exact | 2026-07-20 | `cee0ba5c` | RTX PRO 6000 | [WEBGPU-DET-BIN](performance/results.md) |
| [**WebGPU**](platforms/webgpu.md) | `256x256` | Explicit crop | `192x192` | 4 | `48x48` | `uint16` | `uint8` | `float32` | Prepared source | Page profile | — | p50 | **0.755 s** | Browser peak | **Pending** | Exact | 2026-07-20 | `cee0ba5c` | RTX PRO 6000 | [WEBGPU-DET-BIN](performance/results.md) |
| [**WebGPU**](platforms/webgpu.md) | `256x256` | Explicit crop | `192x192` | 8 | `24x24` | `uint16` | `uint8` | `float32` | Prepared source | Page profile | — | p50 | **0.733 s** | Browser peak | **Pending** | Exact | 2026-07-20 | `cee0ba5c` | RTX PRO 6000 | [WEBGPU-DET-BIN](performance/results.md) |
| [**WebGPU**](platforms/webgpu.md) | `512x512` | Full | `192x192` | 1 | `192x192` | `uint16` | `uint8` | `uint8` | Prepared source | Full-stack load | — | p50 | **0.772 s** | Decoded payload | **9.7 GB** | Exact | 2026-07-20 | `b61572e4` | Apple Metal-3 | [WEBGPU-512-FULL](performance/results.md) |
| [**WebGPU**](platforms/webgpu.md) | `512x512` | Full | `192x192` | 2 | `96x96` | `uint16` | `uint8` | `float32` | Prepared source | Page profile | — | Single profile | **1.199 s** | Browser peak | **Pending** | Exact | 2026-07-20 | `cee0ba5c` | RTX PRO 6000 | [WEBGPU-DET-BIN](performance/results.md) |
| [**WebGPU**](platforms/webgpu.md) | `512x512` | Full | `192x192` | 4 | `48x48` | `uint16` | `uint8` | `float32` | Prepared source | Page profile | — | Single profile | **1.212 s** | Browser peak | **Pending** | Exact | 2026-07-20 | `cee0ba5c` | RTX PRO 6000 | [WEBGPU-DET-BIN](performance/results.md) |
| [**WebGPU**](platforms/webgpu.md) | `512x512` | Full | `192x192` | 8 | `24x24` | `uint16` | `uint8` | `float32` | Prepared source | Page profile | — | Single profile | **1.106 s** | Browser peak | **Pending** | Exact | 2026-07-20 | `cee0ba5c` | RTX PRO 6000 | [WEBGPU-DET-BIN](performance/results.md) |
| [**WebGPU**](platforms/webgpu.md) | `512x512` | Full | `192x192` | 2 | `96x96` | `uint16` | `uint16` | `float32` | Prepared source | Two-pass page profile | — | Single profile | **2.651 s** | Browser peak | **Pending** | Exact | 2026-07-20 | `cee0ba5c` | RTX PRO 6000 | [WEBGPU-DET-BIN](performance/results.md) |

The `256x256` entries above are explicit scan-crop experiments. They are useful
configuration evidence, but they are never substituted for full-scan loading
or used as an automatic memory policy. Follow an evidence ID for the date,
revision, physical device, complete timing distribution, and calibration facts.

A [saved-product reopen](performance/results.md) can take **6.8-8.0 ms**, but
that state contains only derived products. It is never called a source load.

(dtype-support-and-peak-memory)=
### Dtype support and peak memory

“Source,” “working,” “accumulation,” and “resident” dtype describe different
stages. A `uint8` row is scientifically exact only when the source is already
`uint8` or a complete source audit proves `maximum <= 255` and
`pixelsAbove255 == 0`. Otherwise an explicit `dtype="u8"` load saturates values
above 255 and is a browse representation, not raw-count evidence.

| Platform | Source dtype | Decode/working dtype | Resident dtype | Precision policy | Support | Memory kind | Retained memory |
|---|---|---|---|---|---|---|---:|
| [**CUDA**](platforms/cuda.md) | `uint8` | — | — | Native compressed source | — | — | — |
| [**CUDA**](platforms/cuda.md) | `uint16` | `uint16` | `uint16` | Exact native counts | ✓ | Process peak | **Pending** |
| [**CUDA**](platforms/cuda.md) | `uint16` | `uint8` | `uint8` | Complete-audit lossless | ✓ | Decoded resident | **9.66 GB** |
| [**CUDA**](platforms/cuda.md) | `uint32` | `uint32` | `uint32` | Exact native counts | ✓ | Process peak | **Pending** |
| [**Python MPS**](platforms/mps.md) | `uint8` | — | — | Native compressed source | — | — | — |
| [**Python MPS**](platforms/mps.md) | `uint16` | `uint16` | `uint16` | Exact native counts | ✓ | Process/Metal peak | **Pending** |
| [**Python MPS**](platforms/mps.md) | `uint16` | `uint8` | `uint8` | Complete-audit lossless | ✓ | Process/Metal peak | **Pending** |
| [**Python MPS**](platforms/mps.md) | `uint32` | `uint16` | `uint16` | Guarded exact narrowing | ✓ | Process/Metal peak | **Pending** |
| [**Python MPS**](platforms/mps.md) | `uint32` | `uint32` | `uint32` | Exact native counts | ✓ | Process/Metal peak | **Pending** |
| [**Native Swift/Metal**](platforms/swift-metal.md) | `uint8` | `uint8` | `uint8` | Exact native counts | ✓ | Process peak | **Pending** |
| [**Native Swift/Metal**](platforms/swift-metal.md) | `uint16` | `uint16` | `uint16` | Exact native counts | ✓ | Process peak | **Pending** |
| [**Native Swift/Metal**](platforms/swift-metal.md) | `uint16` | `uint16` | `uint16` | Audited exact detector sum | ✓ | Process peak | **1.43 GB** |
| [**Native Swift/Metal**](platforms/swift-metal.md) | `uint16` | `uint16` | `uint32` | General exact detector sum | ✓ | Process peak | **Pending** |
| [**WebGPU**](platforms/webgpu.md) | `uint8` | `uint8` | `uint8` | Exact native counts | ✓ | Browser/device peak | **Pending** |
| [**WebGPU**](platforms/webgpu.md) | `uint16` | `uint16` | `uint16` | Exact native counts | ✓ | Browser/device peak | **Pending** |
| [**WebGPU**](platforms/webgpu.md) | `uint16` | `uint8` | `uint8` | Complete-audit lossless | ✓ | Decoded payload | **9.7 GB** |
| [**WebGPU**](platforms/webgpu.md) | `uint16` | `uint8` | `float32` | Exact detector sum | ✓ | Browser/device peak | **Pending** |
| [**WebGPU**](platforms/webgpu.md) | `uint16` | `uint16` | `float32` | Exact detector sum | ✓ | Browser/device peak | **Pending** |
| [**WebGPU**](platforms/webgpu.md) | `uint32` | `uint32` | `uint32` | Exact native counts | ✓ | Browser/device peak | **Pending** |
| **CPU reference** | `uint8` | `uint8` | `uint8` | Exact reference | Ref | Host peak | **Pending** |
| **CPU reference** | `uint16` | `uint16` | `uint16` | Exact reference | Ref | Host peak | **Pending** |

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

| Detector bin | Resident dtype | Resident payload | 4 GiB fit | 6 GiB fit | Evidence state |
|---:|---|---:|---|---|---|
| 1 | `uint16` | **18.00 GiB** | No | No | Calculated full-resident payload |
| 1 | `uint8` | **9.00 GiB** | No | No | Complete-audit lossless path only |
| 2 | `uint32` | **9.00 GiB** | No | No | Calculated exact-sum payload |
| 4 | `uint16` | **1.125 GiB** | Candidate | Candidate | Physical 8 GB M2 Air evidence; 4/6 GiB signoff Pending |
| 4 | `uint32` | **2.25 GiB** | Candidate | Candidate | Calculated exact-sum payload; physical 4/6 GiB signoff Pending |

Streaming is a different allocation plan and therefore has its own atomic rows:

| Memory budget | Source dtype | Chunk rows | Chunk payload | Chunk count | Physical signoff |
|---:|---|---:|---:|---:|---|
| 4 GiB | `uint16` | 56 | **1.97 GiB** | 10 | Pending |
| 6 GiB | `uint16` | 85 | **2.99 GiB** | 7 | Pending |

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

#### Exact configuration gaps

Scan-size and detector-bin evidence are not multiplied into an implied
Cartesian product. The measured rows above are the verified joint
configurations. This table records the priority `256x256` and `512x512`
combinations that still need a complete timing row. Every bin is a separate
row, and every `256x256` entry remains an explicit scan crop.

| Platform | Selected scan | Scan plan | Source detector | Detector bin | Output detector | State | Time | Next gate |
|---|---:|---|---:|---:|---:|---|---:|---|
| **CUDA** | `256x256` | Explicit crop | `192x192` | 1 | `192x192` | ✓ | **Pending** | Retain the existing exact crop gate as a dated load benchmark |
| **CUDA** | `256x256` | Explicit crop | `192x192` | 2 | `96x96` | Pending | **Pending** | Joint crop-plus-bin parity and hardware timing |
| **CUDA** | `256x256` | Explicit crop | `192x192` | 4 | `48x48` | Pending | **Pending** | Joint crop-plus-bin parity and hardware timing |
| **CUDA** | `256x256` | Explicit crop | `192x192` | 8 | `24x24` | Pending | **Pending** | Joint crop-plus-bin parity and hardware timing |
| **CUDA** | `512x512` | Full | `192x192` | 2 | `96x96` | Test | **Pending** | Retained full-source hardware timing |
| **CUDA** | `512x512` | Full | `192x192` | 4 | `48x48` | Pending | **Pending** | Full-source parity and hardware timing |
| **CUDA** | `512x512` | Full | `192x192` | 8 | `24x24` | Pending | **Pending** | Full-source parity and hardware timing |
| **Python MPS** | `256x256` | Explicit crop | `192x192` | 1 | `192x192` | ✓ | **Pending** | Retain the existing exact crop gate as a dated load benchmark |
| **Python MPS** | `256x256` | Explicit crop | `192x192` | 2 | `96x96` | Pending | **Pending** | Joint crop-plus-bin parity and hardware timing |
| **Python MPS** | `256x256` | Explicit crop | `192x192` | 4 | `48x48` | Pending | **Pending** | Joint crop-plus-bin parity and hardware timing |
| **Python MPS** | `256x256` | Explicit crop | `192x192` | 8 | `24x24` | Pending | **Pending** | Joint crop-plus-bin parity and hardware timing |
| **Python MPS** | `512x512` | Full | `192x192` | 1 | `192x192` | ✓ | **Pending** | Isolated load timing with complete physical-device provenance |
| **Python MPS** | `512x512` | Full | `192x192` | 2 | `96x96` | Test | **Pending** | Promote sparse exact-sum coverage to a full-source hardware row |
| **Python MPS** | `512x512` | Full | `192x192` | 4 | `48x48` | Pending | **Pending** | Full-source parity and hardware timing |
| **Python MPS** | `512x512` | Full | `192x192` | 8 | `24x24` | Pending | **Pending** | Full-source parity and hardware timing |
| **Native Swift/Metal** | `256x256` | Explicit crop | `192x192` | 1 | `192x192` | Pending | **Pending** | Physical-device crop parity and timing |
| **Native Swift/Metal** | `256x256` | Explicit crop | `192x192` | 2 | `96x96` | Pending | **Pending** | Physical-device crop-plus-bin parity and timing |
| **Native Swift/Metal** | `256x256` | Explicit crop | `192x192` | 4 | `48x48` | Pending | **Pending** | Physical-device crop-plus-bin parity and timing |
| **Native Swift/Metal** | `256x256` | Explicit crop | `192x192` | 8 | `24x24` | — | — | Public load plan does not support bin 8 |
| **Native Swift/Metal** | `512x512` | Full | `192x192` | 1 | `192x192` | Test | **Pending** | Physical first-process timing with frozen full-product parity |
| **Native Swift/Metal** | `512x512` | Full | `192x192` | 2 | `96x96` | Test | **Pending** | Physical first-process timing with frozen full-product parity |
| **Native Swift/Metal** | `512x512` | Full | `192x192` | 8 | `24x24` | — | — | Public load plan does not support bin 8 |

WebGPU `256x256` bins 1/2/4/8 and `512x512` bins 1/2/4/8 already have
separate measured rows above. The **CPU reference** remains the independent
`Ref` adjudicator, not an accelerator timing placeholder. The additional
`128x128` and `1024x1024`
size-specific gates remain in the detailed evidence pages until a joint
scan/bin measurement is retained.

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

| Platform | Operation | Measured size/bin plan | Latest retained timing | Details |
|---|---|---|---:|---|
| **CUDA** | BF | Resident `512x512x192x192`, bin 1 | **1.35 ms** | [CUDA-BF-512](performance/results.md), 2026-07-19, `0456e15e`; integer max error `0` |
| **CUDA** | ADF | Resident `512x512x192x192`, bin 1 | **3.86 ms** | [CUDA-ADF-512](performance/results.md), 2026-07-19, `0456e15e`; integer max error `0` |
| **CUDA** | DF | Resident `512x512x192x192`, bin 1 | **1.84 ms** | [CUDA-DF-512](performance/results.md), 2026-07-19, `0456e15e`; integer max error `0` |
| **Python MPS** | Virtual-image module | Full `512` bin-1 products through retained screening parity | **Pending** | No isolated MPS virtual-image timing with complete public device provenance |
| **Native Swift/Metal** | Virtual-image module | Full `512`, detector bin 4 in physical application parity | **Pending** | Products are byte-identical in the M2 Air gate; isolated kernel timing is not retained |
| **WebGPU** | BF | Full `512`, bin 1, fixed 30 px BF radius | **0.378 s p50** | [WEBGPU-BF-512](performance/results.md), 2026-07-20, `b61572e4`; prepared selected-block boundary, not isolated kernel time |
| **CPU reference** | Virtual-image module | Reference fixtures | **Pending** | Correctness adjudication only |

### Detector moments and phase contrast — `quantem.gpu.dpc`

| Platform | Operation | Measured size/bin plan | Latest retained timing | Details |
|---|---|---|---:|---|
| **CUDA** | CoM row/column | Resident full `512x512x192x192`, bin 1 | **12.39 ms** | [CUDA-COM-512](performance/results.md), 2026-07-19, `0456e15e`; max error `0` |
| **Python MPS** | Phase-contrast module | Full `512`, bin 1 through retained screening parity | **Pending** | No isolated full-module MPS timing with complete public device provenance |
| **Native Swift/Metal** | Phase-contrast module | Full `512`, detector bin 4 in physical application parity | **Pending** | CoM/DPC/iDPC exports are byte-identical; isolated kernel timing is not retained |
| **WebGPU** | DPC row | Resident full `512x512x192x192`, bin 1 | **14.9 ms p50** | [WEBGPU-DPC-512](performance/results.md), 2026-07-20, `cee0ba5c`; frozen float32 errors retained |
| **WebGPU** | DPC column | Resident full `512x512x192x192`, bin 1 | **13.2 ms p50** | [WEBGPU-DPC-512](performance/results.md), 2026-07-20, `cee0ba5c`; frozen float32 errors retained |
| **WebGPU** | iDPC | Resident full `512x512x192x192`, bin 1 | **13.2 ms p50** | [WEBGPU-DPC-512](performance/results.md), 2026-07-20, `cee0ba5c`; frozen float32 errors retained |
| **CPU reference** | Phase-contrast module | Reference fixtures | **Pending** | Correctness adjudication only |

### Single-sideband ptychography — `quantem.gpu.SSB`

These are square scan-grid sizes, not detector dimensions.

| Platform | Scan grid | Evidence type | BF policy | State | Statistic | Time | Evidence / next gap |
|---|---:|---|---|---|---|---:|---|
| **CUDA** | `128x128` | Fixed-size parity | Frozen fixture | ✓ | — | **Pending** | Retain a dated device timing |
| **CUDA** | `256x256` | Fixed-size parity | Frozen fixture | ✓ | — | **Pending** | Retain a dated device timing |
| **CUDA** | `512x512` | Native real acquisition | Full active BF | ✓ | p50 | **32.2 ms** | [SSB-CUDA-512-FULL](performance/results.md), 2026-07-19, `0456e15e` |
| **CUDA** | `1024x1024` | Fixed-size parity | Frozen fixture | ✓ | — | **Pending** | Retain a dated device timing |
| **Python MPS** | `128x128` | Resized/synthetic | Fixed-size fixture | ✓ | — | **Pending** | Retain a comparable policy timing |
| **Python MPS** | `256x256` | Resized/synthetic | Fixed-size fixture | ✓ | — | **Pending** | Retain a comparable policy timing |
| **Python MPS** | `512x512` | Native real acquisition | Full active BF | ✓ | p50 | **537.58 ms** | [SSB-MPS-512-FULL](performance/results.md), 2026-07-28, `e8d49866` |
| **Python MPS** | `1024x1024` | Synthetic | 8,809 BF | ✓ | p50 | **669.1 ms** | [SSB-MPS-1024-SYNTH](performance/results.md); scaling evidence only |
| **Native Swift/Metal** | `128x128` | — | — | — | — | — | No native Swift SSB kernel |
| **Native Swift/Metal** | `256x256` | — | — | — | — | — | No native Swift SSB kernel |
| **Native Swift/Metal** | `512x512` | — | — | — | — | — | No native Swift SSB kernel |
| **Native Swift/Metal** | `1024x1024` | — | — | — | — | — | No native Swift SSB kernel |
| **WebGPU** | `128x128` | Real BF30 parity | Radius 30 px | ✓ | — | **Pending** | Retain a dated isolated timing |
| **WebGPU** | `256x256` | Deterministic test | Test fixture | Test | — | **Pending** | Retain physical real-data parity and timing |
| **WebGPU** | `512x512` | Real interaction | Incomplete frozen reference | Partial | — | **Pending** | Complete the CUDA artifact gate |
| **WebGPU** | `1024x1024` | Real interaction | Incomplete frozen reference | Partial | — | **Pending** | Complete the CUDA artifact gate |
| **CPU reference** | `128x128` | — | — | — | — | — | Not a retained adjudication size |
| **CPU reference** | `256x256` | — | — | — | — | — | Not a retained adjudication size |
| **CPU reference** | `512x512` | Independent adjudication | Frozen fixture | Ref | — | **Pending** | Correctness reference only |
| **CPU reference** | `1024x1024` | — | — | — | — | — | Not a retained adjudication size |

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
Keep every row atomic: a different bin, dtype path, fixture, cache state, or
statistic is another row. The documentation tests compare the landing-page and
dashboard configuration keys so the two views cannot silently drift.

Accepted and rejected experiments remain in the
[optimization ledger](maintainer/backend-optimization-matrix.md), and the
machine-readable evidence fingerprints are in
[`performance/evidence_manifest.json`](performance/evidence_manifest.json).
