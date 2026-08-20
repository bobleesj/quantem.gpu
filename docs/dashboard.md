# Kernel and benchmark dashboard

This is the one-page technical overview of `quantem.gpu`: what the scientific
kernels compute, where each runtime implements them, how parity is proved, and
what the latest retained measurements actually mean.

```{admonition} Read the state before comparing the number
:class: important
First-process source load, prepared-source load, warm resident compute, and
saved-result reopen are different experiments. A binned or cropped source is
never presented as native resolution. Check the complete provenance ledger
before using a number in a design or release decision.
```

**Dashboard review:** 2026-08-19. Every measured timing below shows the device
and test date. The overview deliberately omits evidence IDs and source revisions;
it does not replace the
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
the timing, memory, parity, device, and date columns are to the right.

| Platform | Selected scan | Scan plan | Source detector | Detector bin | Output detector | Source dtype | Decode dtype | Resident dtype | Cache state | Wall boundary | Fixture | Statistic | Time | Memory kind | Memory | Parity | Device tested | Date tested |
|---|---:|---|---:|---:|---:|---|---|---|---|---|---|---|---:|---|---:|---|---|---|
| [**CUDA**](platforms/cuda.md) | `512x512` | Full | `192x192` | 1 | `192x192` | `uint16` | `uint16` | `uint16` | Warm source | Load/decompress | C | p50 | **0.588 s** | Resident payload | **18.00 GiB** | Partial | NVIDIA RTX PRO 6000 Blackwell, GPU 0 | 2026-08-19 |
| [**CUDA**](platforms/cuda.md) | `512x512` | Full | `192x192` | 2 | `96x96` | `uint16` | `uint16` | `uint32` | Warm source | Load/decompress | C | p50 | **0.626 s** | Resident payload | **9.00 GiB** | Timing only | NVIDIA RTX PRO 6000 Blackwell, GPU 0 | 2026-08-19 |
| [**CUDA**](platforms/cuda.md) | `512x512` | Full | `192x192` | 4 | `48x48` | `uint16` | `uint16` | `uint32` | Warm source | Load/decompress | C | p50 | **0.553 s** | Resident payload | **2.25 GiB** | Partial | NVIDIA RTX PRO 6000 Blackwell, GPU 0 | 2026-08-19 |
| [**CUDA**](platforms/cuda.md) | `512x512` | Full | `192x192` | 8 | `24x24` | `uint16` | `uint16` | `uint32` | Warm source | Load/decompress | C | p50 | **0.599 s** | Resident payload | **0.5625 GiB** | Timing only | NVIDIA RTX PRO 6000 Blackwell, GPU 0 | 2026-08-19 |
| [**Python MPS**](platforms/mps.md) | `512x512` | Full | `192x192` | 1 | `192x192` | `uint16` | `uint16` | `uint16` | Warm source | Load/decompress | C | p50 | **2.164 s** | Resident payload | **18.00 GiB** | Partial | Apple M5 Max (`Mac17,6`, 40-core GPU, 128 GB) | 2026-08-19 |
| [**Python MPS**](platforms/mps.md) | `512x512` | Full | `192x192` | 2 | `96x96` | `uint16` | `uint16` | `uint16` | Warm source | Load/decompress | C | p50 | **0.691 s** | Resident payload | **4.50 GiB** | Timing only | Apple M5 Max (`Mac17,6`, 40-core GPU, 128 GB) | 2026-08-19 |
| [**Python MPS**](platforms/mps.md) | `512x512` | Full | `192x192` | 4 | `48x48` | `uint16` | `uint16` | `uint16` | Warm source | Load/decompress | C | p50 | **0.586 s** | Resident payload | **1.125 GiB** | Partial | Apple M5 Max (`Mac17,6`, 40-core GPU, 128 GB) | 2026-08-19 |
| [**Python MPS**](platforms/mps.md) | `512x512` | Full | `192x192` | 8 | `24x24` | `uint16` | `uint16` | `uint16` | Warm source | Load/decompress | C | p50 | **0.575 s** | Resident payload | **0.28125 GiB** | Timing only | Apple M5 Max (`Mac17,6`, 40-core GPU, 128 GB) | 2026-08-19 |
| [**Python MPS**](platforms/mps.md) | `512x512` | Full | `192x192` | 1 | `192x192` | `uint16` | `uint16` | `uint16` | Admission check | Pre-decode memory guard | C | Guard result | — | Estimated payload | **18.00 GiB** | Blocked safely | Apple M5 MacBook Pro (`Mac17,2`, 10-core GPU, 24 GB) | 2026-08-19 |
| [**Python MPS**](platforms/mps.md) | `512x512` | Full | `192x192` | 2 | `96x96` | `uint16` | `uint16` | `uint16` | Warm source | Load/decompress | C | p50 | **2.224 s** | Resident payload | **4.50 GiB** | Timing only | Apple M5 MacBook Pro (`Mac17,2`, 10-core GPU, 24 GB) | 2026-08-19 |
| [**Python MPS**](platforms/mps.md) | `512x512` | Full | `192x192` | 4 | `48x48` | `uint16` | `uint16` | `uint16` | Warm source | Load/decompress | C | p50 | **1.695 s** | Resident payload | **1.125 GiB** | Partial | Apple M5 MacBook Pro (`Mac17,2`, 10-core GPU, 24 GB) | 2026-08-19 |
| [**Python MPS**](platforms/mps.md) | `512x512` | Full | `192x192` | 8 | `24x24` | `uint16` | `uint16` | `uint16` | Warm source | Load/decompress | C | p50 | **1.580 s** | Resident payload | **0.28125 GiB** | Timing only | Apple M5 MacBook Pro (`Mac17,2`, 10-core GPU, 24 GB) | 2026-08-19 |
| [**CUDA**](platforms/cuda.md) | `512x512` | Full | `192x192` | 4 | `48x48` | `uint16` | `uint16` | `uint32` | First campaign encounter | Load + products | C | Single run | **2.027 s** | Process RSS maximum | **3.42 GiB** | Partial | NVIDIA RTX PRO 6000 Blackwell, GPU 0 | 2026-08-19 |
| [**Python MPS**](platforms/mps.md) | `512x512` | Full | `192x192` | 4 | `48x48` | `uint16` | `uint16` | `uint16` | First campaign encounter | Load + products | C | Single run | **1.982 s** | Peak memory footprint | **2.93 GiB** | Partial | Apple M5 Max (`Mac17,6`, 40-core GPU, 128 GB) | 2026-08-19 |
| [**Python MPS**](platforms/mps.md) | `512x512` | Full | `192x192` | 4 | `48x48` | `uint16` | `uint16` | `uint16` | First campaign encounter | Load + products | C | Single run | **2.775 s** | Peak memory footprint | **2.93 GiB** | Partial | Apple M5 MacBook Pro (`Mac17,2`, 10-core GPU, 24 GB) | 2026-08-19 |
| [**Native Swift/Metal**](platforms/swift-metal.md) | `512x512` | Full | `192x192` | 4 | `48x48` | `uint16` | `uint16` | `uint16` | First process | First complete product | A | p50 | **1.985 s** | Process peak | **1.43 GB** | Byte-identical | Apple M2 MacBook Air (`Mac14,2`, 8 GB) | 2026-08-18 |
| [**Native Swift/Metal**](platforms/swift-metal.md) | `512x512` | Full | `192x192` | 4 | `48x48` | `uint16` | `uint16` | `uint16` | First process | First complete product | B | p50 | **2.043 s** | Process peak | **1.43 GB** | Byte-identical | Apple M2 MacBook Air (`Mac14,2`, 8 GB) | 2026-08-18 |
| [**WebGPU**](platforms/webgpu.md) | `256x256` | Explicit crop | `192x192` | 1 | `192x192` | `uint16` | `uint8` | `uint8` | Prepared source | Full-stack load | — | p50 | **0.338 s** | Browser peak | **Pending** | Exact | Apple Metal-3 adapter (Mac model not retained) | 2026-07-20 |
| [**WebGPU**](platforms/webgpu.md) | `256x256` | Explicit crop | `192x192` | 2 | `96x96` | `uint16` | `uint8` | `float32` | Prepared source | Page profile | — | p50 | **0.774 s** | Browser peak | **Pending** | Exact | NVIDIA RTX PRO 6000 Blackwell | 2026-07-20 |
| [**WebGPU**](platforms/webgpu.md) | `256x256` | Explicit crop | `192x192` | 4 | `48x48` | `uint16` | `uint8` | `float32` | Prepared source | Page profile | — | p50 | **0.755 s** | Browser peak | **Pending** | Exact | NVIDIA RTX PRO 6000 Blackwell | 2026-07-20 |
| [**WebGPU**](platforms/webgpu.md) | `256x256` | Explicit crop | `192x192` | 8 | `24x24` | `uint16` | `uint8` | `float32` | Prepared source | Page profile | — | p50 | **0.733 s** | Browser peak | **Pending** | Exact | NVIDIA RTX PRO 6000 Blackwell | 2026-07-20 |
| [**WebGPU**](platforms/webgpu.md) | `512x512` | Full | `192x192` | 1 | `192x192` | `uint16` | `uint8` | `uint8` | Prepared source | Full-stack load | — | p50 | **0.772 s** | Decoded payload | **9.7 GB** | Exact | Apple Metal-3 adapter (Mac model not retained) | 2026-07-20 |
| [**WebGPU**](platforms/webgpu.md) | `512x512` | Full | `192x192` | 2 | `96x96` | `uint16` | `uint8` | `float32` | Prepared source | Page profile | — | Single profile | **1.199 s** | Browser peak | **Pending** | Exact | NVIDIA RTX PRO 6000 Blackwell | 2026-07-20 |
| [**WebGPU**](platforms/webgpu.md) | `512x512` | Full | `192x192` | 4 | `48x48` | `uint16` | `uint8` | `float32` | Prepared source | Page profile | — | Single profile | **1.212 s** | Browser peak | **Pending** | Exact | NVIDIA RTX PRO 6000 Blackwell | 2026-07-20 |
| [**WebGPU**](platforms/webgpu.md) | `512x512` | Full | `192x192` | 8 | `24x24` | `uint16` | `uint8` | `float32` | Prepared source | Page profile | — | Single profile | **1.106 s** | Browser peak | **Pending** | Exact | NVIDIA RTX PRO 6000 Blackwell | 2026-07-20 |
| [**WebGPU**](platforms/webgpu.md) | `512x512` | Full | `192x192` | 2 | `96x96` | `uint16` | `uint16` | `float32` | Prepared source | Two-pass page profile | — | Single profile | **2.651 s** | Browser peak | **Pending** | Exact | NVIDIA RTX PRO 6000 Blackwell | 2026-07-20 |

Fixture C is the current `512x512x192x192` 27-shard compressed-HDF5 fixture:
`uint16`, 3,169,920,193 compressed bytes, with all 28 file hashes identical on
the three machines. Its aggregate file-manifest SHA-256 begins `741e7bcf`.
Every C row uses the complete scan, no scan crop, scan bin 1, and the detector
bin shown in the row. The measured source revision is `8c47a466` (source tree
`c3094dcf`).

The C “first campaign encounter” rows are not cold: no operating-system cache
purge or reboot was performed. Their wall boundary is synchronized public load
plus mean diffraction, exact total/BF/DF, CoM row/column, and fixed-orientation
iDPC; campaign artifact serialization is excluded. They are single runs, while
the warm-source rows retain five or seven repetitions with p50/p95/max in the
[complete ledger](performance/results.md).

“Partial” is deliberate. At detector bin 4, integer mean/total/BF/DF products
are byte-exact across all three machines and CoM passes the frozen `1e-5` gate,
but CUDA-versus-MPS iDPC reaches `2.84e-5` maximum absolute error and fails the
frozen `rtol=1e-5`, `atol=1e-5` gate. At native detector resolution, the public
MPS interaction sidecar changes CoM and iDPC; the direct full-resolution Metal
CoM kernel passes, but iDPC still fails. Detector-bin-2 and detector-bin-8 rows
are therefore timing evidence only until equivalent real-data parity artifacts
are retained.

The `256x256` entries above are explicit scan-crop experiments. They are useful
configuration evidence, but they are never substituted for full-scan loading
or used as an automatic memory policy. The
[complete benchmark ledger](performance/results.md) retains the source revision,
complete timing distribution, and calibration facts behind these concise rows.

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
| [**CUDA**](platforms/cuda.md) | `uint16` | `uint16` | `uint16` | Exact native counts | ✓ | Native-detector payload | **18.00 GiB** |
| [**CUDA**](platforms/cuda.md) | `uint16` | `uint8` | `uint8` | Complete-audit lossless | ✓ | Decoded resident | **9.66 GB** |
| [**CUDA**](platforms/cuda.md) | `uint32` | `uint32` | `uint32` | Exact native counts | ✓ | Process peak | **Pending** |
| [**Python MPS**](platforms/mps.md) | `uint8` | — | — | Native compressed source | — | — | — |
| [**Python MPS**](platforms/mps.md) | `uint16` | `uint16` | `uint16` | Exact native counts | ✓ | Native-detector payload | **18.00 GiB** |
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

The current C matrix measures native `uint16` input at detector bins 1/2/4/8.
On Python MPS those resident payloads are 18.00/4.50/1.125/0.28125 GiB. On
CUDA, exact detector summation retains `uint32` for bins 2/4/8, giving
9.00/2.25/0.5625 GiB. The 24 GB Apple M5 rejects the 18.00 GiB native-detector
allocation before decode because its conservative 70% working-set limit is
12.4 GiB; it does not silently select bin 2.

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
| **Python MPS** | `256x256` | Explicit crop | `192x192` | 1 | `192x192` | ✓ | **Pending** | Retain the existing exact crop gate as a dated load benchmark |
| **Python MPS** | `256x256` | Explicit crop | `192x192` | 2 | `96x96` | Pending | **Pending** | Joint crop-plus-bin parity and hardware timing |
| **Python MPS** | `256x256` | Explicit crop | `192x192` | 4 | `48x48` | Pending | **Pending** | Joint crop-plus-bin parity and hardware timing |
| **Python MPS** | `256x256` | Explicit crop | `192x192` | 8 | `24x24` | Pending | **Pending** | Joint crop-plus-bin parity and hardware timing |
| **Native Swift/Metal** | `256x256` | Explicit crop | `192x192` | 1 | `192x192` | Pending | **Pending** | Physical-device crop parity and timing |
| **Native Swift/Metal** | `256x256` | Explicit crop | `192x192` | 2 | `96x96` | Pending | **Pending** | Physical-device crop-plus-bin parity and timing |
| **Native Swift/Metal** | `256x256` | Explicit crop | `192x192` | 4 | `48x48` | Pending | **Pending** | Physical-device crop-plus-bin parity and timing |
| **Native Swift/Metal** | `256x256` | Explicit crop | `192x192` | 8 | `24x24` | — | — | Public load plan does not support bin 8 |
| **Native Swift/Metal** | `512x512` | Full | `192x192` | 1 | `192x192` | Test | **Pending** | Physical first-process timing with frozen full-product parity |
| **Native Swift/Metal** | `512x512` | Full | `192x192` | 2 | `96x96` | Test | **Pending** | Physical first-process timing with frozen full-product parity |
| **Native Swift/Metal** | `512x512` | Full | `192x192` | 8 | `24x24` | — | — | Public load plan does not support bin 8 |

CUDA and Python MPS full-`512x512` bins 1/2/4/8 now have separate current
timing rows above. Detector-bin-2 and detector-bin-8 real-data parity remains
pending and is recorded in the parity column instead of being confused with a
timing gap. WebGPU `256x256` bins 1/2/4/8 and `512x512` bins 1/2/4/8 also have
separate retained rows above. The **CPU reference** remains the independent
`Ref` adjudicator, not an accelerator timing placeholder. The additional
`128x128` and `1024x1024`
size-specific gates remain in the detailed evidence pages until a joint
scan/bin measurement is retained.

### Screening and prepared-product caches — `quantem.gpu.screening`

| Platform | Support | Scan grid | Detector | Source dtype | Detector bin | Chunk plan | Statistic | Time | Device tested | Date tested |
|---|---|---:|---:|---|---:|---|---|---:|---|---|
| **CUDA** | ✓ | `1024x1024` | `192x192` | `uint16` | 1 | 12 GB allocator cap | Single build | **12.31 s** | NVIDIA RTX PRO 6000 Blackwell | 2026-07-28 |
| **Python MPS** | ✓ | `512x512` | `192x192` | `uint16` | 1 | 64 scan rows | Single build | **3.96 s** | Apple Metal GPU (model not retained) | 2026-07-21 |
| **Native Swift/Metal** | — | — | — | — | — | — | — | — | — | — |
| **WebGPU** | — | — | — | — | — | — | — | — | — | — |
| **CPU reference** | Ref | — | — | — | — | Reference fixtures | — | **Pending** | — | — |

Backend-neutral saved-product reopen is a separate state:
[PRODUCT-CACHE-REOPEN](performance/results.md) retained **6.8-8.0 ms** for five
repeats of full-`1024` BF/DF/CoM/rotation products on 2026-07-20. The host model
was not retained. It is never represented as source load or cache construction.

### Virtual images — `quantem.gpu.detector`

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

The current detector-bin-4 mean diffraction and exact total/BF/DF arrays are
byte-identical across Phil, Rodman, and MJGOAT GPU 0. These are first-resident
single-pass diagnostics; they are not source-load times or repeated kernel
distributions.

### Detector moments and phase contrast — `quantem.gpu.dpc`

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

Phil and Rodman detector-bin-4 CoM/iDPC arrays are byte-identical. CUDA differs
by at most `1.91e-6` for CoM, which passes the frozen `1e-5` gate, but iDPC
differs by `2.84e-5` and fails its frozen gate. At detector bin 1, the direct
full-resolution Metal CoM pass takes `83.3 ms` on Phil and agrees with CUDA
within `7.63e-6`; the public automatic bin-2 interaction sidecar does not, so
native-detector MPS CoM/iDPC remains blocked rather than promoted as exact.

### Single-sideband ptychography — `quantem.gpu.SSB`

These are square scan-grid sizes, not detector dimensions.

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

See the [full SSB performance record](maintainer/ssb-performance.md) for the
12-cell redraw matrix, size-specific timings, memory, and rejected experiments.

### Cross-module platform map

| Platform | I/O | Screening | Virtual images | CoM/DPC/iDPC | SSB | Display/movie and other boundaries |
|---|---|---|---|---|---|---|
| **CUDA** | ✓ | ✓ | ✓ | CoM ✓; cross-backend iDPC Block | ✓ | Display ✓; movie via NVENC; parallax CUDA-only |
| **Python MPS** | ✓ | ✓ | ✓ | Bin-4 CoM ✓; native sidecar Block | ✓ | Display ✓; movie via VideoToolbox; parallax — |
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
