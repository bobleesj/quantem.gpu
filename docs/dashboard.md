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

**Dashboard review:** 2026-08-22. Every measured timing below shows the device
and test date. The overview deliberately omits evidence IDs and source revisions;
it does not replace the
[complete benchmark provenance ledger](performance/results.md).

(coverage-and-next-runs)=
## Coverage and next runs

The [filterable coverage registry](performance/coverage.md) now keeps every
required configuration visible, including configurations that have never run.
It separates complete measurements, partial evidence, pending work, refuted
experiments, and fail-closed unsupported paths. Each open row names a stable
runbook, its physical owner, and the exact artifact required for promotion.

```{include} _generated/benchmark_coverage.md
:start-after: <!-- benchmark-coverage-summary-start -->
:end-before: <!-- benchmark-coverage-summary-end -->
```

Agents and maintainers should begin with:

```bash
python scripts/benchmark_registry.py next --limit 10
python scripts/benchmark_registry.py command GATE_ID
```

The overview tables below retain current measurements, qualified probes, and
clearly marked diagnostics. Read the **State** cell before the time. The
coverage registry is the canonical place to find missing combinations and
their reproduction entry points.

(speed-and-memory-at-a-glance)=
## Speed and memory at a glance

The rows below are deliberately not a leaderboard. Each one keeps its cache
state, scientific plan, device, wall-clock boundary, and memory observation in
the same row. Measurement tables are keyed first by **Platform**, then by the
reproducible **Computer** class; local host nicknames are never public benchmark
identifiers.

### Measured load configurations

One row is one exact configuration and one headline statistic. Source, decode,
and resident dtype are separate columns. A second fixture, bin, dtype path,
cache state, or timing statistic requires another row; slash-delimited bundles
are not permitted. **Pending** means the combination is tracked but has no
retained timing with complete provenance.

The WebGPU rows in this measured table are retained historical implementation
measurements. In particular, detector bins 2/4/8 used a `float32` resident
representation; they are not evidence for the pending exact-integer production
contract shown in the coverage matrix.

The rendered site adds local filters for platform, computer, device, detector bin,
cache/process state, and free text. Filtering changes only which retained rows
are visible; the static table remains the canonical source. On narrow screens
the table scrolls so timing, resident payload, measured peak, parity, device,
and date stay separate.

| Platform | Computer | State | Selected scan | Scan plan | Source detector | Detector bin | Output detector | Source dtype | Decode dtype | Resident dtype | Cache/process state | Wall boundary | Fixture | Statistic | Time | Logical resident | Device/driver boundary | Device/driver peak | Process/tree RSS | Parity | Device tested | Date tested |
|---|---|---|---:|---|---:|---:|---:|---|---|---|---|---|---|---|---:|---:|---|---:|---:|---|---|---|
| [**CUDA**](platforms/cuda.md) | Linux CUDA workstation (dual 96 GB Blackwell GPUs) | Partial | `512x512` | Full | `192x192` | 1 | `192x192` | `uint16` | `uint16` | `uint16` | Warm source | First usable resident | D | p50 | **0.386 s** | **18.00 GiB** | Total-card occupancy | **21.215 GiB** | Pending | Qualified products | NVIDIA RTX PRO 6000 Blackwell Max-Q, GPU 1 | 2026-08-19 |
| [**CUDA**](platforms/cuda.md) | Linux CUDA workstation (dual 96 GB Blackwell GPUs) | Partial | `512x512` | Full | `192x192` | 2 | `96x96` | `uint16` | `uint16` | `uint32` | Warm source | First usable resident | D | p50 | **0.396 s** | **9.00 GiB** | Total-card occupancy | **11.561 GiB** | Pending | Qualified products | NVIDIA RTX PRO 6000 Blackwell Max-Q, GPU 1 | 2026-08-19 |
| [**CUDA**](platforms/cuda.md) | Linux CUDA workstation (dual 96 GB Blackwell GPUs) | Partial | `512x512` | Full | `192x192` | 4 | `48x48` | `uint16` | `uint16` | `uint32` | Warm source | First usable resident | D | p50 | **0.390 s** | **2.25 GiB** | Total-card occupancy | **3.756 GiB** | Pending | Qualified products | NVIDIA RTX PRO 6000 Blackwell Max-Q, GPU 1 | 2026-08-19 |
| [**CUDA**](platforms/cuda.md) | Linux CUDA workstation (dual 96 GB Blackwell GPUs) | Partial | `512x512` | Full | `192x192` | 8 | `24x24` | `uint16` | `uint16` | `uint32` | Warm source | First usable resident | D | p50 | **0.381 s** | **0.5625 GiB** | Total-card occupancy | **1.805 GiB** | Pending | Qualified products | NVIDIA RTX PRO 6000 Blackwell Max-Q, GPU 1 | 2026-08-19 |
| [**Native Swift/Metal**](platforms/swift-metal.md) | MacBook Pro (M5 Max, 128 GB) | Measured | `512x512` | Full | `192x192` | 1 | `192x192` | `uint16` | audit-bound `uint8` | `uint16` | Controlled `F_NOCACHE`; new index | Exact complete resident | C | p50 | **0.578 s** | **18.00 GiB** | After-load Metal allocation; sampled peak pending | **>=18.571 GiB** | **0.874 GiB** | Full volume + products exact | Apple M5 Max (`Mac17,6`, 40-core GPU, 128 GB) | 2026-08-22 |
| [**Native Swift/Metal**](platforms/swift-metal.md) | MacBook Pro (M5, 24 GB) | Partial | `512x512` | Full | `192x192` | 2 | `96x96` | `uint16` | audit-bound `uint8` | `uint16` | Prepared index; `F_NOCACHE`; destination reused | Exact indexed load and seven products | C | p50 | **0.671 s** | **4.50 GiB** | Maximum Metal allocation with boundary hash | **5.071 GiB** | **0.646 GiB** | Full volume + products exact; terminal manifest pending | Apple M5 (`Mac17,2`, 10-core GPU, 24 GB) | 2026-08-22 |
| [**Native Swift/Metal**](platforms/swift-metal.md) | MacBook Pro (M5, 24 GB) | Partial | `512x512` | Full | `192x192` | 4 | `48x48` | `uint16` | audit-bound `uint8` | `uint16` | Prepared index; `F_NOCACHE`; destination reused | Exact indexed load and seven products | C | p50 | **0.631 s** | **1.125 GiB** | Maximum Metal allocation with boundary hash | **1.696 GiB** | **0.650 GiB** | Full volume + products exact; terminal manifest pending | Apple M5 (`Mac17,2`, 10-core GPU, 24 GB) | 2026-08-22 |
| [**WebGPU**](platforms/webgpu.md) | MacBook Pro (M5 Max, 128 GB) | Historical diagnostic | `512x512` | Full | `192x192` | 1 | `192x192` | `uint16` | `uint8` | `uint8` | Warm OS cache | First usable resident | D | p50 | **0.824 s** | **9.00 GiB** | Device allocation incomplete | Pending | **5.020 GiB** | Sampled frames only | Chrome 151, Apple M5 Max Metal-3 | 2026-08-19 |
| [**WebGPU**](platforms/webgpu.md) | MacBook Pro (M5 Max, 128 GB) | Historical diagnostic | `512x512` | Full | `192x192` | 2 | `96x96` | `uint16` | `uint16` | `float32` | Warm OS cache | First usable resident | D | p50 | **1.281 s** | **9.00 GiB** | Device allocation incomplete | Pending | **5.363 GiB** | Sampled frames; wrong resident dtype for exact gate | Chrome 151, Apple M5 Max Metal-3 | 2026-08-19 |
| [**WebGPU**](platforms/webgpu.md) | MacBook Pro (M5 Max, 128 GB) | Historical diagnostic | `512x512` | Full | `192x192` | 4 | `48x48` | `uint16` | `uint16` | `float32` | Warm OS cache | First usable resident | D | p50 | **1.044 s** | **2.25 GiB** | Device allocation incomplete | Pending | **5.188 GiB** | Sampled frames; wrong resident dtype for exact gate | Chrome 151, Apple M5 Max Metal-3 | 2026-08-19 |
| [**WebGPU**](platforms/webgpu.md) | MacBook Pro (M5 Max, 128 GB) | Historical diagnostic | `512x512` | Full | `192x192` | 8 | `24x24` | `uint16` | `uint16` | `float32` | Warm OS cache | First usable resident | D | p50 | **0.979 s** | **0.5625 GiB** | Device allocation incomplete | Pending | **5.184 GiB** | Sampled frames; wrong resident dtype for exact gate | Chrome 151, Apple M5 Max Metal-3 | 2026-08-19 |
| **CPU reference** | MacBook Pro (M5 Max, 128 GB) | Reference | `512x512` | Full | `192x192` | 1 | `192x192` | `uint16` | `uint16` | `uint16` | Reference traversal | First usable host array | D | Single run | **34.37 s** | **18.00 GiB** | Not separate | — | **36.450 GiB** | Reference | Apple M5 Max CPU | 2026-08-19 |
| **CPU reference** | MacBook Pro (M5 Max, 128 GB) | Reference | `512x512` | Full | `192x192` | 2 | `96x96` | `uint16` | `uint16` | `uint16` | Reference traversal | First usable host array | D | Single run | **54.22 s** | **4.50 GiB** | Not separate | — | **9.634 GiB** | Reference | Apple M5 Max CPU | 2026-08-19 |
| **CPU reference** | MacBook Pro (M5 Max, 128 GB) | Reference | `512x512` | Full | `192x192` | 4 | `48x48` | `uint16` | `uint16` | `uint16` | Reference traversal | First usable host array | D | Single run | **43.04 s** | **1.125 GiB** | Not separate | — | **2.978 GiB** | Reference | Apple M5 Max CPU | 2026-08-19 |
| **CPU reference** | MacBook Pro (M5 Max, 128 GB) | Reference | `512x512` | Full | `192x192` | 8 | `24x24` | `uint16` | `uint16` | `uint16` | Reference traversal | First usable host array | D | Single run | **38.13 s** | **0.28125 GiB** | Not separate | — | **2.034 GiB** | Reference | Apple M5 Max CPU | 2026-08-19 |

#### Current Python MPS resident lifecycle

These current-head rows use fixture C, the full `512x512` scan, native
`192x192 uint16` source data, scan bin 1, no crop, and exact `uint16`
outputs. Revision `0bc9378` was measured after one same-process warmup; OS
source pages were uncontrolled and no eviction was performed. Every trial
returned a fresh resident destination and explicitly released it before the
next trial. The timed outputs retained six selected-frame parity probes rather
than a full-volume hash, so every row remains **Qualified probes / Partial**.
These are post-warmup package measurements, not cold-source or application
timings.

The MacBook Pro (M5, 24 GB) Python MPS rows remain **Pending** in the coverage
matrix. Its 2026-08-22 preflight stopped before loading because the baseline
runner did not hash the complete resident output or record observed dtype. The
current runbook now requires a full-volume SHA-256, dtype, shape, and sampled
peak-memory check before any timing can be promoted.

| Platform | Computer | Evidence state | Detector bin | Output detector | Samples | p50 | p95 | Maximum | Logical resident | Driver allocated after load | Driver allocated after release | Process RSS high-water | Whole-system swap delta | Date tested |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| [**Python MPS**](platforms/mps.md) | MacBook Pro (M5 Max, 128 GB) | Qualified probes / Partial | 1 | `192x192` | 7 | **0.414824 s** | **0.457261 s** | **0.457261 s** | **19,327,352,832 B** | **19,801,456,640 B** | **474,103,808 B** | **741,818,368 B** | **0 B** | 2026-08-22 |
| [**Python MPS**](platforms/mps.md) | MacBook Pro (M5 Max, 128 GB) | Qualified probes / Partial | 2 | `96x96` | 7 | **0.457153 s** | **0.461730 s** | **0.461730 s** | **4,831,838,208 B** | **6,107,774,976 B** | **1,275,936,768 B** | **616,054,784 B** | **0 B** | 2026-08-22 |
| [**Python MPS**](platforms/mps.md) | MacBook Pro (M5 Max, 128 GB) | Qualified probes / Partial | 4 | `48x48` | 7 | **0.382109 s** | **0.384353 s** | **0.384353 s** | **1,207,959,552 B** | **2,483,896,320 B** | **1,275,936,768 B** | **615,825,408 B** | **0 B** | 2026-08-22 |
| [**Python MPS**](platforms/mps.md) | MacBook Pro (M5 Max, 128 GB) | Qualified probes / Partial | 8 | `24x24` | 7 | **0.356258 s** | **0.358652 s** | **0.358652 s** | **301,989,888 B** | **1,577,926,656 B** | **1,275,936,768 B** | **616,054,784 B** | **0 B** | 2026-08-22 |

“Driver allocated after load” and “after release” are instantaneous
`torch.mps.driver_allocated_memory()` samples, not continuously sampled
peaks. Process RSS is the process-lifetime high-water; swap is whole-system
usage and remained unchanged during each retained run.

##### Historical explicit destination reuse

The accepted isolated-candidate ABBA experiment remains useful because it
measures a different lifecycle: the caller reuses a compatible resident
destination. It is retained as history and is not substituted for the
current-head fresh-destination rows.

| Detector bin | Output detector | Destination | Samples | p50 | p95 | Maximum | Source revision |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | `192x192` | Explicitly recycled | 8 | **0.259189 s** | **0.263118 s** | **0.263375 s** | `b7f8ef3` |
| 2 | `96x96` | Explicitly recycled | 8 | **0.359606 s** | **0.361384 s** | **0.361995 s** | `b7f8ef3` |
| 4 | `48x48` | Explicitly recycled | 8 | **0.352990 s** | **0.355048 s** | **0.355062 s** | `b7f8ef3` |

Fixtures C and D are independent real `512x512x192x192`, native-`uint16`,
27-shard compressed-HDF5 sources. Fixture C contains 3,169,489,846 indexed
compressed bytes plus its 430,347-byte master file. Every row selects the
complete scan, uses no scan crop, keeps scan bin 1, and records the detector bin
explicitly. CUDA and WebGPU use D; Python MPS and Native Swift/Metal use C; CPU
is an independent reference, never a silent fallback. Different fixtures and
boundaries are not ranked.

The 2026-08-19 source rows did not forcibly evict the source cache and therefore
remain labeled warm. The final-head MPS rows also used uncontrolled source
pages after one same-process warmup. Imports and Metal-library initialization
were complete before the measured package load. None is cold-storage or
application evidence.

The former 2.273-second MPS bin-1 headline is superseded: its benchmark cleared
the Torch cache but did not release direct PyObjC Metal buffers, so repeated
18 GiB outputs drove system free memory from 86% to 41%. The retained historical
artifact remains in the results ledger. The exact full bin-1 resident payload
is 19,327,352,832 bytes (18.00 GiB); driver allocation sampled immediately
after load is 19,801,456,640 bytes (18.442 GiB), while process RSS high-water is
only 0.691 GiB. This sampling boundary does not prove a continuously observed
Metal peak.
The binned `uint16` rows are fixture-specific: source identity
`9f0ddb93...` has a complete maximum-count audit of 53, so even an 8x8 exact
sum is at most 3,392 and fits `uint16`. This does not authorize `uint16`
summation for an unaudited source; such a source needs a sufficient wider dtype
or a fail-closed range audit.

The 2026-08-22 native row instead applies macOS
`F_NOCACHE` to source hashing and every indexed source descriptor, creates a new
index root and private Metal destination per process, and stops at the complete
exact 18 GiB resident volume plus products. Its identity-bound value audit
already exists, so it is controlled uncached-source-page evidence—not an
audit-free arbitrary-source cold load and not application end to end. The
0.029-second native prepared exact-summary reopen is intentionally kept in
the screening section rather than this load table because it does not decode or
traverse the resident 4D volume. The results ledger owns p95/max, exact
revisions, fixture hashes, stage intervals, logical payloads, and parity
artifacts.

Logical resident payload, device/driver peak, and process/tree RSS are different
columns by design. Process RSS does not include every direct Metal or WebGPU
allocation. A peak whose boundary is incomplete is labeled incomplete or as a
lower bound; it is never allowed to stand in for total unified-memory demand.
Historical cropped, superseded prepared, first-campaign, and application-level
rows remain in the maintainer records linked from
[Verified benchmark results](performance/results.md).

(dtype-support-and-peak-memory)=
### Dtype support and peak memory

“Source,” “working,” “accumulation,” and “resident” dtype describe different
stages. A `uint8` row is scientifically exact only when the source is already
`uint8` or a complete source audit proves `maximum <= 255` and
`pixelsAbove255 == 0`. Otherwise an explicit `dtype="u8"` load saturates values
above 255 and is a browse representation, not raw-count evidence.

| Platform | Computer | Source dtype | Decode/working dtype | Resident dtype | Precision policy | Support | Resident payload example | Measured memory example | Measurement boundary |
|---|---|---|---|---|---|---|---:|---:|---|
| [**CUDA**](platforms/cuda.md) | No physical result | `uint8` | — | — | Native compressed source | — | — | — | Capability only |
| [**CUDA**](platforms/cuda.md) | Linux CUDA workstation (dual 96 GB Blackwell GPUs) | `uint16` | `uint16` | `uint16` | Exact native counts | ✓ | **18.00 GiB** | **21.215 GiB** | Full bin1 total-card peak |
| [**CUDA**](platforms/cuda.md) | No physical result | `uint16` | `uint8` | `uint8` | Complete-audit lossless | ✓ | **9.00 GiB** | **Pending** | Full native detector |
| [**CUDA**](platforms/cuda.md) | No physical result | `uint32` | `uint32` | `uint32` | Exact native counts | ✓ | Shape/bin dependent | **Pending** | Capability only |
| [**Python MPS**](platforms/mps.md) | No physical result | `uint8` | — | — | Native compressed source | — | — | — | Capability only |
| [**Python MPS**](platforms/mps.md) | MacBook Pro (M5 Max, 128 GB) | `uint16` | `uint16` | `uint16` | Exact native counts | ✓ | **18.00 GiB** | **18.442 GiB** | After-load driver allocation; qualified probes only; continuous peak pending |
| [**Python MPS**](platforms/mps.md) | No physical result | `uint16` | `uint8` | `uint8` | Complete-audit lossless | ✓ | **9.00 GiB** | **Pending** | Full native detector |
| [**Python MPS**](platforms/mps.md) | No physical result | `uint32` | `uint16` | `uint16` | Guarded exact narrowing | ✓ | Shape/bin dependent | **Pending** | Capability only |
| [**Python MPS**](platforms/mps.md) | No physical result | `uint32` | `uint32` | `uint32` | Exact native counts | ✓ | Shape/bin dependent | **Pending** | Capability only |
| [**Native Swift/Metal**](platforms/swift-metal.md) | No physical result | `uint8` | — | — | Native compressed-source decode | Pending | — | — | No retained native-`uint8` bitshuffle-source decode evidence |
| [**Native Swift/Metal**](platforms/swift-metal.md) | MacBook Pro (M5 Max, 128 GB) | `uint16` | audit-bound `uint8` | `uint16` | Exact native counts | ✓ | **18.00 GiB** | **>=18.571 GiB** | Full bin1 after-load allocation; sampled peak pending |
| [**Native Swift/Metal**](platforms/swift-metal.md) | MacBook Air (M2, 8 GB) | `uint16` | `uint16` | `uint16` | Audited exact detector sum | ✓ | **1.125 GiB** | **1.332 GiB** | Historical physical bin4 process-footprint example |
| [**Native Swift/Metal**](platforms/swift-metal.md) | No physical result | `uint16` | `uint16` | `uint32` | General exact detector sum | ✓ | Shape/bin dependent | **Pending** | Capability only |
| [**WebGPU**](platforms/webgpu.md) | No physical result | `uint8` | `uint8` | `uint8` | Exact native counts | ✓ | Shape/bin dependent | **Pending** | Capability only |
| [**WebGPU**](platforms/webgpu.md) | No physical result | `uint16` | `uint16` | `uint16` | Exact native counts | Pending | Shape/bin dependent | **Pending** | Production full-volume `uint16` resident proof not retained |
| [**WebGPU**](platforms/webgpu.md) | MacBook Pro (M5 Max, 128 GB) | `uint16` | `uint8` | `uint8` | Complete-audit lossless | Diagnostic | **9.00 GiB** | **5.020 GiB** | Historical Chrome-tree RSS; device peak and full-volume parity incomplete |
| [**WebGPU**](platforms/webgpu.md) | No physical result | `uint16` | `uint32` | `uint32` | General exact detector sum | Pending | Shape/bin dependent | **Pending** | Exact-integer production binning is not integrated |
| [**WebGPU**](platforms/webgpu.md) | No physical result | `uint32` | `uint32` | `uint32` | Exact native counts | Pending | Shape/bin dependent | **Pending** | Hardware full-volume proof not retained |
| **CPU reference** | Portable CI runner | `uint8` | `uint8` | `uint8` | Exact reference | Ref | Shape/bin dependent | **Pending** | Reference contract |
| **CPU reference** | Portable CI runner | `uint16` | `uint16` | `uint16` | Exact reference | Ref | Shape/bin dependent | **Pending** | See measured load rows |

Memory examples in this capability table are configuration-specific, not a
promise for every shape. The main load table is authoritative for the matching
scan, detector bin, dtype, device, and boundary. In particular, a browser RSS
sample can be smaller than a resident WebGPU payload because it does not capture
all device allocations; that is an incomplete peak, not evidence that the
payload disappeared.

The current C matrix measures native `uint16` input at detector bins 1/2/4/8.
On Python MPS those resident payloads are 18.00/4.50/1.125/0.28125 GiB. On
CUDA, exact detector summation retains `uint32` for bins 2/4/8, giving
9.00/2.25/0.5625 GiB. The native primitive has admitted one full 18.00 GiB
resident load on the 24 GB Apple M5, but repeated uncontended memory and paging
evidence remains incomplete. A consumer may apply a conservative working-set
limit; that application policy is separate from primitive capability and may
not silently select detector bin 2.

The current production WebGPU detector-bin-2/4/8 path accumulates into and
stores `float32`. Those historical timings remain useful implementation
history, but they do not satisfy the exact-integer resident contract. The
platform/computer registry therefore marks exact WebGPU bins 2/4/8 **blocked**
until integer accumulation and residency pass full-volume parity.

The public Python selector is intentionally explicit:

- `dtype="u16"` requests unsigned 16-bit resident counts;
- `dtype="u8"` requests saturating unsigned 8-bit browse counts;
- `dtype="native"` preserves the source dtype; and
- `dtype="auto"` is an advisory convenience, not a substitute for a retained
  complete-source value-range audit.

(minimum-device-memory-gates)=
### Minimum-device memory gates

The public release floors are **6 GiB of dedicated VRAM for CUDA** and **8 GB
of total laptop RAM for WebGPU**. The WebGPU number is the entire machine
budget shared by the operating system, browser, JavaScript heap, staging
buffers, and GPU—not memory available exclusively to one `GPUBuffer`.

**✓** is awarded only after the complete load-and-product pipeline runs on a
physical device at or below the stated floor with retained peak memory,
pressure/swap, output parity, and responsiveness evidence. A calculated
payload can prove **No**, but it can only establish a **Pending** candidate.
An allocator cap on a larger GPU is a useful pre-check, not physical-device
signoff.

| Platform | Computer | Minimum device | Selected scan | Scan plan | Source detector | Detector bin | Output detector | Resident dtype | Resident payload | Gate | Reason | Device tested | Date tested |
|---|---|---|---:|---|---:|---:|---:|---|---:|---|---|---|---|
| [**CUDA**](platforms/cuda.md) | N/A (calculated 6 GiB floor) | 6 GiB VRAM | `512x512` | Full | `192x192` | 1 | `192x192` | `uint16` | **18.00 GiB** | **No** | Payload exceeds device floor | — | — |
| [**CUDA**](platforms/cuda.md) | N/A (calculated 6 GiB floor) | 6 GiB VRAM | `512x512` | Full | `192x192` | 2 | `96x96` | `uint32` | **9.00 GiB** | **No** | Payload exceeds device floor | — | — |
| [**CUDA**](platforms/cuda.md) | Physical 6 GiB CUDA computer pending | 6 GiB VRAM | `512x512` | Full | `192x192` | 4 | `48x48` | `uint32` | **2.25 GiB** | **Pending** | Complete physical 6 GiB peak not retained | — | — |
| [**CUDA**](platforms/cuda.md) | Physical 6 GiB CUDA computer pending | 6 GiB VRAM | `512x512` | Full | `192x192` | 8 | `24x24` | `uint32` | **0.5625 GiB** | **Pending** | Complete physical 6 GiB peak not retained | — | — |
| [**Python MPS**](platforms/mps.md) | MacBook Air (M2, 8 GB) | 8 GB unified RAM | `512x512` | Full | `192x192` | 1 | `192x192` | `uint16` | **18.00 GiB** | **Blocked** | Resident payload exceeds physical memory | — | — |
| [**Python MPS**](platforms/mps.md) | MacBook Air (M2, 8 GB) | 8 GB unified RAM | `512x512` | Full | `192x192` | 2 | `96x96` | `uint16` | **4.50 GiB** | **Pending** | Physical 8 GB pressure/parity run not retained | — | — |
| [**Python MPS**](platforms/mps.md) | MacBook Air (M2, 8 GB) | 8 GB unified RAM | `512x512` | Full | `192x192` | 4 | `48x48` | `uint16` | **1.125 GiB** | **Pending** | Physical 8 GB pressure/parity run not retained | — | — |
| [**Python MPS**](platforms/mps.md) | MacBook Air (M2, 8 GB) | 8 GB unified RAM | `512x512` | Full | `192x192` | 8 | `24x24` | `uint16` | **0.28125 GiB** | **Pending** | Physical 8 GB pressure/parity run not retained | — | — |
| [**Native Swift/Metal**](platforms/swift-metal.md) | MacBook Air (M2, 8 GB) | 8 GB unified RAM | `512x512` | Full | `192x192` | 1 | `192x192` | `uint16` | **18.00 GiB** | **Blocked** | Resident payload exceeds physical memory | — | — |
| [**Native Swift/Metal**](platforms/swift-metal.md) | MacBook Air (M2, 8 GB) | 8 GB unified RAM | `512x512` | Full | `192x192` | 2 | `96x96` | `uint16` | **4.50 GiB** | **Pending** | Current clean-revision pressure/parity run not retained | — | — |
| [**Native Swift/Metal**](platforms/swift-metal.md) | MacBook Air (M2, 8 GB) | 8 GB unified RAM | `512x512` | Full | `192x192` | 4 | `48x48` | `uint16` | **1.125 GiB** | **Historical** | Earlier physical run passed; current clean-revision repeat remains pending | Apple M2 MacBook Air (`Mac14,2`, 8 GB) | 2026-08-18 |
| [**Native Swift/Metal**](platforms/swift-metal.md) | MacBook Air (M2, 8 GB) | 8 GB unified RAM | `512x512` | Full | `192x192` | 8 | `24x24` | — | — | **Not supported** | Current native load-plan contract supports bins 1, 2, and 4 | — | — |
| [**WebGPU**](platforms/webgpu.md) | MacBook Air (M2, 8 GB) | 8 GB total RAM | `512x512` | Full | `192x192` | 1 | `192x192` | `uint16` | **18.00 GiB** | **Blocked** | Resident payload exceeds total machine RAM | — | — |
| [**WebGPU**](platforms/webgpu.md) | MacBook Air (M2, 8 GB) | 8 GB total RAM | `512x512` | Full | `192x192` | 2 | `96x96` | `uint16` | **4.50 GiB** | **Pending** | Exact-integer production path and complete physical peak are not retained | — | — |
| [**WebGPU**](platforms/webgpu.md) | MacBook Air (M2, 8 GB) | 8 GB total RAM | `512x512` | Full | `192x192` | 4 | `48x48` | `uint16` | **1.125 GiB** | **Pending** | Exact-integer production path and complete physical peak are not retained | — | — |
| [**WebGPU**](platforms/webgpu.md) | MacBook Air (M2, 8 GB) | 8 GB total RAM | `512x512` | Full | `192x192` | 8 | `24x24` | `uint16` | **0.28125 GiB** | **Pending** | Exact-integer production path and complete physical peak are not retained | — | — |

The exact integer bin-2/4/8 payloads can fit within the nominal 8 GB total-RAM
floor, but payload arithmetic alone is not acceptance. Production WebGPU still
needs integer accumulation/residency plus a physical browser run that captures
browser, adapter, staging, operating-system, pressure, swap, and scientific
parity. Likewise, the current Blackwell timings do not prove a physical 6 GiB
CUDA floor.

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

The measured table above owns retained full-scan timings. The
[filterable coverage registry](performance/coverage.md) owns every missing
combination as one **Platform → Computer → detector-bin → cache-state** row;
this page does not maintain a second bundled list. Real-space crop remains an
explicit selector and is never an implicit performance policy.

#### Selective scan rectangles

The physical WebGPU rows below are prepared **whole-shard-selective** rectangle
loads. A retained frame-span manifest lets the loader omit nonintersecting
shards and decode/upload only selected scan rows. It does not issue byte-range
reads inside an intersecting shard and does not implement arbitrary ordered or
duplicate position selectors.

| Platform | Computer | Selected scan | Rectangle `(row_start,row_stop,column_start,column_stop)` | Source detector | Source dtype | Shards read | Storage bytes read | Samples | Loader p50 | Loader p95 | Loader maximum | Logical resident | Browser-tree RSS peak | Observed swap delta | Parity | Device tested | Date tested |
|---|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| **WebGPU** | MacBook Pro (M5 Max, 128 GB) | `64x64` | `(0,64,0,64)` | `192x192` | `uint16` | 4 of 27 | 488,224,242 B | 5 | **0.147 s** | **0.1544 s** | **0.156 s** | 301,989,888 B | 1,724,317,696 B | 0 B | ✓ | Chrome 151, Apple M5 Max Metal | 2026-08-22 |
| **WebGPU** | MacBook Pro (M5 Max, 128 GB) | `256x256` | `(128,384,128,384)` | `192x192` | `uint16` | 14 of 27 | 1,705,556,941 B | 5 | **0.381 s** | **0.3924 s** | **0.394 s** | 4,831,838,208 B | 3,002,875,904 B | 0 B | ✓ | Chrome 151, Apple M5 Max Metal | 2026-08-22 |
| **WebGPU** | MacBook Pro (M5 Max, 128 GB) | `384x384` | `(64,448,64,448)` | `192x192` | `uint16` | 20 of 27 | 2,432,636,897 B | 5 | **0.574 s** | **0.582 s** | **0.584 s** | 10,871,635,968 B | 3,896,934,400 B | 0 B | ✓ | Chrome 151, Apple M5 Max Metal | 2026-08-22 |

All rows use prepared block indexes and a prepared frame-span manifest. The
operating-system source-page state was uncontrolled/unspecified and no eviction
was performed. Frame-manifest read/encoding, DevTools
injection, checksum harness, products, and application E2E are excluded. A
negative control without the frame-span manifest read all 27 shards (3.17 GB),
so it is explicitly **not** selective evidence. CUDA and Python MPS selector
semantics have source/portable qualification, while native Swift/Metal bulk
selection and physical cross-backend timing remain pending.

The parity check mark means all five runs for that rectangle passed independent
CPU `uint16` checksum probes at the first, middle, and last retained raw frames,
plus output shape, dtype, row-major order, and selection metadata. Full-tensor
readback parity was not performed. The WebGPU fixture view records source
identity `1be810b9...`; fixture C records `c9c0d968...`, and the CUDA master
record is `4802ec16...`. These rows are not a cross-lane fixture-controlled
comparison.

### Screening and prepared-product caches — `quantem.gpu.screening`

| Platform | Computer | Support | Operation | Source plan | Statistic | Time | Device tested | Date tested |
|---|---|---|---|---|---|---:|---|---|
| **CUDA** | Linux CUDA workstation (dual 96 GB Blackwell GPUs) | ✓ | Exact streamed screening | Full `512x512x192x192` `uint16`; no crop/bin; warm source pages unspecified; empty result cache | p50 of 6 | **1.205 s** | NVIDIA RTX PRO 6000 Blackwell Workstation Edition, GPU 0 | 2026-08-22 |
| **Python MPS** | MacBook Pro (M5 Max, 128 GB) | ✓ | Exact screening build | Full `512x512x192x192` `uint16`; no crop/bin; exact fallback pass | Single run | **6.711 s** | Apple M5 Max (`Mac17,6`, 40-core GPU) | 2026-08-19 |
| **Python MPS** | MacBook Pro (M5 Max, 128 GB) | ✓ | Validated screening-v3 reopen | Prepared derived products | p50 | **20.803 ms** | Apple M5 Max (`Mac17,6`, 40-core GPU) | 2026-08-19 |
| **Native Swift/Metal** | MacBook Air (M2, 8 GB) | ✓ | Validated exact-summary reopen | Prepared derived products | p50 | **0.029 s** | Apple M2 MacBook Air (`Mac14,2`, 8 GB) | 2026-08-19 |
| **WebGPU** | N/A (not implemented) | — | Prepared screening workflow | — | — | — | — | — |
| **CPU reference** | Portable CI runner | Ref | Independent adjudication | Reference fixtures | — | **Pending** | — | — |

The promoted MPS build derives its detector mean and masks from the full scan.
When the provisional and final masks differ, BF/DF are recomputed exactly. The
first-chunk-mask candidate failed parity and is retained only as a rejected
experiment; its timing is not repeated on a current page. Saved-product reopen
is likewise never presented as source load.

The current CUDA pinned-slot candidate reduced the like-for-like package p50
from `1.356516 s` to `1.204713 s`; candidate p95/max were
`1.325760/1.329731 s`. All six public arrays were byte exact in every trial.
This is warm/source-pages-unspecified exact screening, not cold HDF5 and not a
prepared-result reopen. Detailed pinned-registration and memory measurements
remain in the provenance ledger.

### Virtual images — `quantem.gpu.detector`

| Platform | Computer | Operation | Scan grid | Detector | Detector bin | Input state | Statistic | Time | Device tested | Date tested |
|---|---|---|---:|---:|---:|---|---|---:|---|---|
| **CUDA** | Linux CUDA workstation (dual 96 GB Blackwell GPUs) | Mean diffraction | `512x512` | `192x192` | 1 | Warm resident | p50 | **18.392 ms** | NVIDIA RTX PRO 6000 Blackwell Max-Q, GPU 1 | 2026-08-19 |
| **CUDA** | Linux CUDA workstation (dual 96 GB Blackwell GPUs) | BF exact sum | `512x512` | `192x192` | 1 | Warm resident | p50 | **3.768 ms** | NVIDIA RTX PRO 6000 Blackwell Max-Q, GPU 1 | 2026-08-19 |
| **CUDA** | Linux CUDA workstation (dual 96 GB Blackwell GPUs) | ADF exact sum | `512x512` | `192x192` | 1 | Warm resident | p50 | **5.586 ms** | NVIDIA RTX PRO 6000 Blackwell Max-Q, GPU 1 | 2026-08-19 |
| **CUDA** | Linux CUDA workstation (dual 96 GB Blackwell GPUs) | DF exact sum | `512x512` | `192x192` | 1 | Warm resident | p50 | **3.747 ms** | NVIDIA RTX PRO 6000 Blackwell Max-Q, GPU 1 | 2026-08-19 |
| **Python MPS** | MacBook Pro (M5 Max, 128 GB) | Mean diffraction | `512x512` | `48x48` | 4 | Warm resident | p50 | **74.805 ms** | Apple M5 Max (`Mac17,6`, 40-core GPU) | 2026-08-19 |
| **Python MPS** | MacBook Pro (M5 Max, 128 GB) | BF exact sum | `512x512` | `48x48` | 4 | Warm resident | p50 | **2.502 ms** | Apple M5 Max (`Mac17,6`, 40-core GPU) | 2026-08-19 |
| **Python MPS** | MacBook Pro (M5 Max, 128 GB) | ADF exact sum | `512x512` | `48x48` | 4 | Warm resident | p50 | **4.404 ms** | Apple M5 Max (`Mac17,6`, 40-core GPU) | 2026-08-19 |
| **Python MPS** | MacBook Pro (M5 Max, 128 GB) | DF exact sum | `512x512` | `48x48` | 4 | Warm resident | p50 | **2.642 ms** | Apple M5 Max (`Mac17,6`, 40-core GPU) | 2026-08-19 |
| **Native Swift/Metal** | MacBook Pro (M5 Max, 128 GB) | Fused BF, ABF, ADF, total, and row/column moments | `512x512` | `192x192` | 1 | Controlled source load | p50 | **119.040 ms** | Apple M5 Max (`Mac17,6`, 40-core GPU) | 2026-08-22 |
| **Native Swift/Metal** | MacBook Air (M2, 8 GB) | BF, ABF, ADF, total, and row/column moments | `512x512` | `48x48` | 4 | Prepared resident-cache fallback | Single run | **103.0 ms** | Apple M2 MacBook Air (`Mac14,2`, 8 GB) | 2026-08-19 |
| **Native Swift/Metal** | MacBook Air (M2, 8 GB) | Exact moment widening for resident summary | `512x512` | `48x48` | 4 | Same validated fused source pass; no resident traversal | Single run | **0.569 ms** | Apple M2 MacBook Air (`Mac14,2`, 8 GB) | 2026-08-20 |
| **WebGPU** | MacBook Pro (M5 Max, 128 GB) | Mean diffraction | `512x512` | `192x192` | 1 | Warm resident | p50 | **50.9 ms** | Chrome 151, Apple M5 Max Metal-3 | 2026-08-19 |
| **WebGPU** | MacBook Pro (M5 Max, 128 GB) | BF exact sum | `512x512` | `192x192` | 1 | Warm resident | p50 | **5.5 ms** | Chrome 151, Apple M5 Max Metal-3 | 2026-08-19 |
| **WebGPU** | MacBook Pro (M5 Max, 128 GB) | ADF exact sum | `512x512` | `192x192` | 1 | Warm resident | p50 | **15.0 ms** | Chrome 151, Apple M5 Max Metal-3 | 2026-08-19 |
| **WebGPU** | MacBook Pro (M5 Max, 128 GB) | DF exact sum | `512x512` | `192x192` | 1 | Warm resident | p50 | **43.4 ms** | Chrome 151, Apple M5 Max Metal-3 | 2026-08-19 |
| **CPU reference** | MacBook Pro (M5 Max, 128 GB) | Virtual-image adjudication | `512x512` | `192x192` | 1 | One independent traversal | Single run | **31.08 s** | Apple M5 Max CPU | 2026-08-19 |

The current integer and mean-DP rows pass their independent CPU reference. CUDA
uses native detector resolution on fixture D; MPS uses explicit detector bin 4
on fixture C. WebGPU uses native detector resolution on D. The current native
bin-1 source pass produces three virtual images, detector sums, and exact
`uint32` total and detector moments in one fused stage. That stage overlaps
source IO and is not additive with the package-wall measurement.
Revision `70bc366` proves when bin-4 accumulators fit, then widens the three
moment maps to `uint64` in one small dispatch. The 0.569 ms row is only that
incremental widening; it is not the full source pass. The cache-only fallback
row remains because it has a different input state. Neither historical bin-4
row is a compressed-source load time.

### Detector moments and phase contrast — `quantem.gpu.dpc`

| Platform | Computer | Operation | Scan grid | Detector | Detector bin | Input state | Statistic | Time | Device tested | Date tested |
|---|---|---|---:|---:|---:|---|---|---:|---|---|
| **CUDA** | Linux CUDA workstation (dual 96 GB Blackwell GPUs) | CoM row and column | `512x512` | `192x192` | 1 | Warm resident | p50 | **13.002 ms** | NVIDIA RTX PRO 6000 Blackwell Max-Q, GPU 1 | 2026-08-19 |
| **CUDA** | Linux CUDA workstation (dual 96 GB Blackwell GPUs) | Fixed-orientation iDPC | `512x512` | `192x192` | 1 | CPU small-field integration | p50 | **21.272 ms** | NVIDIA RTX PRO 6000 Blackwell Max-Q, GPU 1 | 2026-08-19 |
| **Python MPS** | MacBook Pro (M5 Max, 128 GB) | CoM row and column | `512x512` | `48x48` | 4 | Warm resident | p50 | **4.637 ms** | Apple M5 Max (`Mac17,6`, 40-core GPU) | 2026-08-19 |
| **Python MPS** | MacBook Pro (M5 Max, 128 GB) | Fixed-orientation iDPC | `512x512` | `48x48` | 4 | CPU small-field integration | p50 | **12.678 ms** | Apple M5 Max (`Mac17,6`, 40-core GPU) | 2026-08-19 |
| **Native Swift/Metal** | MacBook Air (M2, 8 GB) | CoM, DPC, iDPC, and display statistics | `512x512` | `48x48` | 4 | Prepared exact `uint64` moments | Single run | **11.389 ms** | Apple M2 MacBook Air (`Mac14,2`, 8 GB) | 2026-08-19 |
| **WebGPU** | MacBook Pro (M5 Max, 128 GB) | DPC row | `512x512` | `192x192` | 1 | Warm cached CoM | p50 | **0.9 ms** | Chrome 151, Apple M5 Max Metal-3 | 2026-08-19 |
| **WebGPU** | MacBook Pro (M5 Max, 128 GB) | DPC column | `512x512` | `192x192` | 1 | Warm cached CoM | p50 | **0.7 ms** | Chrome 151, Apple M5 Max Metal-3 | 2026-08-19 |
| **WebGPU** | MacBook Pro (M5 Max, 128 GB) | iDPC | `512x512` | `192x192` | 1 | Explicit 0-degree rotation | p50 | **1.4 ms** | Chrome 151, Apple M5 Max Metal-3 | 2026-08-19 |
| **CPU reference** | MacBook Pro (M5 Max, 128 GB) | Rotation and iDPC adjudication | `512x512` | `192x192` | 1 | CPU reference | Single run | **177.6 ms** | Apple M5 Max CPU | 2026-08-19 |

The current CUDA and MPS CoM rows pass their respective independent references.
The native row is the complete derived-product stage after exact moments exist;
it includes center/mean, alignment, Metal iDPC, and float-surface construction.
It does not include resident-cache traversal or source loading.
The CUDA/MPS cross-fixture timing rows are not compared numerically. The prior
same-fixture detector-bin-4 comparison remains a historical block for iDPC at
`2.84e-5` maximum error. On the retained M5 Max hardware run, WebGPU DPC row
and column were byte exact. Optimized and zero-rotation iDPC had zero frozen
tolerance violations; optimized rotation had maximum absolute error
`1.52587890625e-5` and maximum tolerance ratio `0.8993483035`.

### Single-sideband ptychography — `quantem.gpu.SSB`

These are square scan-grid sizes, not detector dimensions.

Current `512x512` operation timing is separated from the size-support matrix.
Source loading, `G(\mathbf k,\boldsymbol{\nu})` preparation, and UI paint are
excluded.

| Platform | Computer | State | Operation | Detector plan | BF policy | Boundary | Statistic | Time | Device tested | Date tested |
|---|---|---|---|---|---|---|---|---:|---|---|
| **CUDA** | Linux CUDA workstation (dual 96 GB Blackwell GPUs) | Partial | Complex object | Native `192x192` | 8,928 active | Warm resident GPU | p50 | **13.883 ms** | NVIDIA RTX PRO 6000 Blackwell Max-Q, GPU 1 | 2026-08-19 |
| **CUDA** | Linux CUDA workstation (dual 96 GB Blackwell GPUs) | Partial | Exact phase | Native `192x192` | 8,928 active | Warm resident GPU | p50 | **32.035 ms** | NVIDIA RTX PRO 6000 Blackwell Max-Q, GPU 1 | 2026-08-19 |
| **CUDA** | Linux CUDA workstation (dual 96 GB Blackwell GPUs) | Partial | Exact phase and loss | Native `192x192` | 8,928 active | Warm resident GPU | p50 | **32.335 ms** | NVIDIA RTX PRO 6000 Blackwell Max-Q, GPU 1 | 2026-08-19 |
| **Python MPS** | MacBook Pro (M5 Max, 128 GB) | Partial | Exact phase and loss | Explicit detector bin 2 to `96x96` | 2,275 calibrated | Single synchronized reconstruction | Single run | **497.187 ms** | Apple M5 Max (`Mac17,6`, 40-core GPU) | 2026-08-19 |
| **Native Swift/Metal** | MacBook Pro (M5 Max, 128 GB) | Partial | Complex object | Native-detector exact BF columns | 9,074 logical / 2,459 executed | Warm complete Hermitian cache | p50 | **8.911 ms** | Apple M5 Max (`Mac17,6`, 40-core GPU) | 2026-08-19 |
| **Native Swift/Metal** | MacBook Pro (M5 Max, 128 GB) | Partial | Exact phase-variance loss | Native-detector exact BF columns | 9,074 logical / 2,459 executed | Warm complete Hermitian cache | p50 | **25.120 ms** | Apple M5 Max (`Mac17,6`, 40-core GPU) | 2026-08-19 |
| **WebGPU** | MacBook Pro (M5 Max, 128 GB) | Refuted diagnostic | Complex object | Native `192x192` companion | 3,418 active | Readback-complete compute wall | p50 | **32.5 ms** | Chrome 151, Apple M5 Max Metal-3 | 2026-08-19 |
| **WebGPU** | MacBook Pro (M5 Max, 128 GB) | Refuted diagnostic | Exact phase | Native `192x192` companion | 3,418 active | Readback-complete compute wall | p50 | **102.1 ms** | Chrome 151, Apple M5 Max Metal-3 | 2026-08-19 |
| **WebGPU** | MacBook Pro (M5 Max, 128 GB) | Refuted diagnostic | Exact phase and loss | Native `192x192` companion | 3,418 active | Readback-complete compute wall | p50 | **189.4 ms** | Chrome 151, Apple M5 Max Metal-3 | 2026-08-19 |
| **CPU reference** | Portable CI runner | Reference | SSB | Frozen adjudication only | Frozen fixture | Reference | — | **Pending** | — | — |

The WebGPU SSB values are retained diagnostic timings, not accepted scientific
performance. All 262,144 phase values differed from the frozen reference and
the wrapped maximum error was `0.0597773 rad` against a `0.0002 rad` gate.
That implementation remains refuted until phase parity is restored.

The raw detector-bin-2 MPS phase agrees with an independent CUDA reference to
`1.2815e-6` wrapped radians maximum; loss differs by `7.45e-9`. A prepared
BF-column companion from the same campaign is rejected because its stored
columns do not match the declared detector-bin coordinate grid. Its faster
timings are not published as scientific results.

Calibration is a separate operation:

| Platform | Computer | Search | Refinement | Repetitions | Statistic | Time | Result | Device tested | Date tested |
|---|---|---|---|---:|---|---:|---|---|---|
| **CUDA** | Linux CUDA workstation (dual 96 GB Blackwell GPUs) | Seeded Optuna TPE, 200 trials | Nelder–Mead | 3 | p50 | **11.168 s** | Byte-deterministic parameters, phase, object, and loss | NVIDIA RTX PRO 6000 Blackwell Max-Q, GPU 1 | 2026-08-19 |
| **Python MPS** | MacBook Pro (M5 Max, 128 GB) | Optuna TPE, 200 trials | Nelder–Mead | — | — | **Pending** | Current compatible source not profiled | — | — |
| **Native Swift/Metal** | MacBook Pro (M5 Max, 128 GB) | Seeded TPE, 200 trials | Nelder–Mead | 3 | p50 | **6.061 s** | Deterministic parameters and loss | Apple M5 Max (`Mac17,6`, 40-core GPU) | 2026-08-19 |
| **WebGPU** | N/A (not implemented) | — | — | — | — | — | Unsupported | — | — |
| **CPU reference** | Portable CI runner | — | — | — | — | — | Reference only | — | — |

Levenberg–Marquardt is not implemented in any current SSB backend. An earlier
CUDA atomic-objective calibration split into two different fitted minima under
an identical seed; it is retained as a rejected experiment, not a benchmark
row.

| Platform | Computer | Scan grid | Source kind | BF policy | State |
|---|---|---:|---|---|---|
| **CUDA** | Linux CUDA workstation (dual 96 GB Blackwell GPUs) | `128x128` | Fixed-size parity | Frozen fixture | Test |
| **CUDA** | Linux CUDA workstation (dual 96 GB Blackwell GPUs) | `256x256` | Fixed-size parity | Frozen fixture | Test |
| **CUDA** | Linux CUDA workstation (dual 96 GB Blackwell GPUs) | `512x512` | Native real acquisition | Full active BF | Partial physical evidence |
| **CUDA** | Linux CUDA workstation (dual 96 GB Blackwell GPUs) | `1024x1024` | Fixed-size parity | Frozen fixture | Test |
| **Python MPS** | MacBook Pro (M5 Max, 128 GB) | `128x128` | Resized/synthetic | Fixed-size fixture | Test |
| **Python MPS** | MacBook Pro (M5 Max, 128 GB) | `256x256` | Resized/synthetic | Fixed-size fixture | Test |
| **Python MPS** | MacBook Pro (M5 Max, 128 GB) | `512x512` | Native real acquisition | Full active BF | Partial physical evidence |
| **Python MPS** | MacBook Pro (M5 Max, 128 GB) | `1024x1024` | Synthetic | Fixed-size fixture | Test |
| **Native Swift/Metal** | MacBook Pro (M5 Max, 128 GB) | `128x128` | — | — | Not supported |
| **Native Swift/Metal** | MacBook Pro (M5 Max, 128 GB) | `256x256` | — | — | Not supported |
| **Native Swift/Metal** | MacBook Pro (M5 Max, 128 GB) | `512x512` | Native real acquisition | Full active BF | Partial physical evidence |
| **Native Swift/Metal** | MacBook Pro (M5 Max, 128 GB) | `1024x1024` | — | — | Not supported |
| **WebGPU** | MacBook Pro (M5 Max, 128 GB) | `128x128` | Real BF30 parity | Radius 30 px | Partial |
| **WebGPU** | MacBook Pro (M5 Max, 128 GB) | `256x256` | Deterministic fixture | Test BF | Test |
| **WebGPU** | MacBook Pro (M5 Max, 128 GB) | `512x512` | Real interaction | Frozen phase reference | Refuted |
| **WebGPU** | MacBook Pro (M5 Max, 128 GB) | `1024x1024` | Real interaction | Incomplete frozen reference | Partial |
| **CPU reference** | Portable CI runner | `128x128` | — | — | Not retained |
| **CPU reference** | Portable CI runner | `256x256` | — | — | Not retained |
| **CPU reference** | Portable CI runner | `512x512` | Independent adjudication | Frozen fixture | Ref |
| **CPU reference** | Portable CI runner | `1024x1024` | — | — | Not retained |

Native Swift/Metal has a package-owned 512×512 implementation. Other native
Swift scan sizes remain unsupported rather than inferred from CUDA/MPS. Untimed
CUDA and MPS sizes retain fixed-size parity coverage. The WebGPU 512×512 phase
result is explicitly refuted; its timing cannot be promoted until parity is
restored.

See the [SSB performance history](maintainer/ssb-performance.md) for
size-specific historical experiments. Current numerical rows remain in this
dashboard and the verified-results ledger.

### Cross-module platform map

A one-row-per-platform map hides the computer, configuration, and evidence
state, and previously made a refuted WebGPU SSB result look accepted. It is no
longer maintained. Use the [filterable atomic coverage
matrix](performance/coverage.md), where every row begins with **Platform** and
**Computer** and keeps module, bin, dtype, cache state, and next gate separate.

A warm resident kernel, prepared source, first-process application load, and
saved-result reopen answer different questions. See [Benchmark
methodology](performance/methodology.md) and [Verified benchmark
results](performance/results.md) before comparing them.

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
| Add or review a benchmark | [Benchmark methodology](performance/methodology.md) | [Continuous profiling](performance/continuous-profiling.md), [parity](performance/parity.md), and the [optimization ledger](maintainer/backend-optimization-matrix.md) | Date, revision, device, source plan, cache state, memory, wall boundary, and parity artifact |

## Dashboard maintenance rule

Update a dashboard row only after its detailed evidence row is complete. The
detail remains authoritative and must record measurement date, exact source
revision, physical device/runtime, source shape and dtype, cache state,
crop/bin/load plan, benchmark definition, peak memory or swap where available,
and numerical or hash parity. Keep an older result when the newer experiment
changes any of those conditions; label both instead of silently replacing one.
Keep every row atomic: a different bin, dtype path, fixture, cache state, or
statistic is another row. Documentation tests enforce that the landing page
stays timing-free and that current overview values remain owned by this
dashboard.

Accepted and rejected experiments remain in the
[optimization ledger](maintainer/backend-optimization-matrix.md), and the
machine-readable evidence fingerprints are in
[`performance/evidence_manifest.json`](performance/evidence_manifest.json).
The platform/module cadence, runner ownership, and known harness gaps are
machine-checked from `benchmarks/profile_matrix.json`; see
[Continuous profiling](performance/continuous-profiling.md).
