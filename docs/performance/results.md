# Verified benchmark results

This page is the concise provenance index for numerical performance claims in
the public documentation. It does not turn historical diagnostics into current
release promises. Each row names the original measurement revision, cache
state, load plan, benchmark definition, and scientific agreement gate.

“First process” means a process-isolated first source encounter for which the
operating-system storage cache was not forcibly evicted. It is not called cold.
“Prepared” means an index, sidecar, or derived product cache already existed.

## Current platform profile

- **Date tested:** the cross-platform profile was measured 2026-08-19 local
  time; retained follow-up artifacts extend through 2026-08-22.
- **Baseline revisions:** Phil `334b7b5135fe29787540370a00f280fa138430a2`;
  CUDA execution mirror `8c47a466d573f74e425faff611939a17fa6efbf2`.
  Their production compute trees are byte-equivalent for the profiled paths.
- **Clean follow-up stack:** local branch `mps-subsecond-pipeline` at
  `70bc3663c1c7cc495e77348c9fe7594545c66fa8`. It contains exact streamed
  screening (`5d56535`), strict MPS source validation (`1d2e3c9`),
  deterministic CUDA SSB fitting (`fa9ab6f`), native Metal SSB (`e1da9bc`),
  exact prepared QH5 binning (`e0e92b4`), optimized word-major binning
  (`ff3c7fd`), provenance-bound resident summaries (`d65911a`), and exact
  fused-accumulator widening (`70bc366`). These unpublished commits do not
  retroactively change baseline timings.
- **Current native load stack:** clean local branch `metal-cold-300ms` at
  `c0ea44465e6346a8436a0b74f491a04af0b5dc32`. It adds bounded indexed loading,
  exact private Metal residency, controlled macOS source-page measurement, and
  state-consistent benchmark provenance. It does not relabel the 2026-08-19
  cross-platform rows.
- **Current Python MPS lifecycle evidence:** clean local evidence revision
  `3fbd87a53acc1f4ab165b841175681b166bcb629` records accepted historical
  performance for source `b7f8ef3ff2a2d8458944e1e55a3296a39c854357`.
  It extends explicit caller-owned resident-destination recycling to detector
  bins 2/4/8. Consumer-safe exception cleanup is separately sealed at source
  `3c4d903ea62e5e7b19c760efde908c75544b5eba` and evidence
  `08e50c5baab0ad3ff492a48ff1ff4b723a9da876`; its single smoke does not
  replace the historical timing distributions.
- **Fixture C:** independent real `512x512x192x192` native-`uint16` source,
  27 compressed shards plus one master file, 3,169,920,193 total bytes and
  3,169,489,846 indexed compressed bytes, 28-file manifest SHA-256
  `741e7bcf13ffd77bcacfeeabc0b7edb7b427448273ceba2a166426b8f73f509a`.
- **Fixture D:** independent real `512x512x192x192` native-`uint16` source,
  27 compressed shards, 3,165,551,746 bytes, master SHA-256
  `4802ec16ba241fef439e9dcb1c28e94f9cf9d95f773df9c5c8c3b5f7ed8192c4`;
  dataset/v0.1 identity
  `1be810b96fdff8e384ad4cb6ebd49adff9b4ab0a6503cd5fed9106e09f5aa286`.

Every current load uses the complete `512x512` scan, no scan or detector crop,
scan bin 1, and the explicit detector bin shown. The 2026-08-19 source profile
did not forcibly evict the operating-system storage cache and is therefore
**warm**, not cold. The separate 2026-08-22 native load controls source-page
reuse explicitly and is documented in its own section. CUDA and WebGPU use D;
MPS/Swift use C. They are not a fixture-controlled backend ranking.

### Current warm load/decode/bin

The boundary is synchronized first-usable resident output from the public
loader. WebGPU uses the loader's internal library boundary rather than the
outer browser harness. Resident payload and process/card peaks remain distinct.

| Platform | Revision | Detector bin | Output detector | Resident dtype | Repetitions | p50 | p95 | Maximum | Logical resident | Device/driver peak | Process/tree RSS | Peak boundary | Parity | Device tested |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| **CUDA** | `8c47a466` | 1 | `192x192` | `uint16` | 7 | **0.386 s** | **0.396 s** | **0.397 s** | **18.00 GiB** | **21.215 GiB** | Pending | Total-card occupancy | Pass | NVIDIA RTX PRO 6000 Blackwell Max-Q, GPU 1 |
| **CUDA** | `8c47a466` | 2 | `96x96` | `uint32` | 7 | **0.396 s** | **0.401 s** | **0.402 s** | **9.00 GiB** | **11.561 GiB** | Pending | Total-card occupancy | Pass | NVIDIA RTX PRO 6000 Blackwell Max-Q, GPU 1 |
| **CUDA** | `8c47a466` | 4 | `48x48` | `uint32` | 7 | **0.390 s** | **0.413 s** | **0.419 s** | **2.25 GiB** | **3.756 GiB** | Pending | Total-card occupancy | Pass | NVIDIA RTX PRO 6000 Blackwell Max-Q, GPU 1 |
| **CUDA** | `8c47a466` | 8 | `24x24` | `uint32` | 7 | **0.381 s** | **0.401 s** | **0.402 s** | **0.5625 GiB** | **1.805 GiB** | Pending | Total-card occupancy | Pass | NVIDIA RTX PRO 6000 Blackwell Max-Q, GPU 1 |
| **Python MPS** | `0bc9378` | 1 | `192x192` | `uint16` | 7 | **0.414824 s** | **0.457261 s** | **0.457261 s** | **18.00 GiB** | **19,801,456,640 B** | **741,818,368 B** | Driver allocation sampled after load; continuous peak pending | Native bin1 reference plus generic parity suite | Apple M5 Max, 40-core GPU, 128 GB |
| **Python MPS** | `0bc9378` | 2 | `96x96` | `uint16` | 7 | **0.457153 s** | **0.461730 s** | **0.461730 s** | **4.50 GiB** | **6,107,774,976 B** | **616,054,784 B** | Driver allocation sampled after load; continuous peak pending | Exact sparse and six-frame full-scan sums | Apple M5 Max, 40-core GPU, 128 GB |
| **Python MPS** | `0bc9378` | 4 | `48x48` | `uint16` | 7 | **0.382109 s** | **0.384353 s** | **0.384353 s** | **1.125 GiB** | **2,483,896,320 B** | **615,825,408 B** | Driver allocation sampled after load; continuous peak pending | Exact six-frame full-scan sums | Apple M5 Max, 40-core GPU, 128 GB |
| **Python MPS** | `0bc9378` | 8 | `24x24` | `uint16` | 7 | **0.356258 s** | **0.358652 s** | **0.358652 s** | **0.28125 GiB** | **1,577,926,656 B** | **616,054,784 B** | Driver allocation sampled after load; continuous peak pending | Exact six-frame full-scan sums | Apple M5 Max, 40-core GPU, 128 GB |
| **WebGPU** | `334b7b5` | 1 | `192x192` | `uint8` | 5 | **0.824 s** | **0.892 s** | **0.892 s** | **9.00 GiB** | Pending | **5.020 GiB** | Device allocation incomplete; Chrome-tree RSS retained | Exact tested frames and products | Chrome 151, Apple M5 Max Metal-3 |
| **WebGPU** | `334b7b5` | 2 | `96x96` | `float32` sums | 5 | **1.281 s** | **1.300 s** | **1.300 s** | **9.00 GiB** | Pending | **5.363 GiB** | Device allocation incomplete; Chrome-tree RSS retained | Exact sampled count sums | Chrome 151, Apple M5 Max Metal-3 |
| **WebGPU** | `334b7b5` | 4 | `48x48` | `float32` sums | 5 | **1.044 s** | **1.050 s** | **1.050 s** | **2.25 GiB** | Pending | **5.188 GiB** | Device allocation incomplete; Chrome-tree RSS retained | Exact sampled count sums | Chrome 151, Apple M5 Max Metal-3 |
| **WebGPU** | `334b7b5` | 8 | `24x24` | `float32` sums | 5 | **0.979 s** | **0.986 s** | **0.986 s** | **0.5625 GiB** | Pending | **5.184 GiB** | Device allocation incomplete; Chrome-tree RSS retained | Exact sampled count sums | Chrome 151, Apple M5 Max Metal-3 |
| **CPU reference** | `334b7b5` | 1 | `192x192` | `uint16` | 1 | **34.37 s** | — | — | **18.00 GiB** | — | **36.450 GiB** | Host allocation not separated | Independent exact adjudicator | Apple M5 Max CPU |
| **CPU reference** | `334b7b5` | 2 | `96x96` | `uint16` | 1 | **54.22 s** | — | — | **4.50 GiB** | — | **9.634 GiB** | Host allocation not separated | Independent exact adjudicator | Apple M5 Max CPU |
| **CPU reference** | `334b7b5` | 4 | `48x48` | `uint16` | 1 | **43.04 s** | — | — | **1.125 GiB** | — | **2.978 GiB** | Host allocation not separated | Independent exact adjudicator | Apple M5 Max CPU |
| **CPU reference** | `334b7b5` | 8 | `24x24` | `uint16` | 1 | **38.13 s** | — | — | **0.28125 GiB** | — | **2.034 GiB** | Host allocation not separated | Independent exact adjudicator | Apple M5 Max CPU |

Fixture D bin 1 is value-audited lossless `uint8`; bins 2/4/8 are exact detector
sums stored as `float32`. WebGPU's Chrome RSS is not a complete device-memory
measurement and does not prove the physical 8 GB laptop gate. CPU timings are
diagnostic adjudication only, never a silent production fallback.

### Current prepared WebGPU shard-selective rectangles

Evidence revision `54303cb88aab76c29ff884261b5973c8795c2495` measures the
existing rectangle contract on physical Apple Metal WebGPU. The retained
frame-span manifest permits whole nonintersecting shards to be omitted; selected
row windows alone are decoded and uploaded. Reads inside each intersecting
shard remain whole-file reads, so these rows are not intra-shard byte-range I/O.

| Selected scan | Rectangle `(row_start,row_stop,column_start,column_stop)` | Shards read | Storage bytes read | Samples | Loader p50 | Loader p95 | Loader maximum | Logical resident | Browser-tree RSS peak | Observed swap delta |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `64x64` | `(0,64,0,64)` | 4 of 27 | 488,224,242 B | 5 | **0.147 s** | **0.1544 s** | **0.156 s** | 301,989,888 B | 1,724,317,696 B | 0 B |
| `256x256` | `(128,384,128,384)` | 14 of 27 | 1,705,556,941 B | 5 | **0.381 s** | **0.3924 s** | **0.394 s** | 4,831,838,208 B | 3,002,875,904 B | 0 B |
| `384x384` | `(64,448,64,448)` | 20 of 27 | 2,432,636,897 B | 5 | **0.574 s** | **0.582 s** | **0.584 s** | 10,871,635,968 B | 3,896,934,400 B | 0 B |

All 15 physical runs passed independent CPU `uint16` checksum probes at the
first, middle, and last retained raw frames, as well as output shape, dtype,
row-major order, and selection metadata. Full-tensor readback parity was not
performed. Every row uses the native `192x192 uint16`
detector, scan bin 1, detector bin 1, no detector crop, prepared block indexes,
and a prepared frame-span manifest. Operating-system source-page state was
uncontrolled/unspecified and no eviction was performed.
Frame-manifest preparation/injection, the checksum harness, products, and
application E2E are excluded from loader wall time.

Without the frame-span manifest, the `64x64` negative control read all 27 shards
and 3.17 GB; that path is explicitly non-selective. Arbitrary ordered/duplicate
positions and intra-shard range reads remain unsupported or unqualified.
This WebGPU fixture view records source identity `1be810b9...`; fixture C is
`c9c0d968...`, while the CUDA master record is `4802ec16...`. The rows are not
a cross-lane fixture-controlled comparison.

### Current exact Python MPS resident lifecycle

All rows below use fixture C, full `512x512` scan coverage, native
`192x192 uint16` detector data, scan bin 1, no crop, and exact `uint16` output.
The current source-page state is uncontrolled after one same-process warmup.
The timer spans the exact public package load, backend synchronization, and
return of a fresh destination. Each result is explicitly released before the
next trial.

| Detector bin | Output detector | Repetitions | p50 | p95 | Maximum | Logical resident | Driver after load | Driver after release | Process RSS high-water | Whole-system swap delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `192x192` | 7 | **0.414824 s** | **0.457261 s** | **0.457261 s** | **19,327,352,832 B** | **19,801,456,640 B** | **474,103,808 B** | **741,818,368 B** | **0 B** |
| 2 | `96x96` | 7 | **0.457153 s** | **0.461730 s** | **0.461730 s** | **4,831,838,208 B** | **6,107,774,976 B** | **1,275,936,768 B** | **616,054,784 B** | **0 B** |
| 4 | `48x48` | 7 | **0.382109 s** | **0.384353 s** | **0.384353 s** | **1,207,959,552 B** | **2,483,896,320 B** | **1,275,936,768 B** | **615,825,408 B** | **0 B** |
| 8 | `24x24` | 7 | **0.356258 s** | **0.358652 s** | **0.358652 s** | **301,989,888 B** | **1,577,926,656 B** | **1,275,936,768 B** | **616,054,784 B** | **0 B** |

The driver values are instantaneous samples after load and after release, not a
continuously sampled accelerator peak. RSS is a process-lifetime high-water;
swap is whole-system absolute use and did not change during these runs.

Real-data exactness covers sparse bin2 plus six selected frames from each full
bin2/bin4/bin8 result against native bin1 integer sums. Generic DPC rotation,
NumPy CoM, integer-sum, and display-FFT checks also passed. Widget and CuPy
product tests were skipped, so this evidence is not labeled complete
full-volume product or iDPC parity.

#### Historical explicit destination reuse

The older ABBA rows remain accepted for their distinct caller-recycled
destination boundary:

| Detector bin | Output detector | Repetitions | p50 | p95 | Maximum | Source revision |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | `192x192` | 8 | **0.259189 s** | **0.263118 s** | **0.263375 s** | `b7f8ef3` |
| 2 | `96x96` | 8 | **0.359606 s** | **0.361384 s** | **0.361995 s** | `b7f8ef3` |
| 4 | `48x48` | 8 | **0.352990 s** | **0.355048 s** | **0.355062 s** | `b7f8ef3` |

The current `0bc9378` source incorporates the accepted reuse behavior and its
exception-safe cleanup. The historical timings remain a different lifecycle
measurement; none of these rows is cold HDF5 or application E2E.

### Historical Python MPS topology comparison

Revision `f0f39c9` measures fixture C through the public `quantem.gpu.io.load`
boundary after the final MPS command buffer completes. The exact full-detector
path decodes without a full-volume scratch buffer and pipelines three compressed
inputs. The detector-bin paths fuse bit-unshuffle and exact integer sum for bins
2/4/8; bin 2 has a specialized kernel. LZ4 scratch is allocated only when a
chunk actually needs it. Source-shard-sized batches preserve the 10,000-frame
storage layout, with a 1 GiB safety bound for unusually large shards.

The comparable ABBA experiment ran one excluded warm-up and six retained loads
for each candidate in one persistent process. Every output was explicitly freed
after sampling. Source pages were warm and uncontrolled. “Before” is the
non-fused topology under the same revision, process, source-page state, and
lifecycle; “after” enables the accepted exact fused topology. This is a causal
kernel/pipeline comparison, not a cold-source benchmark.

| Detector bin | Before p50 | After p50 | After p95 | After maximum | p50 reduction | Logical resident | Driver sampled after load | Driver after release | Process RSS maximum |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | **0.689148 s** | **0.522754 s** | **0.540934 s** | **0.544737 s** | **24.1%** | **18.00 GiB** | **18.441544 GiB** | **0.441544 GiB** | **0.689148 GiB** |
| 2 | **0.573434 s** | **0.498435 s** | **0.514828 s** | **0.517209 s** | **13.1%** | **4.50 GiB** | **5.688309 GiB** | **1.188309 GiB** | **0.572571 GiB** |
| 4 | **0.461250 s** | **0.421396 s** | **0.423067 s** | **0.423377 s** | **8.6%** | **1.125 GiB** | **2.313309 GiB** | **1.188309 GiB** | **0.571854 GiB** |
| 8 | **0.441671 s** | **0.416601 s** | **0.422252 s** | **0.423076 s** | **5.7%** | **0.28125 GiB** | **1.469559 GiB** | **1.188309 GiB** | **0.573853 GiB** |

Bins 2/4/8 meet the strict p50 at or below 0.5 seconds. Full native bin 1 is
approximately 0.5 seconds but remains 22.754 ms above the strict threshold. The
system had no reported thermal or memory-pressure throttling, but unrelated
desktop and virtualization work remained active. Whole-system swap was already
nonzero and the final ABBA harness did not measure a per-trial swap delta, so no
zero-swap claim is attached to this table.

A second protocol launched seven independent Python processes per bin. Imports
and package Metal-library initialization completed before the timer, then the
first public load was measured. Source pages were again warm and uncontrolled.
It measures per-process decoder and pipeline setup, not cold storage and not
application launch.

| Detector bin | Repetitions | p50 | p95 | Maximum | Logical resident | Driver sampled after load | Process RSS maximum |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7 | **0.648989 s** | **0.662009 s** | **0.663333 s** | **18.00 GiB** | **18.441544 GiB** | **0.687225 GiB** |
| 2 | 7 | **0.700366 s** | **0.752164 s** | **0.768494 s** | **4.50 GiB** | **5.688309 GiB** | **0.573914 GiB** |
| 4 | 7 | **0.642743 s** | **0.654920 s** | **0.654962 s** | **1.125 GiB** | **2.313309 GiB** | **0.571991 GiB** |
| 8 | 7 | **0.632086 s** | **0.633783 s** | **0.634237 s** | **0.28125 GiB** | **1.469559 GiB** | **0.571960 GiB** |

The full native detector payload is exactly 19,327,352,832 bytes: **18.00
GiB**, not process RSS and not total device pressure. Direct buffers are
allocated through Metal/PyObjC, so process RSS alone is incomplete. The driver
values are instantaneous samples after load, not continuously observed peaks.
The allocation remaining after output release is reusable decoder/runtime
state, not resident 4D data.

Fixture C's retained complete value-range audit records a maximum source count
of 53. The worst possible exact sums are therefore 212 at bin2, 848 at bin4,
and 3,392 at bin8, all within `uint16`. An unaudited `uint16` source must use a
provably sufficient wider result dtype or fail closed.

Full-output parity hashed every result byte for bins 1/2/4/8 and compared shape,
dtype, total, maximum, logical resident bytes, and selected provenance against
the pre-optimization path. Every comparison is exact. The parity artifact
SHA-256 is
`fc61130c120c1713614235f5c2b1eb8ea05b84ada4f78c8f300110ae4eed3d0a`.
The independent-process artifact SHA-256 is
`640313f559145126b40e6f6d9467206a666775b0c99ea3d9aff9a33663a149d0`.
The ABBA artifact SHA-256 values for bins 1/2/4/8 are
`0731c34b4c7ce1959141abb8641976e55ba46768290a9b9871250c9cb7af102a`,
`9f29c4790e51bacf99299a98f74ad34e6bf93b48416568e0d1f00be8ff76a916`,
`7f76b5c795452d5caaf04c59dbb0008f1ffc6eb22de1e1a0be5468eeb35b087a`,
and `d13f0414238387cc105daa9ddc432b95bb9d7bc417cda102486283834606dd62`.

One instrumented bin-1 run measured 0.621986 seconds wall, 0.210796 seconds of
cumulative source reads, 0.389971 seconds in the GPU interval union, and 0.092500
seconds of gaps inside the GPU span. Reads and GPU work overlap, so these
intervals must not be added. The remaining work is GPU scheduling and read
jitter, not a scientifically removable detector traversal. Artifact SHA-256:
`e285f610def7a175555da316557a4b10c3cb58462dd87f8c73e8230c5c295bed`.

Revision `be035c4` remains the lifecycle-correct historical reference: its warm
process p50 values were 0.903/0.907/0.764/0.751 seconds for bins 1/2/4/8. The
older 2.273-second bin-1 result is retained only as pressure evidence because
that harness did not free direct Metal outputs between repetitions.

### Current CUDA exact screening

Source `023a6c497b106b216c87205d3fbec63377d77177` and sealed evidence
`5bcc89ebdb77663ea8c035a255a218079ee1ab31` compare a baseline with two
bounded rounded pinned-host registration slots. The full
`512x512x192x192 uint16` source is streamed at scan bin 1, detector bin 1, and
no crop. Source pages were warm or unspecified and the result cache was empty;
this is neither cold HDF5 nor prepared-result reopen.

| Arm | Samples | Package p50 | Package p95 | Package maximum | Pinned registration p50 | Host RSS p50 | GPU allocated peak | GPU reserved peak | Total-card peak | Swap delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 6 | **1.356516 s** | **1.642386 s** | **1.730328 s** | **0.274093 s** | **2,287,351,808 B** | **2,238,152,192 B** | **5,050,266,112 B** | **6,532,890,624 B** | **0 B** |
| Reused pinned slots | 6 | **1.204713 s** | **1.325760 s** | **1.329731 s** | **0.184592 s** | **2,102,042,624 B** | **2,238,152,192 B** | **5,050,266,112 B** | **6,532,890,624 B** | **0 B** |

All six public screening arrays were byte exact in every trial. First usable is
exact complete; no preview is counted. GPU allocated and reserved bytes are
process metrics, while total-card peak includes the coexisting service
baseline and must not be interpreted as process residency.

### Current controlled native exact resident load

Revision `c0ea444` measures fixture C with a fresh process, a new empty QH5
index root, a fresh private Metal destination, and macOS `F_NOCACHE` on the
exact source hash plus every indexed source descriptor. The immutable source
already has a complete identity-bound value-range audit. The state is therefore
**controlled uncached source pages for an audited source**—not an audit-free
arbitrary-source cold encounter, not a warm reopen, and not application end to
end.

The fixture master SHA-256 is
`c9c0d968fae70b8911ae925d676b90007886970fa99fe296a47cfe07844bbfe9`;
the ordered source identity is
`9f0ddb932c631b63cb573c38d747fa41941ee585c5389d33bdafb4add962b768`.
The retained audit digest is
`2107fefc8bff91e5907e76cc84f53270c547479a626a6ea1271a9e8f317c3d41`:
maximum source count 53, zero pixels above 255, and bad-pixel indices 5,319,
15,050, 21,710, and 29,965. The three-band membership digest is
`669882a145976986ebe08795e39198742a5c40a9302e410a12c8d576c94954e4`.
The device ran macOS 26.4 (25E246) with Metal 4.

The exact plan is the full `512x512x192x192 uint16` volume, scan bin 1,
detector bin 1, crop none, audit-bound lossless `uint8` staging, `uint16`
resident output, and a private Metal allocation. The timed package boundary
ends only after the complete 18 GiB resident volume, exact BF/ABF/ADF, detector
sum, total, detector-row moment, detector-column moment, and provenance are
available. A complete resident-volume SHA-256 is computed afterward and is not
included in package wall.

| Device | Revision | Repetitions | Source/index state | Boundary | p50 | p95 | Maximum | Maximum process RSS | Metal allocated after load | Parity | Date tested |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---|---|
| Apple M5 Max (`Mac17,6`, 40-core GPU, 128 GB) | `c0ea444` | 7 fresh processes | Controlled `F_NOCACHE`; new QH5 index root; existing sealed audit | Exact complete private resident plus products | **0.577793 s** | **0.900979 s** | **0.900979 s** | **938.52 MB** | **19.94 GB** | 7/7 exact volume and product hashes | 2026-08-22 |

The first 0.900979-second storage outlier is retained. Stage intervals explain
the package wall but are not all additive: source reads overlap Metal decode,
reduction, and transfer.

| Stage | p50 | p95 | Maximum | Boundary |
|---|---:|---:|---:|---|
| Catalog, source identity, and new index | **0.208247 s** | **0.229903 s** | **0.229903 s** | Package wall component before exact load |
| Exact indexed load | **0.365434 s** | **0.682858 s** | **0.682858 s** | Synchronized complete resident and products |
| Compressed source reads | **0.521686 s** | **0.751283 s** | **0.751283 s** | Accumulated overlapping read intervals |
| Metal active | **0.276298 s** | **0.286221 s** | **0.286221 s** | Synchronized command-buffer intervals |
| Decode and value audit | **0.152240 s** | **0.157220 s** | **0.157220 s** | GPU stage within Metal active time |
| Exact products and detector-bin stage | **0.119040 s** | **0.127788 s** | **0.127788 s** | GPU stage; detector bin 1 preserves native sampling |
| Complete resident-volume SHA-256 | **6.576119 s** | **6.636689 s** | **6.636689 s** | Post-boundary parity check; excluded from package wall |

The logical resident payload is 19,327,352,832 bytes (18 GiB). The loader
estimated 19,663,855,616 allocated Metal bytes excluding mapped source and
20,519,411,712 bytes including source transfer; the observed post-load Metal
allocation was 19,940,737,024 bytes. Peak retained source-buffer bytes were
610,828,288; maximum process RSS was 938,524,672 bytes; Metal allocation fell
to 720,896 bytes after release. On unified memory, RSS is not a substitute for
Metal allocation.

The working-volume SHA-256 was
`1b555fb64b2c54d4f58b750c381d69ed5fe5452361d39cd65e36cf4d5d7358e5`.
All seven exact product hashes repeated in every trial. Package-wall p50 is
0.577793 seconds, so the 0.3-second target was not met. The previous schema-v7
measurement is preserved but superseded for durable reporting because its raw
state said source pages were unspecified while the command had applied
`F_NOCACHE`; the v8 run fixes that label without changing IO or scientific
arithmetic.

### Current native exact resident summary

Revision `d65911a` adds a package-owned exact summary beside a validated native
resident cache. Revision `70bc366` adds an overflow-checked path that widens
the exact `uint32` accumulators already produced by a fused source pass, so an
integrator need not reread the resident detector volume. The measured plan is
the complete `512x512` scan from fixture C,
no scan or detector crop, scan bin 1, exact detector sum bin 4 from `192x192` to
`48x48`, native `uint16` source, and `uint16` resident output. The source
identity is
`9f0ddb932c631b63cb573c38d747fa41941ee585c5389d33bdafb4add962b768`;
the resident payload SHA-256 is
`2a876d00ca1512955006a40433341b26aee766dec077ddced8368011f4ec52b3`.

The summary stores exact BF, ABF, and ADF `uint32` maps; total, detector-row,
and detector-column `uint64` moments; and one selected `uint32` diffraction
pattern. Read validates source identity, resident payload, geometry, dtype,
scan region/bin, detector bin, count audit, detector-band definition, selected
scan coordinate, artifact sizes, and every artifact SHA-256 before returning a
product. It never represents a prepared summary as source loading.

| Device | Revision | Repetitions | Cache state | First complete product p50 | p95 | Maximum | Process wall p50 | p95 | Maximum | Maximum RSS | Parity |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| Apple M5 Max, 40-core GPU, 128 GB | `d65911a` | 7 fresh processes | Prepared exact summary | **0.026 s** | **0.027 s** | **0.027 s** | **0.120 s** | **0.144 s** | **0.150 s** | **97.4 MB** | Nine same-device products byte exact |
| Apple M2 MacBook Air (`Mac14,2`, 8 GB) | `d65911a` | 7 fresh processes | Prepared exact summary | **0.029 s** | **0.030 s** | **0.030 s** | **0.110 s** | **0.124 s** | **0.130 s** | **92.0 MB** | Nine same-device products byte exact |

The original `d65911a` creation path remains a valid fallback when only a
prepared resident cache exists. It traverses the validated 1.208 GB resident
cache, but it does not open or decompress the 3.17 GB compressed-HDF5 source.
These historical fallback measurements remain here because their input state
differs from source-fused creation.

| Device | Revision | Repetitions | Cache state | Resident load wall | Metal product/moment kernel | Summary write | Process wall | Process swaps |
|---|---|---:|---|---:|---:|---:|---:|---:|
| Apple M5 Max, 40-core GPU, 128 GB | `d65911a` | 1 | Prepared resident cache, summary absent | **0.369 s** | **8.0 ms** | **8.0 ms** | **0.86 s** | **0** |
| Apple M2 MacBook Air (`Mac14,2`, 8 GB) | `d65911a` | 1 | Prepared resident cache, summary absent | **1.397 s** | **103.0 ms** | **18.0 ms** | **2.61 s** | **0** |

On the Air, the resident-cache pass took 1.313 s while the fused exact Metal
product/moment kernel took 103 ms. Mapped-page population, not reduction, is
therefore the dominant creation cost. Once exact moments exist, one
instrumented derived-product stage took **11.389 ms**: 0.565 ms center/mean on
CPU, 1.107 ms rotation/alignment on CPU, 2.095 ms Metal iDPC, and 7.622 ms
float-surface statistics. The legacy `gpu=0.000` aggregate excludes those small
command buffers and is rejected as GPU telemetry for this path.

Revision `70bc366` removes that additional resident traversal when the same
process has just completed the validated fused source pass. The consumer
rechecks a conservative accumulator bound, widens the three exact moment maps
in one small Metal dispatch, and writes the unchanged
`quantem.gpu.resident-summary/v1` schema. The table reports only the incremental
summary work; it is not a compressed-source load time.

| Device | QuantEM.GPU revision | Consumer overlay | Repetitions | Input state | Additional resident traversal | Widen wall | Widen GPU | Summary write | Process swaps | Parity | Date tested |
|---|---|---|---:|---|---|---:|---:|---:|---:|---|---|
| Apple M5 Max, 40-core GPU, 128 GB | `70bc366` | `105942d3` tracked-diff SHA-256 | 1 | Same validated fused source pass | None | **0.703 ms** | **0.328 ms** | **11 ms** | **0** | Seven summary artifacts byte exact; nine products same-device byte exact | 2026-08-20 |
| Apple M2 MacBook Air (`Mac14,2`, 8 GB) | `70bc366` | `105942d3` tracked-diff SHA-256 | 1 | Same validated fused source pass; low page residency and pre-existing swap | None | **0.569 ms** | **0.104 ms** | **18 ms** | **0** | Seven cross-device integer artifacts byte exact; all floating products below `1e-5` | 2026-08-20 |

The Air's surrounding source encounter took 6.444 s in a low-residency state;
three sequential rechecks improved from 5.534 s to 2.723 s to 2.101 s while
Metal stayed between 1.450 s and 1.542 s. Those values are a cache-state
sequence, not independent repetitions, so none is promoted into the current
warm-load table. The follow-up changed only post-pass summary materialization;
it did not make the compressed source sub-second.

All seven summary artifact hashes match across Phil and the Air. Every reopen
also reproduced same-device BF, ABF, ADF, CoM row/column, DPC row/column, iDPC,
and selected diffraction byte-for-byte against the full resident calculation.
This is prepared-product evidence, not the original compressed-source first
encounter and not a headed application-paint measurement.

### Current streamed screening

Screening is a separate build/reopen boundary, not source loading. The accepted
MPS follow-up accumulates the complete detector sum in `uint64`, validates the
provisional mask against the final full-scan mask, and reruns BF/DF only when
those masks differ.

| Platform | Revision | Operation | Source plan | Statistic | Time | Memory state | Numerical state | Device tested |
|---|---|---|---|---|---:|---|---|---|
| **Python MPS** | `5d56535` | Exact screening build | Full `512x512x192x192` `uint16`; no crop/bin; 170 rows, four chunks; exact fallback pass | Single run | **6.711 s** | Streamed source; derived products retained | Mean DP and CoM byte-exact; BF/DF value-exact | Apple M5 Max, 40-core GPU, 128 GB |
| **Python MPS** | `5d56535` | Validated screening-v3 reopen | Prepared derived products from the same source identity | p50 | **20.803 ms** | Saved-result reopen | Cache identity and products validated | Apple M5 Max, 40-core GPU, 128 GB |

The rejected one-pass candidate derived its mask from only the first chunk and
failed full-scan mean-DP/BF/DF parity. Its timing is intentionally absent from
the current ledger and remains discoverable only as a rejected experiment in
the [optimization ledger](../maintainer/backend-optimization-matrix.md).

### Current resident products

All rows exclude source loading. CUDA uses D at detector bin 1; MPS uses C at
detector bin 4; WebGPU uses D at detector bin 1. CUDA and MPS rotation/iDPC are
small-field CPU operations after GPU CoM, not GPU-kernel claims.

| Platform | Mean DP | BF | ADF | DF | CoM row/column | DPC row/column | iDPC | State and parity |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **CUDA** | **18.392 ms** | **3.768 ms** | **5.586 ms** | **3.747 ms** | **13.002 ms** | — | **21.272 ms** | p50 of 7; integer/mean exact; CoM and iDPC pass current CPU gates |
| **Python MPS** | **74.805 ms** | **2.502 ms** | **4.404 ms** | **2.642 ms** | **4.637 ms** | — | **12.678 ms** | p50 of 21; integer/mean/CoM exact; same-runtime iDPC exact |
| **WebGPU** | **50.9 ms** | **5.5 ms** | **15.0 ms** | **43.4 ms** | **82.9 ms** | **0.9/0.7 ms** | **1.4 ms** | p50 of 5; integer/mean exact; per-pixel float errors not retained |
| **CPU reference** | — | — | — | — | — | — | **177.6 ms** | Independent product traversal was 31.08 s; reference only |

### Current SSB reconstruction and calibration

CUDA calibration uses calibration SHA-256
`4a2d9cc36943973dbe0f1d5e40858160f0a6393cd56d9f54e073a358b3eff8e8`,
200 kV, 21.4 mrad semiangle, 0.49492961 Å scan sampling, full automatically
detected BF disk, float32/complex64, seeded Optuna TPE 200 trials, and
Nelder–Mead refinement. The MPS row uses C, explicit detector bin 2, 2,275
calibrated BF positions, and calibration SHA-256
`8815ddd710f33973ac11d504cd679f16d9f5d6bf3043d0480e682ecc0a053941`.
WebGPU uses a separately frozen native-detector exact `uint8` companion, 3,418
active aperture positions, and compute-matched revision `5cd285250911974c738e9c911bd00a170873bf45`.
Native Swift/Metal uses a separate frozen 512×512 exact-`uint8` full-BF
fixture at `e1da9bc86a0c1ae6edc60e1205a9966e6826f315`: 9,074 logical BF
planes, 2,459 executed aperture planes, no scan crop or bin, detector bin 1,
and float32/complex64 compute. Its source SHA-256 is
`6046f7855b6925aafc86a52cc9ef06156ebf617d63b25c5a2a10fd94762ae3ae`.
The operating-system page cache was warm.

| Platform | Operation | Statistic | Time | Numerical state | Device tested |
|---|---|---|---:|---|---|
| **CUDA** | Complex object | p50 of 7 | **13.883 ms** | Deterministic | NVIDIA RTX PRO 6000 Blackwell Max-Q, GPU 1 |
| **CUDA** | Exact phase | p50 of 7 | **32.035 ms** | Deterministic | NVIDIA RTX PRO 6000 Blackwell Max-Q, GPU 1 |
| **CUDA** | Exact phase and loss | p50 of 7 | **32.335 ms** | Deterministic | NVIDIA RTX PRO 6000 Blackwell Max-Q, GPU 1 |
| **CUDA** | 200-trial TPE plus Nelder–Mead | p50/p95/max of 3 | **11.168/11.235/11.242 s** | Byte-identical fitted parameters, phase, object, and loss | NVIDIA RTX PRO 6000 Blackwell Max-Q, GPU 1 |
| **Python MPS** | Exact phase and loss | One synchronized run | **497.187 ms** | CUDA phase max `1.2815e-6` rad; loss error `7.45e-9` | Apple M5 Max, 40-core GPU |
| **Native Swift/Metal** | Complex object, complete Hermitian cache | p50 of 7 | **8.911 ms** | CUDA phase relative L2 `5.8695e-5`; maximum `5.6288e-6` rad | Apple M5 Max, 40-core GPU |
| **Native Swift/Metal** | Exact phase-variance loss, complete Hermitian cache | p50 of 7 | **25.120 ms** | Cached-versus-streamed loss relative error below `5e-5` | Apple M5 Max, 40-core GPU |
| **Native Swift/Metal** | 200-trial TPE plus Nelder–Mead | p50 of 3 | **6.061 s** | Identical fitted parameters and loss, seed 42 | Apple M5 Max, 40-core GPU |
| **WebGPU** | Complex object | Readback wall p50 of 5 | **32.5 ms** | Deterministic hash 5/5 | Chrome 151, Apple M5 Max Metal-3 |
| **WebGPU** | Exact phase | Readback wall p50 of 5 | **102.1 ms** | Deterministic hash; one retained scheduling outlier | Chrome 151, Apple M5 Max Metal-3 |
| **WebGPU** | Exact phase and loss | Readback wall p50 of 5 | **189.4 ms** | Phase byte-identical to no-loss 5/5 | Chrome 151, Apple M5 Max Metal-3 |

An earlier CUDA atomic-objective fit split into two fitted minima under the
same seed and therefore failed the frozen repeatability gate. It is not a
current benchmark row. Native Swift calibration also repeats deterministically
but uses a different real fixture and optimizer implementation, so its timing
is not a direct CUDA ranking. The prepared MPS companion candidate is rejected
because its stored columns do not match its declared detector-bin coordinate
grid. WebGPU implements reconstruction but not TPE or Nelder–Mead calibration.
Levenberg–Marquardt is not implemented in any current SSB backend.

### Current native Swift/Metal boundary

At `d65911a`, the release suite executed 76 tests, skipped five opt-in
real-QH5/performance cases, and had no failures. A separate real-QH5 run
executed 71 tests with one performance skip and no failures. Four real QH5
frames matched the Swift CPU reference exactly for decode, detector bin 4,
BF/ABF/DF/total, and row/column moments. The physical full-scan summary evidence
above adds exact BF/ABF/ADF and overflow-safe total/row/column moments plus
same-device derived-product parity. Package-level numerical DPC/iDPC unit tests
remain pending; the derived-product timing came from an isolated headless
consumer harness, not a GUI or application-paint boundary.

At `70bc366`, the expanded release suite executed 81 tests, skipped five
opt-in/data cases, and had no failures. Its new focused checks cover the
audited low-count plan, scan-bin and incomplete-edge contribution bounds,
high-dynamic-range rejection, arithmetic overflow, and exact Metal widening.
The real-QH5 gate again passed four of four checks.

At `c0ea444`, the release suite executed 111 tests, skipped eight explicit
opt-in real-data/performance cases, and had no failures. Strict recursive Swift
formatting and the release indexed-load benchmark build also passed. The added
tests prove that controlled source-page reporting cannot be enabled through the
private environment variable alone, that default and controlled small-fixture
paths are byte-identical, and that a private-resident plan can create and reopen
the same exact transactional cache. This remains package evidence, not a native
application-paint or physical 8 GB/24 GB admission gate.

The earlier `334b7b5` profile remains the owner of prepared-index reopen
**5.339/5.760/5.842 ms** p50/p95/max and 512×512 FFT **15.622 ms** first,
**0.291/0.584 ms** warm p50/p95. Those measurements were not rerun or silently
reassigned to `d65911a`.

The later native SSB follow-up `e1da9bc` adds `MetalSSBKernels`, complete-cache
and bounded-memory exact policies, and a standalone release benchmark. Its
accepted reconstruction, loss, and fit values appear once in the SSB table
above. The [native Metal SSB migration record](../maintainer/native-metal-ssb-migration.md)
owns API lineage, cache-policy details, and artifact fingerprints. Neither page
represents warm library compute as application wall time or physical 8 GB
device signoff.

### Current evidence fingerprints

| Evidence bundle | SHA-256 |
|---|---|
| Apple atomic matrix | `358da791e27433a7f2cd5bab3e7880d1907a8f07976bd9fa529882685de0c84e` |
| CUDA atomic rows | `e68c6b56a95b6844a9eb354b1c68ca68e2f2f815395dc74ee6b4a3e6f7945272` |
| WebGPU/CPU report | `a2f75c7a595d5fc457b5c32afd0ea8aac5dd20f5db30c496b386e30acedce7d1` |
| Exact MPS screening adjudication | `1094cc68e2bf9952916fb12ac6489119a4f4ee4be2ece7f8b1c1a4f1ed411fa3` |
| Raw MPS/CUDA SSB parity | `17a7ef5750444377c7d16c18bfadef39607ea3d684c91dad93681ca887d7154e` |
| Current MPS fresh-process load/memory audit | `a332552a98d8bc718b30f083a6e4afd4c6e7d56f10cbb1e09639cfab0042bc0d` |
| Current MPS warm-process explicit-release audit | `07affa9decf0a8c99ac10b32db961c2c2681f5baf5f91a151ea121d93811bb50` |
| MPS bin-2/bin-4 resident-lifecycle timing manifest | `e2b57949709c2bf70d0b987b3ca9a83950c8d80d92eaef1cfce850ab010cc7f9` |
| MPS binned exception-cleanup manifest | `4cac5fa7f9cdb647f7e943e46bff27e6244459702b35680d2dde39ba0027a4ac` |
| Current MPS exact sparse detector-bin/product parity | `96a986274dd2ba2b8f175d5003eabffc3c52e94d069f1c4e899d669e710832be` |
| CUDA pinned-slot screening manifest | `2103a425526ebd2caf15d053867a6d8526481f9e833e19047bcaaf0e49c304de` |
| WebGPU prepared shard-selective rectangle manifest | `98e1b24a4f80885213477aaaccc02cc60770284b3287a27cac6b66977ae26b59` |
| Deterministic CUDA full fit | `d262c1ed8fa55728811735bc974ef4fcc413e60aff83dfcd7396b4ad681f4527` |
| Exact Air resident summary | `4f8f366553cf8ae13b5b732a24a070f6cc404127ef6e421599d00b5c27a3688c` |
| Source-fused Air summary follow-up | `123b77e3424994980379a942da21dfbdf0d0921b2de0a862652831c4bfe814a9` |
| Controlled full-native exact resident load, schema v8 | `2480cfd2ea78f24637cc542edceb1c4f39f7e4ba324d15969acdc3115a9dfcee` |

## Historical and rejected results

Current tables stop above. Older campaigns are retained once in their
domain-specific maintainer records instead of being copied into this page.

| Record | Status | Canonical owner |
|---|---|---|
| Earlier three-host full-scan campaign | Superseded for current headline timing; still useful same-fixture history | [Load acceptance evidence](../maintainer/backend-4dstem-load-checklist.md) |
| July native CUDA/MPS IO and compressed-save work | Historical diagnostic; some host/storage fields were not retained | [Optimization ledger](../maintainer/backend-optimization-matrix.md) |
| WebGPU local-file, selected-block, and display campaigns | Historical implementation evidence, not current full-stack or physical-8-GB signoff | [Optimization ledger](../maintainer/backend-optimization-matrix.md) and [WebGPU history](../maintainer/history/index.md) |
| WebGPU full-native batch-2 load candidate | Refuted for performance: exact output but prepared-index/source-pages-unspecified p50/p95/max `1.128484/1.140200/1.141501 s` missed the strict subsecond gate and did not replace the accepted `1.080860 s` single observation | Sealed evidence `75eec9cccd5c1a9814068a83a23e67253227a363` |
| SSB size sweeps and rejected kernel layouts | Historical or rejected unless promoted into the current table above | [SSB performance history](../maintainer/ssb-performance.md) |
| Physical M2 Air application loading | Separate application-level evidence; never substituted for a library benchmark | [M2 Air Metal evidence](../maintainer/m2-air-lz4-match-unroll-2026-08-18.md) |
| Rejected first-chunk screening and nondeterministic CUDA calibration | Failed scientific gates; retained to prevent repetition, never shown as current speed | [Revision and change ledger](changes.md) and [optimization ledger](../maintainer/backend-optimization-matrix.md) |

A historical number may be quoted only with its original revision, fixture,
cache state, scientific plan, hardware, and acceptance status. The
[revision and change ledger](changes.md) records why a current row replaced or
remained separate from an older campaign.
