# QuantEM.GPU cross-backend optimization report

Date: 2026-08-22
Campaign authority: Phil source worktrees; MJGOAT CUDA/data backend; Rodman
physical 24 GB observation gate
Scientific source: `512 x 512 x 192 x 192 uint16`, full scan, no crop
Policy: detector bins are explicit exact integer sums only; no downcast,
sampling, preview, reduced products, or tolerance relaxation

## Timing vocabulary

The following timing boundaries are not interchangeable. A row is reported
only under the boundary that was actually measured.

| ID | Boundary | Included work |
|---|---|---|
| A | Cold original source | New process and original compressed source with an explicitly controlled source-page state, including discovery, read, decode, and required products |
| B | First usable product | Wall time until the first complete scientifically valid requested product can be published |
| C | Exact complete resident state | Wall time until the declared working volume and complete requested product set are exact and resident under the stated plan |
| D | Warm source reopen | Original source is reopened with source pages allowed to remain warm; no prepared-result shortcut |
| E | Prepared/cache reopen | A validated prepared index, resident destination, or saved result is reused exactly as declared |

No accepted row in this campaign establishes boundary A or Live4DSTEM
wall-to-wall time. The Apple package rows cover prepared-index work with
uncontrolled source pages; the fastest CUDA row is boundary E and does not
decode the original source.

## Outcome

The campaign produced separate reviewable wins in native Metal, Python MPS,
hardware WebGPU, and CUDA. It did **not** prove a subsecond cold original-HDF5
load or Live4DSTEM application E2E on every device. Prepared-index warm loads,
resident-destination reuse, and saved-result reopen remain distinct timing
definitions throughout this report.

| Module / host | Accepted source | Evidence | Before -> after p50 | Exactness / memory |
|---|---|---|---:|---|
| Full-native private Metal / Phil | `531e5001` | `afd5e06b` | shared `0.530581 s` -> private `0.313870 s` | complete 18 GiB plus seven products exact; 19.941 GB Metal; 0.685 GB RSS; swap +0 |
| Full-native resident lifecycle / Phil MPS | `e8293a1f` | `b7c22861` | fresh destination `0.425533 s` -> recycled destination `0.259189 s` | complete 18 GiB and selected hashes exact; 19.801 GB Metal; ~0.738 GB RSS; swap +0 |
| Detector bin 2 resident lifecycle / Phil MPS | `b7f8ef3f` | `3fbd87a5` | fresh destination `0.462541 s` -> recycled destination `0.359606 s` | ABBA selected-frame hashes exact; separate complete-volume smoke exact; 6.108 GB Metal; ~0.613 GB RSS; swap +0; cleanup closed at `3c4d903e` / `08e50c5b` |
| Detector bin 4 resident lifecycle / Phil MPS | `b7f8ef3f` | `3fbd87a5` | fresh destination `0.384264 s` -> recycled destination `0.352990 s` | ABBA selected-frame hashes exact; separate complete-volume smoke exact; 2.484 GB Metal; ~0.613 GB RSS; swap +0; cleanup closed at `3c4d903e` / `08e50c5b` |
| Detector bin 2 / Phil MPS | `341a036` | `b8188bef` | `0.509299 s` -> `0.498708 s` | complete 4,831,838,208-byte output exact |
| Detector bin 4 / Phil MPS | `0a0d60ee` | `6ed319b8` | `0.421165 s` -> `0.383772 s` | complete 1,207,959,552-byte output exact |
| Detector bin 8 / Phil MPS | `38bf65c` | `de4cf1ab` | `0.418280 s` -> `0.356969 s` | complete 301,989,888-byte output exact |
| Full-native CoM/DPC/iDPC / Phil WebGPU | `64b6eec6` | `08c071fe` | prior failing path -> accepted `1.080860 s` prepared-index load | DPC byte exact; frozen iDPC contract passes; 6.835 GB Chrome-tree RSS; swap +0 |
| Rectangle-selective IO / Phil WebGPU | `23d25619` | `54303cb8` | full-source fallback `3.17 GB` -> 4/14/20-shard prepared reads for 64/256/384-square regions | 15/15 runs passed three raw-frame probes plus shape/dtype/order/metadata; full-tensor readback not performed; loader p50 0.147/0.381/0.574 s; intra-shard range IO pending |
| Full-native exact screening / MJGOAT CUDA | `5ee2016e` | `47bb6e42` | generic plan `1.317555 s` -> contiguous plan `1.208146 s` | six arrays byte exact; 5.050 GB process reserve; 6.533 GB total card including service baseline |
| Pinned staging lifecycle / MJGOAT CUDA | `023a6c49` | `5bcc89eb` | three registrations `1.356516 s` -> two registrations `1.204713 s` | six arrays byte exact; RSS 2.287 -> 2.102 GB; GPU reserve and total-card peak unchanged; swap 0 |
| Saved screening-result reopen / MJGOAT CUDA | `8258ac2c` | `91ef7c7e` | `0.092367 s` -> `0.005599 s` | six arrays byte exact; 1.028 -> 0.939 GB RSS; swap 0; cross-alias changes fail closed |

All Apple load rows above use prepared indexes. Their OS source-page state was
warm or uncontrolled/unspecified; no row is promoted as a forced-eviction cold
load. The 0.259-second MPS row additionally reuses a caller-owned resident
destination. The 5.599-millisecond CUDA row reopens a 5.458 MB prepared result
cache and performs no new raw-source decode. None is a cold arbitrary-source or
application-E2E claim.

The lane manifests retain distinct fixture identities: Phil fixture C, the
WebGPU fixture view/source-identity records, and the MJGOAT CUDA master are not
silently treated as one hash-equivalent input. Cross-lane timing comparisons
are not fixture-controlled unless their declared input identifiers and hashes
match.

## Scientific and resource gates

- Exact detector sums now fail before publication when the requested output
  dtype cannot represent the range (`8045d4e` / evidence `72df01c`).
- Partial final bitshuffle blocks are covered across native, mask, detector
  bins 2/4/8, row-prefix, selective order/duplicates, and sharded offsets
  (`c10117c` / evidence `83a61411`).
- CUDA automatic 6 GiB planning uses 64 rows for the retained source and
  measured 2.238 GB allocated, 5.050 GB process reserve, and 6.533 GB total-card
  peak including a 1.472 GB pre-existing service baseline.
- Selective-loading source semantics are audited at `58577c81`: CUDA and MPS
  support true rectangular/arbitrary-order/duplicate compressed-range reads;
  Swift lacks bulk selective loading and WebGPU remains rectangle-only. New
  WebGPU hardware evidence at `54303cb8` proves prepared shard omission and
  selected-window decode/upload, but not intra-shard byte-range reads or
  arbitrary ordered positions; the no-frame-span fallback reads all shards.
- The deliberately dirty WebGPU exact-integer candidate was rechecked without
  changing its preserved patch. Physical Metal-3 execution proved complete
  small-case `uint8 -> uint16` and `uint16 -> uint32` bin-2 sums, fail-closed
  `uint32` planning, and all 4,800,000 outputs for a 300,000-scan tiled
  dispatch. The final raw retained rerun was 7.5 ms plus 14.1 ms for readback
  and complete CPU validation; its tiny-repeat p50/p95/max was
  0.3/21.2/21.2 ms, retaining the outlier. It is synthetic kernel-only
  evidence—not HDF5 loading, a full real-volume/product test, a memory gate, or
  a clean handoff. The earlier 7.8 ms probe result is summary-only because its
  overwritten harness could not be retained.
- Rodman full-native bin 1 passed once at 3.827780 seconds with exact products,
  but swap increased by 1,243.19 MiB. Repetition stopped. Later Rodman windows
  remained contaminated by Screen Sharing/WindowServer activity and existing
  swap, so no clean distribution was invented.

## Refuted or blocked work retained

- Two Swift queue/preparation concurrency designs were exact but neutral or
  slower; the two-stage queue candidate regressed package p50 from 0.529075 to
  0.565636 seconds.
- CUDA 56-row batching regressed against the accepted 64-row plan and increased
  allocator reserve. The production policy remains 64 rows.
- A canonical-path CUDA cache shortcut appeared exact on one symlink pair, but
  review proved that two aliases to one master inode can resolve relative HDF5
  links to different shards. Revision `a23ae3b` and its 6.209 ms alias smoke are
  refuted. The accepted `8258ac2c` fast path requires the exact normalized
  master spelling and fully reinspects any alias change.
- Hardware-WebGPU SSB reached 17.9 ms warm phase-plus-loss but failed phase
  parity by 0.0597773 rad against the frozen 0.0002 rad gate. It is diagnostic
  timing only.
- A full-native WebGPU batch-2 overlap candidate reduced its paired internal
  p50 from 1.0735 to 0.9910 seconds and Chrome-tree RSS from 6.865 to 6.118 GB,
  but package p50 was 1.128484 seconds and remained slower than the retained
  accepted 1.080860-second path. Exact parity passed; no source change was
  promoted (`75eec9cc`).
- The binned MPS recycle candidate passed exact bin-8 science and a 0.349807
  second smoke, but its ABBA sequence drifted chronologically while Screen
  Sharing and WindowServer were active. The prior accepted 0.356969-second
  bin-8 distribution remains authoritative; the contaminated sequence is
  retained but not promoted.
- Independent review found that a binned MPS exception after taking a recycled
  destination could strand that large allocation. Source `3c4d903e` and
  evidence `08e50c5b` close the blocker: injected pre-allocation and
  second-shard failures preserve earlier caller-owned output, release only the
  failed invocation's destination exactly once, and pass a subsequent exact
  load. Its single timing smoke is non-regression evidence, not a new speed
  distribution.
- Corrected MPS/CUDA SSB one-shot parity passed, but the frozen 200-trial plus
  Nelder-Mead engine blob is absent from all retained Git/source snapshots.
  The long calibration comparison remains blocked rather than silently
  substituting another engine.
- No physical Steve Kerr run or Live4DSTEM headed/app E2E was performed by this
  package campaign. Device policy, UI disclosure, cache lifecycle, packaging,
  and headed acceptance remain Live4DSTEM-owned.

## Verification summary

These are lane-local checks against each listed evidence revision; counts are
not added across branches because their source revisions differ.

| Lane | Focused / package checks | Broader checks |
|---|---|---|
| Native private Metal | private readback 1/1; strict Swift format and release benchmark build pass | 111 tests passed; 8 explicit opt-in skips |
| MPS bin-1 lifecycle | 108 passed; 2 skipped; critical Ruff and registry pass | 528 passed; 87 explicit skips |
| MPS binned lifecycle | historical performance checks 115 passed / 2 explicit skips; cleanup follow-up 114 passed / 2 explicit skips | exact real-volume hashes/totals/maxima for bins 2/4/8; injected ownership failures pass |
| WebGPU CoM/DPC/iDPC | 31 focused checks passed; hardware adapter and exact packed-u16 smoke pass | 513 passed; 87 explicit skips; full-native hardware parity pass |
| WebGPU exact-integer candidate | 10 static source checks and browser bundle pass | physical Metal-3 small outputs and complete 19.2 MB tiled output byte exact; candidate remains deliberately dirty |
| CUDA exact screening | Phil 84 passed / 23 skipped; MJGOAT 107 passed | MJGOAT 536 passed / 59 skipped; Ruff and diff pass |
| CUDA pinned staging | Phil 86 passed / 1 skipped; MJGOAT 87 passed | all 12 ABBA product sets byte exact |
| CUDA result reopen | Phil 91 passed / 1 skipped; MJGOAT 92 passed | Phil 506/88 and MJGOAT 544/59 passed/skipped; Ruff pass |
| Campaign registry | frozen coordinator source 35 cells / 10 experiments; WebGPU/docs lane 35/11; MPS cleanup lane 35/19 | parity and documentation checks 36/36; counts remain revision-specific |

## Clean evidence boundaries

Each accepted lane remains an isolated local branch; nothing was pushed,
merged, released, or published.

| Branch | Clean evidence HEAD | Purpose |
|---|---|---|
| `metal-private-output-bin1-20260822` | `afd5e06b2afa2c9c30e69ff5dc8d7825c28068f1` | private full-native Metal residency |
| `mps-bin1-resident-lifecycle-20260822` | `b7c22861ab3d987ca4e7076c791c195c735298f7` | explicit MPS destination recycle contract |
| `mps-binned-resident-lifecycle-20260822` | `08e50c5baab0ad3ff492a48ff1ff4b723a9da876` | detector-bin-2/4 performance evidence plus accepted cleanup source `3c4d903e`; bin-8 timing remains inconclusive |
| `webgpu-com-idpc-fullnative-phil-20260822` | `08c071fe96ab5afc725da6725c2afe72b2945e4b` | full-native WebGPU CoM/DPC/iDPC parity |
| `webgpu-rectangle-selective-io-20260822` | `54303cb88aab76c29ff884261b5973c8795c2495` | prepared shard-selective rectangle evidence with three raw-frame probes; full-tensor readback and intra-shard range pending |
| `cuda-screening-20260822` | `47bb6e4276afde0387475b063a9744aba7f0b421` | exact one-pass CUDA screening/read-plan/index evidence |
| `cuda-screening-pinned-slot-audit-20260822` | `5bcc89ebdb77663ea8c035a255a218079ee1ab31` | bounded two-slot pinned-registration reuse |
| `cuda-screening-cache-reopen-20260822` | `91ef7c7e3528aaa8dbaa4e19ed48e05d939097bc` | exact fail-closed prepared-result reopen |
| `selective-load-contract-audit-20260822` | `58577c81390419d6237465092d50a8fb80dbe36d` | backend-neutral selective-load audit |
| `morning-evidence-index-20260822` | `9184aef2d0695b13be2fc1e30128a75cde05ddcc` | 27-artifact accepted/refuted/pending reconciliation plus local documentation build and independent review |

The separately preserved `webgpu-exact-integer-binning` branch remains dirty
at base `a07121d7` with full-index tracked patch SHA-256 `c702888c...` and
untracked binner SHA-256 `5bdee03b...`. Its hardware follow-up is evidence for
continued review, not a clean source boundary and not eligible for consumer
integration.

The public consumer should pin a qualified source commit, not an evidence-only
commit, and should consume APIs from QuantEM.GPU rather than copy kernels.
Live4DSTEM retains device admission, detector-bin policy, product lifecycle,
provenance presentation, cancellation/latest-wins, and UI acceptance.

## Final host and process state

At 10:27 PDT, Phil and Rodman each reported 81% memory free. Phil used
1,911.50 MiB swap and had no campaign browser, server, benchmark, or listener
on ports 8877/9337. Rodman used 2,300.44 MiB swap, had no Live4DSTEM or campaign
process, and retained its pre-existing relay unchanged. MJGOAT had 58 GiB RAM
available, zero swap, and both GPUs at 0% utilization. The existing private
service retained 552 MiB on each GPU and was not stopped. No task-owned loop,
server, tunnel, or GPU process remained.

## Remaining high-value gates

1. Measure cold original compressed-source, first usable, exact complete, warm
   source reopen, and prepared/cache reopen as separate distributions on Steve
   Kerr and Rodman through the owning app task.
2. Complete Swift/Metal bulk selective IO plus WebGPU intra-shard byte-range
   and arbitrary-position parity before claiming cross-backend equivalence.
3. Qualify the WebGPU exact-integer candidate on the full real source with
   complete-output and product parity, immutable memory telemetry, and a clean
   source revision before any adoption.
4. Recover or deliberately re-freeze the missing SSB 200-trial optimizer engine
   before any calibration-speed comparison.
5. Keep the subsecond target honest: current package evidence reaches it for
   prepared/reused Apple paths and CUDA result reopen, but not for a verified
   cold arbitrary-source first encounter or Live4DSTEM wall-to-wall load.
