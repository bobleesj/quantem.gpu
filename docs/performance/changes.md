# Revision and change ledger

## 2026-10-23: IO publication review

The FP8 detector-product comparison now explicitly requests
`hot_pixel_correction="none"` for both its FP8 input and float32 reference.
The test compares original values with flagged pixels excluded; requesting
the default integer-only median correction contradicted that intent. All
numerical assertions and tolerances remain unchanged. Historical failed
acceptance reports remain in the private evidence archive.

## 2026-10-23: Apple loading checks and persistent IO history

The routine loading runner now supports native Swift/Metal alongside Python
CUDA/MPS. Native timing includes HDF5 catalog preparation and checks frozen
raw diffraction frames outside the timer; QEM preparation uses the accepted
MPS representation. Native warm inputs are explicitly primed when needed.
File dtype is checked separately for uint16 HDF5 and losslessly packed QEM.

The [three-backend record](2026-10-23-io-loading-limits.md#routine-cross-backend-loading-check)
adds fresh MPS and Swift/Metal five-trial runs and matched replays. Both Apple
replays stayed within the 20% review threshold. All references passed; full
original-count verification was not rerun. A source-drift run and harness
controls remain in the private evidence history. The final native replay used
a frozen source snapshot; no production loader or saving code changed.

`--history-dir` automatically registers immutable reports and generates JSON,
CSV and Markdown indexes. `benchmark_io_history.py` imports older reports.
Baselines stay explicit and are pinned by SHA-256 in new reports. Concurrent
source edits invalidate a run. Private evidence: `20261023-io-limits`.

## 2026-10-23: repeatable IO loading regression check

`benchmark_io_scale.py --load-only` separates first observations, controlled
disk reads and cached loads, with five trials recommended per format/input.
QEM preparation is outside loading time; encoded-body/layout identity and
indexed products are compared with frozen references without repeating full
original-count verification. A supplied prior report must match the physical
device, storage, inputs and timing/cache/allocator policies. Median slowdowns
over 20% trigger review; references are never automatically replaced.

The [fresh CUDA check](2026-10-23-io-loading-limits.md#routine-512-cuda-loading-check)
passed all reference checks on two full 512² × 192² inputs. A matched replay
stayed within the threshold in all eight comparisons. QEM disk medians were
0.363 / 0.395 s and cached medians 0.165 / 0.191 s. A separate harness control
that reset CUDA memory pools explains higher preliminary timings; normal
allocator reuse is now explicit. No production loader or saving code changed.
The runner is manual, not an automatic hardware CI gate; MPS/native timings
were not refreshed in this campaign. Private evidence: `20261023-io-limits`.

## 2026-10-23: optional conversion checks

Routine Python and CLI QEM conversion now defaults to `verify=False` and
`source_hashes=False`. `--verify` compares every saved value; `--source-hashes`
adds complete original-file fingerprints. They are independent opt-ins, while
QEM checksums, embedded-master restoration, file-data fsync and no-overwrite
publication remain enabled. Existing `--no-verify` calls remain valid. The
verified benchmark requests both checks explicitly and retains its old timing
boundary; earlier results are not relabeled as routine conversion timings.
Portable checks passed 69 tests. Physical conversion checks passed 18 tests on
MPS and 17 on CUDA, with only the opposite-backend lifetime checks skipped.
The broader acceptance runs retained their existing FP8 product failures and
the existing MPS uint64 SSB failure; no new failed test appeared. Private
evidence is retained in experiment `20261023-io-limits`.

Two subsequent routine 2048² conversions per backend, with both optional checks
off, took 35.876 / 24.743 s on MPS and 46.721 / 62.164 s on CUDA. All started with
zero measured source-file cache residency. CUDA saving still varied from
27.145 to 43.089 s including fsync; no stable total speedup is established.
The [routine conversion table](2026-10-23-io-loading-limits.md#routine-conversion-with-optional-checks-disabled)
keeps these measurements separate from the earlier verified results.
One subsequent instrumented CUDA save took 25.858 s; write and fsync intervals
covered 94.4% of it. The raw trace and a reproducible agent handoff are archived
privately. This is diagnostic evidence; no writer change or speedup is claimed.

## 2026-10-23: complete HDF5 and QEM loading optimization

The [qualified workflow comparison](2026-10-23-io-loading-limits.md) adds bounded
HDF5 prefetch and verification windows on CUDA, fewer MPS range/serialization
passes, and direct native QEM reads into final Metal owners. Native HDF5 uses
bounded 8,192-frame windows; native saving bounds private readback lifetimes and
overlaps one next chunk. Full original-count and checksum checks are retained.
All five acquisitions were rerun from a common source fingerprint with measured
empty-cache QEM controls and explicit first/repeated HDF5 timing. Local default
HDF5/QEM paths were smoke-tested. Broader baseline format failures remain visible.
The first CUDA save profile finds that file writes and final fsync dominate.
A later direct-write candidate passed exact real-data checks but did not show a
repeatable save gain and was not adopted. The buffered default and canonical
measurements remain unchanged. A subsequent accepted source-SHA overlap reduces
complete MPS conversion to 41.135 s against 48.692 and 52.212 s baselines; CUDA
preparation improves by 6.1–6.7 s, but variable saving prevents an end-to-end CUDA
speedup claim. All original counts and provenance checks remain exact. Storage
maintenance and its retest separately await administrator access. Older
measurements keep their original implementations and cache conditions.
A subsequent four-worker buffered writer was rejected: its largest save took
54.310 s versus a nearly read-quiet current-writer control of 45.070 s. Exact
counts and hashes passed. A fresh full control conversion took 70.399 s;
writeback telemetry attributes the long fsync tail mainly to pending data writes,
not the much shorter device flush commands. No writer speedup is claimed.
A private final-fsync overlap hid 9.033 s of flushing but complete conversion
took 78.363 s versus 68.423 and 78.704 s controls, with competing SSD reads in
the later runs. All three authenticated envelopes and exhaustive comparisons
matched. Process attribution identified two report-inventory readers accounting
for 14.182 GB of observed reads during the closing control. This identifies a
separate source of storage contention; it does not establish a writer speedup.
The private receiver now reuses fallback report hashes only while full file
identities match and serializes concurrent inventory calls. Fourteen tests
passed; the real repeated inventory took 0.108 s with zero report-body opens
and identical returned records. Publishing rules and the QEM writer are unchanged.
One subsequent full default CUDA conversion took 61.292 s: preparation 21.687 s,
saving with fsync 22.804 s, and full verification 16.669 s. All 154,618,822,656
original values and complete authenticated envelopes matched. Large background
inventory reads disappeared; smaller report-server reads remained. This is a
new qualified observation, not an isolated QEM-code speedup or a hardware limit.

## 2026-10-23: CUDA and MPS loading interpretation

The [five-acquisition comparison](2026-10-23-default-five-io.md) now explains
that 5.10 s is not a theoretical CUDA minimum. The CUDA trial read 24.7 GB from
storage; the MPS trial did not measure physical reads or cache residency. The
frozen MPS path reads into shared GPU-accessible buffers and avoids a separate
host-to-device transfer. Cache benefit remains unquantified. This is a
clarification of existing evidence, not a new timing run or GPU ranking.

## 2026-10-23: CUDA QEM read scheduling

The [loader fix](2026-10-23-cuda-qem-loader-fix.md) uses up to sixteen persistent
readers, cached-block priority and upload on checksum completion. Matched
empty-cache loading improves from 10.61 to 6.59 s; half-cache loading improves
from 10.44 to 4.78 s. Full source verification remains exact on all five default
acquisitions. Pinned staging increases from 256 MiB to at most 1 GiB.

## 2026-10-23: 2048² CUDA QEM cache controls

The [cache recheck](2026-10-23-cuda-qem-cache-recheck.md) repeats complete lossless
verification and isolates ordinary loads, per-file cache eviction, sequential
pre-reads and a full-byte-count pinned-copy control. The production loader and
codec are unchanged; the original comparison medians remain intact.

## 2026-10-23: five-input default selection

The [current Disk–GPU comparison](2026-10-23-default-five-io.md) excludes 96 × 96
detector acquisitions from future default/full-scale benchmarks and active
examples. The five retained 192 × 192 inputs reuse completed measurements;
no new timing or codec improvement is claimed. Historical six-input results
remain unchanged. The private static report separates acquisition and benchmark
dates, labels HDF5 as bitshuffle + LZ4, and identifies the M5 Max MacBook Pro.

This ledger separates documentation revisions from the implementation revision
that produced a benchmark. A documentation edit never makes an older timing a
measurement of the current code.

## 2026-10-23: spatial-index accounting and CUDA read profiling

The [layout and profile follow-up](2026-10-23-qem-layout-cuda-profile.md)
recreates all six QEM files with exhaustive source verification and accounts
for their payload, stream offsets, models, spatial indexes and envelope.
The benchmark gains optional host-call interval profiling after ordinary
reloads. Device/process observations and transfer controls bound the loading
interpretation; the existing source implementation and original timing table
remain unchanged. Static IEEE-style tables are retained privately.

## 2026-10-23: QEM compression and loading interpretation

The [compression comparison](2026-10-23-qem-compression.md) derives exact size
ratios from the retained six-acquisition conversion records and verifies every
source dataset's bitshuffle/LZ4 filter and complete HDF5 file total. It records
29.0% aggregate savings, the 25.5% expansion case, and the distinction between
storage savings and backend loading costs. This is analysis of the existing
benchmark; no codec change or timing rerun is claimed.

## 2026-10-23: selected six-acquisition comparison

The [default-six rerun](2026-10-23-default-six-io.md) adds a single comparison
table across physical CUDA, MPS and native Swift/Metal. Benchmark runners now
accept a checked private selection manifest and fresh first-trial frame indexes,
and record a complete verified conversion wall time. Numerical compute is
unchanged. Native conversion includes its round-trip/product checks followed by
MPS exhaustive verification; the table labels this different workload.

## 2026-10-23: bounded full-count QEM verification

The [QEM verification record](2026-10-23-qem-verification.md) measures bounded
CUDA and Metal source decoding, overlapped batches, exact unsigned GPU comparison,
and parallel source hashing. The native writer now plans its authenticated
header before streaming the body once, removing the second full-size temporary
copy. Separate [CUDA](2026-10-23-io-cuda.md), [MPS](2026-10-23-io-mps.md), and
[Swift/Metal](2026-10-23-io-swift-metal.md) scale reports preserve their timing
boundaries, complete source verification, and unsupported input cases.

## 2026-10-23: prepare loader code before acquisition arrival

The [dated startup record](2026-10-23-loader-startup.md) measures a private
preparation helper and the application H5-first path. Source data, decoding
arithmetic and scientific fitting budgets are unchanged. Loading, preparation
and browser-visible times have separate boundaries; older rows stay historical.

## 2026-10-23: bounded MPS ingestion pipeline

The [dated loader record](2026-10-23-mps-hdf5-pipeline.md) measures one-batch
read/decode overlap and reuse of decoded and ANS scratch buffers. Kernels and
native counts remain unchanged. The benchmark now distinguishes the public
scientific read dtype from the narrower lossless backing representation.

## 2026-10-22: exact independent proposal reuse

The [dated broad-search record](2026-10-22-independent-tpe-reuse.md) reports
batched inverse-CDF calculations and completed-history reuse, with exact
proposals, objective values and phase arrays. The benchmark now explicitly
selects standard search to retain its documented 200-trial-per-branch workload.
No CUDA reconstruction schedule, precision, BF selection or estimator changes.
The separate shared-FFT compiler-cache investigation remains held.

## 2026-10-21: reusable SSB measurement and agent workflow

Added `quantem-gpu benchmark-ssb`, a repository-local `gpu-benchmark` skill,
and installed-wheel hardware gates. The
[dated record](../maintainer/2026-10-21-ssb-joint-benchmark.md) separates repeated
MPS compute timings from browser interactions and earlier one-off observations.
Raw evidence stays private. This adds reproducible measurement infrastructure;
it does not claim a kernel speedup or completed cross-backend release signoff.

## 2026-10-10: registry retired

The benchmark registry (`benchmarks/benchmark_registry.json`), the profile
matrix, the backend-status generator, the evidence fingerprint manifest and the
pages generated from them were removed. No measurement changed. The generated
current-load table is kept as a static dated table in
[Verified benchmark results](results.md); the implementation dashboard's
timings already lived there, and its minimum-device memory table moved to the
{ref}`benchmark methodology <minimum-device-memory-gates>`. Hot-path timing
now uses the scripts listed in the [test guide](../developer/testing.md).

## 2026-09-08: unpublished native packed-resident candidate

The [dated investigation](../maintainer/2026-09-08-packed-resident-optimization.md)
records exact original-HDF5 reconstruction, seven-resident detector reductions,
native presentation measurements, rejected experiments, and corrected timing
assumptions. It is not a release signoff or a replacement for the older source
checkpoints below. The 120 Hz wide-detector and four-to-five-second seven-load
targets remain unmet.

## 2026-08-22: previous review

- **Ledger reviewed:** 2026-08-22
- **Integration base:** origin/main `75be74e`
- **Current clean benchmark source checkpoints:** canonical Python/CPU metadata
  and parity `68dbe3a`; canonical native Swift/Metal load `5106ca4`
- **Historical measured checkouts:** consolidated Python MPS `0bc9378`;
  native controlled-source `c0ea444`; retained CUDA and WebGPU rows remain
  bound to the exact revisions shown in the results ledger
- **Documentation branch:** `benchmark-platform-computer-matrix`

The native stack extends exact prepared QH5 binning (`e0e92b4`), optimized
word-major detector binning (`ff3c7fd`), and exact resident summaries
(`d65911a`) with bounded streaming, private residency, and controlled macOS
source-page measurement through `c0ea444`. It does not relabel an older CUDA,
Python MPS, WebGPU, source-load, or application measurement.

## Documentation changes through 2026-09-08

| Commit | Change | Performance-number effect |
|---|---|---|
| `68dbe3a` | Normalized source and working detector metadata, then reran canonical CPU/Python MPS bins 1/2/4/8 | replaces the `0bc9378` headline with exact full-volume current rows: MPS p50 0.406624/0.477740/0.370645/0.340210 s; cache/source-page states remain explicit and none is arbitrary-source cold |
| `5106ca4` | Added the canonical cross-layout logical-pixel contract used by native Swift/Metal evidence | adds current physical M5 24 GB bin2/bin4 distributions and one pressure-gated full-native bin1 smoke; the bin1 smoke is partial because it caused 758,448,128 B of swapouts |
| `0bc9378` | Consolidated exact MPS bin1/2/4/8 kernels, overflow and partial-tail guards, resident reuse, and exception-safe cleanup | adds final-head post-warmup fresh-destination p50 0.414824/0.457153/0.382109/0.356258 s for bins 1/2/4/8; controlled cold and application E2E remain open |
| `e4a35f9` | Adopted the accepted exact WebGPU integer CoM/DPC/iDPC source without importing obsolete campaign harnesses | no new timing claim; preserves the existing physical-Apple numerical evidence at the consolidated source boundary |
| `f0f39c9` | Added scratch-free exact full-`uint16` decode, fused exact detector-sum kernels for bins 2/4/8, bin-2 specialization, lazy LZ4 scratch, and source-shard-aligned pipelining | historical pre-consolidation MPS rows remain retained; they no longer headline current source |
| Current MPS lifecycle-audit follow-up | Separates resident payload, driver allocation sampled after load and release, process RSS, pressure, and swap | replaces the pressure-contaminated 2026-08-19 MPS load headline with revision `be035c4` fresh-process and explicit-release warm-process rows; no loader arithmetic changed |
| `c0ea444` | Aligned the raw benchmark state with explicit macOS `F_NOCACHE` control and sealed seven replacement runs | promotes a controlled uncached-source-page full-native package row; its distribution replaces schema v7 only because the older raw state was contradictory, not because the IO path changed |
| `00ba8bd` | Reused the immutable source identity while preserving controlled source reads | kept the same exact source/load boundary; the later `c0ea444` run owns durable timing because its state label is internally consistent |
| `3bb3845` | Added exact full-native detector-bin-1 loading into private Metal residency | established the 18 GiB exact resident path; later controlled-source runs own the current distribution |
| `d65911a` | Added provenance-bound exact resident summaries and overflow-safe `uint64` detector moments | adds separate one-time summary-build and prepared-reopen rows; does not replace first compressed-source load or resident-compute rows |
| `e1da9bc` | Added `MetalSSBKernels`, exact 512×512 reconstruction/loss, deterministic 200-trial TPE plus Nelder–Mead fitting, focused tests, and a standalone benchmark | adds separate native Swift/Metal SSB rows; does not replace differently configured CUDA, Python MPS, or WebGPU rows |
| Current profiling-registry follow-up | Added the 35-cell platform/module schedule, human run index, manifest validator, CI gate, and continuous-profiling guide | none; the existing timings and scientific acceptance states are unchanged |
| `e052dfb` | Combined current platform profiling evidence with the three parity-qualified scientific fixes | existing measured baselines remain tied to their recorded revisions |
| `146238a` | Clarified that the SSB PyTorch functions are executable teaching references, not production kernels | none |
| `d83c8a1` | Rewrote SSB examples with explicit named layouts and ordinary PyTorch FFT calls | none |
| `7182ce4` | Made the SSB reference progressive and executable from top to bottom | none |
| `a319507` | Added equation-paired PyTorch SSB reference functions | none |
| `5a897d9` | Added the implementation and benchmark dashboard | no measurement changed; existing evidence was indexed |
| `f16ad05` | Completed operation contracts, source maps, and parity gates | none |
| `6502ed0` | Turned API pages into typed integration contracts | none |
| `de5050c` | Separated kernel implementations from remote compute | none |
| `c7cfccd` | Standardized public `(row, column)` notation | none |
| `a25ab84` | Reorganized the site around scientific operations and runtimes | none |

Local documentation commits are recorded as branch-relative identifiers until
they are published. Do not construct public GitHub commit links for an
unpublished revision.

## Benchmark-evidence changes through 2026-08-22

This table records promotion decisions, not another copy of the measurements.
The current values and distributions live only in
[Verified benchmark results](results.md).

| Evidence change | Current disposition | Why |
|---|---|---|
| `20260822-mps-load-sub500ms` | Promoted for Python MPS exact warm-process and independent-process loading at its revision; `io.load` has since replaced that loader with encoded loading at the native detector | Four-bin full-output parity is byte exact. ABBA measurements preserve process/source state and explicit release. Bins 2/4/8 meet strict p50 <=0.5 s; full native bin1 is 0.523 s. Source pages remain warm and uncontrolled, so the result is not cold storage or application E2E. |
| `20260822-mps-load-memory-audit` | Promoted for Python MPS process-state and memory reporting | Seven fresh processes per detector bin and seven explicit-release warm-process repetitions preserve exact selected-frame hashes, record logical payload plus driver allocation sampled after load and release, process RSS, pressure, and swap, and pass independent exact detector-bin/product parity. Source pages were warm and uncontrolled, so neither protocol is cold-storage evidence. |
| `20260819-platform-profile-mps` load headline | Superseded, preserved | Its repeated-load harness cleared the Torch cache but did not call `MPSChunked4DSTEM.free()` on direct PyObjC Metal outputs. Full bin1 accumulated roughly 18 GiB per repetition, free memory fell from 86% to 41%, and p50 rose to 2.273 s. The artifact remains useful as pressure-failure evidence, not current loader timing. |
| `0822-97-fnocache-provenance-v8` | Promoted as controlled native package evidence | Seven fresh processes use a new index root, fresh private destination, and explicit `F_NOCACHE` source descriptors; volume and product hashes are exact. It is an audited-source package boundary, not arbitrary-source cold or application E2E, and the detailed result remains above the requested target. |
| `0821-96-bin1-controlled-cold` schema v7 | Superseded, preserved | Its command applied `F_NOCACHE` but the raw state token said source pages were unspecified. The timing and raw trials remain retained; schema v8 replaces it only for state-consistent reporting. |
| `20260819-air-exact-resident-summary` | Promoted as separate native Swift/Metal prepared-product evidence | Seven fresh-process reopens reproduce nine same-device products byte-for-byte on MacBook Pro (M5 Max, 128 GB) and MacBook Air (M2, 8 GB). Summary creation, prepared reopen, resident load, compressed-source load, and GUI paint remain distinct boundaries. |
| `20260819-native-metal-ssb` | Promoted as a separate native Swift/Metal SSB row | It uses its own full-BF fixture and optimizer implementation; warm prepared compute is not raw-source load, application wall time, or physical 8 GB signoff. |
| `PLATFORM-PROFILE-2026-08-19` | Current cross-platform profile | It supplies atomic timing, memory, dtype/bin, device, date, and parity fields; fixtures C and D remain explicitly separate. |
| Exact streamed screening follow-up | Promoted | It derives masks from the complete detector sum and transparently reruns BF/DF when the provisional mask differs. The first-chunk-mask candidate failed full-scan parity and remains rejected. |
| Deterministic CUDA SSB calibration follow-up | Promoted | Three seeded fits reproduce fitted parameters, phase, object, and loss. The earlier atomic-objective run produced two fitted minima and remains rejected despite being faster. |
| Current MPS prepared companion | Rejected | Stored columns do not match the declared detector-bin coordinate grid; only the raw detector-bin-2 reconstruction retains current numerical evidence. |
| Current WebGPU profile | Promoted for the measured load and resident-compute boundaries | UI paint, physical 8 GB signoff, per-pixel CoM/DPC/iDPC error arrays, and calibration remain explicit gaps. |
| `THREE-HOST-512-U16-2026-08-19` | Superseded for current headline timing | It remains useful same-fixture history, but newer comparable profile rows own current values. |
| Detector-bin-4 products | Mixed | Integer products and CoM pass their stated gates; cross-backend iDPC remains blocked and cannot inherit an integer-parity check mark. |
| Native-detector MPS CoM/iDPC | Blocked as native-resolution evidence | At that revision the public interaction sidecar used detector bin 2 (Metal detector sessions have since dropped the sidecar); direct full-resolution CoM passes, while iDPC remains blocked. |
| `M2-AIR-BIN4-E2E` | Separate application evidence | It proves a physical 8 GB detector-bin-4 application path, not a no-bin library load or native SSB memory gate. |
| `CUDA-STOCHASTIC-IO` | Historical, labeled first-process | The storage-cache eviction procedure was not retained, so it cannot be called cold. |
| `WEBGPU-VISIBLE-512` | Historical single visible run | No timing distribution was retained, so it is not a median. |
| Other July rows with missing host identity | Historical diagnostic | Their numerical evidence can guide optimization, but incomplete hardware provenance prevents current release signoff. |

Accepted and rejected performance experiments remain in the
[optimization ledger](../maintainer/backend-optimization-matrix.md); older SSB
layout experiments remain in the
[SSB performance history](../maintainer/ssb-performance.md).

## Adding a new revision

1. Add the complete benchmark row with date, exact measured revision, physical
   device/runtime, source shape/dtype, cache state, crop/bin/load plan, timing
   boundary, memory, calibration, and parity artifact.
2. Explain whether it replaces a truly comparable row or remains separate.
3. Keep rejected or incomparable experiments discoverable rather than deleting
   them.
