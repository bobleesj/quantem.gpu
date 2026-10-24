# Disk–GPU loading: HDF5 and QEM, 2026-10-23

**Current status:** routine 512 × 512 × 192 × 192 loading checks passed on
CUDA, Python MPS and native Swift/Metal for the two selected ARINA inputs. Large-QEM CUDA saving remains unresolved and
further optimization is deferred. Loading, saving and complete conversion are
separate measurements. The [benchmark index](../../results.md#hdf5-and-qem-diskgpu-loading-2026-10-23)
links this campaign's current and historical records.

This follow-up measures complete ARINA HDF5 loading, authenticated QEM loading,
and QEM conversion with exhaustive verification on the five default acquisitions.
HDF5 uses **bitshuffle + LZ4**. QEM uses ANS and includes decoding models, stream
offsets, spatial indexes, metadata and checksums. QEM is both a storage format and
an encoded GPU-resident format. A QEM load does not expand the dense count tensor.

The selected geometries are 256², two 512², 1024² and 2048² scans, all with 192²
detectors. Every input is uint16. The established MPS path can losslessly store
values as uint8 when their exact maximum permits it; the original dtype and
this reversible storage choice remain in metadata. No crop, binning, detector
correction, count change or approximate comparison is used.

Python MPS and native Swift/Metal use a MacBook Pro with Apple M5 Max and 128 GB
unified memory. CUDA uses an RTX PRO 6000 Blackwell with 96 GB. The storage,
memory architecture and computers differ. These are complete workflow timings,
not an isolated GPU ranking.

Private experiment: `20261023-io-limits`.
Original five-acquisition source fingerprint: `b87d0c529f6d21c6bc9ec70f4d758df7be5a51fa5ceade967c106de0b2d3774b`.

Routine Python/CLI conversion now defaults to `verify=False, source_hashes=False`.
Enable `verify=True` and/or `source_hashes=True` when needed; the CLI equivalents
are `--verify` and `--source-hashes`. QEM checksums and file-data fsync remain
enabled. The new routine measurements below disable both options. Earlier
verified results retain their original verification boundaries.

## Routine cross-backend loading check

Two complete **512 × 512 × 192 × 192** acquisitions were checked with five disk
and five cached trials per format. The CUDA rows retain the preceding qualified
replay; Python MPS and Swift/Metal were freshly run and repeated against their
own matched references. HDF5 uses **bitshuffle + LZ4**; QEM uses **ANS**.

| Input | Backend | HDF5 disk (s) | HDF5 cached (s) | QEM disk (s) | QEM cached (s) |
|---|---|---:|---:|---:|---:|
| acquisition-01 | CUDA | 0.647 | 0.481 | 0.363 | 0.165 |
| acquisition-04 | CUDA | 0.834 | 0.639 | 0.395 | 0.191 |
| acquisition-01 | Python MPS | 0.755 | 0.704 | 0.175 | 0.078 |
| acquisition-04 | Python MPS | 0.778 | 0.721 | 0.200 | 0.094 |
| acquisition-01 | Swift/Metal | 0.714 | 0.718 | 0.177 | 0.074 |
| acquisition-04 | Swift/Metal | 0.728 | 0.741 | 0.194 | 0.077 |

Values are medians to encoded GPU residency. Apple hardware is a **MacBook Pro,
Apple M5 Max, 128 GB unified memory**, using its internal SSD. CUDA hardware is
an **RTX PRO 6000 Blackwell Workstation Edition, 96 GB**, with one WD_BLACK SN850X
SSD. Different computers, storage and process lifecycles make this a workflow
comparison, not an isolated GPU ranking. QEM remains encoded on the GPU.

All retained disk trials began at 0% measured file-page residency; cached trials
began at at least 99.99% and recorded zero physical reads. Native warm trials explicitly
prime missing master-file pages outside timing. The native HDF5 timer and
physical-read counters include catalog preparation. Native trials use a new
process for each load, excluding process launch; Python reuses the process and
allocator pools. First observations have uncontrolled initial caches and are
kept separate (input 01 / 04):

- CUDA: HDF5 0.915 / 0.886; QEM 0.375 / 0.190 s.
- Python MPS: HDF5 0.848 / 0.747; QEM 0.080 / 0.096 s.
- Swift/Metal: HDF5 1.319 / 1.375; QEM 0.214 / 0.076 s.

Both Apple replays passed all eight comparisons against their same-backend
five-trial references at the 20% median review threshold. MPS changes ranged
from -1.5% to +2.1%; Swift/Metal changes ranged from -2.8% to +0.6%. These are
repeatability checks, not evidence of a new loader optimization.

Each prepared QEM matched the archived encoded body and layout exactly; public
loading retained payload authentication. Python checks the frozen indexed
detector product after every load; native checks three frozen raw diffraction
frames, shape and per-file dtype. Each accepted replay passed all 44 load
checks. No crop, binning or new precision reduction was applied. The original
HDF5 is uint16; the accepted MPS QEM uses existing lossless uint8 packing where
all values fit. Native QEM loads use that same MPS-prepared file. Preparation
and saving are excluded; this does not measure native conversion. Full
original-count verification was not repeated.

The final native replay used a frozen source snapshot after an earlier replay
was invalidated by a concurrent, unrelated source edit. The history preserves
that invalidated run, a dtype-contract harness rejection and the earlier cache
controls. None replaces the accepted reference. Source fingerprints:

- CUDA: `775debceda019b51a2d94adb88e1e42866ad65c1e0932dc24f62104ad231b390`.
- Python MPS: `63207b87bed4b4362bb5098db69ba7f000135e133d0b222baca4719550b3ee62`.
- Swift/Metal: `19d0d1efeb6c4ca3fc716e5429892d8018b21ae70bfba5bd1a5d847d64881dee`.

Private evidence is under `20261023-io-limits/loading-regression`, including
all trials, source references, commands, binary hashes, cache/physical-read
counters, telemetry and the frozen snapshot. The native binary SHA-256 is
`65b71593737fed8622ec3f2661bb785e94e0b743bcc8fd4000eaff0f15db3dd5`.

### Tracking future runs

Use the [routine command](../../../developer/testing.md#routine-io-loading-check) with
`--history-dir` and an explicit `--baseline`. Every report is archived by
content hash; `index.json`, `timings.csv` and `README.md` track its revision,
hardware, storage, cache conditions, checks and baseline. Duplicate registration
does not duplicate evidence, and new timings never automatically replace a
baseline. Use the [history workflow](../../../developer/testing.md#keeping-benchmark-history)
after IO changes and complete qualification before codec changes or releases.
The private history now includes the accepted runs and diagnostic failures.
The runner remains manual; physical IO performance is not an automatic CI gate.

## Routine 512 CUDA loading check

The reusable `benchmark_io_scale.py --load-only` check was run on both complete
512 × 512 × 192 × 192 inputs, then repeated against the first reviewed report.
Hardware: RTX PRO 6000 Blackwell Workstation Edition, 96 GB; one WD_BLACK SN850X
SSD with ext4. The latest replay below uses five trials per cold/cached cell.
HDF5 is bitshuffle + LZ4; QEM is ANS with its normal payload authentication.

| Input | HDF5 disk (s) | HDF5 cached (s) | QEM disk (s) | QEM cached (s) |
|---|---:|---:|---:|---:|
| acquisition-01 | 0.647 | 0.481 | 0.363 | 0.165 |
| acquisition-04 | 0.834 | 0.639 | 0.395 | 0.191 |

Values are medians through synchronized encoded GPU residency. Every disk trial
started at 0% measured source-file residency; every cached trial started at
100% and read zero physical bytes. Physical reads were 3.040 / 3.374 GB for
HDF5 and 2.054 / 2.355 GB for QEM. Source files, original uint16 counts and full
geometry were preserved. The per-acquisition frame index was fresh for the
first HDF5 observation and reused for repeated measurements; GPU/pinned memory
pools were reused normally. First HDF5 observations were 0.915 / 0.886 s and
first QEM observations 0.375 / 0.190 s, with uncontrolled initial cache state.
CUDA context initialization preceded timing; these are not fresh-process
startup measurements.

Both QEM encoded bodies/layouts matched the frozen, previously fully verified
references exactly. All 44 replay loads passed shape, dtype and frozen indexed
detector-product checks outside the timer. Full original-count verification and
source-file hashing were not rerun. Disposable QEM preparation/saving was also
outside load timing. This check adds no new full-verification claim.

All eight median comparisons stayed within the configured 20% review threshold
against the preceding matched five-trial run: changes ranged from -4.5% to
+10.3%. That preceding run measured HDF5 disk 0.678 / 0.833 s, HDF5 cached
0.463 / 0.579 s, QEM disk 0.358 / 0.380 s and QEM cached 0.173 / 0.181 s.
The threshold is a practical review trigger, not a statistical significance
test. Existing desktop processes remained active; GPU preflight was idle, and
GPU/storage telemetry is retained. No loader implementation speedup is claimed.

An earlier harness control emptied GPU and pinned-memory pools before every
trial. Its QEM medians were 0.609 / 0.624 s from disk and 0.421 / 0.416 s cached.
That allocator-reset condition is retained separately and excluded from the
normal-loading baseline. This explains the apparent slowdown found while
building the check; disk-cache eviction and allocator reset are distinct.

Both matched runs used source fingerprint
`775debceda019b51a2d94adb88e1e42866ad65c1e0932dc24f62104ad231b390`;
imports matched the checkout and source remained unchanged. Raw trials,
references, source snapshots and telemetry are in this private experiment's
`loading-regression` folder. The runner and its portable infrastructure checks
passed 33 tests, including report comparison, timing boundaries, existing load
helpers and repository privacy checks. See the
[repeatable command](../../../developer/testing.md#routine-io-loading-check).
This fresh campaign is CUDA only. Python MPS supports the routine command but
was not rerun here; native Swift/Metal retains its separate earlier evidence.
Ordinary CI does not automatically run the physical IO performance gate.

## Validated 512 CUDA loading

Both complete 512 × 512 × 192 × 192 inputs passed exact comparison of
**9,663,676,416 original values each, with zero differences**, including flagged
pixels. No scan or detector crop, binning or pixel correction was applied.
The installed development checkout also passed normal-load product checks.
These are measurements of the tested ARINA HDF5/QEM path, not qualification of
every supported format or a newly published package/app build.

| Input | HDF5 first (s) | HDF5 repeated (s) | QEM empty cache (s) | QEM cached (s) |
|---|---:|---:|---:|---:|
| acquisition-01 | 1.338 | 0.487 | 0.360 | 0.183 |
| acquisition-04 | 1.492 | 0.610 | 0.382 | 0.185 |

Device: RTX PRO 6000 Blackwell, 96 GB. The endpoint is encoded GPU residency,
not expansion of the dense count tensor. HDF5 source-page cache was uncontrolled;
its first observation is not labeled cold. Repeated values are medians of
observations 2–3. QEM empty-cache values are medians of three separately
controlled loads, checked with file residency and physical-read counters.
Full original-count verification was outside these load timings. The
[complete workflow matrix](#complete-workflows) preserves all five acquisitions
and all three backends from the same qualified rerun.

## Routine conversion with optional checks disabled

Two fresh-process conversions per Python backend used the complete
2048 × 2048 × 192 × 192 acquisition: 54.595 GB of HDF5 input and 38.333 GB of
QEM output. Both source hashing and exhaustive original-count comparison were
disabled. QEM payload checksums, file-data fsync and atomic publication remained
enabled. Each run began with zero measured source-file cache residency and a
fresh frame index; GPU context initialization preceded the timer.

| Backend and hardware | Load and encode (s), first / repeat | Save including fsync (s), first / repeat | Complete conversion (s), first / repeat |
|---|---:|---:|---:|
| Python MPS, M5 Max MacBook Pro, 128 GB | 24.550 / 19.455 | 10.810 / 4.811 | **35.876 / 24.743** |
| CUDA, RTX PRO 6000 Blackwell, 96 GB | 19.462 / 18.967 | 27.145 / 43.089 | **46.721 / 62.164** |

These are two observations, not a steady-state median. The repeat also started
with empty source-file cache. Complete conversion includes small metadata,
publication and cleanup costs beyond the two listed stages. CUDA GPU encoding
took 1.216 s in the first run, within loading; saving the 38.333 GB output
dominated and varied substantially. Removing the optional checks does not remove
those writes. The second routine CUDA run was slower than the earlier 61.292 s
verified run, so these observations do not establish a stable end-to-end speedup.
Native Swift/Metal was not rerun with the new Python options.

The runner confirmed successful publication, expected shape, omitted source
hashes and no exhaustive verifier invocation. It did not repeat an original-count
comparison in these four runs. Earlier exact-count qualification is reported
separately below. Original inputs retained their sizes and modification times;
only disposable QEM outputs were removed after recording the results.

Runtime imports matched each measured checkout and remained unchanged during
each trial. MPS source fingerprint:
`f597fbdb1c0833cc681953b77874358d28ad535f9440a9d19f4b57972b50cbaa`.
CUDA source fingerprint:
`d5095e5d4ecf68a2da48eac5e4fa2a65f9c915fd43518298e611b59d81bb2d73`.
The existing runner supports this measurement with
`--no-verify --no-source-hashes --reloads 0`; its qualification defaults still
enable both checks. Raw evidence is in the private experiment's `routine-speed`
folder. These routine timings are separate from the established 15-row matrix.

### Focused CUDA save profile for handoff

A subsequent instrumented routine conversion took 45.181 s, including 19.120 s
loading and 25.858 s inside the save profiler. Source hashing and exhaustive
verification were disabled; the original copy, checksum, write, fsync and
publication calls were unchanged. This is a diagnostic with uncalibrated
profiling overhead, not a new optimization result.

| Save operation | Host duration (s) |
|---|---:|
| Output write calls | 19.425 |
| Final file-data fsync | 4.978 |
| Producing staged GPU-to-host pieces | 2.093 |
| SHA-256 update calls | 9.688 |

These durations overlap and must not be added. The union of write and fsync
intervals covers **94.4%** of saving. During fsync, the device wrote 6.461 GB,
global dirty memory fell from 6.463 GB to 1.384 MB, and device flush commands
totaled 28 ms. Whole-save device reads were 23.732 MB. This supports pending
data writeback as the dominant flush tail in this observation, without
establishing the cause of the SSD's variable write service or a hardware limit.
Device counters include other processes; staged-copy durations are host timings,
not GPU events. Original-count verification was not repeated in this profile.

Runtime source was stable throughout the run. The four core IO modules matched
the preceding routine CUDA trials; unrelated SSB changes account for the new
whole-source fingerprint:
`3a2311a52c3763debf8333e567ee4ddb22833179069aa64a7ba8a90bf47a07e2`.
The private experiment's `agent-handoff` folder retains the raw events, thread
timeline, storage/GPU observations, exact source snapshots, reproduction steps
and previously rejected approaches. Further optimization is deferred.

## SSD writing and flushing follow-up

After fixing redundant background report-inventory reads, one complete verified
CUDA conversion of the 2048² acquisition took **61.292 s**. All 85 source files
had zero measured cache residency initially, with a fresh frame index. The
38.333 GB QEM retained every original count and the same authenticated envelope.

| Latest full CUDA conversion | Time (s) |
|---|---:|
| Preparation: source hashes, HDF5 loading, GPU encoding and indexes | 21.687 |
| QEM saving, including file-data fsync | 22.804 |
| Full original-count verification, including QEM reopen | 16.669 |
| Complete conversion, measured directly | **61.292** |

The final fsync took 4.342 s within saving, draining 8.024 GB to the SSD; QEM
reopen took 4.731 s within verification. All **154,618,822,656 original values
matched exactly**, and source hashes, body hashes and the complete authenticated
envelope matched the earlier controls. The task-created QEM was removed only
after verification and evidence archival. Canonical scientific code was unchanged.

The save stage recorded 128.778 MB of device reads, versus multi-GB background
reads in the affected earlier trials. The observer saw 117.072 MB from the report
server and no large inventory scans. This new 22.804 s save is below the earlier
28–59 s observations, but one run does not isolate the improvement attributable
to the background fix or establish a hardware minimum. The unchanged writer
still averages approximately 1.68 GB/s including checksumming and file fsync.
The established 15-row backend comparison below retains its original results.

A private CUDA writer with four owned 64 MiB buffers retained the same encoded
body, SHA-256 checksums, file-data fsync and atomic publication. It was tested
against the current writer using one resident acquisition per comparison.
**The parallel writer was slower and was rejected.** All times include fsync.

| Scan | Current writer before (s) | Four write workers (s) | Current writer after (s) |
|---|---:|---:|---:|
| 1024² | 4.440 | 5.117 | 7.058 |
| 2048² | 40.957 | 54.310 | 45.070 |

The closing 1024² control and first 2048² control had 4.172 and 2.766 GB of
unrelated device reads. The largest candidate and closing control were nearly
read-quiet, with only 16.0 and 23.8 MB read, yet the candidate was 20.5% slower.
Its shorter final fsync, 6.603 versus 13.242 s, was outweighed by a longer
write pipeline, 47.684 versus 31.813 s. More writer threads were observed waiting
inside ext4; the exact lock was not traced.

The long fsync tail includes **writing buffered data to the SSD**. During the
13.242 s closing-control fsync, the device wrote 8.222 GB while dirty pages fell
from 8.130 GB to approximately 1 MB. Device flush commands accounted for only
110 ms. These counters describe the whole device, but the near-zero concurrent
reads and matching dirty-data decrease support data writeback as the main tail.
They do not establish a theoretical device limit.

All **193,273,528,320 original values** across both candidate acquisitions matched;
all six saved bodies, complete headers and source identities were authenticated.
Four controlled empty-cache reloads retained exact indexed products. The private
candidate fingerprint is
`e08adb6b03c8b67f5427425f07ee8539cfbfe26fc5a9c7e62ef753d71a9a7e11`.

A separate full control conversion of 2048² took **70.399 s**: preparation
21.758 s, saving 30.354 s, and exhaustive verification 18.160 s. Verification
includes the 4.909 s QEM reopen. Every one of its **154,618,822,656 original
values** matched, with unchanged body and source hashes. Source-cache residency
was zero initially; 5.839 GB of unrelated device reads occurred during saving.
This is a new control measurement, not a writer speedup. Its measured source
fingerprint is `d0baea3acdbcbc4cf559bc8552735452a9753e1fb2bba605aabbb909ca934217`.

A second private experiment overlapped the final file fsync with exhaustive
verification after releasing the original GPU resident. Publication required
both operations to succeed. It hid 9.033 s of flushing but did **not** establish
a complete-conversion improvement:

| Complete 2048² conversion | Current before | Flush overlap | Current after |
|---|---:|---:|---:|
| Total (s) | 68.423 | 78.363 | 78.704 |
| File fsync (s), included in total | 8.095 | 9.139 | 9.864 |
| Verification including QEM reopen (s) | 16.638 | 22.560 | 18.172 |

All three runs started with zero measured source-cache residency and fresh frame
indexes. All **154,618,822,656 original values per run** matched exactly, and
all authenticated envelopes were identical. The candidate's internal save return
preceded fsync completion, so it is not a public durable-save timing. The
candidate and closing control experienced competing SSD reads; the overlap
candidate was already slower before its fsync began. The production writer
remains synchronous.

Process-level observation identified two background report-inventory readers
during the closing control. Their sampled reads totaled 14.182 GB. A catalog
change invalidated a shared inventory cache, causing unchanged report files to
be hashed again; concurrent callers could duplicate that work. This explains
the observed later read burst, not every earlier variation. It is separate from
GPU compute and from the approximately 8 GB of buffered QEM output drained by
the final fsync.

The receiver's inventory now caches each fallback checksum against its canonical
path, device, inode, size, modification time and change time. Concurrent callers
share a lock; changed files are fully rehashed with identity checks before and
after reading. Publication rules and stored catalog checksums are unchanged.
Fourteen portable tests passed. On the real catalog, the initial cache build
read 14.598 GB through the process and took 6.083 s. The next call took
**0.108 s with zero report-body opens**, reading only 10.774 MB of catalog/cache
metadata through the process. Both returned exactly the previous inventory.
This is an implemented reduction in redundant background IO, separate from the
unchanged QEM writer.

## Conversion follow-up

An accepted Python conversion change overlaps complete source-file SHA-256 with
HDF5 loading, waits for all ordered fingerprints before saving, and cancels
abandoned hashing. GPU encoding, file-data fsync, publication and exhaustive
verification are unchanged. This later screen used candidate fingerprint
`eb7d42dd58f3d5195b7ca92f4beee65e69f6e1a798b265222e08ad7dc4b67ffd`
and `convert.py` SHA-256 prefix `855e4ceff1e8`, in the same private experiment.

Acquisition-10 (2048² × 192², uint16) was tested on the devices above, once per
arm in **baseline, candidate, closing-baseline order**. Each arm used a fresh
process, an empty frame-index cache and zero measured source-cache residency.
MPS complete conversion took **48.692, 41.135 and 52.212 s**, establishing a
complete-workflow improvement in this screen. Preparation before saving took
22.977, 18.019 and 29.045 s; it includes HDF5 loading, source SHA and metadata.

CUDA preparation improved from 28.108 and 28.690 s to **21.970 s**, a 6.1–6.7 s
reduction. Complete conversion took 73.546, 92.657 and 110.569 s; the unchanged
save stage took 28.174, 52.208 and 58.794 s. The closing baseline also recorded
approximately 12 GB of unrelated device reads. **No CUDA end-to-end improvement
is established** by these variable storage conditions.

CUDA preparation physically read approximately 109 GB in every arm: this
overlaps work rather than eliminating a source pass. MPS preparation reads
were 109.411, 71.473 and 109.323 GB, consistent with source-cache reuse during
overlap. Every trial compared **154,618,822,656 original values with zero
differences**; complete authenticated headers, body checksums and ordered source
hashes matched within each backend. The archived 15-row comparison and its
figure retain their original measurements and source fingerprint.

The overlap candidate repeated both physical Python acceptance gates: every
test status, coverage cell and failure message matched the baseline results
below. Focused checks passed 69 CUDA tests and 70 MPS tests, with only the
opposite-backend lifetime checks skipped; seven portable orchestration checks
also passed. The broader gates retain their existing failures and blocked cells.

## Complete workflows

All times are seconds. Every detector is 192 × 192. HDF5 and repeated QEM
columns use the statistics defined below; conversion is one complete trial.

| Acquisition | Scan | Backend | HDF5 first (s) | HDF5 repeated (s) | Verified conversion (s) | QEM empty cache (s) | QEM repeated (s) | Device tested | Date tested |
|---|---|---|---|---|---|---|---|---|---|
| acquisition-03 | 256² | CUDA | 0.864 | 0.240 | 1.310 | 0.112 | 0.057 | RTX PRO 6000, 96 GB | 2026-10-23 |
| acquisition-03 | 256² | Python MPS | 0.732 | 0.218 | 0.907 | 0.052 | 0.036 | M5 Max, 128 GB | 2026-10-23 |
| acquisition-03 | 256² | Swift/Metal | 0.957 | 0.224 | 1.344 | 0.056 | 0.037 | M5 Max, 128 GB | 2026-10-23 |
| acquisition-01 | 512² | CUDA | 1.338 | 0.487 | 3.050 | 0.360 | 0.183 | RTX PRO 6000, 96 GB | 2026-10-23 |
| acquisition-01 | 512² | Python MPS | 1.391 | 0.723 | 2.332 | 0.170 | 0.084 | M5 Max, 128 GB | 2026-10-23 |
| acquisition-01 | 512² | Swift/Metal | 1.075 | 0.731 | 2.197 | 0.175 | 0.075 | M5 Max, 128 GB | 2026-10-23 |
| acquisition-04 | 512² | CUDA | 1.492 | 0.610 | 3.218 | 0.382 | 0.185 | RTX PRO 6000, 96 GB | 2026-10-23 |
| acquisition-04 | 512² | Python MPS | 1.434 | 0.732 | 2.227 | 0.199 | 0.095 | M5 Max, 128 GB | 2026-10-23 |
| acquisition-04 | 512² | Swift/Metal | 1.123 | 0.743 | 2.311 | 0.191 | 0.075 | M5 Max, 128 GB | 2026-10-23 |
| acquisition-08 | 1024² | CUDA | 4.964 | 1.674 | 9.932 | 1.438 | 0.723 | RTX PRO 6000, 96 GB | 2026-10-23 |
| acquisition-08 | 1024² | Python MPS | 4.618 | 2.679 | 7.399 | 0.743 | 0.362 | M5 Max, 128 GB | 2026-10-23 |
| acquisition-08 | 1024² | Swift/Metal | 3.922 | 2.705 | 7.393 | 0.744 | 0.275 | M5 Max, 128 GB | 2026-10-23 |
| acquisition-10 | 2048² | CUDA | 19.233 | 10.242 | 109.640 | 7.790 | 4.135† | RTX PRO 6000, 96 GB | 2026-10-23 |
| acquisition-10 | 2048² | Python MPS | 13.844 | 12.137 | 40.769 | 2.929 | 1.291 | M5 Max, 128 GB | 2026-10-23 |
| acquisition-10 | 2048² | Swift/Metal | 15.905 | 11.290 | 38.201 | 2.907 | 1.163 | M5 Max, 128 GB | 2026-10-23 |

† The largest repeated CUDA QEM measurement is partially cached. All other
repeated rows were fully cached by the recorded residency and physical-read
checks. The empty-cache column contains separately measured cold loads.

HDF5 reports the first observation and median of observations 2–3. Its OS cache
is uncontrolled. Python starts with an empty per-acquisition frame index and
shares a process; native uses new processes with catalog reuse. Native readiness
includes catalog preparation, source opening, decoding, encoding and indexing.

QEM empty-cache results are medians of three additional per-file eviction
trials. A cold label requires both measured file-page residency and physical
reads covering the file. Repeated results are medians of observations 2–3 after
conversion and verification; cache fractions and physical reads remain in the
evidence. Repetition alone does not establish a fully cached input.

Python conversion is one complete measured workflow: metadata/provenance hashing,
HDF5 load, QEM write with file-data fsync and atomic publication, and exhaustive
source comparison.
Native conversion measures its own load/save/reopen/product checks followed by
MPS comparison of every original detector count; it is a hybrid verification
path with a different metadata and checking workload. Native conversion may
reuse authenticated source hashes from its persistent validation catalog when
all file identity snapshots still match; Python hashes all source files during
each conversion. A fresh native HDF5 catalog hashes the master and all shards.
The conversion total is measured directly, not reconstructed by adding stages
that can overlap.

Save synchronization is not yet normalized across operating systems. The Python
writer uses `os.fsync` on both platforms, but [Apple documents a stronger
`F_FULLFSYNC` operation for flushing the drive's own buffered data](https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/fsync.2.html),
whereas [Linux `fsync` includes writing through or flushing the disk cache](https://man7.org/linux/man-pages/man2/fsync.2.html).
Native saving uses `FileHandle.synchronize()`; its underlying flush operation
was not traced. The recorded 4.502 s MPS and 70.270 s CUDA save stages therefore
do not establish a comparison with matched power-loss durability. No paired
Mac `F_FULLFSYNC` control was measured, so its contribution to the timing gap
is unknown. Full original-count verification remains exact; it checks the
saved contents, not behavior after a power loss.

For the largest CUDA case, the 109.640 s conversion contains 10.617 s for HDF5
loading, GPU decode, ANS encoding and index construction; 70.270 s for saving;
19.272 s for exhaustive verification; and 9.482 s for metadata/provenance and
other workflow work. **GPU compression is already complete before saving**:
the save stage writes the approximately 38.333 GB compressed representation.
The 19.272 s verification stage includes a 5.882 s QEM reopen; adding that reopen
again would double-count it. The writer follow-up below targets saving, not a
70-second GPU compression kernel.

## Accepted implementation changes

- CUDA reads two compressed HDF5 batches ahead, consumes them in source order,
  and drains pending work before closing a failed load. One-time timing
  accounting is protected against concurrent preparation. The current batch
  is the only one decoded on the GPU.
- CUDA exhaustive verification uses an adaptive 128–512 MiB window in the
  resident device context. The extra capacity amortizes dispatch while
  retaining all original-count and decoder-error checks.
- Complete source-file SHA-256 hashing reuses at most sixteen 8 MiB buffers,
  preserves the ordered provenance list, and avoids repeated block allocation.
  On the Mac, matched empty-cache controls improved from 6.015/5.939 to
  4.104 s for 54.595 GB, with identical ordered source hashes.
- MPS QEM loading uses up to sixteen read/hash workers. Stable resident views
  are hashed in parallel while saving writes sequentially; the CUDA streaming
  generator retains its bounded producer-lifetime contract.
- MPS HDF5 encoding records the exact range in a four-byte transient footer
  using a maximum already computed by the encoder. This removes a separate
  full range pass. The flag is reset on every append, published only after
  successful append, and preserves exact existing storage narrowing. Tests
  cover 255, 256 and 65535, constant streams, late outliers and fresh owners.
- Native QEM loading reads and authenticates directly into the final owned
  shared Metal buffers, eliminating the former full-file copy after hashing.
  Alignment padding and the final header identity remain authenticated.
- Native HDF5 uses bounded 8,192-frame windows instead of forcing 512 frames
  for spatial indexing. Both Apple paths enforce codec offset and device
  buffer limits for larger detectors.
- Native saving overlaps hash/write work using at most eight owned 64 MiB
  blocks plus one producer block. Per-chunk autorelease pools bound private
  Metal readback lifetime. One next private chunk, at most 128 MiB, may be
  copied while the current chunk is staged and written. Failure/cancellation
  joins pending work before releasing buffers or closing the destination.

Controlled native private-buffer saving improved 7.095/7.848 to 6.124 s when
adding bounded readback overlap. The earlier lifetime correction reduced peak
Metal allocation from 75.505 to 38.427 GB; prefetch used 38.502 GB. These are
stage experiments, not substitutes for the complete conversion table.

Larger MPS loading/verification batches, 32 MPS QEM readers, extra coalesced
Python write buffers, CUDA direct IO and extra CUDA reader threads were rejected
when they failed to show a reliable benefit or added memory without speed.
A later fused CUDA saved-ANS decode/comparison verifier passed 28 parity checks
but was rejected:
acquisition-08 took 3.039 s against baseline observations of 2.969 and 2.893 s,
and acquisition-10 took 12.420 s against 12.225 and 12.021 s.

## Measured IO limits

The final complete 38.333 GB CUDA QEM empty-cache median is **7.790 s**.
Earlier matched load controls were 6.425–6.439 s, with an independent
pinned-memory read-only control of 6.311–6.431 s. Later complete public loads
took 6.712 and 6.393 s; a later read-only control took 6.113 s. These separate
runs demonstrate storage-path variability. They do not establish why
the final run was slower, and the earlier 6.43 s result does not replace the
final table's 7.79 s. No theoretical minimum or particular competing workload
is proven. The existing sixteen-worker read/hash/upload loader was retained
after other worker counts and direct IO failed to improve it.

On the Mac, independent empty-cache read-only controls took 2.885–2.899 s for
the same body size. Native direct-to-final-buffer loading took 2.946–2.974 s,
with zero cached pages before each load and approximately 38.333 GB physically
read. The final native empty-cache median is 2.907 s, and the final MPS median is
2.929 s. These attained read controls are consistent with a substantial storage
contribution to the cross-computer timing gap, but do not isolate its cause.
Unified memory also avoids a separate host-to-device copy. Faster GPU
arithmetic cannot remove the bytes that must reach memory.

Overlapped read, hash and upload limits must be compared using their maximum,
not added. Small cached reads and small write bursts cannot establish whole-file
throughput. A separate durable 38.333 GB sequential-write control on the CUDA
output drive took 62.112 s, including 12.960 s in fsync; its early burst was much
faster than its sustained writeback tail. Conversion and loading have different
storage limits.

## Original CUDA save profile

Compression already runs on CUDA. One recorded large HDF5 trial measured
1.188 s of GPU ANS encoding and 0.422 s of GPU spatial indexing using CUDA
events; the complete load also reads and decodes the HDF5 input. Saving exports
that encoded representation without expanding or recompressing detector values.

A separate instrumented save isolated the remaining work. Both destinations
wrote approximately 38.333 GB, flushed it before publication, and passed
exhaustive comparison of all 154,618,822,656 original values.

| Save stage or control (s) | 8 TB output SSD | Second 1 TB SSD |
|---|---:|---:|
| Complete QEM save | 53.170 | 55.499 |
| GPU-to-host transfer, active time | 2.064 | 2.155 |
| SHA-256, active time | 9.880 | 10.253 |
| File writes, active time | 26.670 | 34.485 |
| Final fsync | 24.815 | 19.631 |
| Separate durable write-only control | 84.848 | 57.341 |

Transfers, hashing and writes overlap: these active intervals must not all be
added. Writes plus final fsync account for approximately 97% of either complete
save. The write-only controls perform no GPU transfer or QEM hashing and remain
slow. Neither a larger GPU staging ring nor a replacement compressor is
supported as a solution to this measured bottleneck. The second SSD did not
provide a reliable speedup.

These controls followed removal of 52.644 GB of task-created temporary files;
the original HDF5 files were preserved. Free space and drive/cache state differ
from the canonical 70.270 s save. Consequently, 53.170 s is a diagnostic result,
not a claimed software speedup or a replacement for the conversion table.

Both tested ext4 filesystems use PCIe 4.0 ×4 links and lack continuous discard.
The 8 TB output filesystem reported approximately 92% use. A later audit found
1.067 TB of free filesystem blocks, of which 667 GB was available to ordinary
users; the difference is reserved space. These figures do not measure the SSD's
internal free or erased blocks. Periodic TRIM last ran on
October 19 and was next scheduled for October 26. Unreclaimed deleted blocks
are a possible contributor, but a maintenance-and-retest comparison has not
been performed and remains pending operating-system administrator access.
Later read-only UDisks2 access returned cached NVMe health snapshots that were
31–32 seconds old. Both drives reported zero media errors, no critical warning,
and zero accumulated warning/critical-temperature time. A requested refresh was
not authorized; fresh per-trial throttling counters were not acquired. These
cached snapshots do not prove that throttling never occurred or establish a
particular SSD cache state.

The numerical implementation remains the qualified version. A subsequent
`SaveResult` docstring correction clarifies that QEM save timing includes
flushing and publication; its executable Python AST is unchanged.

## Further write-path screening

A second screen used synthetic 38,332,317,853-byte files, with file-data fsync
inside the write timing and exact full-byte readback afterward. Direct writes
without preallocation took 18.767 s; the following buffered control took
26.906 s. Device reads differed: 1.405 MB versus 3.377 GB, so this pair does not
isolate the writer's effect. All eight raw controls passed exact byte comparison.
These are file-write diagnostics, not QEM conversions or detector-count checks.

The private candidate did not establish a repeatable real QEM save improvement.
For acquisition-08, buffered, candidate and closing buffered saves took
3.973, 3.916 and 8.732 s; the small initial difference is not a reliable gain.
For acquisition-10, the candidate took 36.384 s versus 29.591 s buffered, with
17.957 MB and 14.750 MB of device reads respectively. The closing buffered run
took 41.058 s with 3.395 GB of concurrent reads. The actual largest QEM was
38,332,329,252 bytes; its provenance header accounts for the small size difference
from the synthetic control. Four direct-write workers accumulated 132.776 s in
write calls, implying at least 33.194 s elapsed for that work within the
36.384 s save. This is write-path time, not repeated GPU compression.

Both acquisitions passed exhaustive comparison of **193,273,528,320 original
counts with zero differences** in total. Complete headers matched across each
three-save comparison, and all six outputs passed body authentication.
Acquisition-10 cold reopens took 6.268 s buffered and 6.263 s candidate, one
observation each, with zero initial cache residency, file-covering physical
reads and exact indexed products. The private candidate's 34 focused tests
passed; these are separate from the accepted portable checks below. All six
QEM outputs were removed after checking; original source identities were
unchanged.

The direct writer was **not adopted**. The buffered default, canonical conversion
table and figure remain unchanged. No new complete-conversion or broad
acceptance-gate run was claimed for the rejected writer; the raw-write advantage
did not establish an actual QEM save gain.

## Correctness and scope

Each backend compared **215,016,800,256 original detector counts with zero
differences** across the five acquisitions, including flagged pixels. Source
file hashes matched between the two computers; native tests used the same local
masters and shards as MPS. All 45 empty-cache trials had zero measured cached
pages before loading and physical reads covering the file. No original data
was deleted; disposable canonical benchmark QEM outputs were removed after
verification.

The common-source physical acceptance gate is broader than this five-input
benchmark and is **not wholly passing**:

| Gate | Passed | Blocked | Failed | Interpretation |
|---|---:|---:|---:|---|
| CUDA | 145 | 23 | 5 | The five FP8 failures reproduce on the baseline. |
| Python MPS | 143 | 25 | 6 | All six failures reproduce on the baseline. |
| Native Swift/Metal | 9 | 0 | 0 | All selected native cells pass; other format cells are not covered. |
| Portable focused checks | 33 | 0 | 0 | Backend-independent IO, provenance and benchmark checks. |

Blocked or uncovered cells do not establish support. These results qualify the
measured five-acquisition workflows, not universal format coverage, an installed
release, or an application UI. Native targeted tests also passed codec-offset,
Metal-buffer and memory-admission bounds, changed-header rejection, first/middle/
last checksum-block and padding corruption, release/recovery, and cancellation
while a private-buffer readback was pending. Complete detector products and
original counts remained exact.

The accepted changes are ordinary loader defaults in the checked source tree.
Actual normal-environment smoke tests passed original HDF5 loading and QEM
saving/reopening on CUDA and MPS. Matching native release binaries were installed
with previous binaries preserved; both original HDF5 and QEM loads passed from
the canonical executable paths. These local deployment checks do not imply a
published package release.

The additional direct-write candidate passed correctness checks but did not
establish a repeatable save improvement, so the buffered writer remains the
default. Storage maintenance and its retest remain a separate path requiring
operating-system administrator access. Diagnostic timings retain their own
storage conditions and do not replace the frozen conversion table.

The benchmark runner records process physical IO, memory footprint, QEM cache
residency, source-file identities and whether the measured source tree remained
unchanged. Per-file eviction is advisory and occurs outside the load clock.
It does not flush a system-wide cache or mutate the data. Failed eviction is
reported as such rather than silently relabelled as cold.

Use a new private output directory and the checked five-acquisition selection:

```bash
python scripts/benchmark_io_scale.py SOURCE_FOLDER --backend cuda \
  --selection-manifest PRIVATE_SELECTION.json --fresh-index \
  --reps 3 --cold-reloads 3 --output-dir NEW_PRIVATE_DIRECTORY
```

Run separately with `mps` and `native-metal`; the latter also needs the matching
release `--native-binary`. Keep raw reports outside the public repository.
Older comparison pages retain their original sources and cache conditions.
