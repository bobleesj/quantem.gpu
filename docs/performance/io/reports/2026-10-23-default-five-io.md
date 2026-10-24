# Five-acquisition Disk–GPU loading comparison, 2026-10-23

For the latest qualified rerun, see [HDF5 and QEM loading limits](2026-10-23-io-loading-limits.md).
The measurements below retain their original implementation and cache conditions;
the newer results do not relabel this historical table.

The current default set is acquisition-03, acquisition-01, acquisition-04,
acquisition-08 and acquisition-10. All have 192 × 192 detector pixels. The two
96 × 96 acquisitions are excluded from future default and full-scale runs and
active examples, at the user's request. Historical measurements remain archived.

This report selects five acquisitions from the completed
[six-acquisition experiment](2026-10-23-default-six-io.md); it does not claim a
new timing run. HDF5 uses **bitshuffle + LZ4**; QEM uses **ANS** with decoding
models, stream offsets, spatial indexes, metadata and checksums.
QEM serves as both an **on-disk storage format** and a **GPU-resident encoded
format**; loading restores the encoded representation without expanding the
full dense tensor.

MPS and Swift/Metal ran on a **MacBook Pro with Apple M5 Max and 128 GB unified
memory**. CUDA ran on an **NVIDIA RTX PRO 6000 Blackwell Workstation Edition,
96 GB**. The machines and storage differ, so these are workflow measurements,
not an isolated GPU ranking.

Private selection/report experiment: `20261023-five-default-io`. Timing
experiment: `20261023-six-default-io`. Byte-layout experiment:
`20261023-qem-layout-cuda-profile`. Measured scientific source fingerprint:
`2873ac0ffac23c49c89a7d5311ffbcc8836ead6af640acf6df10f5a292290df8`.
Base revision: `d2db0bdb4c9859230d3e9ba2f6b23fcfae8eb0f1`.

## Disk–GPU loading time and verified conversion

| Acquisition | Scan × detector | Backend | HDF5 first / repeated (s) | QEM conversion + full verification (s) | QEM first / repeated (s) |
|---|---|---|---:|---:|---:|
| acquisition-03 | 256² × 192² | CUDA | 4.29 / 0.21 | 3.35 | 0.08 / 0.08 |
| acquisition-03 | 256² × 192² | MPS | 2.49 / 0.24 | 1.04 | 0.03 / 0.03 |
| acquisition-03 | 256² × 192² | Swift/Metal | 0.58 / 0.38 | 1.76* | 0.07 / 0.07 |
| acquisition-01 | 512² × 192² | CUDA | 3.72 / 0.44 | 4.03 | 0.28 / 0.26 |
| acquisition-01 | 512² × 192² | MPS | 3.12 / 0.77 | 2.89 | 0.12 / 0.12 |
| acquisition-01 | 512² × 192² | Swift/Metal | 1.63 / 1.41 | 4.80* | 0.27 / 0.27 |
| acquisition-04 | 512² × 192² | CUDA | 3.85 / 0.55 | 4.03 | 0.29 / 0.29 |
| acquisition-04 | 512² × 192² | MPS | 3.17 / 0.78 | 3.08 | 0.15 / 0.14 |
| acquisition-04 | 512² × 192² | Swift/Metal | 1.91 / 1.53 | 4.64* | 0.29 / 0.29 |
| acquisition-08 | 1024² × 192² | CUDA | 7.40 / 1.48 | 13.66 | 1.19 / 1.12 |
| acquisition-08 | 1024² × 192² | MPS | 7.51 / 3.68 | 10.73 | 0.52 / 0.52 |
| acquisition-08 | 1024² × 192² | Swift/Metal | 6.82 / 6.32 | 18.24* | 1.08 / 1.09 |
| acquisition-10 | 2048² × 192² | CUDA | 22.94 / 11.96 | 100.24 | 10.42 / 10.47 |
| acquisition-10 | 2048² × 192² | MPS | 15.44 / 12.11 | 52.05 | 3.13 / 2.20 |
| acquisition-10 | 2048² × 192² | Swift/Metal | 30.46 / 24.28 | 100.98* | 10.81 / 4.54 |

Repeated loading is the median of observations 2–3; conversion has one
observation. OS caches were uncontrolled and were not evicted. These numbers
must not be interpreted as cold-disk or tail-latency measurements.

Loading ends at authenticated, encoded GPU residency with spatial indexes.
The full logical tensor is not expanded. HDF5 starts from an empty frame
index for its first observation; Python repetitions share a process and native
repetitions use fresh processes with catalog reuse. QEM reloads follow completed
conversion and verification, which already accessed the file.

Conversion includes writing, metadata/source hashing and exhaustive source
comparison. * Swift/Metal includes native round-trip/product checks followed by
full MPS verification; it is a hybrid workflow. Every workflow checks all
215,016,800,256 original values across these five acquisitions, including flagged
detector pixels, with zero differences. No crop, binning, precision change,
correction or relaxed tolerance was applied.

## Storage and CUDA interpretation

Complete HDF5 files total 75.203 GB; CUDA-produced QEM files total 52.644 GB,
**30.0% smaller**. The 2048² input is 54.595 GB HDF5 versus 38.333 GB QEM.
Its spatial index occupies 2.409 GB (6.3% of QEM). Subtracting that index gives
a 35.924 GB size estimate; an index-free file or load was not benchmarked.

The [CUDA profile](2026-10-23-qem-layout-cuda-profile.md) found no competing
sustained scientific GPU compute, while desktop graphics activity was present.
An active file-verification job read other physical disks. A subsequent 12.47 s
QEM profile fetched 37.9 GB from storage; read-call intervals covered 12.34 s
and overlapped checksum work. A separate 64 MiB pinned-buffer transfer control
reached 28.76 GB/s. These observations point to storage reads and the read/hash
pipeline as the main loading limit; the control is not a whole-acquisition
transfer measurement. Later variable profiles do not replace the original
matched loading medians above.

Use `scripts/benchmark_io_scale.py SOURCE_FOLDER --backend cuda
--selection-manifest PRIVATE_SELECTION.json --fresh-index
--output-dir NEW_PRIVATE_DIRECTORY --reps 3` with the current five-input
selection manifest. Use `mps` or `native-metal` for the other backends; the native
path also needs its release binary. Historical experiment manifests are frozen
records, not the current selection. This source-checkout evidence does not
qualify an installed release.

The [subsequent cache-controlled CUDA recheck](2026-10-23-cuda-qem-cache-recheck.md)
confirms roughly ten-second full loads, 97–100% file-size storage reads, and a
separate 1.33 s copy-only control. It retains full count and checksum checks.

The later [CUDA loader fix](2026-10-23-cuda-qem-loader-fix.md) changes the
implementation and measures all five inputs again. Its largest repeated QEM
load is 5.10 s; controlled empty-cache and half-cache observations are 6.59 s
and 4.78 s. The table above remains the original matched-backend record.

## Comparison with the historical 66-acquisition load

Recovered records confirm **66 complete 512² × 192² acquisitions**, totaling
**98.733 GB** in a prepared tANS archive: **17.71 s on one SSD** and **8.04 s
on two SSDs**. The two-SSD loader used direct I/O and performed no per-record
hashing during loading. A further **1.15 s** checked 24 scientific-output
arrays; that was not exhaustive comparison of all original detector counts.
The one-SSD run reused its context and resident owners, while the two-SSD run
started a fresh process. These boundaries and the prepared representation
differ from the current integer QEM container.

The present 2048² × 192² QEM is **38.333 GB**. With every block hash retained,
the corrected one-SSD empty-cache load improved **10.61 → 6.59 s**, about
**5.8 GB/s**, comparable to the historical one-SSD **5.6 GB/s**. Half-cached
loading improved **10.44 → 4.78 s**. The **5.10 s** ordinary repeated load
started approximately 36% cached and read 24.7 GB from storage. A separate
two-SSD, empty-cache diagnostic reached **4.65 s**; it is not a supported
public multi-disk QEM API and does not replace the ordinary loading result.

Current QEM load times include header and block authentication and GPU
synchronization; exhaustive original-count comparisons run separately. All
215,016,800,256 values across the five acquisitions matched exactly. The
observed bottleneck was the storage/read/hash pipeline, including eviction of
cached blocks before use. The fix overlaps reads, hashing and uploads and
prioritizes cached blocks. Monitoring found desktop bursts but no sustained
competing scientific GPU compute during the controlled runs.

Compare compressed bytes, physical disks, cache state and verification work
alongside elapsed time. A 2048² scan has the scan points of sixteen 512² scans;
dataset count alone does not measure the loading work. Different datasets and
representations also prevent attributing the size difference to the codec alone.
The compact private two-table benchmark retains essential methodology notes;
the historical comparison and CUDA diagnosis remain in this detailed report.
Original records, source provenance and the matched two-SSD
diagnostic are documented in the [loader investigation](2026-10-23-cuda-qem-loader-fix.md).

## Why CUDA is 5.10 s while MPS is 2.20 s

**5.10 s is not an established theoretical minimum.** The CUDA measurement
physically read approximately 24.7 GB from its SSD while restoring the complete
38.333 GB encoded representation. At the observed empty-cache loading throughput
of about 5.8 GB/s, that disk traffic alone corresponds to roughly 4.3 s. This is
a scale estimate using measured pipeline throughput, not an independently
measured SSD ceiling or a decomposition of the overlapping stages. The separate
copy-only CUDA control takes 1.33 s for the full byte count and excludes storage,
allocation and checksums. Those times must not simply be added.

The MPS record contains 38.333 GB QEM reloads of 3.128, 2.211 and 2.197 s;
2.20 s is the median of the last two. They followed QEM creation and exhaustive
verification, with uncontrolled OS caches. The record contains no per-load
physical-read count or file-cache residency measurement, so the cached fraction
cannot be recovered. A substantial cache benefit is plausible but was not
quantified. The 17.4 GB/s effective file-size throughput is not a measured SSD
read rate.

The frozen MPS implementation reads and hashes directly into shared Metal
buffers (`MTLResourceStorageModeShared`), using eight read/hash workers. Those
buffers are already GPU-accessible; it does not perform the CUDA path's
additional host-to-device transfer into separate GPU memory. Both paths retain
all block hashes. QEM reload does not expand the dense tensor on either backend.
This establishes a memory-path difference, but does not isolate how much of
the timing gap comes from storage, caching, allocation, hashing or transfer.

The existing results therefore establish the observed workflows, not that MPS
has faster cold-disk loading or that CUDA cannot improve further. A matched
cross-backend comparison still requires measured empty-cache and fully cached
conditions, physical bytes read and the same verified workload. This clarification
uses existing records and the frozen source; it introduces no new measurement.
