# CUDA QEM loading and file-cache recheck, 2026-10-23

The 2048 × 2048 × 192 × 192 acquisition still takes about ten seconds to
restore from its **38.333 GB QEM** into encoded CUDA residency. A separate
copy-only control moves the same byte count into a full-size GPU destination
in **1.333 s**. The complete loader includes storage reads and authentication;
the transfer control excludes both. A one-second complete load was not observed.

## Setup and source identity

Private experiment: `20261023-cuda-qem-cache-recheck`. Only acquisition-10 was
tested, at original uint16 precision, with no correction, crop or binning.
The canonical benchmark recreated the QEM and compared all **154,618,822,656**
source values exactly, including flagged detector pixels: zero differences.
Temporary QEM storage was removed after the controls.

Hardware: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 96 GB, with a
generation-5 eight-lane GPU link; Intel Core Ultra 7 265K; approximately 125 GiB
host RAM. The file was on an ext4 WD_BLACK SN850X 8 TB NVMe, whose negotiated
link was 16 GT/s ×4. CUDA runtime 12.9, CuPy 14.2.0, Python 3.14.6.

The existing conversion/verification runner and its interval profiler were
reused with a private cache-control wrapper. The measured source snapshot
fingerprint, which includes scripts and tests, is
`25bc5ad212e8476b2c981a218182fb886bc9a48c6a4d2d4e18ee81ee029e08e3`.
The acquisition master identity matches the original suite. Loader and envelope
source hashes match the earlier profile and current checkout exactly:

- `io/qem.py`: `ca2971acd642f687d4609bb07ddc027e65fc8c3ac1264a77a474bf0c290996cb`
- `formats/qem/snapshot.py`: `71df8125b3dc3d051d3ccea0b23575d8b3d1659dbcfe5531ce32401530b14faa`

No production loader or codec implementation changed. These source measurements
do not qualify an installed release.

## Measurements

Three uninstrumented QEM reloads took **9.876, 17.055 and 10.214 s**. The later
instrumented controls below each retain header and full-body SHA-256 checks,
then synchronize the GPU before stopping the load timer. `mincore` observes
file-page residency without reading the file. Process IO counters measure
storage reads during each complete load.

| QEM load condition | Load (s) | File pages resident before (%) | Storage read during load (GB) | Read-call active time (s) | SHA-256 active time (s) |
|---|---:|---:|---:|---:|---:|
| Ordinary repeated load | 10.363 | 43.9 | 37.689 | 10.135 | 6.560 |
| Disposable QEM file pages evicted | 10.344 | 0.0 | 38.333 | 9.996 | 6.345 |
| After sequential pre-read 1 | 10.452 | 8.6 | 38.102 | 10.257 | 6.369 |
| After sequential pre-read 2 | 10.196 | 62.6 | 37.320 | 9.833 | 6.370 |

Only the disposable QEM file received `POSIX_FADV_DONTNEED`; the measured
resident-page count then reached zero. No global cache was evicted. This controls
OS file pages, not the storage controller's caches. The sequential pre-reads
took 14.023 and 14.677 s and are excluded from the subsequent loading times.

Read and checksum intervals overlap across four workers; the table reports
the union of intervals for each operation. They must not be added to form
elapsed load time. Read-call intervals include waiting and copying into pinned
staging memory. Host upload-call durations are not GPU transfer durations.

The copy-only control used one repeated 64 MiB pinned host buffer and a full
38.333 GB GPU destination. Three CUDA-event measurements were **1.3330,
1.3331 and 1.3329 s**, approximately **28.76 GB/s**. It excludes file reads,
checksum verification and allocation timing, and does not constitute a QEM
loading result or a retained-host-file benchmark.

## Interpretation and limits

The file-page snapshot does not establish that those pages survive until the
loader reaches them. Every observed load fetched 97–100% of the QEM size from
storage, including after pre-reading. This is strong evidence of a storage-read
limit in the observed pipeline. It is not a fully RAM-cached QEM measurement.
The precise cause of cache churn was not isolated.

Host anonymous memory was about 62.5–64.0 GiB before and during the controls;
background disk verification and desktop services remained active. The GPU was
idle before and after the run, but telemetry recorded desktop bursts and two
brief other Python GPU processes reaching 7% and 5% SM use. Their identities
were not retained after they exited. This was not an exclusive idle-system run;
no other processes were stopped. The cooperative GPU lock was held by the
benchmark throughout. A separate SSH-session hierarchy check found no finite
memory.max or memory.high limit; it is not a capture of the completed process's
now-removed session scope.

QEM avoids re-encoding the acquisition, but it still carries tens of gigabytes
that must be read and authenticated. Improving this loading path requires
addressing storage/cache behavior and the read/checksum pipeline; GPU compute
utilization alone does not measure that bottleneck. The original
[five-acquisition table](2026-10-23-default-five-io.md) retains its matched
10.47 s repeated median. These cache controls are separate measurements, not
an optimization or a replacement selected from the fastest trial.
