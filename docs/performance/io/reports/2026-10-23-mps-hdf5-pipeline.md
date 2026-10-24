# Bounded MPS HDF5 loading, 2026-10-23

A full 2048 × 2048 scan with a 192 × 192 uint16 detector now loads into
lossless count-ANS residency in 16.26–16.54 seconds on the tested Apple M5 Max.
The earlier repeated measurements were 36.11–38.33 seconds; a fresh baseline
control repeated in 34.80 seconds. The conservative matched-control improvement
is 2.12×. This measures the Python MPS API, not native Swift application paint.

## Workload and changes

Base revision: `d2db0bdb4c9859230d3e9ba2f6b23fcfae8eb0f1`, with local MPS
loader changes. Physical device: Apple M5 Max, 128 GB unified memory.
Private experiment: `20261023-mps-h5-pipeline`.

The public `io.load` call uses MPS, `dtype="native"`, no hot-pixel correction,
no crop, and no binning. All 4,194,304 frames are ingested: 288 GiB logical
uint16 counts. The unchanged lossless representation occupies 38,332,415,954
bytes, including its spatial index. Native reads still return uint16 counts.

The previous implementation prefetched compressed bytes but decoded and
processed each batch synchronously. The candidate prepares and decodes one
batch ahead while encoding and indexing the preceding batch. It reuses two
bounded decoded buffers and one encoder scratch buffer. Each GPU command
completes before its buffers are recycled; cancellation and failure drain the
pending batch and free staging. No numerical kernel, codec interval, detector
support, precision, or scientific correction policy changed.

Profiling the baseline found about 10.8 seconds per load inside the ANS encode
command wait, versus about 2.4 seconds of reported GPU execution. Reusing its
large scratch buffer removed much of that allocation/residency overhead.
Decoded-buffer reuse provided a further improvement. GPU timestamps do not
measure occupancy, and overlapping stage wall times must not be summed.

## Timings

The existing `benchmark_hdf5_load.py` runner measures public load through backend
synchronization. Validation and complete-resident hashing are outside that
interval. Residents are released between repetitions. OS file caches are
uncontrolled; none of these measurements is labelled cold-disk.

| 2048² case | First source encounter (s) | Subsequent full loads (s) |
|---|---:|---:|
| Original verified baseline | 45.43 | 38.33, 36.11 |
| Fresh baseline control | 39.55 | 34.80 |
| Final candidate | 20.88 | 16.26, 16.54 |

The final repeated median is 16.40 s, with only two repeated observations;
this is not a long-run tail-latency qualification. Baseline controls bracket
the exploratory pipeline and buffer-reuse probes before final validation.

| Scan / detector | Acquisitions | Earlier repeated range (s) | Candidate repeated range (s) |
|---|---:|---:|---:|
| 256² / 192² | 1 | 0.505–0.511 | 0.250–0.259 |
| 512² / 96² | 2 | 0.728–0.747 | 0.386–0.392 |
| 512² / 192² | 5 | 1.912–2.009 | 0.824–0.892 |
| 1024² / 192² | 2 | 8.722–9.374 | 3.278–3.374 |
| 2048² / 192² | 1 | 36.110–38.327 | 16.261–16.536 |

All 33 candidate loads passed their checks. Sampled peak Metal allocation was
40.31–40.40 GB for the final large case versus 40.31–40.32 GB for the fresh
baseline control. The 20 ms sampler can miss brief peaks; the implementation
permits one additional 576 MiB decoded staging buffer. Metal allocation after
release returned to 9,109,504 bytes. System swap was already present and OS
memory pressure was not experimentally controlled.

## Correctness and limits

Both baseline controls and all three final large loads produced the identical
SHA-256 over every encoded chunk and spatial-index buffer:
`a9e04de2c7c737c3cb3357fdfdcf33e4eb834b5f826cf07f52f64025b41e2dfb`.
This establishes complete resident equivalence to the frozen baseline, rather
than a separately decoded full-volume file oracle. Independent HDF5 comparisons
also checked the first, middle and last frame of every external data shard:
252 exact uint16 frames per large load. All source sizes and modification times,
and master/metadata hashes, were unchanged.

Focused hardware, metadata, benchmark and repository checks: **64 passed**.
The strict acquisition runner reported **145 passed, 6 failed, 23 skipped**.
All six failures reproduced on an isolated export of the unchanged base:
one uint64 SSB dtype-dispatch failure and five FP8 hot-pixel-policy failures.
The two full public collections were unavailable. The broad acceptance gate
therefore remains incomplete; these findings are not a release qualification.
CUDA and native Swift loading are separate paths and are not certified by the
MPS optimization. No release was performed.

The retained evidence includes source-file hashes, an AST fingerprint of the
MPS implementation, full timing samples, memory measurements, failure controls,
and disposable-test cleanup receipts. The retired benchmark registry is not
used; this dated record preserves the earlier results above.

## ANS QEM reopening on the internal SSD

The same full 2048 × 2048 × 192 × 192 acquisition was converted with the
canonical converter on MPS and reopened from the internal APFS SSD. The first
QEM load during conversion verification took **5.077 s**. Three subsequent
public loads took **5.618, 4.128, and 3.687 s** (median **4.128 s**), with the
resident released between runs. Timing includes integrity checking and backend
synchronization, but excludes source-value verification and reconstruction.
These are post-write/repeated observations with uncontrolled OS caches, not
cold-disk performance. Existing system swap and other desktop applications were
present; this is not an isolated tail-latency qualification.

The QEM file is **38,332,680,910 bytes** (35.70 GiB), versus 54,595,035,757 bytes
for the original HDF5 acquisition. Codec: `runtime-column-rans-spatial-v2`.
The converter uses exact uint8 integer packing for these uint16 source counts;
public native reads remain uint16. This is lossless value preservation, with
no crop, binning, clipping, or hot-pixel correction. Exhaustive independent
source comparison checked **154,618,822,656 values with zero differences**.
The three reloads also passed shape, dtype, detector-binning, and source/working
geometry checks. Recorded calibration is carried; missing convergence angle
remains unknown. No SSB result is claimed.

One-time conversion took **780.246 s** overall: HDF5 loading/encoding took
27.636 s, QEM writing including file flush/fsync took 28.759 s, and exhaustive
verification took 698.946 s (including its 5.077 s QEM load). The remaining
24.904 s includes source hashing and metadata/publication overhead. The long
verification time is not incurred by ordinary QEM reopening. Earlier 16.3–16.5 s
HDF5 loads were measured under different runtime/cache conditions; do not use
these observations as a strictly matched speedup ratio.

No QEM implementation was changed for this measurement. Source fingerprints,
raw timings, verification, and cleanup receipts are retained under private
experiment `20261023-mps-h5-pipeline`, QEM follow-up. An external-volume write
attempt was interrupted before any accepted load measurement, then removed.
All internal and external numerical benchmark scratch was removed after use.
