# QEM compression compared with HDF5, 2026-10-23

Historical six-acquisition evidence. The [current five-acquisition selection](2026-10-23-default-five-io.md) excludes 96 × 96 detector inputs from future runs and active examples; the completed measurements below are retained unchanged.

For the six default acquisitions, ANS QEM saves 29.0% of the total HDF5
storage: 76.570 GB becomes 54.360 GB. Five acquisitions become 29.8–32.4%
smaller; acquisition-06 becomes 25.5% larger. These are complete-file sizes
for the exact original counts, not a claim that ANS always compresses better.

The [subsequent index and CUDA profile](2026-10-23-qem-layout-cuda-profile.md)
now isolates the spatial-index bytes and measures storage-read/checksum
intervals. Acquisition-06 still exceeds HDF5 without its spatial index; its
ANS payload alone is larger than the HDF5 acquisition.

## Measurement and lossless coverage

This analysis uses the completed [six-acquisition comparison](2026-10-23-default-six-io.md)
and private experiment `20261023-six-default-io`. There are three backends
and six acquisitions. No conversion or timing was rerun for this analysis.
The measured source fingerprint is
`2873ac0ffac23c49c89a7d5311ffbcc8836ead6af640acf6df10f5a292290df8`,
on base revision `e8750f3e365208421675f47bbdc6314b135f7146`.

HDF5 is a container. Every linked detector dataset in these six inputs uses
bitshuffle plus LZ4, confirmed from filter ID 32008 and compression parameter 2.
QEM saves the resident ANS streams together with decoding models, offsets,
spatial indexes, metadata and block checksums. This compares these two complete
storage layouts; it does not isolate the entropy coders from their indexes or
container overhead.

HDF5 size includes the master and all unique linked detector files. QEM size
is the entire generated file, measured before temporary-file cleanup. The
table uses the MPS-produced QEM size consistently. Among the available CUDA,
MPS and Swift producers, the maximum size spread for any acquisition is below
0.064%; exact per-producer byte counts are retained in the private audit.
There is no native Swift-produced acquisition-06 because that HDF5 layout is
unsupported; its Swift loading test used an MPS-produced QEM.

Original size means the logical tensor at uint16: all scan positions and
detector pixels, including flagged pixels, with no correction, cropping or
binning. Full comparisons covered 217,432,719,360 values across the six inputs,
with zero differences for each completed conversion. The Swift workflow uses
MPS for the exhaustive source comparison, as described in the timing report.

## Compression results

GB means 1,000,000,000 bytes. Compression ratio is original uint16 bytes divided
by complete stored-file bytes; a larger ratio means a smaller file. Space saved
is `100 × (1 − QEM bytes / HDF5 bytes)`. Negative savings mean expansion.
All ratios use exact byte counts before display rounding.

| Acquisition · scan × detector | Original uint16 (GB) | HDF5 (GB) | QEM (GB) | HDF5 ratio | QEM ratio | Space saved vs HDF5 |
|---|---:|---:|---:|---:|---:|---:|
| 03 · 256² × 192² | 4.832 | 0.717 | 0.489 | 6.74× | 9.87× | 31.8% |
| 06 · 512² × 96² | 4.832 | 1.367 | 1.715 | 3.54× | 2.82× | −25.5% |
| 01 · 512² × 192² | 19.327 | 3.040 | 2.054 | 6.36× | 9.41× | 32.4% |
| 04 · 512² × 192² | 19.327 | 3.374 | 2.355 | 5.73× | 8.21× | 30.2% |
| 08 · 1024² × 192² | 77.309 | 13.476 | 9.413 | 5.74× | 8.21× | 30.2% |
| 10 · 2048² × 192² | 309.238 | 54.595 | 38.333 | 5.66× | 8.07× | 29.8% |
| **Total** | **434.865** | **76.570** | **54.360** | **5.68×** | **8.00×** | **29.0%** |

For the 2048² input, QEM is 16.262 GB smaller. HDF5 is 1.42 times the QEM
size: the 8.07× QEM compression ratio is relative to uncompressed uint16,
not relative to HDF5. Across all six, the HDF5-to-QEM size ratio is 1.41×,
weighted by actual bytes rather than an average of the six percentages.

Acquisition-06 retains less compactly in this QEM layout. Its 25.5% expansion
is measured. The initial comparison did not isolate the components; the
follow-up above measures the index contribution, while count statistics and
model selection remain unisolated. The production converter declines
this larger copy. Its reported QEM size comes from the disposable, fully
verified benchmark retry, not a change to that production policy.

## Why the loading gain differs by backend

For the 2048² input, repeated HDF5/QEM loading is 11.96/10.47 seconds on CUDA
(1.14×), 12.11/2.20 seconds on MPS (5.49×), and 24.28/4.54 seconds on
Swift/Metal (5.35×). These medians use trials two and three, with uncontrolled
OS caches, and stop at encoded GPU residency. The machines and storage differ.

QEM avoids HDF5 decoding, ANS re-encoding and spatial-index construction, but
still reads and authenticates approximately 38.3 GB for this acquisition.
CUDA then transfers the encoded arrays into device memory. Its loader already
overlaps bounded reads and SHA-256 checks with uploads using four 64 MiB
pinned staging buffers. In the repeated HDF5 trials, the recorded ANS encoding
time is about 1.20 seconds and index construction about 0.43 seconds. These
stage counters do not independently partition overlapping wall time; they do
show that skipping ANS encoding alone cannot explain an order-of-magnitude
loading improvement.

The MPS loader reads and hashes blocks directly into shared Metal buffers
using up to eight workers. Native Swift verifies a file mapping in parallel,
then copies its arrays into shared Metal buffers. These different allocation,
copying and transfer paths matter independently of compression ratio.

The original QEM load measurements do not separately time storage reads,
SHA-256, allocation, host copies and device transfers. Those costs are
confirmed by source inspection, but assigning the CUDA 10.47 seconds to one
of them requires stage profiling under controlled cache conditions. The
observed throughput alone does not establish a disk, checksum or PCIe limit.
Normal QEM reload checks are distinct from the exhaustive source-count
verification performed during conversion; that full comparison is not repeated
inside each load timing. The follow-up above adds interval profiling and
process storage-read counters without replacing these original observations.

## Evidence

The private `compression-audit.json` retains source identities, every linked
dataset's filter metadata, exact source-file totals, per-producer QEM sizes,
formulas and verification coverage. Source masters and current file totals were
checked against the retained conversion records. Original timing observations
and source data remain unchanged. This is documentation of existing verified
artifacts, not a new codec optimization or release qualification.
