# CUDA streaming index, 2026-11-07

The spatial-index builder now assigns one thread to each 8×8 detector tile,
instead of one warp. This reduces block count and removes the warp reduction.
The tile sums, validity mask, encoded bytes and query API are unchanged.
Even a saturated uint16 tile sums to only 4,194,240, within uint32.

## Matched measurements

RTX PRO 6000 Blackwell, CUDA 13, CuPy 14.2, Holoscan 5, GPU 0. The input is a
512×512×192×192 uint16 experimental acquisition. Each run processes eight scans
from a preloaded raw GPU source, retaining two. Arrival batches contain 2,048
complete patterns; ANS chunks remain 4,096 patterns. Encoding is repeated for
every arrival. Disk reads, initial source preparation and verification are outside
the streaming timer; BF/ADF, encoding, indexing and bounded retention are inside.

| Index construction | Median patterns/s, three runs | Range |
| --- | ---: | ---: |
| Previous warp per tile | 1,111,681 | 1,089,289–1,180,556 |
| One thread per tile | 1,256,081 | 1,228,918–1,323,601 |

The interleaved comparison gives 13.0% more throughput. All runs passed exact
count, BF/ADF, scan order and release checks. A separate 40-scan production run
took 8.608 s, retained the last seven scans and reported zero dropped scans or
mismatched counts. Its retained encoding size was unchanged at 9,920,344,014 bytes.

Nsight Compute measured the index kernel at 508 µs before and 206 µs after,
with DRAM throughput rising from 35.9% to 91.0% (1.56 TB/s). Grid size fell
from 589,824 to 18,432 blocks for a 4,096-pattern chunk. These are instrumented
kernel observations, with SM clocks of 1.58/1.52 GHz, not production wall times.
The full isolated indexing phase fell from approximately 0.487 to 0.310 ms.

## Agreement and limits

The frozen previous CUDA kernel and candidate produced identical payload,
offset, model and index arrays on the experimental input and on uint8/uint16
7×11, 96×96, 192×192 and 257×257 detector cases with masked pixels and partial
chunks. The focused CUDA suite passed 69 tests, including translated masks,
saturated counts, corrupt streams, HDF5 loading and release.

The broader CUDA acquisition gate passed 150 tests. Its 23 external-collection
tests remain blocked without their data roots, so this is not complete release
qualification. One stale test expected the old direct ptychography guidance text; it was updated
to match DirectPtycho without changing any numerical assertion.

Alternative scratch layouts, different block sizes, parallel compaction and an
exact reciprocal replacement for integer division were explored. None established
a sufficiently useful end-to-end gain to retain. Only the simpler tile mapping
was adopted. ANS encoding/compaction and pipeline overhead remain; this is not
the theoretical end-to-end minimum. No RDMA, browser FPS or MPS performance claim
is made by this record.

Baseline: `e1e1ddae1e8e10fbbef2d3410b7474566519b06e`. Candidate kernel SHA-256:
`f7facfafdf74242087f6abd57e01da584bc4eb30c8184605f58263b109ff7380`.
Private experiment: `20261107-native-stream-index`. Raw evidence and experimental
runners stay in the private evidence repository. The application integration
runner records imported Python and CUDA source hashes in each result.
