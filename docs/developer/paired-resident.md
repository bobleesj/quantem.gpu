# Paired-count resident layout

An opt-in second resident layout for native count acquisitions, next to the
byte-rANS `StreamedCounts` default. Every original count is retained exactly;
the layout changes bytes and query kernels only. Nothing in the default load,
dense, packed or ANS-file paths changes when this layout is not requested.

## Operation pipeline

1. Native `uint8`/`uint16` counts arrive as complete 512-scan blocks
   (`PairedCounts.append`). `io.load(..., representation="paired")` produces
   them from original bitshuffle+LZ4 HDF5: reader threads read whole shards
   with direct I/O into page-locked staging and parse the LZ4 block headers
   there; one producer stream copies, LZ4-decodes and bitshuffles into a ring
   of rolling frame buffers; the consumer stream encodes and indexes each
   filled buffer and returns it through an event. Shard reads run ahead across
   acquisitions, so a series is bounded by the drive, not by the GPU.
2. Each detector pixel's 512-scan stream is coded as consecutive count pairs
   with a 1024-state tANS table from 32 Poisson pair models; streams that are
   empty, constant, sparse (few nonzero counts) or incompressible keep exact
   alternative forms. Stream offsets are grouped (one 32-bit base plus 32
   relative 16-bit offsets).
3. Exact per-scan sums over radial-angular pixel groups (64-pixel leaves,
   16-leaf roots) are bit-packed as the interaction index.
4. `detector.prepare` selects `PairedSeriesCompute`. A detector mask (or its
   difference from the previous mask) is decomposed into index fields plus
   residual pixels; the residual streams are decoded by one thread per stream
   with a warp-wide packed reduction per 32 scans. Outputs are exact integer
   virtual images and native diffraction patterns, as for the default layout.
5. `PairedCounts.save` writes the resident arrays once; `PairedCounts.load`
   (or `io.load` on the file) reopens them with direct I/O and no decode.
6. Reconstruction consumers read native count blocks back from the resident
   form: `PairedCounts.decode_blocks(first, scans)` decodes whole 512-scan
   blocks of one chunk, and `PairedFeed(sources, amplitude=True)` iterates the
   blocks of a series on a prefetch stream (`depth` buffers ahead, events in
   both directions) so a joint time-series ptychography update never touches a
   dense copy of the data.

Axes are `(scan_row, scan_col, detector_row, detector_col)`; equations use
$I[R_r,R_c,k_r,k_c]$ with $\mathbf R=(R_r,R_c)$ and $\mathbf k=(k_r,k_c)$.

## Contract

| item | value |
| --- | --- |
| query ABI | `paired-polar-counts-v1` |
| input | complete multiples of 512 scans, native `uint8`/`uint16`, detector up to 65,535 pixels per group of 32 streams |
| H5 loader | `uint16` bitshuffle+LZ4 shards with one frame per chunk; a partial final LZ4 block must hold a multiple of 8 values; one path or a list, every acquisition returned as its own source; `backend="cuda"`, `dtype="native"`, `apply_mask=False`, no selection or binning options |
| virtual image | exact `uint32` sums, `uint64` when a full-detector sum could exceed `2**32` |
| diffraction pattern | native dtype, invalid pixels reported as zero |
| malformed stream | `ValueError` from the query; the result is not returned |
| saved form | `QGPUPAIR` magic, JSON header (`quantem-paired-resident-v1`) plus 4096-aligned arrays; `io.load` detects it; reopen refuses another ABI |

Coding parameters (32 models, 1024 states, 2-byte stream header, sparse mode
preferred unless the paired stream saves at least two bytes) are fixed by the
ABI string; any change needs a new ABI.

## Source map

| piece | location |
| --- | --- |
| kernels: tables, encode, compact, decode, decode_range, frame, plan, residual, polar fields, index pack/sum, offsets, planner weights | `src/quantem/gpu/_compact/kernels/paired.cu` |
| `PairedCounts`, `PairedSeriesCompute`, planner, saved form, `decode_blocks`, `PairedFeed` | `src/quantem/gpu/_compact/paired.py` |
| planner hooks in the streamed base | `src/quantem/gpu/_compact/interaction.py` (`_plan`, `_cost`) |
| dispatch | `src/quantem/gpu/detector/workflow.py` |
| H5 streaming loader, saved-form reopen, `io.load` results | `src/quantem/gpu/io/_paired.py` (`PairedLoader`) |
| representation selector and magic detection | `src/quantem/gpu/io/representation.py`, `src/quantem/gpu/io/_packed.py`, `src/quantem/gpu/io/load.py` |
| tests | `tests/contracts/io/test_paired_counts.py`, `tests/contracts/io/test_paired_loading.py` |

## Verification

The contract tests build synthetic sources with a wide literal row, a
saturated row and an invalid pixel at 17x17, 19x19 and 257x257 detectors,
compare every mask and frame with direct sums (including `uint64` sums above
`2**32`), reject a reserved header bit and a shortened stream extent, and
reopen a saved form byte-identically, and iterate two sources through
`PairedFeed` checking every block and amplitude against the raw counts. The
loading tests write four-shard
Arina-style masters with `save_compressed_arina_h5`, stream one and two of them
through `io.load(representation="paired")`, compare every count, a detector
mask and a frame with direct sums, and reopen a saved form through `io.load`
byte-identically. Set `QUANTEM_GPU_PAIRED_REFERENCE` to a JSON file (`path`,
`poses`, `reference_VI_sha256`) to check frozen full-array virtual-image
digests on a native source through the public loader; on 2026-09-09 all six
digests of one native 512x512x192x192 acquisition matched.

Physical-device evidence (private study, 2026-09-09; one RTX PRO 6000
Blackwell, 69 complete 512x512x192x192 uint16 acquisitions resident): the
paired layout decoded arbitrary-center detector updates at 12.5 ms per batch of
69 full virtual images versus 23.9 ms for the byte-rANS layout on the same
counts, with 414 frozen full-array digests exact; resident bytes per source
1.293 GiB versus 1.307 GiB. Loading from the saved form reached the measured
drive ceiling.

Package loader, single run on the same device with the default
`PairedLoader` (69 complete 512x512x192x192 uint16 acquisitions offered from
one PCIe 4.0 NVMe drive whose direct-read ceiling is 5.5 GB/s; a memory-budget
`admit` callback stopped the series at 66):

| measurement | acquisitions | value |
| --- | --- | --- |
| series load, original HDF5 to resident, wall | 66 | 29.3 s |
| resident-ready wall per acquisition inside the pipeline, median | 66 | 1.50 s |
| encode per acquisition, median | 66 | 0.197 s |
| index per acquisition, median | 66 | 0.120 s |
| resident bytes, sum | 66 | 93.97 GB |
| device bytes in use after load (pool cache and staging included) | 66 | 101.6 GB |
| `detector.prepare` over all sources | 66 | 38 ms |
| first bright-field virtual image, all sources (first source digest exact) | 66 | 8.2 ms |
| first diffraction pattern, all sources | 66 | 1.9 ms |
| save one source as the paired resident form (write plus fsync) | 1 | 1.39 s |
| reopen that form through `io.load`, byte-identical | 1 | 0.53 s |

Shards are read with direct I/O, so the series time is bounded by the drive
(about 145 GB of original chunks in 29.3 s). The complete atomic record with
boundaries, cache states and the admission rule is
{download}`paired-loading-2026-09-09.json <../performance/data/paired-loading-2026-09-09.json>`.
The last acquisitions of a full device fit only with a smaller loader
(`PairedLoader(rolling_scans=512, rings=2)`); that tail policy belongs to the
application.

Feeding a reconstruction from the resident form (three 512x512x192x192 sources,
`PairedFeed(depth=2)`, consumer idle, same device shared with a desktop):

| measurement | per acquisition |
| --- | --- |
| native counts, 512-scan blocks | 76 ms |
| native counts, 8192-scan blocks | 90 ms |
| counts plus float32 sqrt amplitude, 512-scan blocks | 119 ms |
| counts plus float32 sqrt amplitude, 8192-scan blocks | 133 ms |

That is 127 G counts/s decoded into native frames, below the 48 ms per
262,144-position fused ptychography iteration only by a factor of about two, so
one decode per iteration hides behind the update when the two overlap on
separate streams. Record:
{download}`paired-feed-2026-09-09.json <../performance/data/paired-feed-2026-09-09.json>`.

**Device tested**: RTX PRO 6000 Blackwell (CUDA). **Date tested**: 2026-09-09.
