# Paired-count resident layout

An opt-in second resident layout for native count acquisitions, next to the
byte-rANS `StreamedCounts` default. Every original count is retained exactly;
the layout changes bytes and query kernels only. Nothing in the default load,
dense, packed or ANS-file paths changes when this layout is not requested.

## Operation pipeline

1. Native `uint8`/`uint16` counts arrive as complete 512-scan blocks
   (`PairedCounts.append`), for example straight from the CUDA HDF5 decoder.
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
   reopens them with direct I/O and no decode.

Axes are `(scan_row, scan_col, detector_row, detector_col)`; equations use
$I[R_r,R_c,k_r,k_c]$ with $\mathbf R=(R_r,R_c)$ and $\mathbf k=(k_r,k_c)$.

## Contract

| item | value |
| --- | --- |
| query ABI | `paired-polar-counts-v1` |
| input | complete multiples of 512 scans, native `uint8`/`uint16`, detector up to 65,535 pixels per group of 32 streams |
| virtual image | exact `uint32` sums, `uint64` when a full-detector sum could exceed `2**32` |
| diffraction pattern | native dtype, invalid pixels reported as zero |
| malformed stream | `ValueError` from the query; the result is not returned |
| saved form | `quantem-paired-resident-v1`: JSON header plus 4096-aligned arrays; reopen refuses another ABI |

Coding parameters (32 models, 1024 states, 2-byte stream header, sparse mode
preferred unless the paired stream saves at least two bytes) are fixed by the
ABI string; any change needs a new ABI.

## Source map

| piece | location |
| --- | --- |
| kernels: tables, encode, compact, decode, frame, plan, residual, polar fields, index pack/sum, offsets, planner weights | `src/quantem/gpu/_compact/kernels/paired.cu` |
| `PairedCounts`, `PairedSeriesCompute`, planner, saved form | `src/quantem/gpu/_compact/paired.py` |
| planner hooks in the streamed base | `src/quantem/gpu/_compact/interaction.py` (`_plan`, `_cost`) |
| dispatch | `src/quantem/gpu/detector/workflow.py` |
| tests | `tests/contracts/io/test_paired_counts.py` |

## Verification

The contract tests build synthetic sources with a wide literal row, a
saturated row and an invalid pixel at 17x17, 19x19 and 257x257 detectors,
compare every mask and frame with direct sums (including `uint64` sums above
`2**32`), reject a reserved header bit and a shortened stream extent, and
reopen a saved form byte-identically. Set `QUANTEM_GPU_PAIRED_REFERENCE` to a
JSON file (`path`, `poses`, `reference_VI_sha256`) to check frozen full-array
virtual-image digests on a native source.

Physical-device evidence (private study, 2026-09-09; one RTX PRO 6000
Blackwell, 69 complete 512x512x192x192 uint16 acquisitions resident): the
paired layout decoded arbitrary-center detector updates at 12.5 ms per batch of
69 full virtual images versus 23.9 ms for the byte-rANS layout on the same
counts, with 414 frozen full-array digests exact; resident bytes per source
1.293 GiB versus 1.307 GiB. Loading from the saved form reached the measured
drive ceiling. See the evidence ledger for the atomic rows.

**Device tested**: RTX PRO 6000 Blackwell (CUDA). **Date tested**: 2026-09-09.
