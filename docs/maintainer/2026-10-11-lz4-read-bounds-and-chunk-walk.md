# LZ4 read bounds, the bounded chunk walk and the rc15 to rc16 timing comparison - 2026-10-11

> **Historical:** describes the code as of 2026-10-11; the fit it times (200 trials and Nelder–Mead) is retired. See the [current fit methods](../api/ssb.md).

## Question

1. A bitshuffle+LZ4 chunk with intact framing but damaged LZ4 bytes ended the
   process's CUDA context, and a truncated chunk sent the host block walk past
   the chunk's bytes. What does bounding both cost?
2. The rc16 cleanup removed about 23,000 lines. Did any CUDA hot path get
   slower, and did any output change that was not meant to?

## Setup

- One NVIDIA RTX PRO 6000 Blackwell (96 GB). Other jobs held the GPU at 90 to
  100% utilization and the 20-core host at a load average of 16 to 36 during
  every run, so wall-clock medians moved by tens of percent between runs.
  Comparisons therefore alternate the variants (processes in turn, or both
  variants in one process) and report minima as well as medians; kernel times
  use CUDA events, which do not count another process's time slice.
- Python 3.12, CuPy 14.2.0, PyTorch 2.14.1 (CUDA 13).
- Revisions: rc15 `531640d`; "pre-port" `1b6181f` (the merged cleanup before
  the two changes below); "final" `3b2e754`.
- Data: one Arina acquisition, 512 x 512 scan, 192 x 192 detector, uint16
  (27 files, private evidence archive).
- Scripts: `tests/benchmarks/data_paths.py`, `scripts/benchmark_ssb.py`
  (`--voltage-kV 300 --semiangle-mrad 30 --scan-sampling-A 0.264`), and
  throwaway alternation drivers around them.

## The two changes

`io/hdf5/cuda/kernels/bslz4.cu`: `decompressStream` already bounded its writes
(`f26391e`, `5e7fc8f`). It now bounds its reads too: literals lie inside the
compressed stream, two bytes remain before a match offset is read, a match
copies from bytes the block already wrote (`0 < offset <= decoded`), and a
length extension that runs off the stream is refused. Before the change, a
block of zeros (token 0, match offset 0) made the kernel copy from outside the
block: `tests/hardware/cuda/test_streamed_h5.py::test_damaged_lz4_bytes_raise_and_the_gpu_stays_usable`
failed with `cudaErrorIllegalAddress` and every later CUDA call in the process
failed. After it, three damaged variants raise `ValueError` and the same
process decodes the next batch exactly.

`formats/hdf5/frames.py`: `_parse_frame_header` stops at the chunk's last byte,
must end exactly there, and records zero blocks otherwise;
`FrameReader.prepare` refuses such a frame. Before the change, a chunk cut
short, a chunk shorter than its header, an unwritten (zero) chunk and a block
length pointing past the chunk either prepared without an error or raised
`ZeroDivisionError` (`tests/contracts/io/test_hdf5_chunk_index.py`). The paired
loader's `_walk_block_starts` stops the same way.

## Numbers

LZ4 decode kernel `h5lz4dc_batched`, 1000 alternating launches, CUDA events, ms
(output sums equal):

| Frames | rc15, no bounds | write bounds | write and read bounds |
|---|---|---|---|
| 512 | 0.0657 | 0.0689 | 0.0729 |
| 4096 | 0.4073 | 0.4258 | 0.4508 |

The read bounds add 5.8 to 5.9% to the write-bounded kernel, and the two
together 11% to the rc15 kernel.

Chunk header walk of 16384 frames (9 blocks each) as `FrameReader.prepare` calls
it (four workers), 2000 alternating calls, host only, two runs, ms:

| Walk | min | 5th percentile | median |
|---|---|---|---|
| unbounded | 0.43, 0.40 | 0.85, 0.65 | 1.78, 1.64 |
| bounded | 0.58, 0.48 | 0.95, 0.72 | 1.92, 1.81 |

About 0.1 ms per 16384 frames. Block starts and counts are identical.

Whole decode of 65536 frames in four prepared batches (`reader.prepare` then
`decompress_prepared`), both variants alternating in one process, ms:

| Run | pre-port min / median | ported min / median |
|---|---|---|
| 40 repeats | 108.4 / 120.4 | 112.7 / 123.4 |
| 60 repeats | 145.5 / 159.6 | 146.1 / 159.9 |
| 60 repeats | 126.5 / 159.0 | 133.1 / 163.6 |

The medians differ by 0.2 to 2.9%, while the same variant moved by 30% between
runs. The components predict about 0.8 ms per 65536 frames (0.4 ms of kernel
time, 0.4 ms of walk).

rc15 against the final tree, processes alternating, best time of the rounds.
`same` means identical SHA-256 digests of every output.

| Path | rc15 | final | Outputs |
|---|---|---|---|
| SSB session (bright-field crop and Fourier preparation), s | 0.641 | 0.632 | same |
| SSB reconstruct, s | 0.037 | 0.033 | same |
| SSB reconstruct with loss, s | 0.085 | 0.095 | same |
| `find_aberrations`, 200 trials and Nelder-Mead, s | 14.36 | 12.79 | same |
| `find_aberrations(tilt=True)`, s | 8.72 | 8.31 | rc15 differs between repeats; final repeats bit for bit |
| `dpc.run`, s | 0.237 | 0.252 | same |
| `parallax.run` with the aberration fit, s | 0.727 | 0.782 | image, density and shifts same; fitted aberrations differ (the corrected fit) |
| `io.load` (encoded resident), ms | 984 | 955 | same counts |
| prepared decode, 65536 frames, ms | 111.9 | 126.5 | same |
| encoded mean pattern, ms | 230 | 186 | same |
| encoded BF / ADF / DF image, ms | 4.1 / 4.2 / 3.1 | 3.7 / 3.7 / 2.8 | same |
| encoded centre of mass, ms | 206 | 163 | same |
| paired mean / BF / DF, ms | 0.07 / 2.2 / 2.0 | 0.07 / 1.7 / 1.7 (pre-port) | same |

The final tree's paired session never ran in a quiet window; the pre-port
values are given (the same code as the final tree for that path). The
reconstruct-with-loss medians of the quietest round are 0.098 s in both trees,
and the final tree has the lower `dpc.run` minimum in two of three rounds.

Dense CuPy session over 65536 frames, eight alternating rounds of 300 repeats,
median of the best round, ms. The masks select 23% (BF), 64% (ADF) and 77% (DF)
of the detector:

| Query | rc15 | final | Outputs |
|---|---|---|---|
| BF image | 3.15 | 3.16 | same |
| ADF image | 3.74 | 6.91 | same |
| DF image | 3.14 | 7.26 | same |
| centre of mass, repeated | 0.002 | 7.94 | same |

The digest of the encoded resident's retained bytes depends on the staging
batch, which `io.load` sizes from free GPU memory: one of eleven final runs
started with 25 GB free and produced a different layout with identical counts.
Forcing the batch to 16384, 10752, 10240, 9728 and 8192 scans gives the same
five layout digests on rc15 and on the final tree.

## Conclusion

- The read bounds cost 6% of the LZ4 kernel (11% with the write bounds since
  rc15) and the bounded walk 0.1 ms per 16384 frames. A whole decode changes by
  less than its run-to-run spread on a shared GPU.
- No other path is slower beyond that spread, except the two the cleanup
  changed on purpose. A dense CuPy mask that selects more than half of the
  detector (DF, and the ADF annulus here) now sums its selected pixels instead
  of subtracting the complement from cached per-frame totals (3.1 ms to 7.3 ms
  for DF), and the dense centre of mass is recomputed on each call (7.9 ms):
  the cached totals and field went stale when the array changed in place.
- Every output digest is identical except the tilt fit (now reproducible) and
  the parallax aberration fit (corrected).

## Rejected

- Keeping the walk's positions in 64-bit integers with the per-chunk view:
  0.81 ms against 1.05 ms when called serially, but 0.91 ms against 0.52 ms
  through the worker pool that `FrameReader.prepare` uses. Dropped.
- Indexing the staging buffer directly instead of taking a per-chunk view:
  slower both ways (worker pool 0.72 to 0.90 ms against 0.40 to 0.43 ms; serial
  1.65 ms against 1.03 ms). Dropped.
- Comparing whole processes one after another on the shared GPU: one tree can
  land in a busy window in every round (the final tree's dense session did),
  so those rows cannot rank the trees. Alternating both variants in one
  process, or timing with CUDA events, removed that.

## Not done

The Metal LZ4 decoder (`io/hdf5/mps/kernels/bslz4.msl`) bounds neither reads
nor writes. The bounded chunk walk protects both backends from a truncated
chunk; damaged LZ4 bytes inside intact framing have not been tested on Metal.

Done since: [Metal LZ4 bounds](2026-10-11-metal-lz4-bounds.md).
