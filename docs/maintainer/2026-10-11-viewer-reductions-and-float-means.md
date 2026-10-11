# Viewer reductions, float mean accuracy and gzip stack loads - 2026-10-11

## Question

Work done two weeks earlier on a laptop (a full-scan mean that delegates to the
native reduction, 6.2 s to 0.13 s on a 512 x 512 scan) had to be carried onto
the restructured code. Three things were open: is every reduction of a viewer
on the acquisition's own session, does a float32 acquisition give the float64
mean pattern, and which stage of opening an acquisition is slower than it
should be?

## Setup

Nine acquisitions: an Arina 512 x 512 scan of a 192 x 192 detector (uint16), an
EMPAD RAW scan and a background-subtracted EMPAD HDF5 (128 x 128 scans,
float32), three TCMEP stacks from Zenodo 15084123 (a simulation, twisted STO
and PCCO, one gzip chunk per pattern), the SrTiO3 dislocation scan and the
uint8 simulation from Zenodo 7464234, and the WSe2 simulation from Zenodo
15987625. Each was loaded, reduced, opened in `Show4DSTEM`, saved as `.qem` and
reopened on an Apple M5 Max (MPS) and on an RTX PRO 6000 Blackwell (CUDA; the
GPU and its disk were shared with other jobs, so its times are not isolated).
Stage times are from `tests/benchmarks/viewer_stages.py`, median of 3 on fresh
loads, before (378e777) and after on the same machine.

## Float mean patterns were a float32 running total

`FloatANSResident.mean_dp_device` added each decoded chunk to a float32 total.
Against NumPy float64 over the same loaded values, in units in the last place
of float32:

| Acquisition | float32 running total | 32-position reads summed in float32 (the laptop loop) | Now |
|---|---|---|---|
| EMPAD RAW 128 x 128 | 5 (MPS), 4 (CUDA) | 16 | 0 |
| EMPAD HDF5 128 x 128 | 4 | 18 | 0 |
| Twisted STO 100 x 100 | 7 | 17 | 0 |
| PCCO 100 x 100 | 4 | 20 | 0 |
| SrTiO3 dislocations 256 x 256 | 8 | 38 | 0 |
| WSe2 128 x 128 | 4 | 17 | 0 |

CUDA now sums each chunk in float64. Metal has no float64: a kernel
(`resident/mps/kernels/pattern_total.msl`) keeps each detector pixel's float32
sum with two orders of rounding residual (Kahan-Babuska-Klein) and the host
adds the three in float64. Scan-ROI sums and means take the same total. The
mean pattern equals the float64 reference bit for bit on all six, and CUDA and
MPS return the same bits. The virtual images were already exact: their kernel
compensates on both backends.

The first version waited for the GPU three times per chunk (decode, Torch,
kernel) and cost 6% on the mean and up to 25% on a scan ROI. Decoding the chunk
and summing it in one command, with the dark pattern subtracted and the
selected scans flagged in the kernel, removed that: SrTiO3 mean 153 to 147 ms,
scan ROI 37.5 to 22.4 ms.

## Bounded views read every scan position for some acquisitions

`BoundedDetectorCompute` reduced on the acquisition's session only when it was
an encoded `Dataset4dstemGPU`. A GPU tensor held as a dense acquisition (the
widget's tutorial preview), a bare tensor and a paired acquisition were reduced
by 32-position reads. Every one has a backend of its own, so the view now takes
its owner's session for all of them and the read loops are gone. CUDA, GPU
shared:

| Acquisition viewed | Stage | Reads | Session | Values |
|---|---|---|---|---|
| Dense uint16, 192 x 192 x 96 x 96 | mean / BF / scan-ROI mean / viewer | 4.03 / 7.47 / 5.91 / 12.4 s | 0.071 / 0.044 / 0.019 / 0.26 s | identical |
| Dense float32, same shape | mean / BF / viewer | 4.35 / 5.51 / 10.7 s | 0.016 / 0.029 / 0.040 s | mean now 0 instead of 1 unit in the last place off float64; BF identical |
| Paired uint16, 512 x 512 x 192 x 192 | mean / BF / scan-ROI mean / viewer | 84.6 / 69.5 / 8.28 / 65.4 s | 0.014 / 0.036 / 0.166 / 0.14 s | identical |

On MPS the dense uint16 viewer went from 0.46 s to 0.11 s and the float32 one
from 0.39 s to 0.02 s. `TorchBackend` float sums moved to float64 accuracy with
the loops (float64 on CUDA, the Metal kernel on MPS), so no precision was lost.

### Signed data and scan regions of dense owners

Routing dense owners to their backends exposed what those backends did with
data that is not counts. A review of the branch found, and tests now cover:

- int16 data with negative totals (dark-subtracted frames): `ArrayBackend`,
  `TorchBackend.reduce_frames` and the CuPy `reduce_frames` and mean summed in
  uint64 or cast the total to it, so a negative total came back near 1.8e19,
  and the Torch maximum started from 0. Signed integers now accumulate in
  int64 and a maximum starts from the first chunk. Unsigned results are
  unchanged.
- A scan region's mean gathers its patterns: `ArrayBackend` and the CuPy
  backend took the whole selection at once (10.5 GB for a 400 x 400 region of
  128 x 128 float32 patterns). Both now gather in chunks.
- `ArrayBackend.reduce_frames(..., "mean")` of float32 data used a float32
  mean; it now divides the float64 total once.

## Metal scan-ROI reductions summed on the host

MPS `CountDetectorCompute.reduce_frames` decoded each run of selected scans,
copied it to the host and summed in NumPy: 203 ms for the 8245 patterns of a
51-scan-radius circle (98 ms NumPy sum, 79 ms in 103 decode commands, 28 ms
copy). A kernel (`streamed_counts_selected_reduce`) now adds the selected scans
per detector pixel on the device, skipping stream blocks without one: 30 ms for
that circle, 89 ms for 70681 positions, the same sums, maxima and means.

## gzip stacks inflated on one core

A stack of one gzip chunk per pattern is read by HDF5 one chunk at a time:
3.9 s of a 5.8 s load. The stored chunks are now read as they are and inflated
by zlib on up to 8 threads. Every loaded value is unchanged (the loader's
logical SHA-256 is the same).

| Stack | MPS before | MPS after | CUDA host before | CUDA host after |
|---|---|---|---|---|
| Twisted STO, 10000 x 256 x 256 | 3.69 s | 1.74 s | 4.83 s | 2.60 s |
| PCCO, 10000 x 128 x 128 | 2.21 s | 0.76 s | 2.54 s | 1.03 s |
| TCMEP simulation, 676 x 128 x 128 | 0.084 s | 0.049 s | 0.100 s | 0.063 s |

## Stage times on MPS, before and after

Median ms of 3; the float mean and scan-ROI values changed as described, every
other output is identical.

| Acquisition | load | mean | viewer | detector change | scan-ROI mean |
|---|---|---|---|---|---|
| Arina 512 x 512 x 192 x 192 uint16 | 1588 / 1626 | 130 / 131 | 141 / 141 | 6.2 / 6.2 | 449 / 46 |
| EMPAD RAW | 562 / 562 | 42 / 41 | 47 / 49 | 3.0 / 3.0 | 10.5 / 7.6 |
| EMPAD HDF5 | 534 / 544 | 42 / 43 | 51 / 51 | 10.9 / 10.8 | 10.4 / 7.9 |
| TCMEP simulation (676 patterns) | 86 / 49 | 4.2 / 4.5 | 11.7 / 12.6 | 2.2 / 2.2 | 1.1 / 1.7 |
| Twisted STO | 3672 / 1743 | 76 / 64 | 90 / 76 | 20.9 / 20.8 | 20.0 / 10.7 |
| PCCO | 2183 / 759 | 35 / 28 | 44 / 41 | 8.1 / 8.0 | 7.0 / 5.3 |
| SrTiO3 dislocations | 2188 / 2218 | 153 / 147 | 167 / 166 | 30.0 / 30.0 | 37.5 / 22.4 |
| SrTiO3 simulation uint8 | 247 / 246 | 14.3 / 14.3 | 19.2 / 19.0 | 2.3 / 2.2 | 29.3 / 5.3 |
| WSe2 simulation | 556 / 554 | 43 / 41 | 49 / 49 | 4.8 / 4.8 | 10.5 / 7.8 |

A cursor move takes 0.9 to 1.8 ms on every acquisition, before and after. On the
676-pattern simulation the scan-ROI mean is 0.6 ms slower: the fused path's fixed
cost (one window, one flag upload) exceeds what it saves on 13 patterns.

## Rejected

- A pairwise TwoSum tree in Torch operations for the Metal float total: about
  54 operations per 32 MiB window, several times slower than the decode.
- First-order compensation only (one residual): its error grows as
  n times eps squared, about 2e-10 of the total after 65536 patterns, which
  rounds a few pixels in every hundred to the neighbouring float32.
- Queuing the Metal sums of every window and waiting once: a decoded window is
  recycled by the next decode before a queued command has read it.
- Copying decoded patterns to the host and summing in float64 there.
- A tolerance on the float mean test: the comparison is exact.

## Left

- A `.qem` save writes its body twice (a temporary file, then the published
  file) and hashes it on one thread: for 4.3 GB on MPS 1.28 s final write,
  1.24 s SHA-256, 1.02 s temporary write and 0.36 s read back of a 3.9 s save.
- The exact uint64 products (`reduce_frames_exact`, `masked_sum_exact`) are defined
  for counts; signed data with a negative total has no uint64 value there.
- `TorchBackend.masked_sum` of float data sums in float32 (1 unit in the last
  place on the dense float32 stack above); the residents' virtual images are
  exact.
- The compensated Metal kernel costs more than a float32 sum on a dense float32
  stack: 12.8 ms against 2.8 ms for a float32 running sum of 1.36 GB.
