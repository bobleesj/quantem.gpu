# MPS memory: where SSB's gigabytes go and what fits which Mac (2026-10-20)

Apple M5 Max, 128 GB, macOS 26.4, torch 2.13, MLX 0.32. Two acquisitions: "a
512 × 512 scan with a 192 × 192 detector" (uint16, 8,939 bright-field pixels,
2.07 GiB encoded), which is not public, and the public gold scan
(`bobleesj/quantem-data`, `4dstem/gold_512`, 7,644 bright-field pixels, 5.64
GiB encoded as uint16 count ANS). Memory is the process's Metal driver memory
(`torch.mps.driver_allocated_memory`, which counts Torch's and MLX's buffers)
unless a column says footprint (the physical footprint Activity Monitor
shows). The companion speed page is
[MPS performance and regression harness](2026-10-20-mps-performance.md).

## A ShowPtycho session, stage by stage

Load, `data.calibration.fit_detector_sampling()`, `SSB(data)`, the automatic
two-branch `reconstruct()`, `ssb.show(show_fft=True)` at its default
upsampling, 20 slider drags at that factor, 3 slider releases (with the loss),
one 256 × 256 region refit (20 trials) and `save_calibration()`. "Before" is
the integration branch just before the memory work; "after" is the default;
"tiled" adds the opt-in tiled loss route below (1024 pixels). Peak during the
stage / memory after it, GiB.

512 × 512 scan, 192 × 192 detector:

| Stage | Before | After | Tiled 1024 |
| --- | ---: | ---: | ---: |
| `io.load` | 3.9 / 2.1 | 3.9 / 2.1 | 3.9 / 2.1 |
| `SSB(data)` | 8.1 / 8.1 | 2.1 / 2.1 | 2.1 / 2.1 |
| `reconstruct()`, two branches | 34.3 / 26.9 | 20.9 / 10.8 | 16.8 / 10.8 |
| `ssb.show(show_fft=True)` | 27.0 / 27.0 | 20.4 / 20.4 | 14.4 / 14.4 |
| 20 drags (2×), 3 releases | 27.0 / 27.0 | 20.4 / 20.4 | 14.4 / 14.4 |
| region refit | 33.6 / 16.9 | 28.6 / 10.9 | 22.8 / 10.9 |
| `save_calibration()` | 26.4 / 26.4 | 20.4 / 20.4 | 14.4 / 14.4 |
| session peak (footprint) | 34.3 (34.9) | 28.6 (30.0) | 22.8 (24.0) |

Gold:

| Stage | Before | After | Tiled 1024 |
| --- | ---: | ---: | ---: |
| `io.load` | 7.5 / 5.6 | 7.5 / 5.6 | 7.4 / 5.6 |
| `SSB(data)` | 15.9 / 15.9 | 5.6 / 5.6 | 5.6 / 5.6 |
| `reconstruct()`, two branches | 40.2 / 40.2 | 29.9 / 13.1 | 17.9 / 13.1 |
| `ssb.show(show_fft=True)` | 40.5 / 40.5 | 28.4 / 28.4 | 16.4 / 16.4 |
| 20 drags (4×), 3 releases | 40.5 / 40.5 | 28.5 / 28.5 | 16.5 / 16.5 |
| region refit | 46.1 / 23.4 | 34.3 / 13.2 | 22.3 / 13.2 |
| `save_calibration()` | 38.7 / 38.7 | 28.4 / 28.4 | 16.4 / 16.4 |
| session peak (footprint) | 46.1 (47.4) | 34.3 (35.5) | 22.3 (23.5) |

Every fit, image and loss is unchanged: G(q, k) hashes, previews (1× and 2×),
losses and reconstructions are bit-identical before and after on both scans,
and the two-branch fit keeps the same rotation and coefficients to the last
digit. `reconstruct()` took 8.98 → 8.88 s (512² scan) and 20.1 → 20.7 s
(gold), within run-to-run spread; a slider release 0.82 → 0.83 s and 0.84 →
0.83 s. `SSB(data)` is 0.39 → 0.14 s and 0.78 → 0.43 s; the first preview,
which now decodes the bright-field columns, 0.99 → 1.04 s and 0.88 → 0.96 s.

## What the big arrays are

**The encoded acquisition** (2.07 / 5.64 GiB). ANS-encoded counts on the GPU,
one stream per detector pixel and 512 scan positions, so any detector
rectangle decodes without the rest.

**G(q, k)** (8.76 / 7.49 GiB): complex64, `(bright-field pixels, 512, 257)`.
Each bright-field pixel's scan image transformed with a real-to-half FFT; real
input makes the spectrum Hermitian, so half the columns plus one hold all of
it. Every preview, loss and fit reads it, and with the encoded data it is the
session's steady state: 10.8 GiB for the 512² scan, 13.1 GiB for gold.

**The decoded crop** (gone). `SSB(data)` used to decode the square detector
window around the disk for every scan position (6.0 GiB of uint16 for the 512²
scan, 27 % of it outside the disk), keep it on the GPU for the session, and
hold a second copy while masking flagged pixels. The crop now stays encoded
(`EncodedBrightFieldCrop`); while G(q, k) is built, `EncodedBrightFieldFrames`
decodes one block of scan rows (256 MiB) of the selected pixels at a time.
The build runs the real FFT along the scan columns of each block's rows in the
order `rfft2` would, starting on an even row: MLX packs neighbouring real rows
in pairs into one complex FFT, so the row order decides the rounding, and the
same pairs give the same bits. The FFT along the scan rows then runs in place,
256 pixels at a time. Writing G(q, k) in place needs one wait per step: MLX
updates an array in place only while no finished command buffer still holds
its buffer, and one copied update costs one more G(q, k) of peak.

**The build transient**: about 1.3 GiB of MLX memory above G(q, k); the
driver peak while `SSB()` and the first preview run is 12.4 GiB above the
encoded data (512² scan) and 11.7 GiB (gold). The previous chunked build kept
every chunk's spectrum and concatenated them, twice G(q, k), beside the crop:
a peak of 32.2 GiB above the encoded data on the 512² scan (38.2 GiB a day
earlier, when the masking copy stayed too).

**The loss row IFFT** (the "after" column's +9.6 / +15.1 GiB after the first
loss). The 512² loss transforms one chunk of bright-field pixels at a time: 4096
on Macs with 96 GB or more (8.6 GB of complex64 row IFFT), 1024 at 64 GB, 512
below. MLX's allocator keeps the freed buffer, so the next slider release
reuses it instead of mapping 8.6 GB again (clearing it after every release
made a release 2.4 times slower).

**Fit and refit buffers**: the joint fit's candidate batches, about 10 GiB above
G(q, k) on the 512² scan and 17 GiB on gold, and a region refit's own G(q, k) (2.2 GiB for 256 × 256) plus its
fit. `find_aberrations` and `reconstruct()` now return MLX's cached buffers when
they finish, and a region session returns its own on close.

## The loss cache or tiles: a decision

`quantem.gpu.ssb.mps.reconstruct.LOSS_ROW_IFFT_TILE_BF` (default `None`)
sends a 512² loss chunk larger than the tile through the tiled path a
cancellable preview takes: the same reduction boundaries and float32 order, so
the same loss and phase bits, with one tile of row IFFT alive at a time.
Interleaved medians of 3 × 8 thick-sample loss previews on the 512² scan:

| Route | Loss preview | MLX cache kept | MLX peak |
| --- | ---: | ---: | ---: |
| whole chunk (default) | 263 ms | 9.5 GiB | 16.8 GiB |
| tiles of 256 | +7.9 % | 1.0 GiB | 9.3 GiB |
| tiles of 1024 | +4.4 % | 3.5 GiB | 10.8 GiB |
| tiles of 2048 | +3.9 % | 5.5 GiB | 12.8 GiB |
| tiles without a wait between them | +4 to 7 % | 9.5 GiB | 16.8 GiB |

Each tile waits for the GPU before the next is built; without that MLX holds
every tile's row IFFT at once and nothing is saved. In the session above the
tiled releases were not slower (0.77 against 0.83 s), but the interleaved
measurement, on a GPU that the window server shared, puts tiles about 4 %
behind, so the default stays the whole chunk. What would remove the cost: the
row and column IFFT fused per tile, overlap of one tile's row IFFT with the
previous tile's column stage, or the real-output inverse transform on the loss
path, which halves the intermediate.

## What fits which Mac

macOS grants the GPU a working set below the RAM size (this Mac reports
107.5 of 128 GiB in `mx.device_info()["max_recommended_working_set_size"]`;
smaller Macs and older macOS grant less, about two thirds). The session needs
the encoded data plus G(q, k) resident, then one transient at a time; the loss
chunk shrinks with the RAM size (1.1 GB below 64 GB).

| Mac | 512² scan: steady 10.8 GiB | gold: steady 13.1 GiB |
| --- | --- | --- |
| 16 GB | does not fit: G(q, k) alone is 8.8 GiB | does not fit |
| 24 GB | the steady state fits; fits and refits (about +10 GiB) page | steady state at the limit |
| 32 GB | fits, with little room for a fit's batches | tight |
| 64 GB and up | fits | fits |

These rows are estimates from the steady states and transients measured on the
128 GB Mac, not runs on those Macs. A 16 GB Mac would need G(q, k) itself
smaller, for example a half-precision copy for slider previews (4.4 GiB for the
512² scan) with fits rebuilding the exact float32 G(q, k) from the encoded data,
which the encoded crop now makes possible; that is not implemented.

## How to measure

- `python tests/benchmarks/mps_ssb.py run` records, beside the times, the
  memory above the encoded data in `memory_GiB` (`session_steady`,
  `prepare_peak`, `prepared_steady`, `loss_preview_steady`, `fit_peak`,
  `fit_steady`) and the encoded size; `compare` fails a metric that grew by
  more than 10 % and 0.25 GiB.
- Peaks need a sampler: a thread reading `torch.mps.driver_allocated_memory()`
  every 2 to 4 ms while the stage runs (the harness's `PeakMemory`). The
  physical footprint comes from `proc_pid_rusage` (`ri_phys_footprint`).
  `mx.get_active_memory()`, `mx.get_peak_memory()` and `mx.get_cache_memory()`
  split MLX's share into live arrays and its allocator cache.
- Driver memory and footprint can disagree for a moment after a stage: freed
  Metal buffers leave the footprint later than the driver count. Compare
  peaks, and steady values read after a synchronize.
