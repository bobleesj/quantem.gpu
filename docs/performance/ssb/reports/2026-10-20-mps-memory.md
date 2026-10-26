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
the integration branch just before the memory work; "after" is the default
now, with the loss row IFFT in tiles of 1024 pixels; "whole chunk" is the
opt-in loss route that keeps the whole 4096-pixel chunk (below), measured
before region refits released the full field. Peak during the stage / memory
after it, GiB.

512 × 512 scan, 192 × 192 detector:

| Stage | Before | After | Whole chunk |
| --- | ---: | ---: | ---: |
| `io.load` | 3.9 / 2.1 | 3.9 / 2.1 | 3.9 / 2.1 |
| `SSB(data)` | 8.1 / 8.1 | 2.1 / 2.1 | 2.1 / 2.1 |
| `reconstruct()`, two branches | 34.3 / 26.9 | 16.8 / 10.8 | 20.9 / 10.8 |
| `ssb.show(show_fft=True)` | 27.0 / 27.0 | 14.4 / 14.4 | 20.4 / 20.4 |
| 20 drags (2×), 3 releases | 27.0 / 27.0 | 14.4 / 14.4 | 20.4 / 20.4 |
| region refit | 33.6 / 16.9 | 14.4 / 2.1 | 28.6 / 10.9 |
| next full-field preview (rebuilds G) | 27.0 / 27.0 | 16.8 / 14.4 | – |
| `save_calibration()` | 26.4 / 26.4 | 14.4 / 14.4 | 20.4 / 20.4 |
| session peak (footprint) | 34.3 (34.9) | 16.8 (18.1) | 28.6 (30.0) |

Gold:

| Stage | Before | After | Whole chunk |
| --- | ---: | ---: | ---: |
| `io.load` | 7.5 / 5.6 | 7.4 / 5.6 | 7.5 / 5.6 |
| `SSB(data)` | 15.9 / 15.9 | 5.6 / 5.6 | 5.6 / 5.6 |
| `reconstruct()`, two branches | 40.2 / 40.2 | 17.9 / 13.1 | 29.9 / 13.1 |
| `ssb.show(show_fft=True)` | 40.5 / 40.5 | 16.4 / 16.4 | 28.4 / 28.4 |
| 20 drags (4×), 3 releases | 40.5 / 40.5 | 16.5 / 16.5 | 28.5 / 28.5 |
| region refit | 46.1 / 23.4 | 16.5 / 5.7 | 34.3 / 13.2 |
| next full-field preview (rebuilds G) | 40.5 / 40.5 | 17.4 / 16.5 | – |
| `save_calibration()` | 38.7 / 38.7 | 16.5 / 16.5 | 28.4 / 28.4 |
| session peak (footprint) | 46.1 (47.4) | 17.4 (18.6) | 34.3 (35.5) |

Every fit, image and loss is unchanged: G(q, k) hashes, previews (1× and 2×),
losses and reconstructions are bit-identical before and after on both scans,
and the two-branch fit keeps the same rotation and coefficients to the last
digit. `reconstruct()` took 8.98 → 7.82 s (512² scan) and 20.1 → 18.6 s
(gold), and three slider releases 0.82 → 0.73 s and 0.84 → 0.78 s, all within
run-to-run spread on a GPU shared with the window server. `SSB(data)` is 0.39 →
0.14 s and 0.78 → 0.43 s; the first preview, which now decodes the
bright-field columns, 0.99 → 1.04 s and 0.88 → 0.96 s.

A region refit (`select_region`, `find_aberrations(scan_region=...)`, ShowPtycho's
Fit Regions and Crop and Refit) first frees the full field's G(q, k) and MLX's
cache, so the region's own G(q, k) and fit run beside the encoded data alone;
the next full-field preview or fit rebuilds G(q, k) from the encoded data in
about 0.8 s, with the same bits (the preview after the rebuild equals the one
before the refit, and the region's fit is unchanged). A region session frees
its own G(q, k) when it closes; Crop and Refit keeps the session it displays.

## Live arrays and MLX's cache, stage by stage

The driver figure above is MLX's live arrays (active), the buffers MLX keeps
after they are freed (its cache, reused by the next allocation of the same
size), the encoded data and Torch's few decode blocks. After each stage, GiB;
"active peak" is the highest live total during the stage:

| Stage (512² scan) | Driver after | Active after | Cache after | Active peak |
| --- | ---: | ---: | ---: | ---: |
| `SSB(data)` | 2.1 | 0.0 | 0.0 | 0.0 |
| `reconstruct()`, two branches | 10.8 | 8.8 | 0.0 | 11.1 |
| show, drags, releases | 14.4 | 8.8 | 3.6 | 10.8 |
| region refit | 2.1 | 0.0 | 0.0 | 6.7 |
| next full-field preview | 14.4 | 8.8 | 3.6 | 11.2 |

| Stage (gold) | Driver after | Active after | Cache after | Active peak |
| --- | ---: | ---: | ---: | ---: |
| `SSB(data)` | 5.6 | 0.0 | 0.0 | 0.0 |
| `reconstruct()`, two branches | 13.1 | 7.5 | 0.0 | 9.5 |
| show, drags, releases | 16.5 | 7.5 | 3.3 | 9.5 |
| region refit | 5.7 | 0.0 | 0.0 | 5.7 |
| next full-field preview | 16.5 | 7.5 | 3.3 | 9.6 |

The session peak is the G(q, k) build, which `reconstruct()` runs first (and
the preview after a refit runs again): 8.8 GiB of G(q, k), 2.4 GiB of build
blocks and FFT batches, and about 3.6 GiB of MLX cache that the build's blocks
of different widths leave (16.8 GiB with the encoded data). Capping MLX's cache
at 0.5 or 1 GiB during the build cuts its peak by 2.5 GiB but makes the build
37 to 47 % slower (1.25 to 1.35 s against 0.92 s), so the cache is not capped.
Inside `reconstruct()` the fused tilt search (`thick_sample.thick_fit_batch`,
band partial sums of G with no per-pixel row IFFT) adds 0.5 GiB of live
arrays; the branches' loss and phase evaluations run in 256-pixel tiles
(`reconstruct.FIT_LOSS_ROW_IFFT_TILE_BF`), 1.5 GiB less than 1024-pixel tiles
at the same fit time (8.9 s either way). The 3.6 GiB cache after a slider
release is the 1024-pixel loss tile, kept for the next release.

The standard search (`find_aberrations(tilt=False)`) evaluates pairs of
candidates with the exact loss in packs of bright-field pixels. On the 128 GB
M5 Max a pack of 1,024 pixels takes the same time per pair as one of 2,476
(217.5 against 218.1 ms, interleaved medians, the WindowServer using 43 % of
the GPU) and holds 4.0 GiB of row IFFT instead of 9.8 GiB, so the packs are
1,024 pixels; 512 was 2.9 % slower. The interactive loss tile sweep:

| Loss tile (pixels) | Loss preview | Live above G | Cache kept |
| ---: | ---: | ---: | ---: |
| 64 | +22.8 % | 0.14 GiB | 0.23 GiB |
| 128 | +11.8 % | 0.26 GiB | 0.48 GiB |
| 256 | +3.1 % | 0.51 GiB | 0.98 GiB |
| 512 | +3.7 % | 1.01 GiB | 1.48 GiB |
| 1024 (default) | 0 | 2.01 GiB | 3.48 GiB |

Every tile and pack size gives the same bits. Slider releases keep the 1024
tile: smaller tiles cost 3 % or more of a release.

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

**The loss row IFFT** (3.5 GiB kept after a loss by default; +9.6 / +15.1 GiB
with the whole chunk). The 512² loss sums one chunk of bright-field pixels at a
time: 4096 on Macs with 96 GB or more, 1024 at 64 GB, 512 below. Its row IFFT,
2 MB of complex64 per pixel, is computed in tiles of 1024 pixels (2.1 GB), and
MLX's allocator keeps the freed tile, so the next slider release reuses it
instead of mapping it again (clearing it after every release made a release
2.4 times slower).

**Fit and refit buffers**: the joint fit's candidate batches and loss tiles,
about 6 GiB above G(q, k) on the 512² scan and 5 GiB on gold (10 and 17 GiB
with the whole loss chunk), and a region refit's own G(q, k) (2.2 GiB for 256 × 256) plus its
fit. `find_aberrations` and `reconstruct()` now return MLX's cached buffers when
they finish, and a region session returns its own on close.

## The loss route: tiles by default, the whole chunk on request

`quantem.gpu.direct_ptycho.mps.reconstruct.LOSS_ROW_IFFT_TILE_BF` (default 1024) sends a
512² loss chunk larger than the tile through the tiled path a cancellable
preview takes: the same reduction boundaries and float32 order, so the same
loss and phase bits, with one tile of row IFFT alive at a time. On a Mac with
96 GB or more that has memory to spare, set it to `None` before the first
preview to keep the whole 4096-pixel chunk, for loss previews about 4 % faster
and 6 to 12 GB more resident memory. Below 96 GB the chunk is 1024 pixels or
fewer and is not split either way. Interleaved medians of 3 × 8 thick-sample
loss previews on the 512² scan:

| Route | Loss preview | MLX cache kept | MLX peak |
| --- | ---: | ---: | ---: |
| whole chunk (`None`) | 263 ms | 9.5 GiB | 16.8 GiB |
| tiles of 256 | +7.9 % | 1.0 GiB | 9.3 GiB |
| tiles of 1024 (default) | +4.4 % | 3.5 GiB | 10.8 GiB |
| tiles of 2048 | +3.9 % | 5.5 GiB | 12.8 GiB |
| tiles without a wait between them | +4 to 7 % | 9.5 GiB | 16.8 GiB |

Each tile waits for the GPU before the next is built; without that MLX holds
every tile's row IFFT at once and nothing is saved. In the session above the
tiled releases were not slower (0.77 against 0.83 s); the interleaved
measurement, on a GPU that the window server shared, puts tiles about 4 %
behind. What would remove that cost: the
row and column IFFT fused per tile, overlap of one tile's row IFFT with the
previous tile's column stage, or the real-output inverse transform on the loss
path, which halves the intermediate.

## What fits which Mac

macOS grants the GPU a working set below the RAM size (this Mac reports
107.5 of 128 GiB in `mx.device_info()["max_recommended_working_set_size"]`;
smaller Macs and older macOS grant less, about two thirds). The session needs
the encoded data plus G(q, k) resident, then one transient at a time: the
G(q, k) build (about 6 GiB with MLX's cache), a loss tile (2.1 GB, 1.1 GB below
64 GB) or a fit (0.5 to 2 GiB); a region refit runs with the full field's
G(q, k) freed. A whole session peaks at 16.8 GiB (512² scan) and 17.4 GiB
(gold) on this Mac, while G(q, k) is built.

| Mac | 512² scan: steady 10.8 GiB, session 16.8 GiB | gold: steady 13.1 GiB, session 17.4 GiB |
| --- | --- | --- |
| 16 GB | does not fit: G(q, k) alone is 8.8 GiB | does not fit |
| 24 GB | previews fit (about 14 GiB); the G(q, k) build pages | at the limit; the build pages |
| 32 GB | fits | fits, close to the limit while G(q, k) is built |
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
