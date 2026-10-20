# MPS performance: what got faster, why the rest is slow, how to profile (2026-10-20)

This page collects one night of Apple GPU (MPS/Metal) work on the SSB preview,
the aberration search and Show4DSTEM's detector sums, explains where the time
still goes, and describes the regression harness that keeps these paths from
slowing down unnoticed. The browser side is in quantem.widget's
`docs/maintainer/2026-10-20-mps-performance.md`.

Hardware: Apple M5 Max, 128 GB, macOS 26.4, torch 2.13, MLX 0.32. "A
512² × 192² scan" below is one full 512 × 512 × 192 × 192 uint16 acquisition
(8,939 bright-field pixels) that is not public; numbers labelled "gold" are on
the public gold scan (`bobleesj/quantem-data`, `4dstem/gold_512`, 7,644
bright-field pixels) and can be reproduced by anyone with the harness below.

## Summary

| Path | Before | After | Limit now |
| --- | ---: | ---: | --- |
| SSB preview, native 512² (C10 drag) | 55.7 ms | 22.2 ms | DRAM: ~7 GB of `G` reads per redraw |
| SSB preview, 2× 1024² | 37.9 ms | 27.3 ms | DRAM plus the 1024² inverse FFT |
| `find_aberrations(tilt=False)`, 200 trials | 25.6 s | 25.1 s | DRAM: the loss's 2 MB per-BF intermediate |
| Show4DSTEM BF drag, kernel per update | 5.89 ms | 1.88 ms | command waits; changed-pixel decode |
| Show4DSTEM ADF resize, kernel per update | 7.74 ms | 1.82 ms | same |
| Scan-ROI mean pattern (76.8 px radius) | 47 ms | 47 ms | one pass per (pixel, block); not changed |

The preview and detector rows are on the 512² × 192² scan, timed in one process
with the old and new code interleaved; every change kept the results
(phase within float32 rounding, detector sums bit-exact). The search row is
the standard fit; since the same day `find_aberrations()` defaults to the
tilt-corrected joint fit and, for an estimated rotation, fits both 180-degree
branches (24.4 s on gold, against 23.9 s for one standard fit).

## What we optimized and why

### Separable C10/C12 coherent preview

For the two-term probe the aberration function is a quadratic form, so each
transfer phase splits into a per-output term and a product of one row phasor
and one column phasor per bright-field pixel
([details](2026-10-20-mps-ssb-separable-coherent.md)). The per-term kernel
evaluated two soft-edge geometries, two chi values and two precise sincos for
every (output frequency, BF pixel) term; the separable kernel reads two table
phasors instead. Native preview 55.7 → 24.2 ms, 2× 37.9 → 27.6 ms, phase error
against float64 below the old kernel's.

### Nyquist row in the main launch

At native sampling the Nyquist row is its own anti-Hermitian partner and has
to be computed, not mirrored. A separate 255-thread launch added its whole
per-thread loop as a serial tail after the main grid; one extra grid row of
the main launch does the same work in parallel. Native 24.2 → 22.6 ms.
Running the lone Nyquist column as its own launch made things 1.7 ms slower
for the same reason.

### BF-subset previews start with empty caches

A correctness fix found while profiling: a reduced-BF preview context copied
the full session's fine-grid probe and separable geometry and indexed them
with subset BF positions. Subsets now build their own.

### Incremental exact masked sum (`camera_delta`)

A dragged or resized virtual detector changes a few hundred valid detector
pixels per step, but every masked sum on a Metal count resident re-planned and
re-summed the whole mask. `CountDetectorCompute` now keeps the selection and
exact uint64 image of its last sum; a mask that differs in at most 768 valid
pixels adds the counts of the pixels that joined and subtracts those that
left, which equals the full sum exactly (integer arithmetic). Resident-level
step: BF drag 4.49 → 1.39 ms, ADF outer resize 7.94 → 2.12 ms, rect drag
1.86 → 1.12 ms.

### Bounded views keep their masked sum on the host

`BoundedDetectorCompute.masked_sum` moved the mask to the device, read the
native session's host image, uploaded it again and the session downloaded it
back. The overhead went from 0.70 to 0.03 ms per update. With the previous
change, Show4DSTEM's kernel time per detector update is BF drag 5.89 → 1.88 ms
and ADF resize 7.74 → 1.82 ms.

## Where the time goes now

### The preview is close to memory bound

A streaming read of the prepared spectra `G` (9.41 GB on the 512² × 192² scan)
takes 19.2 ms: 489 GB/s. At native sampling the contributing `G` reads alone
are about 7 GB per redraw, about 14 ms at that rate; the kernel takes 22 ms
including the phasor tables, the output writes and the inverse FFT
(0.24 ms). Groups of 8 or 32 BF pixels, other threadgroup shapes, unrolling
with hoisted loads, and reordering alias rows for cache reuse gave no
repeatable gain. Further speed needs fewer bytes, not fewer instructions.

On gold the 2× preview costs 2.6 times the native one (72 versus 28 ms; on
the 512² × 192² scan 2× was only 1.2 times). Gold's 0.5 Å scan step puts the
bright-field support beyond the 2× Nyquist frequency, and the half-spectrum
shortcut applies only when both Nyquist edges lie outside the complete
soft-aperture support (`_coherent_columns`), so the 2× preview computes the
full plane. This is the likely cause; it was not profiled separately.

### The search is bound by one intermediate

The standard fit, `find_aberrations(tilt=False)`, spends 22.8 of its 24.4 s
in the exact phase-variance loss; Optuna takes 0.26 s and Nelder-Mead's 36 evaluations 4.3 s. A TPE pair
costs about 185 ms: the fused correction and row inverse FFT writes a
complex64 512 × 512 image, 2 MB, per BF pixel and candidate, and the column
inverse FFT reads it back. That is about 37.5 GB of traffic per candidate, a
77 ms floor at 489 GB/s; we measure about 93 ms. Smaller packs of 8 to 64
planes did not keep the intermediate in cache (8 planes: 317 ms per pair).

The next step is a real-output inverse FFT on the loss path. Each BF pixel's
corrected spectrum is anti-Hermitian, C(−q) = −conj(C(q)), except at DC and on
the Nyquist lines (checked to 3e-7 relative), so its inverse transform is
i times a real image plus a constant and two separable Nyquist-line terms.
Writing half the spectrum halves the intermediate and should take the search
from about 25 s to about 15 s. MLX's `irfft2` (19.1 µs per 512² plane versus
33.9 µs for `ifft2`) shows the size of the gain but cannot be used directly:
the loss path is a custom kernel at about 10.3 µs per plane and candidate, so
it needs its own real-output kernels.

### uint32 sources take the generic bit-lane resident

The gold masters store uint32 counts. `io.load` keeps them as uint32 (no
narrowing to uint16, even when every count fits), so they live in the generic
bit-lane ANS resident rather than the count resident, and nothing above
applies to them:

- a Show4DSTEM detector sum decodes the whole scan in 1,155 bounded batches
  with a GPU wait after each (of 2.5 s per sum, 1.6 s are
  `torch.mps.synchronize` and 1.0 s Metal command completion): 2.3 s per
  detector change on gold, against 6 ms incremental and 12 ms full on the
  same counts re-encoded as uint16;
- `SSB(data)`, which estimates the rotation from the centre-of-mass curl,
  takes 6.5 to 7.8 s on gold, against 0.6 s on the uint16 512² × 192² scan;
  `io.load` takes 4.3 s against 1.7 s.

Narrowing uint32 counts that fit into the uint16 count resident at load, or
batching the bit-lane decode without a wait per batch, would bring gold to
the count-resident numbers.

### Scan-ROI mean pattern

Averaging the patterns inside a scan circle runs one pass per (detector
pixel, scan block): 47 ms for a 76.8 px radius circle (18,513 scans, 18
updates per second in the browser) and 13 ms for 20 px. It was not changed.

## Things that look like a regression but are not

- **Another GPU client.** WindowServer takes 8 to 35 % of the GPU while the
  screen is shared or streamed; `mediaanalysisd` (Photos and Spotlight media
  analysis) runs on the GPU for minutes after files arrive; another agent's
  benchmark or test suite. Each slows every Metal kernel. The harness lists
  every other process that used the GPU in the two seconds before and after
  the run, from the IOAccelerator statistics (`ioreg -l`), and `compare`
  warns about any at 10 % or more. With the screen shared (WindowServer at
  30 to 50 %) and other agents queued for the GPU, the same search candidate
  measured 81 ms in one run and 141 ms in another an hour earlier.
- **Timings taken without the lock.** On a shared machine run benchmarks
  under the same lock as other GPU work, one command at a time.
- **Cold first calls.** Metal pipeline compilation and the first BF
  preparation are paid once per process; the harness warms every repeated
  stage first and reports `ssb_prepare` separately.
- **The page cache.** `io.load` reads 4.6 GB of gold; the first load after a
  reboot or under memory pressure reads the disk. The harness discards the
  first of four loads.
- **A changed fit.** `find_aberrations()` is timed once. If the fitted
  coefficients change (`compare` prints a note), the trajectory changed and
  the time is not like for like.
- **SwiftShader.** Browser timings on Chrome's CPU renderer say nothing about
  the GPU; the widget harness refuses to run on it.

## How to profile

### The regression harness

`tests/benchmarks/mps_ssb.py` times every path above on the cached gold scan
and compares the result with this chip's committed baseline,
`tests/benchmarks/baselines/apple-m5-max.json`:

```bash
python tests/benchmarks/mps_ssb.py run --out report.json   # about two minutes
python tests/benchmarks/mps_ssb.py compare report.json
pytest -m benchmark tests/benchmarks                         # the same check as a test
```

`run --download` fetches the gold scan (4.6 GB) on first use; `run
--save-baseline` re-records this chip's baseline. On a shared machine,
record it from three runs instead: `merge a.json b.json c.json
--save-baseline` keeps the median run of every metric and refuses runs whose
fits differ. Each stage waits for torch
and MLX before its clock stops, and repeated stages report the p50 and p90 of
their repeats after warm-up calls. The report records the chip, macOS,
Python, torch, MLX, Metal architecture, quantem.gpu commit and whether the
checkout had local edits, and no file paths.

| Metric | What it times |
| --- | --- |
| `io_load` | `io.load(master, hot_pixel_correction="median")` |
| `ssb_session` | `SSB(data)`, including the rotation estimate |
| `ssb_prepare` | the first preview: BF evidence read and transformed |
| `preview_native`, `preview_2x` | a ShowPtycho slider drag step (no loss) |
| `preview_native_loss` | the slider release (with the variance loss) |
| `search_candidate` | half of one TPE pair of the exact loss |
| `find_aberrations` | the default fit, once (joint tilt fit on both branches of the estimated rotation) |
| `find_aberrations_standard` | one standard `tilt=False` fit at the kept rotation, once |
| `detector_full_uint32` | a Show4DSTEM detector sum on gold as loaded |
| `detector_update`, `detector_full` | the same on the counts as a uint16 count resident: a half-pixel disk move (incremental) and an alternating disk / annulus (full) |

`compare` exits 1 when any p50 is slower than the baseline by more than its
tolerance and by more than 0.5 ms. The default tolerance is 20 %; `io_load`
and `ssb_prepare` allow 35 % and `ssb_session` 30 %. Override per metric with
`--tolerance find_aberrations=0.3`. It also fails when the incremental
detector update no longer equals the full sum.

### Gold baseline on the M5 Max

| Metric | p50 | p90 |
| --- | ---: | ---: |
| `io_load` | 4.33 s | 4.39 s |
| `ssb_session` | 6.66 s | 6.76 s |
| `ssb_prepare` | 868 ms | – |
| `preview_native` | 27.7 ms | 29.0 ms |
| `preview_2x` | 71.9 ms | 73.6 ms |
| `preview_native_loss` | 123 ms | 126 ms |
| `search_candidate` | 80.9 ms | 83.3 ms |
| `find_aberrations` | 24.4 s | – |
| `find_aberrations_standard` | 23.9 s | – |
| `detector_full_uint32` | 2.28 s | 2.33 s |
| `detector_update` | 5.9 ms | 6.2 ms |
| `detector_full` | 12.2 ms | 13.7 ms |

The median of three runs (`merge`) on 2026-10-20 with this harness's commit (macOS 26.4, Python 3.13.13,
torch 2.13.0, MLX 0.32.0), each under the shared GPU lock, while WindowServer used 24 to 50 % of the GPU
(the screen was shared). 7,644 bright-field pixels; the uint16 update equalled the uint32 full sum in every run.

### Reading a failure

1. Look at the warnings first. An "other processes used the GPU" warning on
   either side means rerun on a quiet GPU before blaming the change.
2. Rerun the one failing metric's stage two or three times. A real
   regression repeats; contention does not.
3. A slower `preview_*` points at `ssb/mps/upsampling.py`; a slower
   `search_candidate` or `find_aberrations` at `ssb/mps/loss.py` and the row and
   column kernels; `detector_*` at `detector/counts.py` and
   `resident/mps/spatial.py`; `ssb_session` at the rotation estimate and the
   source decode.
4. Bisect with the harness: it runs on any checkout that has the paths.
5. A deliberate trade (more work for a correctness fix) re-records the
   baseline in the same commit, with the reason in the message.

### Deeper profiles

- Interleave the old and new implementation in one process (a module flag
  such as `upsampling._SEPARABLE`), alternating calls, and compare p50s: two
  separate processes minutes apart see different GPU contention.
- `cProfile` around one call separates Python, Metal command completion and
  `torch.mps.synchronize`; many short waits mean many small launches.
- Divide the bytes a kernel must read by its time and compare with a plain
  streaming read of the same buffer (489 GB/s here). Near that rate, only
  fewer bytes help.
