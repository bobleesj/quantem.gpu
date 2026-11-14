# Direct ptychography through fifth order on a 4 GB CUDA device: 512 x 512 x 192 x 192

Date: 2026-11-12

## Question

Can a 4 GB CUDA card (target: NVIDIA RTX A400) load a 512 x 512 x 192 x 192 uint16 acquisition, fit its aberrations
through fifth order, and reconstruct the whole scan with them? Which method costs the least memory, and what does
each choice cost in time?

## Answer

Yes, for a scan that compresses to about 2.6 GB or less. The method:

1. Keep the acquisition encoded on the device (2.46 GiB here) and reconstruct the whole scan from it in batches of
   32 detector columns. Working memory: 58 MiB. Nothing is binned, cropped or approximated.
2. Fit on the central 128 x 128 scan region with its spectra resident (the fast path), while the encoded acquisition
   waits in page-locked host memory, still encoded. Only the chunks that hold the region's rows return to the device
   while its bright-field crop is decoded. The whole acquisition returns after the fit.

Measured on scan A (below) with the device held to 3.9 GiB free at the start: `open` 1.3 s, `aberrations()` 4.7 s,
`run` 0.44 s, process peak 3.43 GiB (3512 MiB). The reconstructed phase equals the one from the same code on the
unrestricted card bit for bit (maximum absolute difference 0.0).

The fit region matters on this scan. The whole-scan fit, which needs 16.2 GiB, finds C10 +8.9 nm where the
128 x 128 and 256 x 256 region fits find -1.5 and -1.6 nm, and its whole-scan phase shows separate atomic columns
where the region fits' phase shows streaks (9 % more phase variance). So the 4 GB path computes, and its fit is not
yet the fit the open card gives. See "Fit region size".

**Superseded on 2026-11-13** ([the whole-scan fit on a 4 GB device](2026-11-13-whole-scan-fit-small-device.md)):
the small device now fits the whole scan on G(q, k) at the fit band's frequencies (1.6 GB), the same fit as the
dense path, from 3600 MiB free. The "about 420 MiB of GPU libraries" below was mostly local memory the resident
search's random-number kernel reserves; the fits now give it back.

## Setup

- Scan A: 512 x 512 scan, 192 x 192 detector, uint16, 2.58 GB as `.qem`. On the device: 2456 MiB encoded, of which
  145 MiB is the spatial index of detector queries. 9052 bright-field pixels. Scan B: same shape, 3.31 GB as `.qem`.
- CUDA: RTX PRO 6000 Blackwell shared with other jobs. A second process held device memory so that a stated amount
  was free when the test started (`memGetInfo`). Process memory was read from `nvidia-smi --query-compute-apps`
  every 50 ms. On this shared card about 370 MiB of the reported free memory could not be allocated (2772 MiB of
  154 MiB blocks out of 3140 MiB reported free), so "3900 MiB free" was about 3.4 GiB usable.
- Metal: Apple M5 Max, `swift run -c release metal-direct-ptycho-redraw-benchmark --scans 256,512 --orders low,fifth
  --factors 1,2 --runs 20 --warmup 3`, with a desktop session active.
- Not run on an RTX A400 or on Windows. Speed on that card is unmeasured.

## Numbers

### The whole flow on scan A, session opened from the file

`DirectPtycho.open(path)`, `aberrations()`, `run(fit, compute_loss=False)`. Process device memory in MiB.

| Free at start | open | aberrations() | run | Process peak | Result |
|---|---|---|---|---|---|
| 4096 MiB | 1.2 s | 6.2 s | 0.44 s | 3512 | passes |
| 3900 MiB | 1.3 s | 4.7 s | 0.44 s | 3512 | passes |
| 3800 MiB | 1.2 s | 4.5 s | fails | 3462 | the fit passes; the reconstruction cannot allocate 34 MB |

Inside `aberrations()`: moving the encoded acquisition to host memory 1.2 to 2.7 s, the fit 2.2 s, returning the
acquisition 0.5 s. The process holds 3028 MiB after `open`, at most that during the fit, and 3462 MiB after it.

Scan B (3.16 GiB encoded) does not load at 3.7 GiB free, so nothing after loading applies to it.

The largest scan of the same session (2.668 GB as `.qem`): passes at 4096 MiB free (fit 5.5 s, peak 3604 MiB); at
3900 MiB the fit passes and the reconstruction fails.

The same flow from a viewer that watches a folder (the viewer's detector sessions close for the fit, the folder
watch waits), 4300 MiB free: scan shown 2.3 s after its file appeared, fit and whole-scan phase 4.9 s, next scan
shown 2.4 s after its file appeared, its fit and phase 4.1 s, process peak 3610 MiB. At 4096 MiB free it passed
twice (peak 3602 MiB) and once could not return the scan to the device.

Batched against the dense path with the same aberrations, scan A, whole scan: 2.5e-5 rad at most, 5.8e-6 rad rms.

### Where the memory goes

| Item | MiB |
|---|---|
| CUDA context and decode kernels | 556 |
| Encoded acquisition (scan A) | 2456 |
| Regional fit, 128 x 128, beyond the context (crop, G(q, k), working planes), acquisition parked | 970 held, 1414 at the peak |
| GPU libraries a fit loads and the process keeps afterwards | 416 to 426 |
| Batched reconstruction at 1x, 2x, 4x output | 58, 82, 178 |

The libraries: with no array alive after the fit closes, the process holds 982 MiB against 556 MiB before it.
Clearing the cuBLAS workspaces, both FFT plan caches and the kernel memo returned 10 MiB. First-use costs measured
alone: Torch context 88, cuBLAS 98, `torch.linalg` 96, `cupy.linalg` 66 MiB.

### Time of the batched reconstruction against the column batch

Scan A, full 512 x 512 output, RTX PRO 6000, best of 3.

| Columns a batch | C10 and C12 | Through fifth order |
|---|---|---|
| 1 | 4332 ms | 4343 ms |
| 4 | 1105 ms | 1117 ms |
| 8 | 586 ms | 596 ms |
| 16 | 338 ms | 339 ms |
| 32 | 198 ms | 210 ms |

At 32 columns through fifth order: 243 ms at 2x output and 312 ms at 4x. On this card the time follows the number of launches, not
the arithmetic.

### What batching costs in time (Metal, Apple M5 Max)

A batched reconstruction transforms every bright-field column again each time. With the spectra resident only the
correction runs.

| Scan | Transform every column once | Redraw, spectra resident, 1x | 2x |
|---|---|---|---|
| 256 x 256, 8804 bright-field pixels | 0.19 s | 6.5 ms (C10, C12), 7.1 ms (fifth) | 9.6 ms, 10.4 ms |
| 512 x 512, 8804 bright-field pixels | 0.66 s | 22.1 ms, 22.2 ms | 31.3 ms, 36.8 ms |

Re-transforming costs about 30 times a resident redraw. The resident spectra of a 512 x 512 scan are 9.3 GB, so a
4 GB device has no choice for the whole scan; a 128 x 128 region's are 0.6 GB.

### Fit region size

Same scan, fits on the central 128 x 128 and 256 x 256 regions and on the whole scan, each followed by the
whole-scan reconstruction:

| | 128 x 128 | 256 x 256 | whole 512 x 512 |
|---|---|---|---|
| C10 (nm) | -1.5 | -1.6 | +8.9 |
| C12 (nm) | 9.7 | 9.8 | 15.7 |
| C21, C23 (nm) | 98.5, 81.0 | 98.5, 84.9 | 77.2, 60.5 |
| Half-band check of the higher orders (limit 0.5 rad) | 2.68 rad | 4.99 rad | 3.35 rad |
| Whole-scan phase, standard deviation (rad) | 0.0544 | 0.0546 | 0.0569 |
| Fit memory beyond the CUDA context, peak | 1414 MiB | 4340 MiB | about 16 000 MiB |

The two region fits agree (their whole-scan phases differ by 0.005 rad rms, 0.024 rad at most). The whole-scan fit
differs from both by 0.039 rad rms and 0.14 rad at most (correlation 0.76) and gives the sharper image. The two were
not scored on one loss: the shared card was too full for a second dense run. The fit's own check declines the
higher orders at all three sizes.

Fit memory, measured on the whole process: 7 bytes a bright-field pixel and scan position, of which 4.1 are
G(q, k) (complex64, the half plane Hermitian symmetry leaves) and 2.7 the decoded counts over the disk's box, plus
about 0.5 GiB that does not grow with the region. For the whole scan G(q, k) is 9.53 GB and the counts 6.5 GB; the
dense session peaked at 16 552 MiB. A 4 GB card has about 2900 MiB beyond the context once the scan is parked, so
it holds the 128 x 128 fit only. Without the decoded counts a 256 x 256 fit would need about 2700 MiB.

### Host memory transfer

2.4 GB from the device to host memory, two runs on the shared test machine: into ordinary memory 35.3 s and 19.5 s
as one array (38.4 s and 4.4 s in 16 pieces); into page-locked memory 2.0 s and 1.1 s to lock the pages and 0.11 s
to copy, both times. Back to the device: 0.4 to 1.5 s. Inside the flow the park step took 11 to 27 s with ordinary
memory and 1.2 to 2.7 s with page-locked memory.

## What changed

- `StreamedCounts.park()` and `restore(first, stop)`: the encoded chunks move to page-locked host memory and back,
  whole or the chunks of a scan range. Reading a parked scan raises.
- `DirectPtycho.aberrations()` on a session that opened its own file and cannot hold the dense spectrum parks the
  acquisition for the regional fit (`_region_for_fit`). The dark-field image the 180-degree test needs is computed
  before parking.
- `DirectPtycho.on_park`: a caller of a session over its own dataset sets it to vouch for the dataset. It runs before
  the acquisition is parked, so a viewer closes its detector sessions there. A failed `restore()` raises
  `MemoryError` and leaves the acquisition intact in host memory.
- `StreamedDirect.phase` takes the 14 higher-order magnitudes and azimuths (`thick_wave_sum_full_kernel`).
- An encoded integer acquisition goes to the batched path when free device memory is under a quarter of its decoded
  size, with one printed line; `aberrations()` then fits the largest central region that fits and prints it.
- Tests: `test_parked_counts_return_the_same_counts_and_refuse_reads_meanwhile`,
  `test_fit_region_parks_the_owned_acquisition_then_restores_it`,
  `test_streamed_fifth_order_phase_matches_stored_spectra`.

## Limits

- The fit a 4 GB card holds (128 x 128) is not the whole-scan fit: on scan A the two differ by 10 nm in C10 and the
  whole-scan fit gives the better image. A whole-scan fit within 4 GB needs G(q, k) built in batches and kept only
  at the frequencies that carry signal, or an objective evaluated in batches; neither exists yet.
- The scan must leave room: about 3.4 GiB usable for scan A. Scan B does not fit a 4 GB card.
- A dataset the caller loaded (`io.load`, then `DirectPtycho(data)`) is parked only when the caller set `on_park`,
  because a viewer on it may hold the same device memory. Without it the fit on a small device raises `MemoryError`.
- Other jobs on the test card changed its free memory by gigabytes between runs, so the budgets are approximate.
- After a fit the process holds about 420 MiB of GPU libraries until the kernel restarts.
- The reconstruction at 3800 MiB free failed with a raw allocation error, not the session's own message: the
  device reported 327 MiB free that it could not allocate.
- One test of the suite did not run to a verdict on the shared card:
  `test_one_decode_batch_beyond_4_gib_keeps_every_frame_in_place` needs more free memory than was available.

## Rejected

- A column batch above 32. The FFT-input kernel (`sc_columns_fft`) moves one tile of 32 columns through shared
  memory, so columns past the 32nd of a batch are never written: 64 and 128 returned phases wrong by 0.022 to
  0.042 rad, and NaN. Batches of 8 and 32 agree to 0.0 rad. The times first measured for 64 and 128 (153 and
  137 ms) are void.
- Releasing the acquisition by closing it and reading the file again. It decoded the fit region after the release
  point and ran out of memory by about 60 MB; reading an Arina master again also means encoding it again.
- Fitting with the acquisition resident. The region's crop and G(q, k) need about 1.0 GiB and 0.7 GiB is left.
- Freeing the libraries a fit loads (cuBLAS workspaces, plan caches, kernel memo): 10 MiB of 426 came back.
- Parking into ordinary host memory: 11 to 27 s inside the flow.
- A 256 x 256 fit region on this device: 4340 MiB beyond the context, about 2900 are there. An earlier estimate of
  13 bytes a pixel and position (7.7 GB at 256 x 256, 31 GB for the whole scan) had the fixed 0.5 GiB folded in;
  the measured cost is 7 bytes plus 0.5 GiB.
- Limiting the CuPy pool to rehearse a small card: Torch and unpooled allocations are not bound by it.
