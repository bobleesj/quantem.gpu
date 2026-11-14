# The whole-scan fifth-order fit on a 4 GB CUDA device

Date: 2026-11-13. Follows [Direct ptychography through fifth order on a 4 GB CUDA device](2026-11-12-small-device-direct-ptycho.md).

## Question

The earlier record fitted a central 128 x 128 region on a small device, and that fit differed from the whole-scan fit
(C10 -1.5 nm against +8.9 nm on scan A, with a visibly worse image). Can a 4 GB device fit the whole 512 x 512 scan,
and give the same fit the large card gives?

## Answer

Yes. The fits read G(q, k) only in their band, 0.2 to 0.9 1/A. On scan A that is 23,448 of the half plane's 131,584
spatial frequencies: 1619 MiB for 9052 bright-field pixels, where the whole half plane is 9.5 GB. One pass of the
batched reconstruction's detector columns gathers G at those frequencies into page-locked host memory. During the
fit the band spectrum and the encoded scan take turns on the device: the band spectrum while trials are scored, the
encoded scan while the fit reconstructs a phase. Nothing is dropped from the band, so the fit is the dense fit:

| | Dense session, whole scan | Encoded session, band spectrum |
|---|---|---|
| C10, C12 (nm), phi12 (deg) | 8.93, 15.67, 16.47 | 8.93, 15.67, 16.47 |
| C21, C23 (nm) | 77.22, 60.52 | 77.21, 60.55 |
| Sample tilt (mrad), depth spread (nm) | (3.677, -2.109), 23.38 | (3.677, -2.110), 23.38 |
| Agreement (fraction of band power) | 0.06642 | 0.06642 |
| Whole-scan phase against the dense session's | | 8.7e-5 rad rms, 4.5e-4 rad at most |

On the rehearsed small card (below) the flow passes with 3600 MiB free at the start for scan A and 3700 MiB for the
largest scan of the session, where the region fit needed 3900 to 4096 MiB. The floor is about the encoded scan plus
1.05 GB: the scan returns to the device next to the CUDA context for each reconstruction.

The match holds for the fit as it stands: on scan A the half-band check rejects the higher orders (3.35 rad against
0.5 rad, on the large card too), so the fit shown is the +-2 rad one and C30 sits on that box's edge in every fit
(3094.68 nm is 2 rad at the 30 mrad aperture edge). C30 to C56 are bounded there, not measured.

## Setup

- Scan A of the earlier record: 512 x 512 scan, 192 x 192 detector, uint16, 2456 MiB encoded on the device, 9052
  bright-field pixels, scan sampling 0.271 A. The largest scan of the same session: 2544 MiB encoded.
- RTX PRO 6000 Blackwell (188 SMs) shared with other jobs. A second process held device memory so that a stated
  amount was free when the test started. Process memory from `nvidia-smi --query-compute-apps` every 20 to 50 ms.
  A run counts only when the free memory after `open` matched the budget (budget minus about 3030 MiB); other jobs
  changed free memory by gigabytes in several runs, which were discarded.
- `DirectPtycho.open(path)`, `aberrations()` (standard fifth-order fit, 200 trials, both 180-degree branches),
  `run(fit, compute_loss=False)`.

## Numbers

### Small-card rehearsal, scan A

| Free at start | `aberrations()` | `run` | Result |
|---|---|---|---|
| 3800 MiB | 9.0 s | 0.55 s | passes, whole-scan fit |
| 3700 MiB | 8.1 s | 0.55 s | passes |
| 3600 MiB | 9.3 s | 0.55 s | passes |
| 3500 MiB | | | fails: the scan cannot come back for a reconstruction |
| 3400 MiB | | | fails while gathering the band spectrum |

The largest scan of the session (2544 MiB encoded) passes at 3800 and 3700 MiB (C10 -22.55 nm, C12 8.73 nm, agreement
0.10387); it failed at 3600 MiB until the parked scan returned into one allocation instead of one per chunk (see "Memory
the fit kept"), and passes there now. Fifteen runs in which other jobs changed free memory, or in which a ballast that
followed other jobs misjudged it, were discarded.

From a viewer that watches a folder (`Show4DSTEM.watch`, then the viewer's direct ptychography), 3800 and 4096 MiB
free: whole-scan fit and phase 4.5 and 4.2 s, process peak 3292 MiB, the next scan shown 2.4 and 2.0 s after its
file appeared and fitted in 2.0 and 1.9 s.

### Where the time goes inside `aberrations()`

| Step | Time |
|---|---|
| Gather G at the band frequencies (one pass of 32-column batches) | 0.8 to 1.8 s |
| Move the encoded scan to page-locked host memory | 1.4 to 1.5 s |
| Each reconstruction during the fit: scan back 0.25 s, phase 0.25 s, scan out again 0.16 s | 3 per fit |
| The scan back for good at the end | 0.25 s |
| The fit itself (`joint_fit_s`) | 3.3 to 6.0 s |

### Process memory during the fit, large card

| Stage | MiB |
|---|---|
| After `open` (context and encoded scan) | 3028 |
| Gathering the band spectrum | 3092 |
| Fitting on the band spectrum, scan parked | 2186 to 2682 |
| A reconstruction during the fit, band spectrum off the device | 3512 to 3586 with the stack limit kept, 3108 to 3182 set back |
| After the fit, scan back | 3498 kept, 3094 set back |

### Six real scans on a rehearsed 4 GB card

Every scan 512 x 512 positions at 300 kV. "Dense" is the whole-scan fit from the dense spectrum on the unrestricted
card; the 4 GB column is `DirectPtycho.open(path)`, `aberrations()`, `run(fit, compute_loss=False)` with 4096 MiB
free at the start, each run checked for other jobs' interference by its free memory after `open`.

| Scan | Detector, semiangle, step | Encoded | Band spectrum | `aberrations()` at 4096 MiB | `run` | 4 GB phase vs dense (rms) |
|---|---|---|---|---|---|---|
| oxide scan A | 192 x 192, 30 mrad, 0.271 A | 2456 MiB | 1.70 GB, on the device | 7.8 s | 0.55 s | 8.7e-5 rad |
| oxide, largest of session | 192 x 192, 30 mrad, 0.271 A | 2544 MiB | 1.67 GB, on the device | 7.8 s | 0.54 s | 8.0e-5 rad |
| oxide zone B | 192 x 192, 30 mrad, 0.383 A | 2284 MiB | 3.41 GB, 2.34 GB on the device | 19.5 s | 0.55 s | 4.7e-5 rad |
| MoS2 bilayer | 192 x 192, 30 mrad, 0.373 A | 1348 MiB | 3.29 GB, 2.34 GB on the device | 18.0 s | 0.55 s | 3.5e-5 rad |
| Logic device (HDF5) | 192 x 192, 30 mrad, 0.5 A | 2173 MiB | 5.67 GB, 2.0 GB on the device | 36.6 to 37.4 s | 0.55 s | 1.4e-5 rad |
| Lamella, low dose (HDF5) | 96 x 96, 22.6 mrad, 0.664 A | 171 MiB | 1.19 GB, on the device | 6.3 s | 0.47 s | 0.044 rad (see below) |
| the second oxide scan | 192 x 192, 30 mrad, 0.383 A | 3160 MiB | | did not load (now loads into host memory, see below) | | |

On the dense path the same fits take 1.3 to 3.5 s. Closer to what an A400 has, with the shared passes and the memory
fixes below (`aberrations()`, each run checked by its free memory after `open`):

| Scan | 3700 MiB, before | 3700 MiB, now | 3400 MiB, before | 3400 MiB, now | Phase vs dense (rms), now |
|---|---|---|---|---|---|
| oxide scan A | 8.1 s | 8.0 to 8.5 s | fails while loading | read from host memory (below) | 8.7e-5 rad |
| oxide, largest of session | 7.6 s | 7.8 to 7.9 s (3600 MiB: 7.9 s, failed before) | fails while loading | read from host memory (below) | 8.0e-5 rad |
| oxide zone B | 21.6 s | 18.5 to 19.9 s | 22.8 s | 20.4 s | 4.6e-5, 5.1e-5 rad |
| MoS2 bilayer | 20.6 s | 18.2 s | 21.5 s | 19.9 s | 3.5e-5, 3.8e-5 rad |
| Logic device (HDF5) | 38.7 s | 31.2 to 33.1 s | fails while loading | 32.7 s | 1.4e-5 rad |
| Lamella, low dose (HDF5) | 6.5 s | 6.3 to 6.4 s | 6.8 s | 6.4 s | 0.044 rad (see below) |

The logic device's three runs at 3700 MiB gave the same phase bit for bit; its fit is the dense fit as before (C10 3.23
nm, C12 4.71 nm, agreement 0.09472). More of its band now stays on the device (2.08 GB against 1.65 GB) because its
load no longer loses memory to page rounding (see "Memory the fit kept"). Runs in which another job on the shared card
took memory part way through (an allocation failing while the process held about 600 MiB) were repeated; each such
case passed twice on repeat.

At 3400 MiB the encoded oxide scans alone (2.4 to 2.6 GB, one block from `.qem`) do not fit beside the CUDA context,
which takes about 560 MiB of the budget on this 188-SM card; they now stay in host memory and the device reads them
there (see "A scan the device has no room for"). From a viewer that watches a folder at 3700 MiB, the free memory checked
before and after the run: oxide scan A fit and phase 5.2 s (7.4 s with the current code, on a card other jobs kept busy),
zone B shown 1.7 s after its file appeared (2.2 s) and fitted in 18.5 s (19.7 s), process peak 3278 MiB (3206 MiB).

The low-dose lamella's band and dense fits land on different points (C12 9.80 against 9.14 nm, C21 171 against 193 nm)
whose agreements on one objective differ by 0.011 % (1.49205e11 against 1.49188e11, the band fit higher); both are
reproducible, and the two phase images look alike by eye. Its agreement is 0.18 % of the band power, so the objective
is flat and float32 rounding differences steer the optimizer: the fit is not determined to better than this at this
dose.

The lamella first ran out of memory: the session chose the dense path, because the old test (free memory against a
quarter of the decoded scan) and then 85 % of the free memory against the 7-byte rule both said it fit, but the dense
fit's working arrays grow with the 117,439 band frequencies. `_exceeds_device` now counts 70 % of the free memory
against 7 bytes a bright-field pixel and scan position plus 0.5 GiB.

### A band spectrum larger than the device: the logic device

The band grows with the field of view: the number of band frequencies goes as (scan step x scan side)^2. At 0.5 A on a
512 x 512 scan (the logic device, 8928 bright-field pixels) it is 79,441 frequencies, 5.67 GB, more than a 4 GB card holds
next to anything. At 0.373 A (MoS2, 9298 pixels) it is 3.29 GB.

Every fit objective is a sum over bright-field pixels (N_q = sum_k G conj(gamma), D_q = sum_k |gamma|^2 and their
derivatives). So the band spectrum is split by pixels: the first pixels stay on the device, the rest stay in
page-locked host memory and are copied in slabs of 256 pixels into two alternating device buffers on a second
stream while the kernels read the previous slab (`DirectEngine.band_slabs`); each objective adds the slabs' sums in
float64. A spectrum the device holds whole is one slab, the launch the dense path has always made.

| Logic device, whole-scan fit | `aberrations()` | C10, C12 (nm) | Agreement |
|---|---|---|---|
| Dense spectrum, full card | 1.9 s | 3.23, 4.71 | 0.09472 |
| Band spectrum on the device, full card | 8.0 s | 3.23, 4.71 | 0.09472 |
| Band spectrum read in place from host memory (zero-copy), full card | 99.7 s | 3.23, 4.71 | 0.09472 |
| 2.0 GB on the device, rest streamed, every slab whole | 62.4 s | 3.23, 4.71 | 0.09472 |
| Same, subsampled passes copy only their rows (pitched copy) | 34.6 to 35.3 s | 3.23, 4.71 | 0.09472 |
| 4096 MiB free (rehearsal, three runs) | 36.6 to 37.4 s | 3.23, 4.71 | 0.09472 |

A fit makes 152 passes over every pixel and 242 over every 8th or 64th (the refinement's steering subsamples), almost
all with one candidate: the optimizer is sequential, so candidates cannot share a pass. With 2.0 GB resident, each full
pass copies 4.0 GB; the subsampled passes now copy an eighth of that. The time is the bus: about 27 GB/s on this
card's PCIe 5.0 x8 link. An RTX A400 has PCIe 4.0 x8, about half that, so expect roughly twice the time there.

### Fewer passes over a streamed spectrum

A streamed fit's time is the number of passes over G times the bytes each streams. Three changes cut it without
changing any number the fit computes:

- The fifth-order stage ran four independent refinements one after another: the +-20 rad fit, the two half-band fits
  of the higher-order check and the +-2 rad fallback. They now run together (`SharedPasses`, one thread each): every
  round, the waiting requests of all four are evaluated in one values pass and one gradients pass per stride. A
  candidate's sums do not depend on the others in its pass; values and gradients are bit for bit those of the
  candidate alone (test `test_shared_passes_give_every_refinement_what_it_gets_alone`). The fallback now always runs
  and is kept only when the check declines the higher orders, as before.
- The joint search and the standard comparison's search (for `tilt_fit_gain`) are independent too and share their
  rounds (`_MergedEvaluations`) when they run in the same call, which is when the rotation is given or recorded;
  `_fit_encoded` now loops slabs outside the candidate groups so a round of 16 trials streams G once. With an
  estimated rotation the comparison runs after the branch decision and is not merged.
- The working reserve beside the band spectrum depends on the device and the band: 64 MiB plus 8.5 KiB a band
  frequency, plus the local memory the random-number kernel makes CUDA reserve for every thread (404 MiB here, about
  13 MiB on a 6-SM A400), instead of a flat 768 MiB. The per-frequency part is measured: the peak memory in use over a
  whole fit (an allocator hook), with the band held whole (the most launch partials), was 206 MiB above the band at
  23,448 frequencies, 379 MiB at 46,673 and 676 MiB at 79,441. A flat 256 MiB ran out at 3400 MiB free on the logic
  device once the refinements shared their rounds, and a first per-frequency rule (4.5 KiB, from pool samples taken
  only after each pass) ran out on the second oxide scan at 3400 MiB and oxide scan A at 3000 MiB. On this card such a shortfall can hide
  behind the 404 MiB of local memory the fit releases part way; an A400 has no such slack. The larger reserve keeps
  less of the logic device's band on the device at 3700 MiB (1.59 against 1.92 GB): 32.8 and 34.5 s against 31.9 and
  31.5 s, alternated in one sitting.

| Whole-scan fit, full card | Before | After | Phase vs dense (rms) |
|---|---|---|---|
| oxide scan A, band on the device | 5.5 s | 3.4 s | 8.7e-5 rad, the same bits as before |
| Logic device, 1.65 GB of 5.67 GB on the device (as at 3700 MiB) | 35.3 s | 28.4 s | 1.4e-5 rad, the same bits as before |

### Memory the fit kept between its own steps

Three holders of device memory that a small device needs, found by listing live arrays at each return of the scan:

- Encoded chunks loaded from HDF5 each had their own allocation, rounded up to whole 2 MiB pages: 100 allocations of
  8.45 MiB took 1000 MiB. Over the logic device's 256 chunks the load used 2736 MiB of device memory for 1903 MiB
  encoded when it ran out at 3700 MiB free. Chunks now pack back to back into blocks of up to 256 MiB, and a scan
  returning from host memory takes one allocation: 2206 MiB for 2173 MiB encoded, and 937 MiB stays free after the load
  where it used to run out.
- The fifth-order objective kept launch buffers for every launch layout it had used (batch size, stride, resident or
  streamed slab): 87 MiB on a 23,448-frequency band, about 300 MiB on 79,441, alive until the session closed. They come
  from the memory pool at each launch now; only the layout in use holds memory.
- The shared refinements' last sums (60 candidates x band frequencies) and the objective's latest values stayed alive
  through the reconstruction that follows the fit; they are dropped before it. A shared round also wrote each batch
  into a list joined at the end, holding the sums twice; batches now write into their rows, and the gradient pass
  builds its per-frequency terms in one kernel instead of joining the refinements' sums (same bits, checked).

### A scan the device has no room for: read in place from host memory

The floor at 3400 MiB was the encoded scan itself: oxide scan A (2456 MiB encoded) and oxide largest (2544 MiB) could not
load beside the CUDA context, and the second oxide scan (3160 MiB) did not load even at 4096 MiB. Under unified addressing a kernel reads
page-locked host memory by its own address, so the column decoder needs no copy of the scan on the device. Read in
place, with the same bits, against the scan in device memory (oxide scan A on the full card):

| Read | Scan on the device | Scan in host memory |
|---|---|---|
| Whole-scan phase (one reconstruction) | 0.20 s | 0.96 s |
| A new virtual image (bright field or annular) | 5 ms | 237 ms |
| A 1-pixel detector drag | 2.3 ms | 53 ms |
| One diffraction pattern | 2.4 ms | 4.2 ms |
| Detector total and centre of mass at `open`, read in place | | 8.7 s and 8.6 s |
| The same, each chunk copied to the device first (`Staging`) | | 0.2 s for both |

So a scan stays on the device when it fits; when an allocation for it has no room, the rest stays in page-locked host
memory instead of failing: a `.qem` load puts the whole scan there (reads go straight into it, no device upload), an
HDF5 load keeps the chunks that do not fit, and `restore` maps the parked copies. Each prints one line saying so.
Passes that decode every stream of every chunk (the detector total, the centre-of-mass moments) copy one chunk at a
time into one device buffer the size of the largest chunk; a buffer per chunk had grown the memory pool by 1.8 GB.
Opening oxide scan A this way takes 2.3 s, 1.0 s of it locking 2.5 GB of host pages.

Rehearsed (the card shared with other jobs at full use during these runs, so the times are upper bounds):

| Scan | 3700 MiB | 3400 MiB | 3000 MiB |
|---|---|---|---|
| oxide scan A | on the device, 11.1 s | **in host memory, 9.9 s** (failed to load before) | **in host memory, 15.2 s** |
| oxide, largest of session | on the device, 10.4 s | **in host memory, 9.1 s** (failed to load before) | |
| oxide zone B | on the device, 23.6 s | on the device, 24.6 s | |
| MoS2 bilayer | on the device, 23.7 s | on the device, 24.5 s | |
| Logic device (HDF5) | on the device, 35.2 s | on the device, 36.9 s | **0.42 GB in host memory at load, all of it after the first restore, 42.2 s** |
| the second oxide scan | **in host memory, 24.4 s** (did not load before) | **in host memory, 25.5 s** | |
| Lamella (HDF5) | on the device, 9.0 s | on the device, 20.4 s | |

Every fit's phase against the dense fit's: 2.2e-5 to 1.3e-4 rad rms (the lamella 0.044 rad, as before); oxide scan A
read from host memory at 3400 MiB gave the same bits as from the device at 3700 MiB. The times are slower than the
table above by a third because the card was at full use by other jobs throughout (the lamella, which holds 171 MiB,
took 20.4 s against 6.4 s); compare times within this table, not across.

From the viewer that watches a folder, at 3400 MiB free (two runs, the free memory checked by a fresh process before
and after): oxide scan A shown from host memory 3.2 to 3.3 s after its file appeared, `direct_phase` 4.3 and 4.5 s with
the same fit (C10 8.93 nm, C12 15.67 nm), then the second oxide scan shown 3.6 s after its file appeared and fitted in 20.5 s; process
peak 2912 MiB. Three changes were needed for that flow, each found there:

- The fit empties the memory pool's idle blocks before its band spectrum goes up. The placement counts them as free,
  but what is allocated outside the pool cannot use them: a viewer left 305 MiB idle, and the fit's first cuBLAS
  handle found no memory (`CUBLAS_STATUS_ALLOC_FAILED`).
- The cuBLAS handle and the Newton step's batched matmul are created before the placement measures free memory
  (`_load_fit_libraries`, 12 MiB), not part way through the fit.
- The Newton step's derivative partial sums (positions x chunks x 18 x band frequencies, float32) are evaluated a few
  positions at a time within 64 MiB (`DERIVATIVE_PARTIAL_BYTES`): three positions on the oxide scan's band held whole took
  174 MiB. Each position's sums are its own; one position at a time gives the same bits (test
  `test_low_order_derivatives_of_each_position_do_not_depend_on_its_launch_group`).

Rechecked after these changes, standalone: the logic device at 3400 MiB 34.4 s, the second oxide scan at 3400 MiB 19.1 s, oxide scan A at
3000 MiB 9.2 s.

The logic device was recorded at 300 kV, as its session file says and as every fit here used (confirmed 14 Nov 2026);
its HDF5 header's detector photon energy of 200 keV is not the beam energy.

### The memory a fit kept: local memory, not libraries

The earlier record attributed about 420 MiB kept after a fit to GPU libraries. On the whole-scan path Torch is never
initialized and cuBLAS costs 8 MiB. The 415 MiB arrive with the resident search's random-number kernel
(`direct_joint_random_words`): its Mersenne Twister state takes 2496 bytes of local memory a thread, so CUDA raises
the stack limit from 1024 to 2496 bytes and reserves that much for every thread the device can run at once, then keeps
it. Setting the limit back after the fit returned 404 MiB (3462 to 3058 MiB). The reservation scales with the SM
count: 188 SMs x 1536 threads here, about 22 MiB on a 6-SM RTX A400. The fits now set the limit back before the scan
returns to the device.

## What changed

- `StreamedDirect.band_spectra(flat)`: G(q, k) at given half-plane frequencies from one pass of the column batches,
  into page-locked host memory. `phase` and `band_spectra` share the batch loop (`_column_spectra`).
- `thick.band_frequencies`: the band's frequencies, weights and flat indices, used by the fit and the gather alike.
- `CudaDirectBackend.from_band_spectra`: a backend whose G holds only the band (`DirectEngine.fit_band_inv_A`). Its
  reconstructions refuse; `release_prepared()` takes G off the device and the engine reads it back from the host
  copy when a fit next needs it.
- `DirectPtycho.aberrations()` on an encoded session that cannot hold the dense spectrum fits the whole scan this way
  (`_whole_scan_for_fit`). The band spectrum takes what the device has left beside a working reserve (see "Fewer passes"); the
  pixels beyond that stay in host memory and stream in 256-pixel slabs (`DirectEngine.band_slabs`, one printed line).
  Only when even two slab buffers do not fit does it fit a central region, with one printed line.
- Every fit objective (`ThickSample.fit`, `_fit_encoded`, `band_power`, `broad_batch.Objective`,
  `FifthOrderGradients`, `LowOrderDerivatives`) sums its kernels' outputs over the slabs (`SlabSums`, float64 across
  slabs); one slab when G is whole, so the dense path launches as before. A subsampled pass copies only its rows
  (`memcpy2DAsync` with a pitch of stride rows).
- `StreamedDirect.__init__` estimated the detector sampling by calling its own `None` argument when the file records
  none (a scan opened from `.qem` without detector calibration crashed on the small-device path); it now calls
  `detector.detector_sampling_mrad`.
- `_exceeds_device` (dense or encoded path): 70 % of free memory against 7 bytes a bright-field pixel and scan
  position plus 0.5 GiB, the pixels counted on the mean pattern; it was a quarter of the decoded scan.
- `resident.cuda.counts.retain` frees the memory pool's idle blocks and retries when an allocation outside the pool
  runs out (HDF5 loads on a small card).
- `resident.cuda.counts.host_mapped` and `pinned`: page-locked host memory as a device array kernels read in place.
  `io.qem._arena` loads a `.qem` scan the device has no room for into it, `Blocks` (given a device) keeps the HDF5
  chunks that do not fit there, and `StreamedCounts.restore` maps the parked copies when the scan cannot come back;
  `StreamedCounts.host_nbytes` says how much is read from host memory, and the loaders and the fit print one line.
  `Staging` copies one chunk at a time to the device for the detector total and the centre-of-mass moments.
- `resident.cuda.counts.Blocks`: an HDF5 load packs its encoded chunks back to back into blocks of up to 256 MiB, each
  sized to the scans still to come at the bytes per scan so far; `restore` returns a parked scan into one allocation.
- `FifthOrderGradients.gradients_many` writes each refinement's gradient terms [a_r, a_i, |a|^2, 2 w] into rows of
  one array with one kernel (`_gradient_scale`), instead of concatenating the refinements' numerators, denominators
  and weights and building five temporaries of candidates x frequencies; the terms are the same bits (explicit
  roundings keep |a|^2 from becoming a fused multiply-add).
- `StreamedCounts.restore(keep_host=True)` keeps the host copy, so parking again only frees the device.
- `_free_fit_memory` sets the stack limit back after a fit, in the region path as well.
- Tests: `test_band_spectrum_scores_trials_as_the_whole_spectrum` (G within 2.3e-7 of the largest |G|, agreements
  within 1.3e-7, fifth-order gradients within 6e-7 of the largest), `test_whole_scan_fit_holds_the_band_spectrum_and_the_encoded_scan_in_turn`,
  `test_streamed_band_slabs_sum_to_the_band_held_whole` (one and no resident slab, four slabs, strides 1 and 14).

## Limits

- Not run on an RTX A400 or on Windows. The A400's context and local-memory reservation are smaller than this card's,
  so the rehearsal should overstate its needs; that is reasoning, not a measurement. If the A400 drives a display,
  its free memory is lower than 4096 MiB.
- Speed on the A400 is unmeasured; it has 6 SMs against 188 here and PCIe 4.0 x8 against 5.0 x8. A streamed fit is
  bound by the bus, so the logic device's 31 to 33 s should take about twice that there.
- A scan whose encoded counts do not fit beside the CUDA context stays in page-locked host memory (2.5 GB for oxide
  scan A, 3.3 GB for the second oxide scan, beside the band spectrum's host copy): reconstructions about 5x slower and new virtual images
  about 50x (237 ms) on this card's PCIe 5.0 x8; a viewer drag 53 ms. It needs unified addressing (the device
  property `unifiedAddressing`), not checked on the A400 or on Windows; without it the load fails as before.
- Host memory: a streamed fit holds the band spectrum (up to 5.7 GB here) and the parked scan page-locked at once.
- The higher orders are not measured on scan A: the half-band check gives 3.35 rad (limit 0.5), as on the large card,
  and the fit shown is the +-2 rad one with C30 on the box edge.
- `aberrations(compute_loss=True)`, `search="primary"` and `tilt=False` still fit a central region: the loss and the
  standard fit's reconstruction need every frequency.
- A run in which another process took memory between the session's choice of path and the fit failed with
  `CUFFT_INTERNAL_ERROR` or a raw allocation error, not the session's own message.

## Rejected

- Keeping only the strongest band frequencies. On scan A the strongest 12,000 and 8,192 (83 % and 77 % of the band
  power) reproduce the full-band fit (phase 1.7e-4 rad rms from it); the strongest 6,000, 4,096 and 2,048 land on a
  different fit (C10 7.7 nm, C12 7.55 nm, sample tilt 10.5 mrad; phase correlation 0.22 with the full-band fit). A
  cliff between 6,000 and 8,192 is no default, and the full band fits the small device.
- Holding the band spectrum and the encoded scan together: 1619 + 2456 MiB plus the context exceeds 4 GB.
- Reading the band spectrum in place from page-locked host memory (zero-copy, unified addressing): the logic device's
  fit took 99.7 s against 34.6 s for slabs with pitched copies; the kernels' small reads reach about 3.4 GB/s.
- Copying whole slabs for subsampled passes: 62.4 s; the 242 passes at stride 8 or 64 read an eighth or less.
- Staging fewer scans per HDF5 batch to leave room for the encoded chunks: batches of 512 scans made 512 chunks, each
  rounded up to whole 2 MiB pages, and the load ran out earlier (at 1903 of 2173 MiB encoded) than with 1024.
- A copy stream that does not first wait for the compute stream: the first slabs of freshly allocated buffers came
  back as garbage (band power NaN), because the memory pool hands the compute stream memory that work still queued
  there owns.
- A ballast that follows other jobs' memory through `nvidia-smi`: it either grabbed what the test process had just
  freed (the per-process report lags) or missed memory `nvidia-smi` does not list. A fixed ballast with runs checked
  after the fact was used instead.
