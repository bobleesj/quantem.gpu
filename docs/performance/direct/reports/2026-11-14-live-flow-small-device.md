# Scans arriving one after another on a 4 GB CUDA device: the live viewer with direct ptychography below it

14 Nov 2026. Follows [the whole-scan fit on a 4 GB device](2026-11-13-whole-scan-fit-small-device.md).

## Question

Does the two-cell live flow hold on a 4 GB device when scans keep arriving: `Show4DSTEM.live` shows each scan,
`ShowPtycho.live` fits its aberrations through fifth order on a 256 x 256 scan region and shows the whole phase, the
reader moves the fit region, and the next scan arrives?

## Setup

- One RTX PRO 6000 (188 SMs), shared with other jobs, so a ballast process could not hold a fixed free-memory budget
  (other jobs moved free memory by gigabytes within a run). A cap inside the test process replaced it:
  `tests/tools/small_gpu.py`, `small_gpu.cap(megabytes)`. After the call `cudaMemGetInfo` and PyTorch's
  `mem_get_info` answer `megabytes` minus what the process holds, and the CuPy pool, allocations made outside the
  pool (`cp.cuda.using_allocator()`, the encoded scan's blocks) and PyTorch refuse anything beyond it with their
  usual errors. The context is counted as 560 MiB. Against `nvidia-smi` for the process the count was 99 MiB low
  (3109 counted, 3208 measured) on one scan.
- Caps: 3700 and 3400 MiB (the free memory a fresh process sees before its CUDA context exists).
- Each run: scans copied into the folder one after another; for each, the first fit, the fit region moved to the
  top-left 256 x 256, back to the centre, and one diffraction pattern read from the viewer.
- Scans: three 512 x 512 x 192 x 192 oxide scans (2.3 to 2.6 GB encoded), a MoS2 bilayer scan, an oxide scan whose
  3.2 GB of encoded counts stay in host memory, and the logic device from HDF5.

## Result

Seconds from the fit starting to the phase shown; every row also passed the region moves and the pattern read.

| Scans in one session | 3700 MiB | 3400 MiB |
|---|---|---|
| oxide A, oxide B, oxide C (first fit each) | 6.9, 5.1, 5.3 | 6.6, 6.2, 7.9 |
| the same, fit region moved | 7.4, 3.1, 8.5 | 8.5, 7.6, 8.4 |
| MoS2 bilayer, then the host-memory oxide | 9.4, 9.7 | 9.3, 15.3 |
| logic device (HDF5) | 6.5 | 9.4 |

The most the process held: 3692 MiB at the 3700 cap, 3316 MiB at the 3400 cap. Without the fixes below the same
runs failed in about every second run at 3400 MiB and on the third scan at 3700 MiB.

## What was wrong, and the fixes

1. **Parked chunks stayed on the device.** `StreamedCounts.park` dropped its chunks, but a finished decode or
   detector pass still referred to them through a reference cycle, so their device memory (the whole 2.5 GB scan)
   stayed until Python's collector next ran. The fit that parked the scan for room then ran out, in about every
   second run. `park` now collects. Found because logging that happened to call `gc.collect()` made every run pass.
2. **The fit's fixed working reserve was 64 MiB.** A scan-region fit beside a viewer's resident scan ran out by
   24 MB in the tilt step's band power: with 64 MiB the placement judged the scan could stay on the device. With
   96 MiB and more it moves the scan to host memory first. The reserve is 128 MiB.
3. **Reading one diffraction pattern was refused.** `io.read` kept a flat 256 MiB in reserve for the decoder's
   scratch, so a 36 KB read failed with 190 MiB free. The reserve is eight times the output plus 16 MiB, at most
   256 MiB.

Fixed in quantem.widget's `ShowPtycho.live` at the same time: a refit lets the shown session go before its panel
is built (the 1536 x 1536 upsampled phase of the MoS2 scan did not fit beside it at 3400 MiB), and a failure there
is the status line, not a panel left on "fitting".

## Rejected

- A ballast process on a shared GPU: other jobs changed free memory during a run and the process under test took
  what they freed (5.7 GB peaks under a "3700 MiB" ballast). Only a run's first scan could be trusted.
- Retrying the fit after an out-of-memory error: the second try failed the same way, because the cause was state
  (uncollected chunks), not timing.

## Limits

- The cap is not the card. Not modelled: memory CUDA takes outside the allocators (a kernel's local-memory
  reservation, about 400 MiB on 188 SMs and about 13 MiB on a 6-SM card), page rounding of single allocations, a
  display's share of the card, and the smaller context of a smaller card. A run on the RTX A400 itself is still to do.
- The damped Phase and Complex reconstructions need the dense bright-field spectrum; on these caps the session
  reads the scan in batches and shows the phase of the mean wave.
