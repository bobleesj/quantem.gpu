# EMPAD native interaction: 2026-09-08

The experimental application now opens an original EMPAD acquisition into a
lossless float32 GPU resident and displays it through the existing native
viewer. This is not a published or notarized release.

## Controlled result

Original AutoDisk Pd@Pt: full `(64, 64, 128, 128)` float32 measurements. Apple M5,
24 GB, 120 Hz display. No binning, cropping, rounding or clipping. All three arms
used the same executable; only the serial detector control changed.

| Actual native presentation | Serial control A | Cooperative | Serial control A2 |
|---|---:|---:|---:|
| Full original open to first presented frame | 1.216 s | 0.942 s | 1.155 s |
| Large ABF center drag, updates/s | 9.4 | 120.0 | 8.9 |
| Large ABF resize, updates/s | 8.9 | 116.0 | 8.8 |
| Large ADF center drag, updates/s | 6.5 | 82.2 | 6.5 |
| Large ADF resize, updates/s | 6.5 | 81.8 | 6.0 |
| Selected-DP drag, updates/s | 108.6 | 108.6 | 108.6 |

The old kernel walked 16,384 detector pixels per thread. The candidate assigns
one 32-lane SIMD group to each scan position, coalesces packed reads and retains
compensated float summation within lanes and across their partial sums. Packed
payload plus descriptors stay exactly **236,838,288 bytes**. No dense resident
or persistent pixel cache was added. Total device allocation and process RSS
are different quantities; the resident number is not a peak-memory claim.

OS page-cache state was not controlled. These are original RAW reread/packing
timings, not cold SSD I/O, retained-resident switching or a cached 2D preview.
120 Hz is not achieved across all gestures. Do not extrapolate this result to
512² scans, other EMPAD sources, seven EMPAD residents or other GPUs.

The final FFT-corrected executable repeated the full native journey: 0.942 s
to first presentation, 115.5–118.0 ABF updates/s, 80.5–80.8 ADF updates/s and
108.6 selected-DP updates/s. No console errors; normal exit. Its SHA-256 is
`0b322b6d4ba80ab938d7708ce2ca5f25a660d5646d24893ee21eb8d5f33c58d4`;
raw evidence is `empad-native-final` in the retained Documents folder.

## Correctness and visible behavior

- All 67,108,864 original detector words matched bit-for-bit after packing and
  extraction. BF/ABF/ADF/total, mean DP and CoM passed independent float64
  references at the unchanged `rtol=1e-6, atol=1e-6`.
- Seven synthetic source tests supplement the full acquisition: special float
  bits, signed and zero-intensity coordinates, footer separation, rectangular
  scans, bad XML, source mutation, budgets, cancellation and retry.
- The actual viewer DP buffer matched the original central pattern bit-for-bit.
  Its actual BF/ABF/ADF buffers passed independent original-file sums with
  maximum relative errors below `7.04e-8`.
- Native controller-driven checks exercised BF/ABF/ADF/iDPC, scan and detector
  gestures, gray/viridis, contrast, light/dark appearance, FFT reveal, failed
  opens, forced reload, mixed-format folder replacement and normal quit.
  These are live native-window tests, not physical mouse or Finder/drop tests.
- Seven original full `(512,512,192,192)` ARINA acquisitions were visited.
  Fifteen navigation requests each applied exactly once, including rapid
  switching and returns. Seven-way comparison reached 14.75 GB resident and
  14.886 GB sampled device allocation within the 17.163 GB budget. No crash.
  This is a regression run, not an ARINA kernel speedup claim.

## Failure retained and fixed

The first controls test incorrectly passed while the FFT panel remained black
after returning from ARINA to EMPAD. Screenshot review exposed the gap. A
stronger test timed out waiting for the visible FFT on the old executable.
The EMPAD publication used `.disabled` rather than the shared `.automatic`
FFT policy. Correcting it keeps FFT hidden on initial launch, but recomputes it
when a visible panel receives a new dataset. The strengthened test then passed
and the restored FFT hash matched its pre-switch value.

Raw evidence is retained under the local `Live4DSTEM Validation/2026-09-08 EMPAD`
Documents folder. `result.json` identifies the exact executable and raw JSON
hashes. The earliest screenshot was captured after app exit and is explicitly
not visual proof; later runs capture the exact active process before gestures.

## Repeat and remaining gates

Build the local EMPAD backend and app integration. Development currently uses
SwiftPM edit mode for `quantem.gpu`; the release dependency pin is unchanged.
Run `Tests/NativeUI/drive_folder.py` with
`LIVE4DSTEM_ENABLE_EXPERIMENTAL_EMPAD=1`, then run
`Tests/NativeUI/drive_empad.py` with `--exe`, `--folder`, `--raw`, `--arina` and
a fresh `--out` directory. The latter requires NumPy. Use
`QGPU_EMPAD_SERIAL_DETECTOR=1` only to reproduce the retained serial control.

Remaining: connect and qualify the average-DP UI; extend float display/audit
coverage; test additional genuinely distinct EMPAD acquisitions and real
Finder/drop interactions; finish dependency pinning and remove local edit mode;
complete the release gates, signing, notarization and fresh-ZIP validation.
Installed v0.0.9 and its colleague archive were not changed.

The test data came from the public AutoDisk demonstration, not the downloaded
Cornell MATLAB export. A local XML sidecar states the 64² scan documented by
AutoDisk; it is test metadata, not original instrument metadata. The RAW file
was not modified. Attribution and references are in
`docs/maintainer/empad-contract.md`.
