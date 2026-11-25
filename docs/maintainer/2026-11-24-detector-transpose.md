# Loading a transposed detector, and a disk rule that read float data as markers - 2026-11-24

## 1. `io.load(source, detector_transpose=True)`

**Question.** Some acquisitions record the detector transposed relative to the scan. The EMPAD-G1 WSe2 scan of
Zenodo 10621271 (128 x 128 positions, 130 x 128 float32 records) is one: DPC finds the scan-detector rotation flipped,
a reflection that no rotation produces. Direct ptychography can take the flip as a setting, but a reconstruction that
needs the detector in the scan's frame has to transpose the array by hand. Can the loader swap the detector axes as it
ingests, for every source, at the speed of a plain load?

**What the option does.** Each block of frames is transposed on the GPU after its flagged pixels are corrected (and,
for an ARINA master, after `detector_bin`) and just before it is encoded, so the resident holds `data[r, c, j, i]` of
the stored `data[r, c, i, j]`. EMPAD RAW, NumPy, Zarr, generic HDF5, ARINA masters and `.qem` copies take it; a
`.qem` reopens its encoded arrays without decoding, so a transposed reopen decodes them on the GPU in bounded blocks of
scan rows and encodes them again. Every per-axis detector value of the record follows its axis (`detector_shape`, the
sampling pairs, the pixel mask and the flagged-pixel coordinates, the bright-field disk centre, the last two entries of
the four-axis lists, and the schema-2 detector axes, quantities and overrides); source fields keep describing the file
as stored. `detector_transpose: true` is kept in the retained metadata and `detector_transpose` is declared as a
processing step, and `validate_declared_processing` requires the two to agree. A copy saved transposed reopens
transposed; asking a saved copy (`.qem`, or an HDF5 master whose attributes carry the record) for
`detector_transpose=True` again is refused before anything is read, rather than swapping back. A recorded
`scan_detector_rotation_flipped` is inverted, since direct ptychography transposes the frames when it is set. CUDA and
Apple GPUs (MPS: the same 32 x 32 tile as a Metal kernel, for NumPy, ARINA masters and `.qem` copies; tested on an M5
Max); DM, scaled precision and the paired layout refuse the option.

**Setup.** An RTX PRO 6000 Blackwell Workstation Edition whose host ran another job on its second GPU (the first three
rows), and an RTX PRO 6000 Blackwell Server Edition on an otherwise idle host (the last row). The WSe2 RAW above (1.07 GB
of float32 values) and an ARINA master of 512 x 512 positions, 192 x 192 uint16 pixels (19.3 GB of counts in 3.1 GB of
bitshuffle-LZ4 chunks), from page cache (the files are re-cached before each load: on the busy host another job evicted
them and the same load took 0.5 to 5 s). Each source loaded once to warm, then plain and transposed loads alternated,
3 times each for the first two rows and 7 for the last two; medians, wall time including the device synchronization.
Every transposed resident was compared on the GPU, bit for bit, with the GPU transpose of the plain resident: equal
for all three sources on both hosts.

**Where the time went, in the order it was found.**

| change | WSe2 RAW plain / transposed (s) | ARINA master plain / transposed (s) |
|---|---|---|
| CuPy strided transpose of each block; float checksum read back through pageable memory | 0.67 / 0.91 | 0.61 / 0.67 |
| encoder reads each stored pixel from its transposed place (no copy); checksum hashed on a worker thread | 0.42 / 0.46 | 0.64 / 0.80 |
| tiled shared-memory transpose; checksum read back asynchronously into page-locked buffers | 0.432 / 0.453 | 0.666 / 0.686 |
| the same, idle host | 0.536 / 0.539 | 0.449 / 0.472 |

- The plain float32 load spends most of its time on the host hashing the values for the float codec's
  `logical_sha256` (about 0.27 s for 1.07 GB). Hashing on one worker thread beside the reads made the plain load
  0.67 to 0.42 s, with the same checksum. A transposed float load needs the checksum of the values as stored, so its
  transposed blocks are read back for it: pageable read-backs cost 0.24 s; page-locked ones, copied on their own stream
  and hashed while the next block is read, cost 3 ms on the idle host (0.536 to 0.539 s, within the plain loads' spread
  of 0.535 to 0.538 s).
- Reading transposed inside the encoder (`sc_encode`, `sc_compact`, `sc_fields` taking a per-pixel source table)
  removed the copy but made the master slower, 0.80 s against 0.67 s with the copy: neighbouring threads no longer
  read neighbouring pixels. Rejected and reverted.
- One transpose of a full ARINA block (16384 x 192 x 192 uint16, 1.2 GB) takes 3.63 ms as CuPy's strided copy and
  1.63 ms with a 32 x 32 shared-memory tile (1.48 TB/s of reads and writes); a 512 x 128 x 128 float32 block 0.058
  and 0.011 ms (the 34 MB block stays in the GPU's cache when repeated, so this one is not a memory-rate number).
- The ARINA master's remaining 23 ms (5 %) is about that one pass: 16 blocks of 1.64 ms, 26 ms alone. Only a
  transpose inside the decoder's or the encoder's own memory accesses would remove it, and the encoder's was slower.
- A `.qem` reopen uploads its encoded arrays without decoding them (0.87 s for the WSe2 copy on the idle host); a
  transposed reopen decodes them on the GPU and encodes them again, 1.37 s. Loading the original RAW transposed is
  faster (0.54 s), and its transposed copy reopens without decoding.

## 2. The half-plateau disk rule read the WSe2 disk as 4.98 px

**Question.** On the same scan, `detector.detector_sampling_mrad` (the radius where the bright-field disk of the mean
pattern falls to half its plateau) gave a 4.98 px edge, while a circle fit gave 11.97 px. Why?

**Cause.** Not the background-subtracted negative values (the mean pattern has none) and not the mean + std
threshold. Before measuring, `_unflagged` zeroes every mean-pattern value at or above 65535, ARINA's marker for a
flagged pixel. This scan's bright-field plateau is about 66000 per pixel: 349 of the 444 pixels above mean + std were
zeroed, and the rule measured the edge ring that was left. Without that step the same rule reads 11.64 px; the
half-plateau disk's second moments give radii of 11.60 and 11.72 px, and the radial profile falls from 83 % to 35 %
of the plateau between 11.0 and 11.5 px.

**Fix.** A marker in every frame averages to exactly 2**16 - 1 (a uint32 marker's float32 mean rounds up to 2**32),
so only those values are flagged. `exclude_marked_pixels` skips float scans (markers are integer counts) for the same
reason, and for integer scans takes the stored type's largest value as the marker instead of anything at or above
65535. The rule then
gives 11.645 px, 24.9 / 11.645 = 2.138 mrad per pixel.

**Robust rule.** Markers come from the detector mask or from an exact marker value in every frame, never from a
brightness threshold, and the rule should be told which pixels are invalid rather than guess them from the mean. The
remaining 2.8 % between 11.64 px and the circle fit's 11.97 px is the edge definition: the half-plateau contour sits
at the geometric edge for a symmetric blur, while a fit to the disk's outer extent includes the foot of the edge
(4 % of the plateau at 12.0 px).
