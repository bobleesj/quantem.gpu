# Parallax and scan rotation on the encoded acquisition - 2026-10-05

## Question

`io.load` returns an encoded acquisition on CUDA (`Dataset4dstemGPU` holding
entropy-coded counts), but `parallax.run` called `cp.asarray` on its input and
`geometry.rotate_scan` accepted only dense arrays, so neither worked on the data
a scientist actually loads. Can both take the encoded acquisition, read it only
in bounded parts, and give exactly what the old dense path gives for the same
counts?

## Setup

- One NVIDIA RTX PRO 6000 Blackwell (96 GB), shared with other jobs. Python 3.14,
  CuPy 14.2.0, PyTorch 2.13.0.
- Base: quantem.gpu `a8065dc9` (dense CuPy input). Branch: `ea42d293` (parallax),
  `ee113bcb` (rotation).
- Synthetic: Arina-style bitshuffle/LZ4 masters written with h5py. Parallax:
  24 x 28 scan, 32 x 32 detector, Poisson counts of one scene shifted in
  proportion to the bright-field tilt. Rotation: 13 x 21 scan, 16 x 16 detector,
  Poisson(30) counts.
- Real: one native Arina acquisition, 256 x 256 scan, 192 x 192 detector, uint32
  file whose counts fit uint16 (no flagged pixels).
- Method: the base code ran on the dense counts read with h5py (`cp.asarray`);
  the branch ran on `io.load(master, backend="cuda")`. Arrays were compared
  byte for byte (SHA-256 for arrays over 64 MiB). For parallax, spies recorded
  the deterministic inputs of the later stages: the mean pattern given to the
  disk detection and the float32 bright-field stack given to the alignment.

## Results

Parallax (`voltage_kV=300`, `scan_sampling=0.5`, `fit_aberrations=True`,
`upsampling_factor=2`):

| Case | Mean pattern, detected disk | BF stack | Shifts, image, aberrations |
|---|---|---|---|
| Synthetic, auto disk (149 px) and given disk (113 px) | identical | identical | identical with fixed-order binning; otherwise within the run-to-run spread below |
| Real, radius 20 (1257 px), auto and given disk | identical: center (95, 96), radius 55 | identical | identical with fixed-order binning |
| Real, full disk (9477 px) | identical | identical (SHA-256) | branch: 13.8 s; base dense path ran out of memory (34 GB allocated) on the shared GPU |

The alignment bins the detector with `cp.add.at`, a float32 atomic addition, so
its result changes in the last bits from run to run on identical input. Two base
runs on the same synthetic counts differed by up to 1e-4 px in the shifts and
0.02 in image values near 23000; base against branch by up to 1e-4 px and 0.035,
the same order. Replacing `cp.add.at` with a fixed-order sum in both runs (proof only)
made every output byte-identical between base and branch.

Scan rotation, encoded result read back and compared with the base dense result:

| Case | Data and validity mask | Encoded time | Dense CuPy time |
|---|---|---|---|
| Synthetic: 0, 90, -90, 180, 270, 450 degrees (full, same); 30, -45, 123.4 degrees nearest with fill 0 and 7 (full, same) | identical to base NumPy and base CuPy in all 24 cases | 0.045-0.156 s | 0.004-0.074 s |
| Real, -90 degrees, full | identical (SHA-256) | 5.06 s | 0.054 s |
| Real, 180 degrees, same | identical (SHA-256) | 0.95 s | 0.046 s |
| Real, 33 degrees nearest, full (355 x 355 scan) | identical (SHA-256) | 4.58 s | 0.178 s |

The dense times exclude building the 4.8 GB dense acquisition the dense path
needs; the encoded path never holds more than 256 MiB of decoded frames per band
and per read. The rotated acquisition is itself encoded: detector means, exact
masked sums and `io.save`/`io.load` round trips match the rotated counts.

## Conclusion

- `parallax.run(io.load(...))` reads the bright-field pixels in bands of scan rows
  cropped to the disk's bounding box (`Dataset4dstemGPU.read`) and fits the disk
  from `detector.mean`. Every stage input equals the old dense path's, byte for
  byte, on synthetic and real data.
- `geometry.rotate_scan(io.load(...), angle)` maps every output scan position to a
  source position with the existing dense rotation applied to the scan-index
  grid, gathers each band of output rows from bounded reads, and encodes it into
  a new acquisition. Quarter turns and nearest-neighbor rotations are exact;
  bilinear rotation of encoded counts raises because it would produce float32.
- Neither operation has an Apple GPU (MPS) implementation; both raise
  `NotImplementedError` naming MPS.

## Findings left as they are

- Parallax spectrum tiling inserts zeros: with `upsampling_factor=2` (the default)
  every odd row and column of the image is exactly 0 on the real acquisition
  (base and branch alike), because tiling the summed spectrum before the inverse
  FFT is zero insertion, not interpolation.
- Parallax applies the measured shifts twice: `align_vbf_stack_multiscale_cp`
  returns a stack already shifted by its phase ramps, and the upsampling step
  shifts it again by the same global shifts.

## Rejected

- One exact detector query per bright-field pixel (`masked_sum_exact` with a
  one-pixel mask): one launch and one partial decode per pixel, thousands of
  launches for a real disk; a band read decodes each needed pixel once.
- Reading column bands for 90-degree rotations: a read narrower than the scan
  decodes one 512-scan interval per scan row (0.85 s for a 256 x 8 band of the
  real acquisition, against 0.086 s to decode it whole), so the rotation reads
  whole source rows and keeps only the frames each output band needs.
- Bilinear rotation of encoded counts: the result is float32, which the count
  encoder does not store; use quarter turns or `interpolation="nearest"`, or
  rotate the reconstructed scan images.
