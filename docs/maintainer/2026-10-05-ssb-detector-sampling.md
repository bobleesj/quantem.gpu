# SSB detector sampling, losses, kernel cache and the MPS fit state - 2026-10-05

## 0. Merging main's joint-calibration commits, and a shared kernel cache that hid it

main's bbe978e1, dee4a1e4 and c33b8e8e (coalesced 512 calibration FFT writes, shared probe geometry in the thick-sample
batch kernel, joint-fit reuse for the 180-degree check) merged into `ssb/cuda/engine.py`, `ssb/cuda/kernels/engine.py`
and `ssb/cuda/kernels/fft512.py`. With the default CuPy kernel cache (`~/.cupy/kernel_cache`) the SSB fingerprints
(scratch `probes/ssb_fingerprint.py`: previews, reconstructions, batch losses and fits at 128, 256 and 512) of main and
of the restructure base 059ada7c differed at 128 and 256, and the merged tree differed from both at 512.

Cause: the cache, not the code. NVRTC compiles the same module source with the same options to different cubins
depending on what the process compiled before (the 128 module, for example, lowers `sqrtf` to `MUFU.SQRT` in one process
and to a multiply in another: 157 of 1524 SASS lines of its fused row kernel differ). CuPy's cache key cannot see that
state, so the first process to compile a module source decides the binary every later process loads. Entries written
by earlier processes on 2026-10-04 made main and the restructure disagree; a new module source (the merge) compiled
fresh.

With a fresh `CUPY_CACHE_DIR` per tree, every deterministic entry agrees:

| entry (default cache: main differs from 059ada7c) | fresh: main vs 059ada7c | main vs merge | main vs this branch |
|---|---|---|---|
| 128 batch losses (16, 4), mean-phase preview | equal | equal | equal (batch losses: see 6) |
| 128 complex-wave reconstruction loss, `SSB.open` fit | equal | equal | differ on purpose (2; 1) |
| 256 fits, rotation-check fit, dataset fit | equal | equal | equal |
| 256 higher-order and mean-phase previews, batch loss | equal | equal | equal (batch losses: see 6) |
| 256 complex-wave loss, `SSB.open` fit, fixed-probe series | equal | equal | differ on purpose (2; 1) |
| every 512 entry, a 1024 preview | equal | equal | `SSB.open` and series entries differ on purpose (1) |

The name-expression change first made for the 512 difference (8f26fc81) only moved the cache key and is reverted
(cce53026): with fresh caches the tree with and without it fingerprints identically. Every comparison in this record
uses a fresh cache per tree. Maintainer note: results in the last float32 bits depend on which process first compiled each
module into the shared cache; a fingerprint run on a cache that pytest processes had filled differed at 256 and 512
from the same tree on a fresh cache.

## 1. Automatic detector sampling was about twice the physical value, and differed by backend

**Question.** `SSB(..., det_sampling=None)` derives the detector sampling (mrad per detector pixel) from the
bright-field disk. What is the right rule, and did CUDA and MPS use it?

**Physics.** The bright-field disk is the image of the probe-forming aperture, so its edge sits at the convergence
semiangle. With the disk edge radius R in detector pixels, `det_sampling = semiangle_mrad / R`. A symmetric blur of
the aperture edge (detector point spread, partial coherence) leaves the half-intensity contour at the geometric edge,
so R is the radius where the disk falls to half its plateau.

**Before.** Six sites used `2 * semiangle / R` with three different radii:

| path | radius | formula |
|---|---|---|
| CUDA array (`ssb/cuda/backend.py`) | integer half-maximum of the azimuthal profile (`detector.cuda.probe.detect_bf_radius`) | 2 alpha / R |
| CUDA dataset and `SSB.open` (`ssb/workflow.py`) | same, on the full-detector mean | 2 alpha / R |
| MPS array, no `bf_radius` | equal-area radius of the pixels above mean + std (`detector.fit_probe`) | 2 alpha / R |
| MPS array with `bf_radius` | NumPy copy of the integer half-maximum | 2 alpha / R |
| MPS dataset and `SSB.open` | integer half-maximum of the bright-field CROP (no disk edge in it) | 2 alpha / R |
| CUDA WebGPU export (`detected_radius_px`) | inverse of the above | 2 alpha / det |
| regional fit (`find_aberrations(scan_region=...)`) | `bf_radius` or the equal-area radius | alpha / R |

**After.** One radius, one formula, both backends and every input: `ssb.brightfield.disk_edge_radius(mean_pattern)`
is the radius of a disk with the area of the pixels brighter than half the plateau (the median of the pixels above
mean + std), computed on the host from the full-detector mean pattern, and `det_sampling = semiangle_mrad / R`.
Loaded datasets and `SSB.open` calibrate on the full detector before cropping the disk; the regional fit uses the
session's calibration. `BrightfieldDisk.detected_radius_px` (and `SSBResult.detected_bf_radius`) is the radius the
calibration uses, `semiangle / det_sampling` on both backends.

**Oracle 1: abTEM with a known detector sampling.** One BaTiO3 cell, 300 kV, 30 mrad, 100 A defocus, 64 x 64 scan at
0.25 A, 168 x 168 detector; abTEM's sampling is 0.546875 mrad per pixel (disk radius 54.86 px).
Script: scratch `probes/abtem_radii.py`; test `tests/hardware/cuda/test_ssb_units.py`.

| radius estimate | R (px) | alpha / R | error | 2 alpha / R | error |
|---|---|---|---|---|---|
| half-plateau equal area (new) | 54.680 | 0.5486 | +0.32 % | 1.0973 | +100.7 % |
| equal area, mean + std | 54.493 | 0.5505 | +0.67 % | 1.1011 | +101.3 % |
| integer half maximum | 55 | 0.5455 | -0.26 % | 1.0909 | +99.5 % |

Fitted C10 (truth -10 nm, `find_aberrations(trials=200)`): exact sampling -10.03 nm; new automatic sampling -9.67 nm;
equal-area -9.60 nm; integer half maximum -9.77 nm; old CUDA automatic (1.0909) -4.36 nm; old MPS array automatic
(1.1011) -4.29 nm.

**Oracle 2: a real acquisition.** Arina, 300 kV, 30 mrad, 91 mm camera length, 256 x 256 scan, 192 x 192 detector.
The measured calibration of this camera at 91 mm is 0.554 mrad per pixel
(quantem.widget `planptycho.ARINA_MRAD_PER_PX`, median over reconstructions), and the documented Arina SSB fixture uses
0.5554. Script: scratch `probes/real_calibration.py` (base and fixed trees).

| rule | R (px) | det_sampling (mrad/px) | vs 0.554 |
|---|---|---|---|
| new: half-plateau equal area | 54.31 | 0.5524 | -0.3 % |
| equal area, mean + std, alpha / R | 53.16 | 0.5643 | +1.9 % |
| integer half maximum, alpha / R | 55 | 0.5455 | -1.5 % |
| old CUDA (all inputs) | 55 | 1.0909 | +97 % |
| old MPS array | 53.16 | 1.1287 | +104 % |
| old MPS dataset / `SSB.open` (crop) | 26 | 2.3077 | +317 % |

Effect on the CUDA fit of that acquisition (`SSB(io.load(...))`, rotation 10.06 deg, 200 trials): the old calibration put
6418 of the 8878 selected bright-field pixels beyond the semiangle, so their aperture weight was zero and only 2460
pixels carried signal; the new calibration keeps all 8878 inside the aperture. Fitted C10 1.71 nm (old) and 4.16 nm (new).
The phase-variance loss is not comparable across samplings, so it was not used to choose the radius: pixels with zero
aperture weight lower it (best loss 0.069 at 1.0909 against 0.225 at 0.5556 mrad per pixel), and over 0.50 to 0.60 it
falls as the sampling grows.

**Rejected.** Keeping `detector.fit_probe`'s equal-area radius (the bright-field selection rule) for the calibration too:
its mean + std threshold sits at about 0.68 of the plateau on this acquisition, so it measures the disk 1.1 px small
(+1.9 % sampling). Keeping the integer half maximum: its first-bin-below-half rule is up to one pixel large (-1.5 %).
The radial-profile half crossing interpolated to sub-pixel gives the same 54.0 px as the equal-area rule but needs a
centre, binning and smoothing; the area count needs none.

**Tests changed on purpose.** `tests/hardware/cuda/test_ssb_cuda_128.py` export state: `detected_radius_px` 42.8 ->
21.4 (21.4 mrad / 1 mrad per pixel). `tests/hardware/mps/test_ssb_mps_cuda_reference.py` flat-pattern selection tests
monkeypatch `disk_edge_radius` (a flat pattern has no edge). New: `tests/hardware/test_native_ssb.py`
(array, dataset and `SSB.open` reconstruct exactly as with `det_sampling = semiangle / R`, CUDA and MPS) and the abTEM
known answer above.

## 2. The CUDA result loss depended on whether a fit had run

`CudaSSBBackend.result()` reported `objective.loss`, which is the batched optimizer evaluator until a fit sets
`objective.exact`, and the exact reconstruction afterwards. The batched evaluator is a different quantity on 256 and
1024 scans: its transposes write only a subset of the scan rows, so a session that had not fit reported a loss about
0.4 times the exact one (Poisson counts, C10 -8 nm, C12 3 nm; scratch `probes/batched_vs_exact.py`):

| scan | BF pixels | pre-fit loss (batched) | exact loss |
|---|---|---|---|
| 128 | 80 / 448 | 0.140269 / 0.138036 | 0.140269 / 0.138036 |
| 256 | 80 / 448 | 0.056643 / 0.055427 | 0.140460 / 0.138221 |
| 1024 | 32 | 0.021868 | 0.140105 |

It now always reports the exact reconstruction loss, the evaluator every fit uses
(`test_result_loss_is_the_exact_objective_with_or_without_a_fit`: 0.0578 before, 0.1427 after). The batched evaluator
is deleted (section 6).

## 3. The MPS fit prepared its own copy of the evidence

`MpsSSBBackend.fit` called `optimizer.optimize(source, ...)`, which recomputed the mean pattern, the selection and the
whole G(q, k) stack with its own DC rule, then replaced the session's prepared state. `optimize` now takes the session's
prepared state and selection, so previews before and after a fit and the fit itself read one preparation
(`test_mps_fit_reads_the_prepared_session_evidence`). The test-only `ssb.mps.reconstruct.reconstruct`, a second
from-source preparation, is deleted with its environment-gated test (that test passed an encoded resident, which SSB
refuses since the restructure).

## 4. MPS export of a NumPy session

`export_brightfield` streams columns through `columns_float32_into`, which the NumPy source (`ArrayFrames`) lacked, so
exporting raised `AttributeError`. `ArrayFrames` now has it. Reading the export back reconstructs to within 6e-10 rad of
the NumPy session: MLX rounds the FFT of the NumPy session's column stack (bright-field-minor strides) differently from
the contiguous stack read from the file. `detector.mean` of the exported columns, built without detector sums, is the
exact mean of every exported pixel.

## 5. `MpsBfColumnFrames` without detector sums and `detector.mean`

Not reproduced at 059ada7c on an Apple M5: `detector.mean` of BF columns built with `detector_sum=None` (with or without
a stored DC value) returns the exact mean of every exported pixel, because `MpsBfColumnFrames.detector_sum` computes the
column sums on first use. The export test above now pins it.

## 6. The batched variance evaluator is deleted

Fits switch the objective to the exact reconstruction before their first evaluation, and results now use it too, so
the batched evaluator had no caller: `PhaseVarianceObjective`'s batched branch and staging buffers, the `exact` flag,
`CustomFFTBase.ifft2_inplace_batch_fused_pk_variance`, the `*_rows_fused_pk_batch_*_transpose_packed_b4` kernels of the
128, 256, 512 and 1024 modules, the 128/256/1024 row-variance kernels, `variance_from_sums_batch_kernel` and
`loss_from_sums_batch_kernel`, the device functions only they used, and the optimizer's padding to four candidates
(the exact evaluator runs each candidate once; padded candidates were discarded). Every remaining kernel compiles to the
same SASS (scratch `probes/module_sass.py`: both module versions compiled in one process with CuPy's NVRTC options,
`cuobjdump -sass` per kernel), and every fingerprint entry is unchanged except the four 128/256 batch losses, which
measured the deleted evaluator.

## 7. Chunked previews of large scans are fixed-order

Scans with more than 6 GB of corrected planes run a chunked path. At 512 its previews (phase, and phase with loss)
summed the 32-BF groups with atomic adds through mirror-pair and dual kernels, so the phase changed in its last bits
from call to call; fits used the fixed-order reduction. Every call now writes one partial plane per group and sums the
planes in a fixed order; the direct, paired and dual accumulate methods, their two row kernels and the mirror-pair
cache are deleted (remaining kernels SASS-identical). `test_cuda_512_chunked_preview_is_repeatable` fails before (98 of
262144 phase pixels differ between calls) and passes after.

Timing (scratch `probes/large_preview.py`: 512 x 512 scan, 3388 BF pixels, 7.1 GB of corrected planes; RTX PRO 6000,
idle, median of five calls after a warm-up):

| call | atomic (before) | fixed order (before, fit path) | now |
|---|---|---|---|
| preview phase + loss | 10.4 ms | 10.4 ms | 10.3 ms |
| preview phase only | 10.1 ms | 10.5 ms | 10.4 ms |

The loss of every variant was 0.0011000020895153284, so the preview loss equals the fit's loss bit for bit.

## 8. `SSB.open` on exported BF columns uses the stored detector sampling

ShowPtycho's BF-column companion (`cal.json`) stores `det_sampling_mrad_px`; `load_bf_columns_mps` ignored it, and a
session given no `det_sampling` divided the semiangle by the stored selection radius, which is the selection's, not the
disk edge the export was calibrated on. `MpsBfColumnFrames` now carries the stored sampling and the MPS backend uses it
when given none (`test_bf_column_session_uses_the_exported_detector_sampling`: (16, 16) mrad per pixel before, the
stored (0.7, 0.8) after); an MPS export writes its sampling into the replacement columns.
