# CUDA SSB region latency — 2026-10-21

Closing a regional SSB session used to call Python's global garbage collector.
That scanned unrelated objects in the notebook process on every region change.
The CUDA backend now breaks its known engine/ThickSample ownership cycle
directly before returning unused memory to the device pool. Saved phase results
remain valid after closing the session.

This changes resource cleanup only. The FFT kernels and their specialized
512/1024 accumulation schedules are unchanged; no new kernel framework was
introduced.

## Matched measurement

Baseline: `4528aef5`. Evidence: private experiment `1021-05`. Both versions
used the shipped `quantem.gpu.benchmark.run_ssb` runner, with an explicit CUDA
phase download added to its reporting boundary. The previous implicit NumPy
conversion prevented the CUDA benchmark from finishing.

The physical device was an RTX PRO 6000 Blackwell Workstation Edition. A
512 × 512 × 192 × 192 uint16 acquisition retained all 8,807 bright-field
pixels. Each fit used one seed, 200 trials and Newton refinement; automatic
full-field rotation checked both branches, for 400 trials. Regional parameters
were applied to the full scan before taking the corrected crop. Auto output
remained 1536 × 1536, with unchanged precision and calibration.

The sequence was baseline, cleanup only, cleanup plus fewer search waits,
cleanup plus fewer search waits, cleanup only, baseline. Each variant ran four
measurements; the first observation of each was excluded from the warm median.
The table compares the six warm observations of the baseline and retained
cleanup-only version. GPU0 was observed idle before the sequence; desktop
services remained resident. Earlier contended measurements are archived
separately and are not used here.

| Operation | Before | Retained change |
| --- | ---: | ---: |
| Full-field fit, both rotation branches | 1.240 s | 1.228 s |
| 256 × 256 region workflow | 0.408 s | 0.264 s |
| 128 × 128 region workflow | 0.285 s | 0.133 s |
| Move 128 × 128 region | 0.269 s | 0.117 s |
| Resize to 127 × 128 | 0.272 s | 0.122 s |

Regional workflows include preparation, joint fitting, cleanup, full-field
reconstruction and the corrected crop. Browser transfer and paint are excluded.
The existing CUDA resize path pads the 127-row scan with its mean pattern;
every measured position is retained.

Median cleanup time fell from 149.5 ms to 1.6 ms. Full-field fitting is
essentially unchanged; its roughly 1% difference is not treated as a speedup.
Full-field Auto reconstruction remained about 26 ms. Newton refinement remains
the largest measured fitting stage, and different regions require different
numbers of Newton evaluations.

## Qualification and rejected candidate

The complete trial sequences, Newton results, fitted parameters and image
hashes matched exactly against the frozen baseline. The real-data workflow
also ran again using the retained source, without the variant method patches.
The lifetime test confirms that closing releases the engine, thick model and
spectra while preserving saved results. Retained device-pool allocation after
each paired benchmark stayed identical at 36 MiB; this is not a peak-memory
measurement.

The candidate that removed CUDA's per-batch search waits also preserved exact
results. It saved only a few milliseconds on small regions and did not improve
the full-field fit consistently. That candidate was not retained.

The regression test for the public benchmark reproduces the implicit CUDA
array-conversion failure on the previous runner and passes with the explicit
download. The focused physical CUDA suite covers loading, both rotation
branches, full and regional fitting, movement, resizing, repeated phase
agreement, lifetime and GPU-owned search. It does not qualify browser gestures,
MPS, or the separate experimental higher-order optimizer.

Run the relevant hardware checks with:

```bash
CUDA_VISIBLE_DEVICES=0 python -m pytest -q -m "" \
    tests/hardware/cuda/test_ssb_close.py \
    tests/hardware/cuda/test_ssb_benchmark.py \
    tests/hardware/test_ssb_resident_joint.py -k "not mps"
```

See [the reproducible SSB benchmark](ssb-joint-fit.md) for the public command
and timing boundaries.

## Separate integration check

The export and cleanup fixes were also tested on `6fce2932`, without the
shared-FFT refactor. The focused CUDA checks passed (19 tests). Two full
512 × 512 × 192 × 192 acquisitions retained all 8,807 and 8,930 valid BF
pixels, with Auto 3× and 4× output respectively. Each benchmark ran four
complete full-field, regional, move and resize sequences with 200 trials per
rotation branch and Newton refinement. All 80 complete phase arrays and all
48 fit records were exactly equal to the frozen baseline. This separate check
qualifies the export and cleanup changes; it does not qualify the shared-FFT
refactor or resolve its compiler-cache discrepancy.
