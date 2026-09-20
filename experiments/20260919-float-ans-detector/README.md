# Exact float32 ANS detector interaction

The steady detector update is substantially faster, but **sustained native
120 FPS has not passed**. The 8.33 ms budget must include scientific image
updates and presentation, not just the compute kernel.

## Scope and measurement

Apple M5, 24 GB, Swift/Metal. Public Zenodo 15987625 MOSS6 and ZSM5 scans
are 256 by 256; WSe2 is 128 by 128. All have 128 by 128 float32 detectors.
There is no binning, cropping, precision conversion, or full decoded resident.
ANS retains the two UInt16 bit lanes of each original IEEE float32 word.
Only bounded query scratch and image-sized accumulated sums are decoded.

The scientific quantity is the compensated virtual-detector intensity sum.
The same 96 masks and existing reduction order are used for every arm.
Timing starts at `encodeVirtualImage` and ends at Metal command completion.
Mask construction and validation hashes are outside this interval. Each arm
uses a fresh process and resident; OS file pages are uncontrolled. These are
not cold SSD loading measurements.

## Backend results

Milliseconds, steps 3 through 95 inclusive, nearest-rank p95. The periodic
rebase at step 63 is included. Initial masks and the first large jump are
retained in raw evidence, but excluded from this steady-sequence summary.

| Source | Full-decode control median, A1 / A2 | Candidate median | Candidate p95 | Candidate max |
| --- | ---: | ---: | ---: | ---: |
| MOSS6 | 229.873 / 230.082 | 6.252 | 6.512 | 104.346 |
| ZSM5 | 231.579 / 231.759 | 6.257 | 6.616 | 104.070 |
| WSe2 | 57.769 / 57.629 | 1.736 | 1.827 | 26.374 |

The final candidate was repeated at medians 6.205, 6.218, and 1.714 ms.
That repetition overlapped CPU Swift compilation; it is corroboration, not an
idle-machine A/B/A. The original A/B/A qualified selective decode against full
decode; later probes were compared to those frozen outputs and timings.

Every recorded DP, product, mean, CoM, aperture-sequence, and mask hash matches
the frozen GPU control for all three sources. This is product parity, not an
independent full-volume CPU audit.

Earlier exploratory summaries mistakenly excluded ordinary step 67. The
retained raw records are unchanged; [results.json](results.json) recomputes
all summaries using steps 3 through 95 and includes the actual step-63 rebase.

## What changed and what did not work

- Decode changed detector entropy streams rather than every detector stream.
- Consume literal, zero, and constant ANS lanes directly during reduction.
  This removes unnecessary scratch writes without changing arithmetic order.
- Specialize already decoded word access instead of using the legacy XOR
  accessor.
- One serial encoder alone did not improve the approximately 39 ms selective
  path. Parallel literal decoding reached approximately 17 ms, but direct
  consumption superseded it.
- Off-by-default stage instrumentation helped isolate decode and reduction.
  Its timings include instrumentation overhead and are not native FPS proof.

## Native workflow

An ordinary release test bundle was driven with native pointer controls:
open MOSS6, drag the detector, select ADF/ABF/BF, save `.qem`, enable `.qem
only`, reopen the saved sidebar row, drag again, and return to BF.

- Original and reopened BF hash: `9a910b2df5d2ba49`.
- Original and reopened selected DP hash: `bb87255844447954`.
- Saved file: 4,336,963,161 bytes, less than 1% smaller than this float source.
  This dataset does not substantiate a large compression-ratio claim.
- Python validation reported verified integrity and codec layout. Calibrated
  scan axes and normalized voltage, camera length, dwell time, and row/column
  sampling were retained. Full-volume decoded parity was not run.
- The native ANS residency gate passed. The test app quit with status zero.
  Managed cleanup removed staged data, generated `.qem`, and per-run caches.

Short native drags are **not** a sustained 120 Hz test. Of 12 accepted detector
submissions, six had positive confirmed presentation timestamps; six remained
unconfirmed. Confirmed input-to-presentation median was 41.312 ms, p95/max
55.985 ms. These are latency, not frame-rate measurements. Missing timestamps
were not replaced by callback times.

The final native executable SHA-256 is
`c9971c5926d3210adbc691f9773bf866af98e81ed418acc6433acb98952ca859`.
The backend parity executable is
`d9bf69c1953dc49a0a7306d80950a262ad406d188de9e659c1c8563fc6fee9ff`.
Source snapshots and hashed raw records are linked from the manifest/results.
Later source edits only corrected kernel comments and a Python docstring.

## Remaining gates

1. The existing exact rebase every 64 changes still spikes to approximately
   104 ms on the larger scans. Do not remove it or alter accumulation order
   without a new scientific parity qualification.
2. The hot native interaction path still submits statistics after the detector
   batch, then publishes through the main thread and Metal drawable queue.
   The combined submission implemented for staged/loading images does not
   eliminate this separate hot-path synchronization.
3. Measure sustained distinct presentations, frame gaps, and missing events
   under continuous native input before claiming 120 FPS.
4. Python CUDA/MPS use of this new float codec and float64 sources are not
   qualified by these Metal tests. No all-formats/all-backends claim is made.

Regression checks: 31 native tests and 19 subtests passed; 29 focused Python
reference/layout/metadata tests passed with one explicit skip. Registry and
whitespace validation are separate checks. No release, push, or installed-app
replacement is part of this experiment.
