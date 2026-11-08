# Native Direct Ptycho preparation, 2026-11-05

The complete five-second native-app target is not yet
established. These measurements concern Swift/Metal; they do not qualify the
Python MPS or CUDA implementations.

## Workload and reference

Apple M5 Max, 128 GiB unified memory, AC power. Full 512 × 512 scan positions
and 192 × 192 detector pixels, original integer counts, 200 search trials,
both estimated-rotation branches, fifth-order refinement and the existing
half-band consistency check. Source calibration, BF selection, precision,
search settings and fallback rules remain fixed within each comparison.

The private experiment ID is `live4dstem-rc2-optimization-20261105`. Its frozen
backend control is `99bf9026`, based on `373725d8` with the preceding native
integration changes preserved. Raw source identities, command receipts,
patches, arrays and app screenshots remain in that private archive.

## Verified preparation improvements

- Fuse the final FFT transpose with the Hermitian-cache write, retaining
  exactly the original Fourier values.
- Fuse the integer conversion with the first 512-point FFT stage. Threadgroup
  staging preserves the original arithmetic for uint8, uint16 and uint32.
  A register-only alternative failed exact parity and was rejected.
- Gather cached spectra in larger batches and traverse the existing power
  buffer contiguously. Each frequency retains its original Double summation
  order. Partly cached and fully streamed cases retain bounded batches.
- Decode QEM columns with contiguous stores through a bounded shared tile.
  Specialize native-width output to avoid unnecessary row-padding indexing.
  Padded scans use the same exact count decoder with the original layout.

Matched warm Fourier preparation with the first fusion already enabled fell
from 0.59–0.63 s to 0.54–0.56 s across three acquisitions. Model preparation
fell from approximately 0.26–0.33 s to 0.15–0.24 s. These are isolated stages,
not total interaction times.

For two QEM acquisitions with 9,081 and 9,110 selected detector columns,
interleaved column-decoder measurements gave:

| Selected-column GPU decoding | Original | Coalesced stores | Native-width | Parallel chunks |
|---|---:|---:|---:|---:|
| Acquisition A | 1.88–1.90 s | 1.45 s | 1.25 s | 0.331 s |
| Acquisition B | 1.89–1.91 s | 1.47 s | 1.28 s | 0.333 s |

All returned column hashes match the frozen original. The reusable runner is
`metal-qem-detector-update-benchmark --columns acquisition.qem pixels.json 3`;
the JSON pixel list contains detector indices in the requested order.
It hashes all returned counts outside synchronized GPU timing and checks
original diffraction samples independently. The parallel-chunk comparison uses
three repetitions per condition, interleaved serial/parallel/serial. Independent
chunks write disjoint positions of the existing bounded 32-plane output buffer;
the ANS stream order and integer counts are unchanged. An argument table adds
32 bytes per chunk (1 KiB for these acquisitions, bounded at 4 KiB), included in
resident accounting. The Metal allocator reports the same allocation totals;
this is not a claim of zero extra metadata bytes or a 16 GiB working set.

Synthetic tests additionally cover partial restart blocks, multiple chunks
starting inside a scan row, flagged pixels, repeated indices, scan padding,
and zero, constant, sparse, entropy and literal count streams.

## Correctness and acceptance boundaries

QEM stored moments include flagged counts, whereas ordinary selected-column
reads mask them. `includeFlaggedPixels: true` explicitly retrieves original
counts for exact moment replacement; the default still masks them. Neither
the source nor its validity mask is changed. Native app rotation now uses the
original flagged values and verifies the remaining independent valid total.

Full 200-trial, fifth-order benchmark results on three acquisitions match the
frozen fitted parameters, branch decisions, objectives and consistency checks
exactly. Independent count tests and native SSB/Phase/Complex image comparisons
are separate gates. Format-specific BF estimation differences must be recorded;
matched comparisons use explicit session-only geometry, without modifying
source calibration.

Small-batch and eight-candidate layouts, paired Double normalization,
additional zero-support culling and several decoder layouts preserved values but failed to establish a speed improvement;
they are archived, not retained as production tuning choices. GPU and host
timings drift under normal desktop conditions. Report first observations
separately from repeated timings, and do not call a warm minimum a cold result.

The first cleaned default passed 17 focused backend tests and seven native
reconstruction journeys plus three native binding-driven slider journeys.
The registered final candidate passed 15 uninterrupted native journeys (three
repetitions for five acquisition/format combinations), with exact initial/final
fits, exact initial-method images and positive first/final drawable receipts.
The first repetition of each combination also checked all three reconstruction
methods. Two injected memory-pressure recovery journeys passed.

Matched serial/parallel/serial native Fourier preparation was 1.801/0.639/1.828 s
for acquisition A and 1.920/0.688/1.920 s for acquisition B. All fits and method
images matched. This supports a roughly 2.8-fold preparation improvement; it is
not a whole-pipeline speedup. Lazy compilation now builds only the requested
column layout and the separately requested totals pipeline.

The final three-repeat native measurements are:

| Acquisition/source | Run to checked image, median (range) | Launch to checked image, median (range) |
|---|---:|---:|
| A / HDF5 | 7.85 (5.11–9.18) s | 10.27 (7.84–11.81) s |
| A / QEM | 7.09 (4.93–8.94) s | 8.62 (7.42–10.33) s |
| B / HDF5 | 11.26 (6.61–13.35) s | 14.06 (9.26–15.90) s |
| B / QEM | 11.17 (9.18–13.75) s | 12.57 (10.79–15.15) s |
| C / HDF5 | 8.96 (5.22–9.37) s | 11.31 (7.79–11.83) s |

Native test hooks request Run and Refine in separate immediately consecutive
actions. Their action gap and consistency checks are included; screenshots occur
after timed work. The source page-cache state is unspecified. The macOS thermal
state moved from nominal to fair during the run, coincident with increased fit
time. GPU clocks were not measured, so this does not establish the sole cause of
timing drift. Fourier preparation remained comparatively stable. These are
native-window tests, not physical mouse or trackpad acceptance.

Acquisitions A and B retain the original standard-range fallback and untrusted
higher-order flag; C retains the wide-range result. Exact parity includes these
decisions. A completed check is not the same as trusted physical coefficients.

The full Swift backend suite passed 311 tests with 31 fixture-dependent skips
and zero failures. Frozen CUDA image references and native full-count audits are
separate acceptance gates; their completion is recorded in the private archive. Loading,
first image, checked fifth-order result, cache interactions and positive
drawable presentation must be reported separately. An active-stage sum that
excludes loading or the Run-to-Refine action gap is not open-to-final latency.
No release, cross-device speed claim or 16 GiB memory guarantee follows from
these measurements.
