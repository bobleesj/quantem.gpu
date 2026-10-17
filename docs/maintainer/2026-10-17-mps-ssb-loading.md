# Reuse MPS loading buffers and decode detector regions directly

## Question and change

Can original-HDF5 loading and SSB session construction do less work while
returning exactly the same counts and phase?

Two changes reduce allocation, decoding and copying:

- The encoded MPS loader keeps one HDF5 decoder across its bounded batches.
  Compressed-input and block-metadata buffers are reused, with capacity growth
  when a later batch is larger. Decoded output retains independent ownership.
  Decoder scratch is released when loading finishes, including on failure.
- Integer MPS residents decode a requested detector rectangle directly into
  the final Torch tensor. Full scan rows use one command and one synchronization
  boundary. Previously, the same SSB read decoded complete detector frames,
  cropped them and copied the result in 74 batches. Partial scan rows retain
  bounded batching. Calibrated float residents keep their existing decoder.

The acquisition remains complete and losslessly encoded. This implements the
existing SSB detector selection without its intermediate whole-frame buffers;
it does not change BF selection, calibration, precision or the estimator.
All stored values, including values at invalid pixels, retain their existing
read semantics. Default bad-pixel correction during loading is unchanged.

## Fresh-process measurements

Baseline: `e2080809f2bf54813d3281e7126148a8aa2518a7`.
Device: Apple M5 Max, 40 GPU cores, 128 GiB unified memory. Both acquisitions
retain the complete 512 × 512 × 192 × 192 uint16 input. Acquisition A retains
8,807 BF pixels and returns 1536² Auto phase; B retains 8,930 and returns
2048². There is no additional crop, binning or BF subsampling.

Each version ran in six fresh processes per acquisition, alternating baseline
and candidate order. Calibration and rotation were fixed. The joint workflow
uses 200 trials plus Newton, including the existing zero-depth comparison.
GPU stage boundaries synchronize completion. The timer starts before loading
and ends after saving native and Auto phase arrays and metadata. Imports,
later parity checks and teardown are excluded. OS and shader caches were not
cleared. Percentiles from six observations describe these runs, not a latency
guarantee.

| Stage, seconds | A before p50 / p95 / max | A after p50 / p95 / max | B before p50 / p95 / max | B after p50 / p95 / max |
| --- | ---: | ---: | ---: | ---: |
| Load | 2.415 / 3.315 / 3.591 | 2.102 / 2.341 / 2.382 | 2.452 / 2.555 / 2.565 | 2.196 / 2.587 / 2.713 |
| Session setup | 1.234 / 2.076 / 2.352 | 0.981 / 1.058 / 1.072 | 1.229 / 1.282 / 1.286 | 1.002 / 1.034 / 1.039 |
| Load + session | 3.619 / 5.389 / 5.942 | 3.095 / 3.393 / 3.454 | 3.707 / 3.798 / 3.799 | 3.213 / 3.562 / 3.670 |
| Complete workflow | 14.431 / 18.952 / 20.408 | 13.177 / 15.080 / 15.451 | 22.329 / 23.864 / 24.153 | 20.620 / 23.841 / 23.886 |

Loading includes reading original compressed chunks, GPU decompression,
the existing bad-pixel correction, lossless ANS encoding and spatial indexing.
Median loading plus session construction fell by 14% for A and 13% for B.

| Remaining stage, median seconds | A before | A after | B before | B after |
| --- | ---: | ---: | ---: | ---: |
| BF backend selection | 0.108 | 0.107 | 0.107 | 0.108 |
| Fourier preparation | 1.086 | 0.904 | 0.948 | 0.903 |
| Fit and native phase | 8.825 | 8.462 | 15.740 | 15.014 |
| Auto reconstruction | 0.692 | 0.737 | 1.347 | 1.435 |
| Output handoff | 0.007 | 0.008 | 0.012 | 0.012 |

Fitting and reconstruction code did not change. Their timing differences
are not credited to this optimization. Stage medians do not sum to the median
total. Fitting remains the dominant cost.

Peak MLX allocation remained 18.47/18.95 GiB for A/B. The maximum sampled
Metal-driver allocation fell from 57.67/59.09 GiB to 56.70/58.12 GiB, a reduction
of 0.97 GiB for each acquisition. Samples were taken every 50 ms. Peak process
RSS was 3.06/1.55 GiB. These overlapping memory accounts must not be added.

## Numerical checks

- Every working count after the existing bad-pixel correction, every retained
  encoded buffer and every spatial index matched the frozen GPU baseline
  exactly for both complete acquisitions.
- Complete native and Auto phase arrays matched bit-for-bit for ordinary SSB
  and tilt/depth-corrected SSB in all 24 runs. Fitted parameters, objective
  values and trial histories also matched exactly. Existing tolerances were
  not changed and frozen references were not regenerated.
- Ordinary reconstruction controls disabled tilt and depth on the same fixed
  optics. Separate ordinary-mode application runs exercised their own fitting
  workflow.
- Hardware tests cover uint8/uint16, rectangular and single-pixel detector
  reads, full/partial scan rows, codec boundaries and tails, stored invalid
  pixels, changing compressed batch sizes and repeated loads.

The focused suite passed 107 tests, with three unavailable external-fixture
tests skipped. Skipped cases are not passes. The separate compressed-acquisition
IO gate passed 16 workflow categories. Its two full external collections were
unavailable, so the overall IO gate remains incomplete. The complete coverage
table is retained in the private evidence.

## Consuming application

QuantEM.Live `16cc9a1e` processed three fresh arrivals per version and
acquisition through its folder watcher, automatic calibration, 200-trial
joint search with Newton, native save, Auto reconstruction and first HTTP
image. Each version used a persistent isolated server/worker. Server startup
was outside the arrival timer; detection and application overhead were inside.
No browser interaction ran during the timing comparison.

| Arrival through first served Auto image | Before p50 / p95 / max | After p50 / p95 / max |
| --- | ---: | ---: |
| A | 15.47 / 16.72 / 16.86 s | 17.72 / 22.59 / 23.13 s |
| B | 27.28 / 27.44 / 27.46 s | 26.79 / 33.75 / 34.52 s |

These application timings do **not** establish a consistent wall-time gain.
The unchanged fitting stage varied substantially on the shared interactive
machine, including slower candidate runs. All observations are retained.
The alternating fresh-process comparison supports the narrower loading and
session improvement; it does not justify attributing every wall-time change
to the new loader. Application correctness and timing are separate results.

All six paired joint application results retained bit-exact native and Auto
arrays and identical fitted parameters and objective values. Both separate
ordinary-mode runs retained bit-exact native and Auto arrays against the frozen
application results. Original HDF5 files were unchanged.
A separate browser check displayed acquisition A's saved Auto phase and FFT
with the expected tilt/depth metadata, 200 trials and Newton refinement.

This change qualifies Python MPS on the tested device. It does not qualify
CUDA, native Swift or a release. Evidence identifier:
`20261017-mps-ssb-loading`. Original inputs, complete arrays, timing records,
commands and application evidence remain private.
