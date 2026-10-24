# MPS SSB correction and reduction fusion — 17 October 2026

Experiment: `ssb-parameter-reuse-20261017`. Acquisition identities, full-array
comparisons, profiler traces and application evidence remain private.

## Change

The coherent upsampling kernel now corrects and accumulates 64 existing
bright-field groups per launch. Each frequency reuses its coordinates and tilt
shift across groups. Partial group spectra stay in registers instead of being
written to memory and read by a second reduction kernel.

The BF group boundaries, within-group additions and sequential group additions
are unchanged. So are aperture support, Hermitian handling, complex64 source
spectra, float32 output, the phase-of-mean estimator and native loss evaluation.
Cancellation checkpoints remain between bounded launches. No BF pixels are
removed, and output sampling is unchanged.

## Measurements

Two full 512 × 512 scans with 192 × 192 detectors were measured on an Apple
M5 Max with 128 GiB memory. Each row contains 20 alternating reference/candidate
pairs in one process, after preparation. Every call changed parameters. These
are complete prepared phase evaluations, including the NumPy output handoff;
they exclude loading, fitting, transport and browser drawing.

| BF pixels / output | Change | Reference median | Candidate median |
| --- | --- | ---: | ---: |
| 8,807 / 1536² | C10, joint model | 123.3 ms | 121.8 ms |
| 8,807 / 1536² | Tilt, joint model | 124.0 ms | 123.7 ms |
| 8,807 / 1536² | C10, ordinary control | 110.8 ms | 108.8 ms |
| 8,930 / 2048² | C10, joint model | 299.7 ms | 246.9 ms |
| 8,930 / 2048² | Tilt, joint model | 294.8 ms | 239.4 ms |
| 8,930 / 2048² | C10, ordinary control | 271.4 ms | 211.9 ms |

The smaller output has no meaningful speed gain. The larger output improves
by 18–22% in this paired run. Tail latency remains variable: its C10 p95/max
were 478/891 ms before and 393/460 ms after. These measurements do not establish
a hardware limit or a 60 FPS interaction rate.

Temporary peak MLX allocation falls from 545 to 187 MB for the smaller output
and from 555 to 332 MB for the larger output. The latter previously wrote about
17.1 GB of logical partial-spectrum data per redraw; fusion writes about
0.27 GB of accumulated spectra. Those are array-size calculations, not DRAM
counter readings, and exclude source reads and subsequent FFT buffers.

Controlled non-scientific ablations on the larger case gave median times of
276 ms for the reference, 256 ms without source-spectrum reads, and 149 ms
without optical transfer calculations. These variants diagnose cost only;
their results are not reconstructions. Optical arithmetic and intermediate
traffic both matter. Removing source reads alone did not remove most of the
cost. The Metal trace did not provide hardware ALU or DRAM counters.

## Rejected alternatives and validation

- A separable geometry cache added about 497 MB, ran slower and changed
  rounding. It is not used.
- Full-array addressing with offsets was slower than the existing sliced
  inputs in a paired test. The existing addressing is retained.
- All 120 real full-image comparisons were bit-identical to the frozen GPU
  implementation, across ordinary and joint C10/tilt changes.
- The MPS upsampling suite passed 43 tests, including incomplete BF groups
  across multiple launches, higher-order aberrations, cancellation, anisotropic
  sampling and saved-result behavior.
- Native reconstruction, the 200-trial native fit and saved-run round trip
  passed. The separate external frozen fit-trajectory gate could not run
  because its acquisition fixture was unavailable. No pin was changed.

The change targets reconstruction redraw. It does not change the fitting
objective or promise faster optimization, loading, CUDA or browser rendering.
