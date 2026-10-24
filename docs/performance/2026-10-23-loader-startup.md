# Acquisition-worker loader preparation, 2026-10-23

Private experiment: `quantem-progressive-latency-20261023`. Unpublished source
checkout, not an installed-wheel qualification or a release promise.

## Change and workload

A private CUDA loader preparation helper initializes the existing header parser,
decode module, and shared ANS coding tables before an acquisition arrives. It
opens no source and allocates no detector-count volume. Loading still uses the
public `io.load` path. No decoding arithmetic, scan positions, BF pixels,
precision, fitting budget, or calibration changes.

Both real acquisitions contain a complete 512 × 512 scan and native 192 × 192
uint16 detector. CUDA ran on an RTX PRO 6000 Blackwell Workstation GPU, with
Python 3.14.6, CuPy 14.2.0 and NumPy 2.4.6. The browser ran separately on an
Apple M5 Max using hardware WebGPU. There was no competing scientific job;
existing desktop/background contexts remained running.

GPU base: `d2db0bdb4c9859230d3e9ba2f6b23fcfae8eb0f1` plus local changes.
The full-volume benchmark records diff SHA-256
`b51b0d24d1506a5b96b97e99d23b8e4efe3629fdecb13b1ee0d9e1b24dbc7dec`.
Concurrent MPS changes are not qualified by these CUDA observations.
Live base: `20aed10c0d3a9b7d6f6db0603ab3f1ea8da723b3` plus staged-result,
H5-first loading, worker preparation and immediate ready-event refresh changes.
The worker source SHA-256 is
`55d8ea285055efccc5a8b747684a3db7b2acfdcb9965e6a6e267932dd0cb9aad`.

## Measured boundaries

These are individual observations, not median/p95 estimates. OS source pages
and compiler caches were retained, not flushed; none is a cold-disk claim.

| Boundary | Acquisition A | Acquisition B |
| --- | ---: | ---: |
| Previous loading including QEM write and verification | 4.610 s | 4.665 s |
| H5-first loading, without startup preparation | 1.907 s | 1.874 s |
| H5-first loading, with startup preparation | 1.295 s | 1.412 s |
| Initial 200-trial fit, prepared worker | 1.084 s | 2.050 s |
| Additional fifth-order refinement | 7.634 s | 7.841 s |
| Initial image visible, prepared worker | 4.288 s | 6.599 s |
| Final image visible, prepared worker | 12.376 s | 14.679 s |

Browser times start when the last staged chunk finishes writing, and include
Watch readiness checks, loading, previews, fitting, saved products, transport
and paint. The test deliberately holds an incomplete final chunk first; it
must not start reconstruction prematurely. The initial image updates to the
final image without navigation or reload. This does not establish a two-second
initial browser result or ten-second final browser result.

Standalone profiling on acquisition A measured 0.307 s startup preparation,
1.019 s first load and 0.601 s repeated load. First-load host registration used
0.234 s. A separate shipped `benchmark_hdf5_load.py` run, after 0.202 s startup
preparation, measured 0.895 s synchronized public loading. Its complete logical
uint16 output hash matched the preserved reference; hashing took another
12.961 s and was excluded from loading. Existing canonical-path indexes and
uncontrolled source-page reuse mean this is not a new-path arrival latency.

## Scientific and UI gates

- Both acquisitions: all nine saved product arrays, the initial native phase,
  13 fitted/calibration fields, and optimizer seeds/bounds/coordinates agree
  exactly with the retained pre-change run. No tolerance or reference changed.
- Complete acquisition A logical pixels match SHA-256
  `9729a14aadd3bb2ffca8d85955c1bb0784a3c4d9eda657120e844b58375983b1`,
  with source/working detector 192 × 192, bin 1, uint16.
- The physical-CUDA synthetic regression retains uint16 values through 65535
  across repeated preparation and loading.
- Both real browser tests passed initial/final image updates, phase/FFT,
  native/Auto switching, and cached reopening without an implicit refit.
- Acquisition B uses a preserved experimental calibration override, not
  independently verified acquisition calibration.

The initial/fifth-order optimizer remains the existing hybrid CPU numerical
optimizer with CUDA objective/reconstruction. This change is startup scheduling
and application IO/presentation work, not a new all-GPU solver or faster kernel.
No production service restart, publication, release, or original-data deletion.
