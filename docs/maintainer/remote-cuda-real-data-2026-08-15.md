# Remote CUDA Real-Data Acceptance Record

Date: 2026-08-15

Hardware: two 96 GB RTX PRO 6000 Blackwell GPUs

Environment: clean Python 3.12 Conda environment created from
`environment-remote-cuda.yml`, with the current branch installed as a wheel.
The environment contained `quantem.gpu[cuda,remote]` and did not contain
`quantem.live`, `quantem.widget`, or a separate web frontend.

## Exact real-data results

All cold times include HDF5 reading, GPU decode, resident-volume creation, and
the first BF response through the HTTP protocol. Warm times are subsequent
responses from the CUDA-resident volume. No crop or binning was used unless
the row says so.

| Evidence | Plan | Cold load | Resident bytes | Exact parity |
| --- | --- | ---: | ---: | --- |
| Seven 512x512x192x192 uint16 tilts | one full tilt per request | 1.998 s median | 19,327,352,832 each | selected DP and custom detector, max error 0 |
| 512x512x192x192 uint32 source | full native detector | 4.113 s | 19,327,352,832 after lossless narrowing | selected DP, max error 0 |
| 1024x1024x192x192 uint16 source | detector bin 2 | 6.156 s | 38,654,705,664 | selected DP, max error 0 |
| same 1024 source | detector bin 2, scan bin 2 | 4.819 s | 9,663,676,416 | selected DP, max error 0 |
| same 1024 source | direct 256x256 scan crop | 0.155 s | 4,831,838,208 | selected DP, max error 0 |

The seven-tilt cold range was 1.528-4.136 seconds. With a bounded two-entry
cache on each GPU, all seven tilts loaded without exceeding 38,654,705,664
resident bytes per GPU. The least-populated-device policy distributed complete
volumes across both GPUs; it did not split a volume or count aggregate memory
as single-volume capacity.

Representative warm responses were BF 1.78 ms, ABF 16.68 ms, ADF 7.48 ms,
CoM row 18.61 ms, CoM column 2.71 ms, selected diffraction 1.41 ms, and custom
detector 3.63 ms. These are service response times, not display-frame-rate
claims.

The unbinned 1024 source was rejected in 1.7 ms before allocation: its exact
transition peak was 72.0 GB, above the configured 52.2 GB per-device budget.
The cropped plan read only the requested scan region. The scan-binned plan
replaced the source representation, and changing plans evicted the stale
same-file representation.

## Failures found and retained fixes

1. An unpinned TestPyPI install selected `0.0.1rc6`, which predates the remote
   service and therefore had no `quantem-gpu` command. The environment now
   requires `>=0.0.1rc7`. Publish that release before distributing the file.
2. Admission assumed two bytes per unbinned value. Catalog inspection now
   reports source dtype and reserves the native item size, including the safe
   `uint32` fallback, before CUDA allocation.
3. File monitoring guessed conventional shard names. It now follows actual
   HDF5 external links, including absolute paths and nonstandard names. The
   1024 evidence correctly reports 2,799,459,964 on-disk bytes instead of only
   the 7,424-byte master file.
4. CuPy scan binning imported Torch even though it did not use Torch. The
   import is now optional, and a minimal-environment regression test covers
   the CuPy path.

## Native application gate

The packaged arm64 Live4DSTEM app connected through an SSH alias, launched the
service from the standard Conda environment path, cataloged all seven tilts,
and displayed non-black diffraction, BF, ADF, thumbnails, histograms, and
metadata. Dataset and product switching were exercised through native UI
controls on the target display. The structural bundle verifier passed; the
full Swift suite passed 39 tests with one opt-in test skipped, and the real
SSH/CUDA test passed separately.

This record contains no private hostnames, filesystem paths, credentials, or
microscope filenames.
