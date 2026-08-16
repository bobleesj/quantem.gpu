# Changelog

One line per release candidate: the main user-facing thing that changed. Newest
first. Add an entry under **Unreleased** as you land a change; move it under the
new `rcN` heading when that rc is published to TestPyPI.

## Unreleased

- Add a loopback-only CUDA browse service for native applications, with exact
  virtual-detector and selected-diffraction transport, acquisition monitoring,
  crop/bin admission, automatic whole-dataset placement across multiple GPUs,
  and bounded per-device resident caches. The service has no `quantem.live`
  dependency and ships through the `[cuda,remote]` install extras. A single
  `quantem-gpu-remote` Conda environment is provided for workstation setup,
  and memory admission preserves native `uint32` capacity when an Arina source
  cannot be narrowed losslessly.
- Add drift-aware sparse ptychography batches to `quantem.gpu.io.load()`: a
  shared random position set can be paired with one dense integer or
  fractional `(row, col)` drift field per source, matching the scan grid.
  Raw diffraction patterns remain unchanged and corrected float32 probe
  positions are recorded in `metadata["drift_batch"]`.
- Add `load(..., output="torch")` for direct Torch tensors on CUDA, MPS, and
  CPU, including recursive conversion of multi-dataset results. Add the
  notebook-friendly `quantem.gpu.device.profile()` diagnostic for reporting
  host, Python environment, Torch version, and the resolved compute backend
  without manual printing.
- Organize public compute around strict scientific domains: `io`, `detector`,
  `dpc`, `parallax`, `screening`, `device`, and `SSB`. CUDA, MPS, and WebGPU
  implementations now live below each domain's `compute` or `backends`
  directory; deleted flat APIs and the top-level WebGPU package are not kept as
  compatibility aliases.
- Add native four-byte unsigned (`dtype='uint32'` / `u32`) load and virtual-image
  support across CUDA, MPS, and WebGPU, and add CUDA packed `dtype='u4'`
  output for true 4-bit counts (`0..15`). CUDA and MPS selected sums use wider
  internal accumulation before float32 display output; MPS and WebGPU load paths
  preserve native `uint32` unless the caller explicitly requests `uint8` browse
  clipping. WebGPU product-first low8 sidecars now reject `uint32` sources
  instead of silently dropping high bits. Public `dtype='u4'` now means packed
  two-counts-per-byte storage with exact range audit and CUDA BF/DF/CoM kernels;
  it no longer aliases NumPy's four-byte `<u4` storage token.
- Add MPS cache-miss generation for `screening.prepare()`. CUDA still
  uses the RawKernel reduction path; MPS now streams raw HDF5 row chunks through
  chunk-backed Metal BF/DF/CoM reducers, records timing/memory metadata, and
  matched CUDA on local anonymized real-data agreement: mean DP/BF/DF exact,
  CoM max abs error `7.63e-6`, and matched rotation/radius.
- Add WebGPU GPU-resident DPC row/col and iDPC reducers to the canonical
  domain-owned Show4DSTEM WebGPU engine. The browser path now computes CoM,
  global CoM mean, centered DPC components, and fixed-rotation iDPC in WGSL,
  with direct browser agreement against NumPy/CUDA references and a local
  anonymized full-512 no-bin real-data NVIDIA WebGPU stress run. The latest headed signoff
  reports DPC row/col max abs error `7.63e-6`, iDPC mean abs error `4.70e-6`
  (`3.05e-5` max, float32 FFT tolerance), GPU-resident display medians of
  about `14.9/13.2/13.2 ms`, and full recompute medians of
  `13.7/19.3/22.7 ms` for DPC row/DPC col/iDPC on an RTX PRO 6000
  Blackwell WebGPU adapter after batching the browser FFT passes into one
  command submission. The browser benchmark harness now has a
  `--require-local-profile` guard so local-file timing runs cannot silently
  accept the URL/fetch fallback path.
- Sign off the Show4DSTEM WebGPU local-H5 detector-bin path for explicit
  `detBin=2/4/8` on full `512x512x192x192` and true crop-256 real evidence.
  The WGSL load path now zeroes raw bad detector pixels before binning and
  keeps the binned output free of raw-detector bad-pixel indices. Headed Chrome
  on an RTX PRO 6000 Blackwell WebGPU adapter matched corrected-frame integer
  checksums exactly against the zero-bad-before-bin reference, with full-load
  count-audited low8 page profiles `1.199/1.212/1.106 s` and crop-256
  20-repeat medians `0.774/0.755/0.733 s` with p95
  `0.798/0.813/0.775 s`; native non-low8 `uint16` `detBin=2` was also exact
  at `2.651 s`.
- Sign off WebGPU product-first BF selected-block loading on true
  real-acquisition `1024x1024x192x192` evidence with BF radius `30`. Headed
  Chrome on an RTX PRO 6000 Blackwell WebGPU adapter matched an independent
  Python reference exactly (`max_abs=0`, `mean_abs=0`, mismatches `0`) with
  4-run median wall `4.92 s`, page/profile `4.85 s`, product stage `1.56 s`,
  selected compressed payload `6.88 GB`, and `4.19 MB` output. This is a
  product-first signoff, not full-stack no-bin browser browse/load signoff.
- Harden the WebGPU selected-block staging uploader so reused staging buffers
  are not remapped until the previous submitted copy work has completed. The
  browser product benchmark now validates fixture existence, mounted file
  count, required reference arrays, and product-debug hooks before reporting
  timings.
- Refresh the MPS SSB performance status on a 24 GB Apple M5 reference laptop:
  radius-30
  `512x512` object steering is real-time (`10.86 ms` mean), radius-30 exact
  phase/loss is reviewable but not CUDA-like (`76.28 ms` mean), and full-active
  `512x512` exact phase/loss remains slow (`528.90 ms` mean). The docs now keep
  BF policy, object-wave steering, and exact phase/loss timing separate.
- Record true real-acquisition `1024x1024x192x192` CUDA and MPS HDF5
  load/decode signoffs: no hidden bin/crop, `uint16` output, selected
  corrected frames bit-exact against direct HDF5, `77.31 GB` resident,
  `4.704 s` wall on CUDA and `4.617 s` wall through the chunk-backed MPS path.
- Keep Show4DSTEM browser VI/DPC ownership beside the detector and DPC domains: detector
  and scan mask builders now live with the canonical WebGPU compute source, and
  the widget source-contract tests verify the frontend does not reintroduce
  local BF/DF/DPC mask helper implementations.
- Add a CUDA RawKernel virtual-image backend for resident CuPy uint8/uint16
  4D-STEM data, wire `compute_backend(cupy_array)` to it, and add exact parity
  tests against the old CuPy selected-pixel reduction. Add a
  `virtual_image_kernel_support()` probe plus a maintainer checklist covering
  CUDA, MPS, and WebGPU browser paths, including the future
  `1024x1024x192x192 uint8` target. The CUDA path now uses warp-shuffle
  selected-pixel reducers, a custom total-count reducer, fused dense
  `total - complement` output, and per-viewer detector-index caching. On a
  local anonymized full 512x512x192x192 real-data benchmark, median BF/ADF/DF drag
  latency improved from 4.96/16.16/62.64 ms on the old widget Torch path to
  1.35/3.86/1.84 ms with bit-exact output. On a local seven-tilt detector-bin2
  benchmark, per-panel BF/ADF/DF medians are now 0.54/1.35/0.53 ms with
  max absolute error 0.
- Add a CUDA RawKernel CoM/DPC reducer for resident CuPy uint8/uint16 data.
  The fused kernel accumulates total intensity, detector-row moment, and
  detector-column moment in one detector pass and caches the full-detector CoM
  field per backend. On a local anonymized full 512x512x192x192 uint16 benchmark, DPC
  CoM improved from 200.42 ms to 12.39 ms with max absolute error 0; on a
  local seven-panel detector-bin2 benchmark, first full-grid DPC improved
  from 373.14 ms to 24.63 ms with max absolute error 0, and repeated DPC reads
  use the backend cache.
- Clarify the cross-backend CoM/DPC product-kernel tracker: MPS uses raw Metal
  `com_u8`/`com_u16`, while WebGPU already has WGSL masked CoM source under
  the DPC domain but still needs the same GPU-resident buffer/cache
  parity path as virtual-image dragging.
- Add canonical reusable WebGPU/TypeScript browser compute beside each
  scientific domain. The Show4DSTEM and ShowPtycho browser engines are shipped
  as package data for widget builds.
- Fix MPS SSB fixed-aberration loss reporting for cached 512x512 geometry. The
  cached path now treats the 512 column-kernel sum-of-squares as a scalar, so
  real-data MPS fixed phase/loss and sparse optimizer parity pass against the
  CUDA reference artifacts.
- Add MPS Metal uint8 virtual-image kernels and route
  `load(..., backend="mps", dtype="u8")` through chunk-backed Metal IO, so
  Show4DSTEM browse loads do not materialize a giant Torch-MPS tensor.
- Add the MPS dense-mask `total - complement` cache path so dark-field style
  Show4DSTEM drags use sparse complement reads on Metal, matching the CUDA
  kernel strategy.
- Make CUDA SSB batch variance deterministic for sparse 256/512/1024 row
  transforms, clarify the ShowPtycho UI handoff, and document that WebGPU/WGSL
  runs in the browser while reusable source lives beside its scientific domain.

## rc5 - 2026-07-14

- Add the first documentation site with install/backend tutorials, simplify the
  tutorial language around BF/DF/ADF/DPC, add movie rendering docs, and add a
  backend coverage matrix for CUDA, MPS, CPU, and remaining migration work.
- Add an Apple Metal/MPS MP4 rendering backend so `save_mp4(...,
  backend="auto")` tries CUDA/NVENC, then MPS/Metal, then CPU/ffmpeg.

## rc4 - 2026-07-14

- Correct installed package version reporting so `quantem.gpu.__version__`
  matches the `quantem.gpu` distribution version from TestPyPI installs.

## rc3 - 2026-07-14

- Add `load(path, scan_region=(row_start, row_stop, col_start, col_stop))` as
  the crop-first HDF5 API.
- Move MPS crop-first sparse HDF5 decode and the lazy multi-dataset MPS loader
  into `quantem.gpu.io`, leaving `quantem.widget.multidataset_mps` as a
  compatibility re-export.
- Add real-data CUDA/MPS parity tests for crop-first HDF5 IO and MPS SSB sparse
  optimizer objective checks on a full 512x512 acquisition.
- Match MPS SSB fixed-preview phase output to CUDA's mean-of-per-BF-phase
  contract, tighten real-data phase parity thresholds, and add a fused
  MLX/Metal correction kernel that reduces MPS sparse objective timing from
  about 26 ms/candidate to about 7 ms/candidate on a 24 GB Apple M5 reference
  laptop.

## rc2 - 2026-07-14

- Publish the first `quantem.gpu` release candidate to TestPyPI as the
  multi-backend accelerated STEM package for QuantEM (`cuda`, `mps`, `cpu`),
  with a Quick Start README showing install, device reporting, HDF5 crop load,
  virtual detector products, and widget migration usage.
- Move the HDF5 GPU IO/decompression hot path into `quantem.gpu.io`, including
  CUDA bitshuffle/LZ4 chunk decode, pinned-buffer master loading, scan-region
  crop loading, MPS Metal bitshuffle/LZ4 chunk IO, and CPU reference decode for
  parity.
- Add device policy helpers (`device_report`, `select_device`) and import-light
  lazy exports so `import quantem.gpu` works without CUDA/CuPy installed.
- Move BF/DF/ADF, mean diffraction pattern, masked-sum, virtual image, CoM/DPC,
  and iDPC compute paths into `quantem.gpu`, with parity tests against the
  legacy widget/live paths.
- Move SSB compute APIs from `quantem.live` into `quantem.gpu.ssb`, including
  CUDA reference parity, MPS/MLX preview and C10/C12/phi12 free-fit paths, and
  real-data parity/speed checks used during migration.
- Move MPS chunk-backed product compute and movie export helpers into
  `quantem.gpu`, leaving widget responsible for frontend display/export
  orchestration.
- Wire the `quantem.widget` migration branch to depend on
  `quantem.gpu>=0.0.1rc2`, so widget HDF5 loading and accelerated products can
  call the new package without changing public widget APIs.
- Add release automation for `gpu-v*` tags, TestPyPI trusted publishing through
  `release.yml`, MIT license packaging, and an NVIDIA nvCOMP CUDA LZ4
  BSD-3-Clause third-party notice.
