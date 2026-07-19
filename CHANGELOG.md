# Changelog

One line per release candidate: the main user-facing thing that changed. Newest
first. Add an entry under **Unreleased** as you land a change; move it under the
new `rcN` heading when that rc is published to TestPyPI.

## Unreleased

- Add WebGPU GPU-resident DPC row/col reducers to the canonical
  `quantem.gpu.webgpu` Show4DSTEM engine. The browser path now computes CoM,
  global CoM mean, and centered DPC components in WGSL, with direct browser
  parity against NumPy and a private real-data NVIDIA WebGPU stress run.
- Keep Show4DSTEM browser VI/DPC ownership in `quantem.gpu.webgpu`: detector
  and scan mask builders now live with the canonical WebGPU compute source, and
  the widget source-contract tests verify the frontend does not reintroduce
  local BF/DF/DPC mask helper implementations.
- Add a CUDA RawKernel virtual-image backend for resident CuPy uint8/uint16
  4D-STEM data, wire `compute_backend(cupy_array)` to it, and add exact parity
  tests against the old CuPy selected-pixel reduction. Add a
  `virtual_image_kernel_support()` probe plus a maintainer checklist covering
  CUDA, MPS, and `quantem.gpu.webgpu` browser paths, including the future
  `1024x1024x192x192 uint8` target. The CUDA path now uses warp-shuffle
  selected-pixel reducers, a custom total-count reducer, fused dense
  `total - complement` output, and per-viewer detector-index caching. On a
  private full 512x512x192x192 real-data benchmark, median BF/ADF/DF drag
  latency improved from 4.96/16.16/62.64 ms on the old widget Torch path to
  1.35/3.86/1.84 ms with bit-exact output. On a private seven-tilt detector-bin2
  benchmark, per-panel BF/ADF/DF medians are now 0.54/1.35/0.53 ms with
  max absolute error 0.
- Add a CUDA RawKernel CoM/DPC reducer for resident CuPy uint8/uint16 data.
  The fused kernel accumulates total intensity, detector-row moment, and
  detector-column moment in one detector pass and caches the full-detector CoM
  field per backend. On a private full 512x512x192x192 uint16 benchmark, DPC
  CoM improved from 200.42 ms to 12.39 ms with max absolute error 0; on a
  private seven-panel detector-bin2 benchmark, first full-grid DPC improved
  from 373.14 ms to 24.63 ms with max absolute error 0, and repeated DPC reads
  use the backend cache.
- Clarify the cross-backend CoM/DPC product-kernel tracker: MPS uses raw Metal
  `com_u8`/`com_u16`, while WebGPU already has WGSL masked CoM source under
  `quantem.gpu.webgpu` but still needs the same GPU-resident buffer/cache
  parity path as virtual-image dragging.
- Add `quantem.gpu.webgpu` as the canonical source package for reusable
  WebGPU/TypeScript browser compute. The existing Show4DSTEM WebGPU engine and
  ShowPtycho SSB browser engine are copied there as package data, with helpers
  for widget build scripts to read the shipped sources.
- Fix MPS SSB fixed-aberration loss reporting for cached 512x512 geometry. The
  cached path now treats the 512 column-kernel sum-of-squares as a scalar, so
  real-data MPS fixed phase/loss and sparse optimizer parity pass against the
  CUDA reference artifacts on Phil.
- Add MPS Metal uint8 virtual-image kernels and route
  `load(..., backend="mps", dtype="u8")` through chunk-backed Metal IO, so
  Show4DSTEM browse loads do not materialize a giant Torch-MPS tensor.
- Add the MPS dense-mask `total - complement` cache path so dark-field style
  Show4DSTEM drags use sparse complement reads on Metal, matching the CUDA
  kernel strategy.
- Make CUDA SSB batch variance deterministic for sparse 256/512/1024 row
  transforms, clarify the ShowPtycho UI handoff, and document that WebGPU/WGSL
  runs in the browser while reusable source lives in `quantem.gpu.webgpu`.

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
  the friendly crop-first HDF5 API. `load_scan_region()` remains available as a
  compatibility helper.
- Move MPS crop-first sparse HDF5 decode and the lazy multi-dataset MPS loader
  into `quantem.gpu.io`, leaving `quantem.widget.multidataset_mps` as a
  compatibility re-export.
- Add real-data CUDA/MPS parity tests for crop-first HDF5 IO and MPS SSB sparse
  optimizer objective checks on the full Samsung 512x512 dataset.
- Match MPS SSB fixed-preview phase output to CUDA's mean-of-per-BF-phase
  contract, tighten real Samsung phase parity thresholds, and add a fused
  MLX/Metal correction kernel that reduces MPS sparse objective timing from
  about 26 ms/candidate to about 7 ms/candidate on Phil.

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
