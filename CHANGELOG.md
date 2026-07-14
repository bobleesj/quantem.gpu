# Changelog

One line per release candidate: the main user-facing thing that changed. Newest
first. Add an entry under **Unreleased** as you land a change; move it under the
new `rcN` heading when that rc is published to TestPyPI.

## Unreleased

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
