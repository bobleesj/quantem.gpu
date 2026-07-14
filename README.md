# quantem.gpu

`quantem.gpu` is the multi-backend accelerated STEM package for QuantEM.
The public brand is `quantem.gpu`, not `quantem.cuda`.

## Charter

`quantem.gpu` owns:

- GPU IO and decompression: bitshuffle/LZ4 HDF5 chunk decode, chunk assembly,
  load-to-device, pinned/zero-copy host transfer paths, and detector masking
  during decode.
- Heavy compute: virtual images, BF/DF/DPC, reductions, and later SSB engines.
- Device policy: explicit `cuda`, `mps`, and `cpu` selection with honest errors.

`quantem.widget` owns frontend behavior: anywidget UI, export, interaction, and
display. It should call `quantem.gpu` for accelerated load and compute.

`quantem.live` apps, CLI, and dashboard should move to `quantem.gpu` over time
instead of keeping second permanent copies of load/decompress/math paths.

Dependency arrow:

```text
file -> quantem.gpu (load + decompress + to_device) -> arrays
     -> quantem.gpu (products / later SSB) -> quantem.widget (display)
```

## Backends

- `cuda`: CuPy RawKernel bitshuffle/LZ4 decompression and GPU arrays. This is
  the phase-1 migrated hot path.
- `mps`: Apple Silicon device selection is reported. Chunk-backed virtual image
  and CoM/DPC product dispatch now lives in `quantem.gpu.compute`. Metal
  bitshuffle/LZ4 chunk IO and zero-copy chunk assembly now live in
  `quantem.gpu.io.backends.mps`. SSB has MLX-backed fixed-aberration preview
  (`ssb_preview_mps`) and C10/C12/phi12 free-fit (`ssb_fit_mps`) APIs that run
  on Apple GPU without Torch.
- `cpu`: h5py/hdf5plugin reference decode for availability and parity.

## Phase-1 Status

Implemented in this package:

- `import quantem.gpu`
- `quantem.gpu.device_report()` and `quantem.gpu.select_device()`
- `quantem.gpu.io.hdf5.load()`, copied from the proven `quantem.widget` HDF5
  loader and kept API-compatible for the migrated slice
- CUDA bitshuffle/LZ4 kernels and pinned-buffer HDF5 master load path
- MPS Metal bitshuffle/LZ4 kernels, chunk-backed zero-copy load path, memory
  guard, and `load_mps_4dstem`
- `quantem.widget.io.hdf5` shim in the migration worktree, re-exporting the new
  `quantem.gpu.io.hdf5` API for one release
- `quantem.gpu.detector` BF/DF/ADF, `mean_dp`, `masked_sum`, `dp_mean`,
  `virtual_image`, and BF disk detection copied from the widget/live paths with
  parity tests
- `quantem.gpu.dpc` CoM/DPC/iDPC copied from the widget path with parity tests
- `quantem.gpu.ssb` SSB engine/API copied from the live compute path, with
  real-data parity and speed tests against the legacy live implementation
- `quantem.gpu.compute` MPS chunk-backed virtual-image and CoM/DPC compute
  copied from widget Metal kernels; Linux CI has dispatch guardrails, and true
  Metal runtime parity must run on macOS
- `quantem.gpu.ssb.mps.ssb_preview` / `quantem.gpu.ssb_preview_mps` and
  `quantem.gpu.ssb.mps.ssb_fit` / `quantem.gpu.ssb_fit_mps`, optional
  MLX-backed MPS SSB preview/free-fit paths for chunk-backed Mac data.

Out of scope for phase 1:

- Full SSB UI inside Show4DSTEM
- Full `quantem.live` CLI/dashboard migration, though SSB/detector production
  call sites have started routing directly through `quantem.gpu`
- Rewriting every `quantem.widget.io` helper
- Paper scripts or denoise sweeps

## Next Phases

- Phase 2: complete product migration coverage, including macOS MPS runtime
  parity and broader real-data product parity.
- Phase 3: broaden SSB real-data parity, including full CUDA-engine optimizer
  parity and dashboard integration for the MPS MLX fit path.
- Phase 4: add Show4DSTEM More -> interactive SSB preview and save.
- Phase 5: move live CLI/dashboard callers onto `quantem.gpu`.
- Phase 6: fold/rename `quantem.cuda` internals under `quantem.gpu` backends.
