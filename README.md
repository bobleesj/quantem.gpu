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
- `mps`: Apple Silicon device selection is reported. The Metal IO implementation
  remains a temporary legacy shim during phase 1 and should move here in a
  later backend consolidation pass.
- `cpu`: h5py/hdf5plugin reference decode for availability and parity.

## Phase-1 Status

Implemented in this package:

- `import quantem.gpu`
- `quantem.gpu.device_report()` and `quantem.gpu.select_device()`
- `quantem.gpu.io.hdf5.load()`, copied from the proven `quantem.widget` HDF5
  loader and kept API-compatible for the migrated slice
- CUDA bitshuffle/LZ4 kernels and pinned-buffer HDF5 master load path
- `quantem.widget.io.hdf5` shim in the migration worktree, re-exporting the new
  `quantem.gpu.io.hdf5` API for one release

Out of scope for phase 1:

- Full SSB UI inside Show4DSTEM
- Full `quantem.live` migration
- Rewriting every `quantem.widget.io` helper
- Paper scripts or denoise sweeps

## Next Phases

- Phase 2: move BF/DF/DPC and other products into `quantem.gpu` with parity.
- Phase 3: add SSB engine as a pure compute API with preview and free-fit modes.
- Phase 4: add Show4DSTEM More -> interactive SSB preview and save.
- Phase 5: move live CLI/dashboard callers onto `quantem.gpu`.
- Phase 6: fold/rename `quantem.cuda` internals under `quantem.gpu` backends.
