# Migration notes

`quantem.gpu` exists to remove permanent duplicate accelerated code from
`quantem.widget`, `quantem.live`, and the legacy `quantem.cuda` name.

## Current ownership

`quantem.gpu` owns:

- GPU IO and decompression.
- Chunk assembly and load-to-device.
- Device selection and backend errors.
- Heavy BF/DF/DPC image compute.
- SSB compute APIs.

`quantem.widget` owns:

- anywidget UI.
- Interaction state.
- HTML/notebook export.
- Display wrappers around arrays and reduced images from `quantem.gpu`.

`quantem.live` calls `quantem.gpu` for product and SSB compute instead of
keeping second copies.

Native macOS Live4DSTEM calls the Swift package products instead of a local
Python backend:

| Client need | Endpoint |
|---|---|
| Browser FFT of BF/ADF/custom | `MetalImageFFT.logMagnitude` |
| Histogram / contrast window | `MetalImageRuntime` |
| HDF5/EMD catalog | `Native4DSTEMIO` |
| Decode / detector / CoM | `Metal4DSTEMKernels` |
| Remote raw 4D on a CUDA workstation | Python `quantem.gpu` CUDA service only |

Do not restore an app-local FFT, histogram, or HDF5 parser. Do not bundle
Python in the signed Mac app. Pin Live4DSTEM to an exact `quantem.gpu`
revision after this package is published; a local path override is only for
integration worktrees.

## Release checks

Before publishing an rc:

1. Run focused GPU parity tests.
2. Build wheel and sdist into a temporary directory.
3. Run `twine check`.
4. Inspect package contents for private data or generated reports.
5. Install from TestPyPI and verify:

   ```python
   import importlib.metadata as md
   import quantem.gpu

   assert md.version("quantem.gpu") == quantem.gpu.__version__
   ```

## Do not regress

- Do not move GPU decompression back into widget.
- Do not make SSB depend on anywidget.
- Do not use `quantem.cuda` as the public package name.
- Do not treat CPU fallback speed as acceptable for GPU workflows.
- Do not use fast-mode SSB as parity evidence.
- Do not copy `MetalImageFFT` or `Native4DSTEMIO` source into Live4DSTEM.
- Do not add a local Python FFT or HDF5 helper to the Mac app.
