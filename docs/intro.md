# quantem.gpu

`quantem.gpu` is the multi-backend accelerated STEM IO and compute package for
QuantEM. It owns the parts that are expensive, backend-specific, or data-path
critical:

- HDF5 master/chunk IO, bitshuffle/LZ4 decompression, chunk assembly, and
  load-to-device paths.
- BF, DF, ADF, mean diffraction pattern, CoM/DPC, and iDPC image computation.
- SSB compute APIs that can be called by `quantem.live`, scripts, and future
  widget workflows.
- Device policy for `cuda`, `mps`, and `cpu` with explicit errors.

`quantem.widget` remains the front end: anywidget views, interactions, exports,
and display. Widget load and compute call sites should route through
`quantem.gpu` instead of keeping long-term duplicate GPU loaders or math.

The intended dependency arrow is:

```text
file -> quantem.gpu (load + decompress + to_device) -> arrays
     -> quantem.gpu (BF/DF/DPC / SSB / movies) -> quantem.widget (display)
```

## Current release candidate

The current release candidate used by the widget display migration branch is
`quantem.gpu==0.0.1rc5`.

```bash
python -m pip install \
  --extra-index-url https://test.pypi.org/simple/ \
  "quantem.gpu==0.0.1rc5"
```

These docs are intentionally compute-facing. If you want interactive viewers,
load with `quantem.gpu` and display the result with `quantem.widget`.

## Agent quick rules

When using or extending this package, keep these rules first:

- Use `quantem.gpu.load(...)` for 4D-STEM HDF5 load/decompress work. Use
  `scan_region=(row_start, row_stop, col_start, col_stop)` for crop-first IO.
- Put reusable CUDA, MPS/Metal, and WebGPU kernels in `quantem.gpu`, then have
  `quantem.widget` bundle or call them. Widget should keep UI, display, and
  export orchestration.
- Before claiming a backend is fast, record backend, hardware, shape, dtype,
  BF policy or mask, parity metric, stage timing, and memory footprint.
- Do not count hidden crop, hidden binning, reduced BF pixels, saved derived
  caches, CPU fallback, or software WebGPU adapters as backend parity.
- Update the backend feature matrix and maintainer checklist whenever a kernel
  moves from `Gap` or `Partial` to `Done`.
