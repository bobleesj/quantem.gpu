# quantem.gpu

`quantem.gpu` is the multi-backend accelerated STEM IO and compute package for
QuantEM. It owns the parts that are expensive, backend-specific, or data-path
critical:

- HDF5 master/chunk IO, bitshuffle/LZ4 decompression, chunk assembly, and
  load-to-device paths.
- Virtual detector products such as BF, DF, ADF, mean diffraction patterns,
  CoM/DPC, and iDPC.
- SSB compute APIs that can be called by `quantem.live`, scripts, and future
  widget workflows.
- Device policy for `cuda`, `mps`, and `cpu` with explicit errors.

`quantem.widget` remains the front end: anywidget views, interactions, exports,
and display. Widget load and compute call sites should route through
`quantem.gpu` instead of keeping long-term duplicate GPU loaders or math.

The intended dependency arrow is:

```text
file -> quantem.gpu (load + decompress + to_device) -> arrays
     -> quantem.gpu (products / SSB) -> quantem.widget (display)
```

## Current release candidate

The current release candidate used by the widget migration branch is
`quantem.gpu==0.0.1rc4`.

```bash
python -m pip install \
  --extra-index-url https://test.pypi.org/simple/ \
  "quantem.gpu==0.0.1rc4"
```

These docs are intentionally compute-facing. If you want interactive viewers,
load with `quantem.gpu` and display the result with `quantem.widget`.
