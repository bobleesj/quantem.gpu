# quantem.gpu WebGPU Sources

This directory is the canonical source home for reusable QuantEM WebGPU
browser-compute code.

`quantem.widget` still bundles the code into anywidget JavaScript and exported
HTML, because browsers cannot import Python package modules directly. The
ownership rule is:

```text
quantem.gpu.webgpu source -> quantem.widget bundle/export -> browser runtime
```

Keep heavy reusable WGSL/TypeScript kernels here when they implement shared
4D-STEM compute such as:

- HDF5 bitshuffle/LZ4 browser decode
- BF/DF/ADF virtual-image reductions
- CoM/DPC/iDPC reductions
- ptychographic SSB browser kernels

Widget-specific UI, React state, controls, layout, and export orchestration
stay in `quantem.widget`.

Current status:

- `compute.ts` contains the Show4DSTEM WebGPU virtual-image engine, including
  selected-index BF/DF/ADF and masked CoM kernels.
- `showptycho-ssb.ts` contains the ShowPtycho WebGPU SSB engine and imports
  shared device, HDF5, and bitshuffle/LZ4 helpers from this package.
- BF/DF/ADF has a GPU-resident `maskedSumBuffer` path in the widget fork.
- CoM/DPC still needs a matching GPU-resident `maskedCoMBuffer` path and
  headed real-adapter parity/performance signoff before it is considered
  equivalent to the CUDA/MPS interaction path.
