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
- BF/DF/ADF has a GPU-resident `maskedSumBuffer` path, including cached
  full-detector total minus complement for dense DF/ADF masks.
- CoM/DPC has GPU-resident `maskedCoMBuffer` and `maskedDpcBuffer` source
  paths. It still needs headed real-adapter agreement/performance signoff before
  it is considered equivalent to the CUDA/MPS interaction path.
- Browser HDF5 source mode currently fetches whole HDF5 data files before
  parsing/decode. Future WebGPU IO work should add range-read or sidecar
  chunk-index loading before making full-size `512` or `1024` performance
  claims.
