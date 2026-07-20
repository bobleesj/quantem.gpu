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
- `local-h5.ts` contains the Show4DSTEM local-file HDF5 acquisition path. When
  the browser is given the local master/data files, it reads them with a classic
  Blob worker pool, parses the HDF5 chunk index, and feeds the same WGSL
  bitshuffle/LZ4 decoder as the URL path.
- Maintainers can prebuild optional metadata-only block-index sidecars with:

  ```bash
  python scripts/build_webgpu_h5_block_index_sidecar.py \
    --frame-index-json frame_index_manifest.json \
    --input-dir /path/to/local/h5/files \
    --output-dir /path/to/local/h5/files/block_index \
    --data-template 'data_{index:06d}.h5' \
    --output-template 'data_{index:06d}.qh5idx' \
    --data-files 27
  ```

  The `.qh5idx` files store block offsets/lengths only. They do not contain raw
  detector frames or LZ4 payloads, and the browser still decodes the original
  HDF5 data files.
- For count-audited browse data where bad-pixel-corrected detector counts are
  proven to fit in `uint8`, the local HDF5 path uses the lossless low8
  frame-cooperative decoder with a staging pipeline. The current default uses
  `wg32`, full-load local-file group size `8`, and worker count `8` for
  metadata-accelerated full loads, while small scan-region crops keep worker
  count `2` and group size `4`. A frame-index manifest can move bslz4 metadata
  parsing into the worker; an optional `QH5IDX01` block-index sidecar goes
  further by storing only deterministic block offsets/lengths so the browser
  does not walk every bslz4 block header on each load. On Apple WebGPU this cut
  the full `512x512x192x192` local-file load from about `3.78 s` to a
  946-cycle soak median of `0.772 s` by page profile time, with range
  `0.726-0.879 s`, real adapter `apple metal-3`, and checksum parity. The
  tested count audit had max unmasked count `57` and zero unmasked pixels above
  `255`.
- The full-load browser path has a corrected-frame checksum gate against CUDA:
  first/middle/last detector frames match after bad-pixel correction, without
  pulling the whole `9.7 GB` decoded stack back to JavaScript.
- The full-stack local-H5 loader is scan-region aware at the lower-level API:
  it slices native bslz4 frame windows before WebGPU upload/decode and uses a
  frame-index manifest to skip data files that cannot intersect the requested
  crop. With the block-index sidecar present, a true `256x256x192x192` crop
  from full local evidence had a 946-cycle soak median of `0.338 s` by page
  profile time, with range `0.316-0.464 s` and exact corrected-frame checksum
  parity against CUDA. This is not wired into the Show4DSTEM display path until
  the frontend also switches to the cropped scan shape.
- `local-h5.ts` also exposes a product-first masked-sum path for BF/DF/ADF-style
  virtual images. It reads local HDF5 data, builds only the requested
  scan-region product, and returns a GPU-resident float32 image buffer without
  materializing the resident `9.7 GB` stack. On real Chrome/Apple WebGPU with
  BF radius `30`, repeated parity gates against CUDA references were exact:
  `256x256` scan crop median about `0.78 s` wall / `0.21 s` product stage,
  full `512x512` median about `1.03 s` wall / `0.51 s` product stage, and a
  `1024x1024` repeat-stress gate about `4.08 s` wall / `1.80 s` product stage.
  The `1024` number exercises a 1,048,576-position output/dispatch using
  repeated real `512` evidence; it is not a true 1024-acquisition signoff.
- The selected-block sidecar path is the current best architecture for
  CUDA-like browser product loading. It stores the exact native bslz4/LZ4
  streams for only the detector bitshuffle blocks touched by the product mask,
  then reuses the same WGSL decoder. The production local-H5 masked-sum API
  discovers these sidecars, verifies detector-block coverage, prefilters
  non-intersecting sidecars for scan crops, and falls back to native HDF5 when
  coverage is missing. For BF radius `30` on real Chrome/Apple WebGPU, exact
  parity (`max_abs=0`) against CUDA was measured with the current auto
  direct-float/staging-pipeline reducer in a 946-cycle soak. Median page totals
  were `0.210 s` for a true `256x256` crop, `0.378 s` for full `512x512`, and
  `1.170 s` for a `1024x1024` repeat-stress gate. The `1024` number exercises a
  1,048,576-position output/dispatch using repeated real `512` evidence; it is
  not a true 1024-acquisition signoff.
- The selected-block reducer has separate grouped-mask, `pixel-wg64`,
  `pixel-wg128`, and `pixel-wg256` kernels for profiling. On Apple WebGPU, the
  production auto route uses pixel `wg64` for `<=256x256` outputs, grouped-mask
  `wg64` above that, and a compact shared-memory grouped-mask route above
  `512x512`. Set
  `globalThis.__QT_BSLZ4_MASKED_SUM_GROUPMASK=true/false` to force a route, and
  `globalThis.__QT_BSLZ4_MASKED_SUM_COMPACT_SHARED=true/false` to force the
  large-scan shared-memory layout, and
  `globalThis.__QT_BSLZ4_MASKED_SUM_PIPELINE=false` to disable the staging
  pipeline during diagnostics.
- The remaining full-load bottleneck is compressed-byte upload plus WGSL
  bslz4 decode/GPU wait, not file reads or HDF5 parse. The block-index sidecar
  reduces full-load parse to about `0.005 s`, but the strict full-stack path
  still uploads about `3.17 GB` of compressed bytes and materializes the `9.7 GB`
  browse cube. A full low8-prefix audit found the useful prefix is still about
  `99.55%` of the native compressed byte stream, so the next large win needs a
  more efficient decoder/upload strategy or a selected-evidence browser payload
  contract, not simple high-bitplane trimming.
- Current off-default WebGPU IO experiments are intentionally kept behind
  runtime flags. Chunked `queue.writeBuffer` upload preserves parity but is
  slower than staging; a single-lane token parser preserves parity only with a
  fixed uniform loop and is also slower; grouping two frames per workgroup is
  parity-clean but not a robust repeatable win over the default one-frame
  schedule; `decodeBatch=2` slightly reduced page profile but increased wall
  time and upload pressure; a packed-word shared-memory low8 decoder preserved
  parity but was slower than the default atomic packed-byte decoder. A
  zero-literal-barrier skip variant and packed decoded-output group buffers both
  preserved parity but regressed, so they were removed from production source. The
  full-stack frame-cooperative bslz4 decoder now defaults to `wg32`; corrected
  parity/timing sweeps showed `wg8`, `wg16`, `wg64`, and `wg128` are slower on
  Apple WebGPU.
