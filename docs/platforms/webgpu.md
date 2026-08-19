# WebGPU

WebGPU provides reusable browser GPU implementations through TypeScript and
WGSL. It is not a Python backend name and is not “Metal” even when a browser
maps WebGPU to Metal internally.

## Dispatch and implementation layers

| Operation | TypeScript/WGSL source |
|---|---|
| Local HDF5 read/decode | `src/quantem/gpu/io/backends/webgpu` |
| BF/DF/ADF and moments | `src/quantem/gpu/detector/compute/webgpu` |
| DPC/iDPC | `src/quantem/gpu/dpc/compute/webgpu` |
| SSB | `src/quantem/gpu/ssb/compute/webgpu` |
| Display statistics/FFT/color | `src/quantem/gpu/display/webgpu` |

The browser call path is:

```text
client bundles canonical TypeScript resources
  → device/webgpu.ts acquires and monitors a GPUDevice
  → IO worker parses local HDF5/block metadata
  → WGSL decode and scientific kernels write GPUBuffer results
  → only requested small arrays or display buffers are read back
```

The IO implementation is split deliberately: `h5reader.ts` parses source and
block metadata, `local-h5.ts` owns browser file handles/workers, and
`bslz4.ts` owns GPU decode/upload variants. Detector geometry is shared in
`detector/geometry.ts`; scientific dispatch lives beside each domain rather
than in a browser UI.

These files are package resources. A browser client bundles the canonical
sources rather than maintaining a second scientific implementation.

## Execution and memory model

Browser file access, worker parsing, queue writes, GPU decode, reductions,
readback, and presentation are distinct stages. Keep detector data and derived
products in `GPUBuffer` objects across compatible kernels. Read back only the
requested small result or parity artifact. Account for browser buffer limits,
alignment, adapter limits, and temporary staging buffers.

Local-file security may require worker or file-handle paths unavailable to a
normal network request. Those acquisition details must not change
$I[R_r,R_c,k_r,k_c]$ or `(row, column) ≡ (r, c)`.

## Source and build checks

```bash
PYTHONPATH=src python -m pytest -q \
  tests/test_webgpu_sources.py \
  tests/test_webgpu_widget_sync.py
```

The source test verifies packaged resources and required kernel contracts. A
consumer-sync test verifies byte identity when a consumer checkout is supplied.
Neither test proves that a physical browser adapter executed the WGSL.

## Profiling and acceptance

Distinguish source presence/build, software-adapter smoke, real hardware
adapter parity, first local-file load, warm interaction, and prepared sidecar
paths. Only real-adapter runs are hardware evidence. Record browser/version,
adapter/device, source bytes, read/parse/upload/decode/compute/readback/present
intervals, output checksum, and load plan.

Integer corrected-frame, bin, mask, histogram, and RGBA outputs are byte-exact
where formats match. Large-source capability requires a physical browser run;
TypeScript compilation alone is not signoff.
