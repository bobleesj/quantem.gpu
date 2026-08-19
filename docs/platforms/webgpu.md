# WebGPU

WebGPU provides reusable browser GPU implementations through TypeScript and
WGSL. It is not a Python backend name and is not “Metal” even when a browser
maps WebGPU to Metal internally.

## Source map

| Operation | TypeScript/WGSL source |
|---|---|
| Local HDF5 read/decode | `src/quantem/gpu/io/backends/webgpu` |
| BF/DF/ADF and moments | `src/quantem/gpu/detector/compute/webgpu` |
| DPC/iDPC | `src/quantem/gpu/dpc/compute/webgpu` |
| SSB | `src/quantem/gpu/ssb/compute/webgpu` |
| Display statistics/FFT/color | `src/quantem/gpu/display/webgpu` |

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
$I[r_y,r_x,q_y,q_x]$ or `(row, column) ≡ (y, x)`.

## Profiling and acceptance

Distinguish source presence/build, software-adapter smoke, real hardware
adapter parity, first local-file load, warm interaction, and prepared sidecar
paths. Only real-adapter runs are hardware evidence. Record browser/version,
adapter/device, source bytes, read/parse/upload/decode/compute/readback/present
intervals, output checksum, and load plan.

Integer corrected-frame, bin, mask, histogram, and RGBA outputs are byte-exact
where formats match. Large-source capability requires a physical browser run;
TypeScript compilation alone is not signoff.
