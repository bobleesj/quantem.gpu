# Choose a kernel runtime

The scientific operation determines what must be computed; the runtime page
explains how to implement it efficiently on a particular device.

| You are implementing for | Start here | Build and memory model |
|---|---|---|
| NVIDIA GPU | [CUDA](cuda.md) | Python/CuPy with CUDA kernels and dedicated VRAM |
| Apple GPU from Python | [Python MPS](mps.md) | Python adapters over MLX/PyObjC/Metal and unified memory |
| Native Apple client/library | [Native Swift and Metal](swift-metal.md) | SwiftPM products, Metal resources, and unified memory |
| Browser GPU | [WebGPU](webgpu.md) | TypeScript/WGSL, browser security, and explicit GPU buffers |
| Independent adjudication | [CPU reference](cpu-reference.md) | deterministic small NumPy/reference implementations |

## Shared implementation shape

Each runtime follows the same layered contract:

```text
public scientific API
    ↓
device and operation dispatcher
    ↓
runtime adapter
    ↓
kernel, shader, or reference operation
    ↓
typed result and scientific provenance
```

The public API owns coordinates, shapes, dtypes, units, errors, and provenance.
The dispatcher selects only an explicitly available implementation. The
runtime adapter owns allocation, layout, compilation, queueing, and
synchronization. Kernels may optimize those private details but must return the
same scientific result.

## Repository map by layer

| Runtime | Discovery/dispatch | IO/decode | Detector and DPC | Reconstruction/display | Primary tests |
|---|---|---|---|---|---|
| CUDA | `device/backend.py`, operation protocols | `io/backends/cuda` | `detector/compute/cuda`, `dpc/compute/cuda` | `ssb/compute/cuda`, `display/cuda.py` | `test_cuda_*`, `test_realdata_parity.py` |
| Python MPS | `device/backend.py`, operation protocols | `io/backends/mps` | `detector/compute/mps`, `dpc/compute/mps` | `ssb/compute/mps` | `test_mps_*`, MPS sections of parity tests |
| Swift/Metal | SwiftPM products in `Package.swift` | `Native4DSTEMIO`, `Metal4DSTEMKernels` | `Metal4DSTEMKernels` | `MetalImageFFT`, `MetalDisplayKernels`, `MetalImageRuntime` | `src/quantem/gpu/swift/Tests` |
| WebGPU | `device/webgpu.ts` and TypeScript adapters | `io/backends/webgpu` | detector/DPC WebGPU modules | SSB and display WebGPU modules | `test_webgpu_*`, browser hardware gates |
| CPU reference | explicit `backend="cpu"` | `io/backends/cpu/reference.py` | NumPy paths in detector/DPC workflows | small independent reference fixtures | product/parity tests |

Open the runtime page for exact source files, call path, memory behavior,
profiling boundaries, build commands, and acceptance gates.

Before working in a platform folder, read the corresponding page under
[Scientific kernels](../kernels/index.md). The platform may optimize layout,
fusion, queueing, and transfers; it must preserve the operation's
`(row, column) ≡ (r, c)` contract and provenance.

`backend="auto"` is suitable for ordinary Python use. Tests and benchmarks
select a runtime explicitly so missing hardware and unsupported paths fail
honestly. Capability status is maintained in [Backend coverage](../backends.md);
current measurements live in
[Verified benchmark results](../performance/results.md).

Serving CUDA to another process is not a kernel implementation. See
[QuantEM.GPU Remote](../remote/index.md) for local loopback and SSH-tunneled
deployment.
