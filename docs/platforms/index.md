# Choose a runtime

The scientific operation determines what must be computed; the runtime page
explains how to implement it efficiently on a particular device.

| You are developing for | Start here | Build and memory model |
|---|---|---|
| NVIDIA GPU | [CUDA](cuda.md) | Python/CuPy with CUDA kernels and dedicated VRAM |
| Apple GPU from Python | [Python MPS](mps.md) | Python adapters over MLX/PyObjC/Metal and unified memory |
| Native Apple client/library | [Native Swift and Metal](swift-metal.md) | SwiftPM products, Metal resources, and unified memory |
| Browser GPU | [WebGPU](webgpu.md) | TypeScript/WGSL, browser security, and explicit GPU buffers |
| Independent adjudication | [CPU reference](cpu-reference.md) | deterministic small NumPy/reference implementations |
| Linux CUDA host or service | [Linux CUDA service](remote-cuda.md) | local loopback or SSH-tunneled access with per-device admission |

Before working in a platform folder, read the corresponding page under
[Scientific kernels](../kernels/index.md). The platform may optimize layout,
fusion, queueing, and transfers; it must preserve the operation's
`(row, column) ≡ (r, c)` contract and provenance.

`backend="auto"` is suitable for ordinary Python use. Tests and benchmarks
select a runtime explicitly so missing hardware and unsupported paths fail
honestly. The complete implementation/evidence status is maintained in
[Current verified results](../backends.md).

The [CUDA](cuda.md) page describes in-process computation. The
[Linux CUDA service](remote-cuda.md) page describes how to expose that same
backend through a versioned service. “Remote” is a client relationship, not a
separate scientific backend.
