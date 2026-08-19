# Choose a platform

Choose a runtime based on where the data and application live. Do not choose a
backend by changing scientific parameters until the workload happens to fit.

| Platform | Use it when | Start here |
|---|---|---|
| CUDA | Data lives on an NVIDIA workstation/server or the workload needs large dedicated VRAM | [CUDA](cuda.md) |
| Python MPS | A Python workflow runs locally on Apple Silicon | [Python MPS](mps.md) |
| Native Swift/Metal | A macOS or iOS application needs native IO and reusable Metal kernels | [Native Swift and Metal](swift-metal.md) |
| WebGPU | Compute runs in a browser or portable exported HTML | [WebGPU](webgpu.md) |
| CPU reference | A small deterministic reference is needed for adjudication | [CPU reference](cpu-reference.md) |

`backend="auto"` is appropriate for ordinary Python workflows. Tests,
benchmarks, and parity reports select the backend explicitly so unsupported
hardware fails honestly.

The complete implementation and evidence status is maintained in
[Current verified results](../backends.md).
