# QuantEM.GPU Remote

QuantEM.GPU Remote exposes the existing CUDA implementation to another
process while keeping raw 4D-STEM data and full-volume computation on a Linux
CUDA host. It is a deployment and communication layer, not another kernel
runtime or numerical implementation.

The product name is **QuantEM.GPU Remote**. Its reproducible Conda environment
is `quantem-gpu-remote`, and its command remains `quantem-gpu serve`.

## Choose the page you need

| Goal | Page |
|---|---|
| Create and verify the service environment | [Deploy a Linux CUDA host](deployment.md) |
| Connect from the same host or another machine | [Connect locally or over SSH](connect.md) |
| Implement a client or evolve the wire contract | [Protocol and integration](protocol.md) |
| Understand GPU selection, memory fit, and eviction | [GPU admission and residency](admission.md) |
| Change CUDA kernels used by the service | [CUDA kernel implementation](../platforms/cuda.md) |

## Component boundary

```text
client process
    │  versioned requests and typed image payloads
    ▼
QuantEM.GPU Remote          src/quantem/gpu/remote/
    │  public QuantEM.GPU calls
    ▼
CUDA implementation         src/quantem/gpu/{io,detector,dpc,ssb}/
    │
    ▼
resident CUDA data and products
```

The service owns catalog discovery, source readiness, CUDA admission, resident
dataset reuse, and response provenance. A client owns presentation,
interaction, and user-visible policy. The service does not import a client UI
framework.

## Scientific invariants

Every response preserves source identity, scan and detector shape, source and
output dtype, half-open regions, scan and detector bins, backend, device, and
implementation revision. Detector binning is explicit and count-preserving.
A cropped or binned result is never represented as native resolution.

Raw detector data stays on the CUDA host unless a requested endpoint explicitly
returns a selected diffraction payload. Missing capacity or unsupported work
returns a typed failure; it never causes an implicit CPU fallback, crop, bin,
or precision change.
