# Kernel architecture

`quantem.gpu` is organized by **scientific operation first** and runtime
second. A developer starts from the operation being implemented, not from a
platform folder and not from a user interface.

```text
scientific contract
    -> backend-neutral public API and result type
        -> CUDA | Python MPS | Swift/Metal | WebGPU | CPU reference
            -> parity bundle and benchmark record
```

This structure prevents four optimized implementations from drifting into four
different definitions of the science.

## Find the code by operation

| Operation | Python contract | Accelerator implementations | Native implementation |
|---|---|---|---|
| Load, decode, crop, and bin | `src/quantem/gpu/io` | `io/backends/{cuda,mps,webgpu}` | `Native4DSTEMIO`, `Metal4DSTEMKernels` |
| BF/DF/ADF and mean diffraction | `src/quantem/gpu/detector` | `detector/compute/{cuda,mps,webgpu}` | `Metal4DSTEMKernels` |
| CoM, DPC, and iDPC | `src/quantem/gpu/dpc` | `dpc/compute/{cuda,mps,webgpu}` | `Metal4DSTEMKernels`, `MetalImageFFT` |
| Display statistics and transforms | `src/quantem/gpu/display` | `display/webgpu` and backend modules | `MetalDisplayKernels`, `MetalImageRuntime` |
| Single-sideband ptychography | `src/quantem/gpu/ssb` | `ssb/compute/{cuda,mps,webgpu}` | backend-specific compute library |

The current `compute` folder name is an internal compatibility boundary. New
public APIs belong to the scientific domain; consumers must not import a
backend module directly.

## Read the docs in two directions

If you are implementing a **scientific operation**, start in
[Scientific kernels](../kernels/index.md). Each page defines the equations,
array axes, exactness rules, reusable optimization opportunities, source map,
and parity gate.

If you are implementing a **runtime**, start in
[Kernel implementations](../platforms/index.md). Each platform page explains
its memory model, source locations, build commands, profiling tools, and the
same cross-backend acceptance boundary.

## What can change behind the contract

Backends may change memory layout, tiling, thread topology, chunk size, queue
depth, buffer reuse, fusion, and caching of prepared state. Backends may not
silently change:

- `(row, column) ≡ (r, c)` axis meaning;
- scan or detector coverage;
- detector or scan binning;
- masks, bad-pixel treatment, or calibration;
- source, accumulation, or output dtype;
- reconstruction objective; or
- provenance describing any of the above.

## One reviewable kernel lifecycle

Every optimization follows the same path:

1. freeze a backend-independent reference and fixture;
2. measure the existing end-to-end stage breakdown;
3. change one topology or memory assumption;
4. compare exact arrays or frozen floating-point metrics;
5. profile on the physical target device;
6. record cold, warm, and prepared/cache-reopen results separately; and
7. update the parity matrix and evidence ledger.

See [Kernel development lifecycle](../developer/kernel-lifecycle.md) for the
review checklist and [Benchmark methodology](../performance/methodology.md)
for the evidence schema.
