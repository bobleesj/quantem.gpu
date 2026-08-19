# quantem.gpu

One scientific GPU contract for electron microscopy—from compressed detector
data to exact products—across NVIDIA CUDA, Apple MPS/Metal, native Swift,
WebGPU, and an explicit CPU reference.

```{admonition} Choose how you want to enter
:class: tip
**Scientific operation:** start with [Scientific kernels](kernels/index.md).

**Runtime implementation:** start with [Backend implementation](platforms/index.md).

**Correctness or speed claim:** start with
[Verification and performance](performance/index.md).
```

## The shared coordinate contract

Every backend interprets 4D-STEM data as

$$
I[r_y,r_x,q_y,q_x],
\qquad (\text{row},\text{column})\equiv(y,x),
$$

where $\mathbf r=(r_y,r_x)$ is the scan coordinate and
$\mathbf q=(q_y,q_x)$ is the detector coordinate. A private device layout may
be flattened, transposed, tiled, packed, or detector-major, but the public
shape, masks, metadata, and results preserve this meaning.

Read [Data model and coordinates](kernels/data-model.md) before implementing a
new kernel.

## Find the operation you are implementing

| Operation | Meaning | Kernel page |
|---|---|---|
| Load/decode/bin | compressed source to typed resident counts | [Load, decode, and bin](kernels/load-decode-bin.md) |
| Virtual detector | BF/DF/ADF and mean-diffraction reductions | [BF, DF, and ADF](kernels/virtual-detectors.md) |
| Detector moments | CoM row/y, CoM column/x, DPC, and iDPC | [CoM, DPC, and iDPC](kernels/com-dpc-idpc.md) |
| Ptychography | SSB object, phase, loss, and aberrations | [Single-sideband ptychography](kernels/ssb.md) |
| Scan selection | explicit half-open real-space subsets | [Explicit scan regions](kernels/scan-regions.md) |
| Presentation math | ranges, histograms, colormaps, FFT views, movies | [Display and export kernels](kernels/display-export.md) |

Each page combines the scientific equations, exactness and provenance rules,
optimization model, backend source map, and parity gate. This keeps the math
beside the operation instead of separating it into a generic tutorial.

## Choose the runtime you are implementing

| Runtime | Start here | Primary implementation boundary |
|---|---|---|
| CUDA | [CUDA](platforms/cuda.md) | Python adapters, CuPy, CUDA C/RawKernel |
| Python MPS | [Python MPS](platforms/mps.md) | Python adapters, MLX/PyObjC, Metal kernels |
| Native Swift/Metal | [Native Swift and Metal](platforms/swift-metal.md) | SwiftPM products and bundled Metal resources |
| WebGPU | [WebGPU](platforms/webgpu.md) | TypeScript adapters and WGSL resources |
| CPU reference | [CPU reference](platforms/cpu-reference.md) | independent NumPy/reference implementation |

All runtimes implement the same operation contract. They do not expose
platform-specific scientific workflows.

## What belongs in this package

`quantem.gpu` owns reusable accelerated IO, math, kernels, result contracts,
resource estimation, and scientific provenance. A consuming application owns
presentation, user-visible resource-policy choices, scheduling, and lifecycle.
No application framework or view state is required to build or test this
package.

Read [Kernel architecture](concepts/kernel-architecture.md) for the source tree
and [Kernel development lifecycle](developer/kernel-lifecycle.md) before adding
an implementation.

## Performance numbers are evidence

The [Performance and parity](performance/index.md) section keeps current and
historical measurements with source revision, hardware, data shape/dtype,
cache state, load plan, memory peak, parity artifact, and benchmark definition.

A cached reopen is not a first source load. A cropped or binned fixture is not
full-resolution evidence. A compile test is not a hardware benchmark. Rejected
experiments remain recorded so kernel developers can avoid repeating known
regressions.

## Start coding

Install the runtime you need, run the smallest relevant parity test, then use
the physical target device for performance evidence:

```bash
python -m pip install -e ".[dev,docs]"
PYTHONPATH=src python -m pytest -q
swift test
```

See [Install](install.md), [API contracts](api/index.md), and
[Contributing](developer/index.md).

## Citing and support

If this package contributed to your research, see
[CITATION.cff](https://github.com/bobleesj/quantem.gpu/blob/main/CITATION.cff).
Questions and bug reports belong in the
[issue tracker](https://github.com/bobleesj/quantem.gpu/issues).
